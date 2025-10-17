from __future__ import annotations

"""
Harmonizer for descriptor;explainer pairs using:
- Stella embeddings (Marqo/dunzhang-stella_en_400M_v5)
- FAISS ANN search over the Schema
- vLLM with guided JSON decoding to choose the best synonym (or none)
"""

# Standard library
import argparse
from dataclasses import dataclass, field
import functools
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import hashlib
import io

# Third‑party
import faiss  # type: ignore
import json_repair  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from pydantic import BaseModel  # type: ignore
from sqlitedict import SqliteDict  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def setup_logging(log_file: Path, verbosity: int = 1) -> None:
    """Configure both file and stdout logging, creating the directory if needed."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(
            "Execution of %s took %s.",
            func.__name__,
            time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        )
        return result

    return wrapper

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Pair:
    id: str
    descriptor: str
    explainer: str

    @property
    def text(self) -> str:
        d = (self.descriptor or "").strip()
        e = (self.explainer or "").strip()
        if d and e:
            return f"{d}; {e}"
        return d or e

@dataclass
class InputRow:
    row_idx: int
    raw: Any           # original parsed JSON object or raw string
    pairs: List[Pair]  # descriptor/explainer items for this row

@dataclass
class Decision:
    chosen_id: Optional[str]  # schema id selected by LLM; None => drop


class VerdictSchema(BaseModel):
    is_synonym: bool
    chosen_id: str  # empty string if none


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _split_descriptor_explainer(s: str) -> Tuple[str, str]:
    if ";" in s:
        left, right = s.split(";", 1)
        return left.strip(), right.strip()
    # If no semicolon, treat the whole string as descriptor
    return s.strip(), ""


def load_schema(path: Path) -> List[Pair]:
    pairs: List[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(Pair(id=obj["id"], descriptor=obj["descriptor"], explainer=obj["explainer"]))
    return pairs


def load_descriptors(path: Path) -> List[InputRow]:
    """Load input JSONL and extract descriptor/explainer pairs per row.

    Expected input schema (per line):
      {
        "similarity": [float, ...],
        "descriptors": [ ["desc; explainer", ...], ... ]
      }
    We take the descriptors list at the argmax(similarity) index.
    """
    rows: List[InputRow] = []
    with path.open("r", encoding="utf-8") as f:
        for doc_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            best_idx = int(np.argmax(obj["similarity"]))
            desc_exp = obj["descriptors"][best_idx]

            pairs_for_row: List[Pair] = []
            for desc_i, d_e in enumerate(desc_exp):
                d, e = _split_descriptor_explainer(d_e)
                pid = f"{doc_i}_{desc_i}"
                pairs_for_row.append(Pair(id=pid, descriptor=d, explainer=e))

            rows.append(InputRow(row_idx=doc_i, raw=obj, pairs=pairs_for_row))

    return rows

# -----------------------------------------------------------------------------
# Embeddings (Stella)
# -----------------------------------------------------------------------------

class StellaEmbedder:
    """Minimal pooled embedding wrapper around Marqo/dunzhang-stella_en_400M_v5."""

    def __init__(self, cache_dir: Optional[Path], batch_size: int = 32, device: str = "cuda:0") -> None:
        cache_dir = os.environ.get("HF_HOME") or (cache_dir if cache_dir else None)
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        self.device = torch.device(device)
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            .to(self.device)
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.batch_size = batch_size

    @log_execution_time
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[i : i + self.batch_size])
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)[0]  # [B, T, H]
                attn = inputs["attention_mask"]  # [B, T]
                masked = outputs.masked_fill(~attn[..., None].bool(), 0.0)
                pooled = masked.sum(dim=1) / attn.sum(dim=1)[..., None]
                
                # Normalize to unit length for distance measuring
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.detach().cpu().numpy().astype("float32"))
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 1024), dtype="float32")
    

def _save_schema_embeds(path: Path, emb: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    np.savez_compressed(str(path), emb=emb, meta=np.frombuffer(meta_bytes, dtype=np.uint8))

def _load_schema_embeds(path: Path) -> Tuple[np.ndarray, dict]:
    with np.load(str(path), allow_pickle=False) as z:
        emb = z["emb"].astype("float32", copy=False)
        meta_bytes = bytes(z["meta"].tolist())
        meta = json.loads(meta_bytes.decode("utf-8"))
        return emb, meta

def _schema_fingerprint(pairs: Sequence[Pair], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for p in pairs:
        # include ids + text so any change invalidates cache
        h.update(b"\x00"); h.update(p.id.encode("utf-8", "ignore"))
        h.update(b"\x00"); h.update((p.descriptor or "").encode("utf-8", "ignore"))
        h.update(b"\x00"); h.update((p.explainer or "").encode("utf-8", "ignore"))
    return h.hexdigest()

# -----------------------------------------------------------------------------
# FAISS index
# -----------------------------------------------------------------------------

class FaissIndex:
    def __init__(self, nlist: int = 100, nprobe: int = 10):
        self.nlist = nlist
        self.nprobe = nprobe

    @log_execution_time
    def build(self, embeddings: np.ndarray) -> faiss.Index:
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        d, n = embeddings.shape[1], embeddings.shape[0]

        # Fallback to a flat index for small n or if IVF training fails
        if n < max(2 * self.nlist, 100):
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            return index

        try:
            # Build index with Inner-Product quantizer
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = 42  # for reproducibility
            
            # Train on sample of embeddings
            rng = np.random.default_rng(42) # Set seed for reproducibility
            train_sample = embeddings[rng.choice(n, size=min(100_000, n), replace=False)]
            index.train(train_sample)
            ids = np.arange(n, dtype=np.int64)
            index.add_with_ids(embeddings, ids)
            index.nprobe = self.nprobe
            index.make_direct_map()
            return index
        except Exception as exc:
            logging.warning("FAISS IVF build failed (%s); falling back to IndexFlatIP", exc)
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            return index

    @log_execution_time
    def neighbors(self, index: faiss.Index, embeddings: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("k must be >= 1")
        
        # Search for k+1 neighbors, so we can afford to drop self
        sims, idxs = index.search(embeddings, k + 1)  # candidate pool
        n = idxs.shape[0]
        out_idx = np.empty((n, 0), dtype=idxs.dtype)
        out_sim = np.empty((n, 0), dtype=sims.dtype)
        for i in range(n):
            row_idx = idxs[i]
            row_sim = sims[i]
            # drop self if present
            # Self might not be among neighbors in IVF
            mask = row_idx != i  
            row_idx = row_idx[mask]
            row_sim = row_sim[mask]
            # pad in case self wasn't present and we still need only top-k
            out_idx = np.vstack([out_idx, row_idx[:k][None, :]]) if out_idx.size else row_idx[:k][None, :]
            out_sim = np.vstack([out_sim, row_sim[:k][None, :]]) if out_sim.size else row_sim[:k][None, :]
        return out_sim, out_idx

def _save_faiss_index(path: Path, index: faiss.Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def _load_faiss_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))

# -----------------------------------------------------------------------------
# vLLM selector
# -----------------------------------------------------------------------------
class SynonymSelector:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[Path],
        temperature: float = 0.1,
        batch_size: int = 64,
        cache_db: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = os.environ.get("HF_HOME") or (str(cache_dir) if cache_dir else None)
        self.temperature = temperature
        self.batch_size = batch_size
        self._llm: Optional[LLM] = None
        self._cache: Optional[SqliteDict] = SqliteDict(str(cache_db), autocommit=True) if cache_db else None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPU available for vLLM.")
            logging.info("Using %d GPU(s) for vLLM.", n_gpus)
            self._llm = LLM(
                model=self.model_name,
                download_dir=self.cache_dir,
                dtype="bfloat16",
                max_model_len=16384,
                tensor_parallel_size=n_gpus,
                enforce_eager=False,
                gpu_memory_utilization=0.90,
            )
        return self._llm

    @staticmethod
    def _prompt(query: Pair, candidates: List[Pair]) -> str:
        cand_lines = []
        for c in candidates:
            cand_lines.append(f"- id={c.id} :: {c.text}")
        candidates_block = "\n".join(cand_lines)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a careful ontology assistant.\n"
            "Task: Given a query pair in the form 'descriptor; explainer' and a list of schema candidates,\n"
            "pick the single candidate that is synonymous or near‑synonymous in most contexts to the query.\n"
            "If many candidates are synonymous, choose the one that is the closest match.\n"
            "Set chosen_id to exactly the chosen candidate's ID.\n"
            "If none are acceptable, return is_synonym=false and chosen_id=\"\".\n"
            "Output MUST be JSON only.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Query: {query.text}\n\n"
            f"Candidates:\n{candidates_block}\n\n"
            "Return JSON with keys: is_synonym (bool), chosen_id (string; empty if none).\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _response_format(self) -> GuidedDecodingParams:
        return GuidedDecodingParams(json=VerdictSchema.model_json_schema())

    @staticmethod
    def _clean_chosen_id(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        s = str(x).strip().strip("`\"'")
        # common pattern from prompt rendering
        if s.lower().startswith("id="):
            s = s[3:].lstrip()
        return s or None

    @staticmethod
    def _cache_key(query: Pair, candidates: List[Pair]) -> str:
        key = {
            "query": {"id": query.id, "d": query.descriptor, "e": query.explainer},
            "candidates": [{"id": c.id, "d": c.descriptor, "e": c.explainer} for c in candidates],
        }
        return json.dumps(key, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _log_throughput(in_tok, gen_tok, elapsed):
        tot_tok = gen_tok + in_tok
        if elapsed > 0 and tot_tok > 0:
            logging.info(
                "LLM throughput: %.1f tok/s (%.1f gen tok/s) — %s tokens in %.2fs",
                tot_tok / elapsed,
                gen_tok / elapsed if gen_tok else 0,
                tot_tok,
                elapsed,
            )

    @staticmethod
    def _extract_tokens(outputs) -> Tuple[int, int]:
        in_tok = 0
        gen_tok = 0
        for o in outputs:
            if getattr(o, "prompt_token_ids", None):
                in_tok += len(o.prompt_token_ids)
            outs = getattr(o, "outputs", None) or []
            if outs:
                cand = outs[0]
                if getattr(cand, "token_ids", None):
                    gen_tok += len(cand.token_ids)
        return in_tok, gen_tok

    @staticmethod
    def _parse_text_to_decision(text: str) -> Optional[Decision]:
        s = (text or "").strip().strip("` ")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        try:
            obj = json_repair.loads(s)
            is_syn = bool(obj.get("is_synonym", False))
            chosen = str(obj.get("chosen_id", "")).strip() or None
            return Decision(chosen_id=chosen if is_syn else None)
        except Exception as exc:
            logging.warning("Failed to parse JSON verdict (%s): %r", exc, s[:200])
            return None

    @log_execution_time
    def select_batch(self, batch: List[Tuple[Pair, List[Pair]]]) -> List[Decision]:
        prompts: List[str] = []
        out_positions: List[int] = []
        decisions: List[Optional[Decision]] = [None] * len(batch)

        # Fill from cache or queue for inference
        if self._cache is not None:
            for i, (q, cands) in enumerate(batch):
                ck = self._cache_key(q, cands)
                cached = self._cache.get(ck)
                if cached:
                    try:
                        obj = json.loads(cached)
                        decisions[i] = Decision(chosen_id=self._clean_chosen_id(obj.get("chosen_id")))
                    except Exception:
                        pass

        for i, (q, cands) in enumerate(batch):
            if decisions[i] is None:
                prompts.append(self._prompt(q, cands))
                out_positions.append(i)

        if prompts:
            params = SamplingParams(
                temperature=self.temperature,
                top_p=0.9,
                max_tokens=512,
                stop=["<|eot_id|>"],
                guided_decoding=self._response_format(),
                seed=42,
            )
            t0 = time.time()
            outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
            t1 = time.time()
            in_tok, gen_tok = self._extract_tokens(outputs)
            tot = in_tok + gen_tok
            self._log_throughput(in_tok, gen_tok, t1 - t0)
            logging.info("LLM tokens: in=%d, gen=%d, total=%d", in_tok, gen_tok, tot)
            # Map back
            for pos, out in zip(out_positions, outputs):
                text = ""
                outs = getattr(out, "outputs", None) or []
                if outs:
                    text = (outs[0].text or "")
                dec = self._parse_text_to_decision(text) or Decision(chosen_id=None)
                # Remove possible "id=" preamble from model output
                dec.chosen_id = self._clean_chosen_id(dec.chosen_id)
                decisions[pos] = dec
                # Write cache
                if self._cache is not None:
                    q, cands = batch[pos]
                    ck = self._cache_key(q, cands)
                    try:
                        self._cache[ck] = json.dumps({"chosen_id": dec.chosen_id or ""}, ensure_ascii=False)
                    except Exception:
                        pass

        return [d if d is not None else Decision(chosen_id=None) for d in decisions]


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

@log_execution_time
def run(
    input_path: Path,
    schema_path: Path,
    run_id: str,
    llm_name: str,
    topk: int = 5,
    min_embed_score: float = 0.0,
    batch_size_embed: int = 64,
    batch_size_llm: int = 1024,
    faiss_nlist: int = 256,
    faiss_nprobe: int = 32,
    cache_dir: Optional[Path] = None,
    cache_db: Optional[Path] = None,
    verbosity: int = 1,
    schema_embeds: Optional[Path] = None,
    schema_index: Optional[Path] = None,
    rebuild_cache: bool = False,    
) -> None:

    results_dir = Path(f"../results/harmonized/{run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / f"{run_id}.log", verbosity=verbosity)

    logging.info("Loading input and schema JSONL ...")
    input_rows = load_descriptors(input_path)
    # Flatten per-descriptor inputs for embed/search/LLM, keeping a map to regroup
    flat_inputs: List[Pair] = []
    owner_row: Dict[str, int] = {}   # pair.id -> row_idx
    for r in input_rows:
        for p in r.pairs:
            flat_inputs.append(p)
            owner_row[p.id] = r.row_idx

    # Load schema
    schema_pairs = load_schema(schema_path)
    if not schema_pairs:
        logging.warning("Schema is empty; nothing to harmonize.")

    # Build lookup by id
    schema_by_id: Dict[str, Pair] = {p.id: p for p in schema_pairs}

    embedder = StellaEmbedder(cache_dir=cache_dir, batch_size=batch_size_embed)

    # Check that schema and embedding models match previously saved embeddings
    schema_fp = _schema_fingerprint(schema_pairs, model_name="Marqo/dunzhang-stella_en_400M_v5")
    schema_emb: Optional[np.ndarray] = None

    if (not rebuild_cache) and schema_embeds and schema_embeds.exists():
        logging.info("Trying to load schema embeddings from %s", schema_embeds)
        try:
            cached_emb, meta = _load_schema_embeds(schema_embeds)
            if meta.get("fingerprint") == schema_fp:
                logging.info("Loaded cached schema embeddings from %s", schema_embeds)
                schema_emb = cached_emb
            else:
                logging.info("Schema/model fingerprint changed; ignoring cached embeddings")
        except Exception as exc:
            logging.warning("Failed to load cached embeddings (%s); will recompute", exc)

    if schema_emb is None:
        if not schema_pairs:
            raise ValueError("Cannot embed empty schema.")
        schema_emb = embedder.embed_texts([p.text for p in schema_pairs])
        if schema_embeds:
            _save_schema_embeds(
                schema_embeds,
                schema_emb,
                meta={
                    "fingerprint": schema_fp,
                    "normalized": True,
                    "dtype": "float32",
                    "shape": list(schema_emb.shape),
                    "model": "Marqo/dunzhang-stella_en_400M_v5",
                },
            )
            logging.info("Saved schema embeddings to %s", schema_embeds)

    # ---- Build or load FAISS index over schema
    faiss_indexer = FaissIndex(nlist=faiss_nlist, nprobe=faiss_nprobe)
    faiss_idx: Optional[faiss.Index] = None

    if (not rebuild_cache) and schema_index and schema_index.exists():
        logging.info("Trying to load schema Faiss index from %s", schema_index)
        try:
            faiss_idx = _load_faiss_index(schema_index)
            # reapply nprobe if IVF
            try:
                faiss_idx.nprobe = faiss_nprobe
            except Exception:
                pass
            logging.info("Loaded cached FAISS index from %s", schema_index)
        except Exception as exc:
            logging.warning("Failed to load FAISS index (%s); will rebuild", exc)

    if faiss_idx is None:
        faiss_idx = faiss_indexer.build(schema_emb)
        if schema_index:
            _save_faiss_index(schema_index, faiss_idx)
            logging.info("Saved FAISS index to %s", schema_index)

    # Embed input
    logging.info("Embedding input descriptors (%d) ...", len(flat_inputs))
    input_emb = embedder.embed_texts([p.text for p in flat_inputs])

    # Search for nearest K neighbors from schema
    logging.info("Finding %s neighbours from schema for each input descriptor", topk)
    sims, idxs = faiss_indexer.neighbors(faiss_idx, input_emb, k=topk)

    # Prepare LLM batches
    selector = SynonymSelector(
        model_name=llm_name,
        cache_dir=cache_dir,
        temperature=0.1,
        batch_size=batch_size_llm,
        cache_db=cache_db,
    )

    kept = 0
    dropped = 0

    llm_jobs: List[Tuple[Pair, List[Pair], Dict[str, float]]] = []
    for qi, q in enumerate(flat_inputs):
        cand_ids = idxs[qi]
        cand_sims = sims[qi]
        cands: List[Pair] = []
        score_map: Dict[str, float] = {}
        for j, cid in enumerate(cand_ids):
            if cid < 0:
                continue
            score = float(cand_sims[j])
            if score < min_embed_score:
                continue
            p = schema_pairs[int(cid)]
            cands.append(p)
            score_map[p.id] = score
        if not cands:
            dropped += 1
            continue
        llm_jobs.append((q, cands, score_map))

    # Iterate in micro‑batches for the LLM
    def _iter_batches(items, bs: int):
        for i in range(0, len(items), bs):
            yield items[i : i + bs]

    # Open file for logging how descriptors are harmonized
    decision_log_path = results_dir / "decision_log.jsonl"
    with decision_log_path.open("w", encoding="utf-8") as audit_f:
        harm_by_row: Dict[int, List[str]] = {}
        for batch in _iter_batches(llm_jobs, selector.batch_size):
            decisions = selector.select_batch([(q, cands) for (q, cands, _scores) in batch])

            for (q, cands, score_map), dec in zip(batch, decisions):
                if dec.chosen_id:
                    chosen = schema_by_id.get(dec.chosen_id)
                    if chosen is None:
                        logging.warning("Chosen id %s not in schema; defaulting to top-1 candidate.", dec.chosen_id)
                        chosen = cands[0]
                    row_idx = owner_row[q.id]
                    harm_by_row.setdefault(row_idx, []).append(f"{chosen.descriptor}; {chosen.explainer}")
                    kept += 1

                    embed_sim = score_map.get(chosen.id)
                    json.dump(
                        {
                            "input_id": q.id,
                            "input_descriptor": q.descriptor,
                            "input_explainer": q.explainer,
                            "chosen_id": chosen.id,
                            "descriptor": chosen.descriptor,
                            "explainer": chosen.explainer,
                            "embed_sim": embed_sim,
                        },
                        audit_f,
                        ensure_ascii=False,
                    )
                    audit_f.write("\n")
                else:
                    dropped += 1

    out_path = results_dir / f"{run_id}_harmonized.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in input_rows:
            out_obj: Dict[str, Any]
            if isinstance(r.raw, dict):
                out_obj = dict(r.raw)  # shallow copy
            else:
                # raw string input -> wrap into object
                out_obj = {"text": r.raw}
            out_obj["harmonized_descriptors"] = harm_by_row.get(r.row_idx, [])
            json.dump(out_obj, f, ensure_ascii=False)
            f.write("\n")

    logging.info("Results saved to %s and %s.", decision_log_path, out_path)
    logging.info("Done. Kept=%d, Dropped=%d", kept, dropped)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replace descriptor;explainer pairs in INPUT with synonyms from SCHEMA")
    p.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    p.add_argument("--schema", type=Path, required=True, help="Path to schema JSONL")
    p.add_argument("--run-id", type=str, required=True, help="Name for run")
    p.add_argument("--llm", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name")

    p.add_argument("--topk", type=int, default=5, help="Number of schema candidates per input for LLM")
    p.add_argument("--min-embed-score", type=float, default=0.0, help="Drop schema candidates below this cosine.")

    p.add_argument("--batch-size-embed", type=int, default=64)
    p.add_argument("--batch-size-llm", type=int, default=1024)

    p.add_argument("--faiss-nlist", type=int, default=256)
    p.add_argument("--faiss-nprobe", type=int, default=32)

    p.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir for models")
    p.add_argument("--cache-db", type=Path, default=None, help="SQLite file to cache LLM decisions")

    p.add_argument("--verbosity", type=int, default=1, choices=[0,1,2])

    p.add_argument("--schema-embeds", type=Path, default="../results/schema_embeddings.npz",
                help="Path to save/load cached schema embeddings (.npz)")
    p.add_argument("--schema-index", type=Path, default="../results/schema_index.faiss",
                help="Path to save/load cached FAISS index (.faiss)")
    p.add_argument("--rebuild-cache", action="store_true",
                help="Force recomputation of schema embeddings/index, ignoring any cache")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(
        input_path=args.input,
        schema_path=args.schema,
        run_id=args.run_id,
        llm_name=args.llm,
        topk=args.topk,
        min_embed_score=args.min_embed_score,
        batch_size_embed=args.batch_size_embed,
        batch_size_llm=args.batch_size_llm,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
        cache_dir=args.cache_dir,
        cache_db=args.cache_db,
        verbosity=args.verbosity,
        schema_embeds=args.schema_embeds,
        schema_index=args.schema_index,
        rebuild_cache=args.rebuild_cache,  
    )


if __name__ == "__main__":
    main()
