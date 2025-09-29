from __future__ import annotations

"""
Harmonizer for descriptor;explainer pairs using:
- Stella embeddings (Marqo/dunzhang-stella_en_400M_v5)
- FAISS ANN search over the Schema
- vLLM with guided JSON decoding to choose the best synonym (or none)

CLI
---
python harmonize_with_schema.py \
  --input input.jsonl \
  --schema schema.jsonl \
  --output output.jsonl \
  --llm qwen2.5-7b-instruct \
  --topk 5

Notes
-----
* Input JSONL can be either raw JSON strings of the form "descriptor; explainer"
  or JSON objects with keys {"descriptor": str, "explainer": str} or {"text": str}.
* Schema JSONL should contain JSON objects with keys {"descriptor": str, "explainer": str}.
  If an "id" field is absent, a stable synthetic id is assigned.
* Embeddings are L2‑normalized and indexed with Inner Product (cosine similarity).
* The LLM is the final authority on synonymy; if it returns "none", the pair is dropped.
"""

# Standard library
import argparse
from dataclasses import dataclass
import functools
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

def setup_logging(logging_dir: Path, verbosity: int = 1) -> None:
    """Configure both file and stdout logging, creating the directory if needed."""
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_file = logging_dir / f"{logging_dir.name}.log"

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
class Decision:
    chosen_id: Optional[str]  # schema id selected by LLM; None => drop
    reason: str


class VerdictSchema(BaseModel):
    is_synonym: bool
    chosen_id: str  # empty string if none
    reason: str


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _parse_pair_like(obj_or_str: object, fallback_id: str) -> Optional[Pair]:
    """Accepts a JSON value that may be a string "d; e" or an object with descriptor/explainer/text.
    Returns a Pair or None if unparsable.
    """
    try:
        if isinstance(obj_or_str, str):
            desc, expl = _split_descriptor_explainer(obj_or_str)
            return Pair(id=fallback_id, descriptor=desc, explainer=expl)
        if isinstance(obj_or_str, dict):
            # prefer explicit fields
            desc = (obj_or_str.get("descriptor") or obj_or_str.get("name") or "").strip()
            expl = (obj_or_str.get("explainer") or obj_or_str.get("description") or "").strip()
            if not (desc or expl):
                # try generic text
                text = str(obj_or_str.get("text", "")).strip()
                desc, expl = _split_descriptor_explainer(text)
            pid = str(obj_or_str.get("id", fallback_id))
            return Pair(id=pid, descriptor=desc, explainer=expl)
    except Exception as exc:
        logging.warning("Failed to parse line as pair (%s): %r", exc, obj_or_str)
    return None


def _split_descriptor_explainer(s: str) -> Tuple[str, str]:
    if ";" in s:
        left, right = s.split(";", 1)
        return left.strip(), right.strip()
    # If no semicolon, treat the whole string as descriptor
    return s.strip(), ""


@log_execution_time
def load_pairs_jsonl(path: Path, expect_objects: bool = False, prefix: str = "p") -> List[Pair]:
    pairs: List[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if expect_objects:
                    logging.warning("Line %d is not JSON; skipping.", i)
                    continue
                obj = line  # treat as JSON string line
            pair = _parse_pair_like(obj, fallback_id=f"{prefix}{i}")
            if pair and (pair.descriptor or pair.explainer):
                pairs.append(pair)
            else:
                logging.warning("Line %d has no usable content; skipped.", i)
    return pairs


# -----------------------------------------------------------------------------
# Embeddings (Stella)
# -----------------------------------------------------------------------------

class StellaEmbedder:
    """Minimal pooled embedding wrapper around Marqo/dunzhang-stella_en_400M_v5."""

    def __init__(self, cache_dir: Optional[Path], batch_size: int = 32, device: str = "cuda:0") -> None:
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            .to(self.device)
            .eval()
        )
        # Use half precision on CUDA to save memory
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.batch_size = batch_size

    @log_execution_time
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        all_chunks: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)[0]  # last hidden states [B, T, H]
                attn = inputs["attention_mask"].unsqueeze(-1).to(outputs.dtype)  # [B, T, 1]
                masked = outputs * attn
                pooled = masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1e-6)  # [B, H]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_chunks.append(pooled.detach().cpu().numpy().astype("float32"))
        arr = np.vstack(all_chunks)
        return arr


# -----------------------------------------------------------------------------
# FAISS index
# -----------------------------------------------------------------------------

class FaissIndex:
    def __init__(self, nlist: int = 100, nprobe: int = 10) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.index: Optional[faiss.Index] = None

    @log_execution_time
    def build(self, embeddings: np.ndarray) -> faiss.Index:
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        n, d = embeddings.shape
        if n == 0:
            raise ValueError("No embeddings to index.")
        # If small dataset, a flat IP index is simple and strong
        if n < max(2 * self.nlist, 100):
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            self.index = index
            return index
        try:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
            # Train on a sample
            sample = embeddings[np.random.choice(n, size=min(100_000, n), replace=False)]
            index.train(sample)
            index.add(embeddings)
            index.nprobe = self.nprobe
            self.index = index
            return index
        except Exception as exc:
            logging.warning("FAISS IVF build failed (%s); falling back to IndexFlatIP", exc)
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            self.index = index
            return index

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        sims, idxs = self.index.search(np.ascontiguousarray(query_embeddings.astype("float32")), k)
        return sims, idxs


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
        guided_backend: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = os.environ.get("HF_HOME") or (str(cache_dir) if cache_dir else None)
        self.temperature = temperature
        self.batch_size = batch_size
        self._llm: Optional[LLM] = None
        self._cache: Optional[SqliteDict] = SqliteDict(str(cache_db), autocommit=True) if cache_db else None
        self.guided_backend = guided_backend  # e.g., "outlines" or "lm-format-enforcer"

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
                guided_decoding_backend=self.guided_backend,
            )
        return self._llm

    @staticmethod
    def _prompt(query: Pair, candidates: List[Pair]) -> str:
        cand_lines = []
        for c in candidates:
            cand_lines.append(f"- id={c.id} :: {c.text}")
        candidates_block = "\n".join(cand_lines)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"  # works well with many instruct models
            "You are a careful ontology assistant.\n"
            "Task: Given a query pair in the form 'descriptor; explainer' and a list of schema candidates,\n"
            "pick the single candidate that is synonymous or near‑synonymous to the query.\n"
            "If none are acceptable, return is_synonym=false and chosen_id=\"\".\n"
            "Guidelines: prefer exact concept match, avoid mismatches in scope, polarity, or domain.\n"
            "Output MUST be JSON only — no extra text.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Query: {query.text}\n\n"
            f"Candidates:\n{candidates_block}\n\n"
            "Return JSON with keys: is_synonym (bool), chosen_id (string; empty if none), reason (short).\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _response_format(self) -> GuidedDecodingParams:
        return GuidedDecodingParams(json=VerdictSchema.model_json_schema())

    @staticmethod
    def _cache_key(query: Pair, candidates: List[Pair]) -> str:
        key = {
            "query": {"id": query.id, "d": query.descriptor, "e": query.explainer},
            "candidates": [{"id": c.id, "d": c.descriptor, "e": c.explainer} for c in candidates],
        }
        return json.dumps(key, ensure_ascii=False, sort_keys=True)

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
            reason = str(obj.get("reason", "")).strip() or ("none" if not is_syn else "")
            return Decision(chosen_id=chosen if is_syn else None, reason=reason)
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
                        decisions[i] = Decision(chosen_id=obj.get("chosen_id"), reason=obj.get("reason", "cache"))
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
            outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
            in_tok, gen_tok = self._extract_tokens(outputs)
            tot = in_tok + gen_tok
            logging.info("LLM tokens: in=%d, gen=%d, total=%d", in_tok, gen_tok, tot)
            # Map back
            for pos, out in zip(out_positions, outputs):
                text = ""
                outs = getattr(out, "outputs", None) or []
                if outs:
                    text = (outs[0].text or "")
                dec = self._parse_text_to_decision(text)
                if dec is None:
                    dec = Decision(chosen_id=None, reason="parse_error")
                decisions[pos] = dec
                # Write cache
                if self._cache is not None:
                    q, cands = batch[pos]
                    ck = self._cache_key(q, cands)
                    try:
                        self._cache[ck] = json.dumps({"chosen_id": dec.chosen_id, "reason": dec.reason}, ensure_ascii=False)
                    except Exception:
                        pass

        # type: ignore[return-value]
        return [d if d is not None else Decision(chosen_id=None, reason="missing") for d in decisions]


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

@log_execution_time
def run(
    input_path: Path,
    schema_path: Path,
    output_path: Path,
    llm_name: str,
    topk: int = 5,
    min_embed_score: float = 0.0,
    batch_size_embed: int = 64,
    batch_size_llm: int = 64,
    faiss_nlist: int = 256,
    faiss_nprobe: int = 32,
    cache_dir: Optional[Path] = None,
    cache_db: Optional[Path] = None,
    device: str = "cuda:0",
    verbosity: int = 1,
    guided_backend: Optional[str] = None,
) -> None:
    setup_logging(Path("logs/synonymizer"), verbosity=verbosity)

    logging.info("Loading input and schema JSONL ...")
    input_pairs = load_pairs_jsonl(input_path, expect_objects=False, prefix="q")
    schema_pairs = load_pairs_jsonl(schema_path, expect_objects=True, prefix="s")
    if not schema_pairs:
        raise ValueError("Schema is empty.")

    # Build lookup by id
    schema_by_id: Dict[str, Pair] = {p.id: p for p in schema_pairs}

    logging.info("Embedding schema (%d) and input (%d) ...", len(schema_pairs), len(input_pairs))
    embedder = StellaEmbedder(cache_dir=cache_dir, batch_size=batch_size_embed, device=device)
    schema_emb = embedder.embed_texts([p.text for p in schema_pairs])
    input_emb = embedder.embed_texts([p.text for p in input_pairs])

    # Build FAISS index on schema
    index = FaissIndex(nlist=faiss_nlist, nprobe=faiss_nprobe)
    index.build(schema_emb)

    # Search top‑K per input
    sims, idxs = index.search(input_emb, k=topk)

    # Prepare LLM batches
    selector = SynonymSelector(
        model_name=llm_name,
        cache_dir=cache_dir,
        temperature=0.1,
        batch_size=batch_size_llm,
        cache_db=cache_db,
        guided_backend=guided_backend,
    )

    accepted: List[Pair] = []
    kept = 0
    dropped = 0

    # Iterate in micro‑batches for the LLM
    def _iter_batches(items: List[Tuple[Pair, List[Pair]]], bs: int) -> Iterable[List[Tuple[Pair, List[Pair]]]]:
        for i in range(0, len(items), bs):
            yield items[i : i + bs]

    llm_jobs: List[Tuple[Pair, List[Pair]]] = []
    for qi, q in enumerate(input_pairs):
        cand_ids = idxs[qi]
        cand_sims = sims[qi]
        cands: List[Pair] = []
        for j, cid in enumerate(cand_ids):
            if cid < 0:
                continue
            score = float(cand_sims[j])
            if score < min_embed_score:
                continue
            cands.append(schema_pairs[int(cid)])
        if not cands:
            dropped += 1
            continue
        llm_jobs.append((q, cands))

    # Process through LLM in batches
    for batch in _iter_batches(llm_jobs, selector.batch_size):
        decisions = selector.select_batch(batch)
        for (q, cands), dec in zip(batch, decisions):
            if dec.chosen_id:
                chosen = schema_by_id.get(dec.chosen_id)
                if chosen is None:
                    # Fallback: if chosen_id wasn't a known id (model hallucination), pick first candidate
                    logging.warning("Chosen id %s not in schema; defaulting to top‑1 candidate.", dec.chosen_id)
                    chosen = cands[0]
                accepted.append(Pair(id=chosen.id, descriptor=chosen.descriptor, explainer=chosen.explainer))
                kept += 1
            else:
                dropped += 1

    # Write output JSONL — only accepted pairs, each line is an object {id, descriptor, explainer}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in accepted:
            json.dump({"id": p.id, "descriptor": p.descriptor, "explainer": p.explainer}, f, ensure_ascii=False)
            f.write("\n")

    logging.info("Done. Kept=%d, Dropped=%d, Out=%s", kept, dropped, str(output_path))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replace descriptor;explainer pairs in INPUT with synonyms from SCHEMA")
    p.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    p.add_argument("--schema", type=Path, required=True, help="Path to schema JSONL")
    p.add_argument("--output", type=Path, required=True, help="Path to write filtered/replaced JSONL")
    p.add_argument("--llm", type=str, required=True, help="vLLM model name, e.g., 'Qwen2.5-7B-Instruct'")

    p.add_argument("--topk", type=int, default=5, help="Number of schema candidates per input for LLM")
    p.add_argument("--min-embed-score", type=float, default=0.0, help="Drop schema candidates below this cosine.")

    p.add_argument("--batch-size-embed", type=int, default=64)
    p.add_argument("--batch-size-llm", type=int, default=64)

    p.add_argument("--faiss-nlist", type=int, default=256)
    p.add_argument("--faiss-nprobe", type=int, default=32)

    p.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir for models")
    p.add_argument("--cache-db", type=Path, default=None, help="SQLite file to cache LLM decisions")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbosity", type=int, default=1, choices=[0,1,2])

    p.add_argument("--guided-backend", type=str, default=None, help="Guided decoding backend: outlines or lm-format-enforcer")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(
        input_path=args.input,
        schema_path=args.schema,
        output_path=args.output,
        llm_name=args.llm,
        topk=args.topk,
        min_embed_score=args.min_embed_score,
        batch_size_embed=args.batch_size_embed,
        batch_size_llm=args.batch_size_llm,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
        cache_dir=args.cache_dir,
        cache_db=args.cache_db,
        device=args.device,
        verbosity=args.verbosity,
        guided_backend=args.guided_backend,
    )


if __name__ == "__main__":
    main()
