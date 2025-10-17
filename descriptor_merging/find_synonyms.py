#!/usr/bin/env python3
"""Synonym Deduper

Pipeline:
1) Read descriptor–explainer pairs from JSONL
2) Embed each pair (descriptor+explainer text)
3) Build a FAISS index
4) For each item, pull its nearest neighbor candidate(s)
5) Min similarity: if a candidate neighbor's similarity < t, skip LLM for that pair
6) Ask an LLM to decide if a remaining candidate pair are synonyms; if yes, keep the more representative one
7) Do not re-merge items within the same iteration
8) Iterate until convergence or a max-iteration cap

Notes
-----
* Uses vLLM Guided Decoding with a Pydantic JSON schema to keep responses tidy.
* Caches LLM decisions in a SqliteDict so repeated runs are faster.
* You can raise --k to consider multiple neighbors per item.
* Falls back to a flat L2 index when IVF training is not appropriate (e.g., small `n`).
* Make sure each item has a unique ID or else you will experience data loss.
"""
from __future__ import annotations

# Standard library imports
import argparse
from collections import defaultdict
from dataclasses import dataclass
import functools
import gc
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Literal

# Third-party imports
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
# Logging
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

    # Reset handlers to avoid duplicates across multiple runs in the same process
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
    """Decorator that logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(
            "Execution of %s took %s.",
            func.__name__,
            time.strftime("%H:%M:%S", time.gmtime(execution_time)),
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
        """Return a compact, combined text used for embeddings."""
        d = self.descriptor.strip()
        e = self.explainer.strip()
        if d and e:
            return f"{d}; {e}"
        return d or e


@dataclass
class Decision:
    is_synonym: bool
    keep_id: str
    drop_id: str
    reason: str

class DecisionSchema(BaseModel):
    is_synonym: bool
    keep_id: str
    drop_id: str

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def read_jsonl(path: Path, sample_size: Optional[int] = None) -> List[Pair]:
    limit = sample_size

    pairs: List[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line %d", i)
                continue
            d = str(obj.get("descriptor", "")).strip()
            e = str(obj.get("explainer", "")).strip()
            if not d and not e:
                # Skip empty rows
                continue
            # Get id or create one based on index
            item_id = obj.get("id") or f"{i:08d}"
            pairs.append(Pair(id=item_id, descriptor=d, explainer=e))
            if limit is not None and len(pairs) >= limit:
                logging.info("Test mode: limiting to %d items", limit)
                break
            
    logging.info("Loaded %d descriptor-explainer pairs", len(pairs))
    
    # Check that there are duplicate IDs in data, as this will cause pairs to be silently dropped
    ids = [p.id for p in pairs]
    dups = {x for x in ids if ids.count(x) > 1}
    if dups:
        raise ValueError(f"Duplicate IDs found: {sorted(list(dups))[:10]} (and more)")

    return pairs

def write_jsonl(path: Path, pairs: Sequence[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {"id": p.id, "descriptor": p.descriptor, "explainer": p.explainer},
                    ensure_ascii=False,
                )
                + "\n",
            )
    logging.info("Wrote %d descriptor-explainer pairs -> %s", len(pairs), path)
    
    
def _log_dropped(pairs: Sequence[Pair], dropped: List[Tuple[int, int, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, j, s in dropped:
            a, b = pairs[i], pairs[j]
            reason = "nan_similarity" if np.isnan(s) else "below_min_similarity"
            rec = {
                "i": i, "j": j, "text_i": a.text, "text_j": b.text, "similarity": round(s, 4), "reason": reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            
def _write_all_decisions_jsonl(
        path: Path,
        ordered_pairs: List[Tuple[Pair, Pair]],
        decisions: List[Decision],
    ) -> None:
    """Write one JSON line per LLM decision (potentially large)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for (a, b), d in zip(ordered_pairs, decisions):
            rec = {
                "a_id": a.id,
                "b_id": b.id,
                "a_descriptor": a.descriptor,
                "b_descriptor": b.descriptor,
                "a_explainer": a.explainer,
                "b_explainer": b.explainer,
                "is_synonym": d.is_synonym,
                "keep_id": d.keep_id,
                "drop_id": d.drop_id,
                "reason": d.reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _shorten(s: str, max_len: int = 200) -> str:
    """One-line, trimmed preview for logs."""
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _log_iteration_decisions(
        iteration: int,
        ordered_pairs: List[Tuple[Pair, Pair]],
        decisions: List[Decision],
        output_dir: Path,
        sample_n: int = 5,
        write_full_jsonl: bool = True,
    ) -> None:
    """Summarize counts to the main log and write a small, human-friendly sample file."""
    syn_examples = []
    non_examples = []

    for (a, b), d in zip(ordered_pairs, decisions):
        rec = {
            "a_id": a.id,
            "b_id": b.id,
            "a_descriptor": a.descriptor,
            "b_descriptor": b.descriptor,
            "a_explainer": a.explainer,
            "b_explainer": b.explainer,
            "is_synonym": d.is_synonym,
            "keep_id": d.keep_id,
            "drop_id": d.drop_id,
            "reason": d.reason,
        }
        (syn_examples if d.is_synonym else non_examples).append(rec)

    n_syn = len(syn_examples)
    n_non = len(non_examples)
    total = n_syn + n_non
    logging.info(
        "Iteration %d — LLM decisions: %d merges, %d non-merges (total %d).",
        iteration, n_syn, n_non, total
    )

    def _sample(xs, k: int):
        if not xs:
            return []
        if len(xs) <= k:
            return xs
        return random.sample(xs, k)

    syn_sample = _sample(syn_examples, sample_n)
    non_sample = _sample(non_examples, sample_n)

    txt_path = output_dir / f"iteration_{iteration}_llm_decisions.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Iteration {iteration} — LLM decision summary\n")
        f.write(f"Merges: {n_syn} | Non-merges: {n_non} | Total: {total}\n")

        f.write("\n--- Examples: MERGES ---\n")
        for rec in syn_sample:
            f.write(
                (
                    f"[MERGE] A({rec['a_id']}): '{rec['a_descriptor']}' — expl: '{_shorten(rec['a_explainer'])}'\n"
                    f"        B({rec['b_id']}): '{rec['b_descriptor']}' — expl: '{_shorten(rec['b_explainer'])}'\n"
                    f"        keep={rec['keep_id']} drop={rec['drop_id']}\n"
                    f"        reason={rec['reason']}\n\n"
                )
            )

        f.write("\n--- Examples: NON-MERGES ---\n")
        for rec in non_sample:
            f.write(
                (
                    f"[NO]    A({rec['a_id']}): '{rec['a_descriptor']}' — expl: '{_shorten(rec['a_explainer'])}'\n"
                    f"        B({rec['b_id']}): '{rec['b_descriptor']}' — expl: '{_shorten(rec['b_explainer'])}'\n"
                    f"        reason={rec['reason']}\n\n"
                )
            )

    logging.info("Wrote iteration decision samples -> %s", txt_path)

    if write_full_jsonl:
        jsonl_path = output_dir / f"iteration_{iteration}_llm_decisions.jsonl"
        _write_all_decisions_jsonl(jsonl_path, ordered_pairs, decisions)
        logging.info("Wrote all iteration decisions -> %s", jsonl_path)


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

class StellaEmbedder:
    """Minimal pooled embedding wrapper around Marqo/dunzhang-stella_en_400M_v5."""

    def __init__(self, cache_dir: Optional[Path], batch_size: int = 32, device: str = "cuda:0") -> None:
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


# -----------------------------------------------------------------------------
# LLM combiner
# -----------------------------------------------------------------------------

class SynonymCombiner:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[Path],
        temperature: float = 0.1,
        cache_db_path: Optional[Path] = None,
        batch_size: int = 512,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = os.environ.get("HF_HOME") or (str(cache_dir) if cache_dir else None)
        self.temperature = temperature
        self._llm = None
        self.cache_db_path = cache_db_path
        self._cache = SqliteDict(str(cache_db_path), autocommit=True) if cache_db_path else None
        self.batch_size = batch_size

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPU available.")
            else:
                logging.info(f"Using {n_gpus} GPU(s).")
            self._llm = LLM(
                model=self.model_name,
                download_dir=self.cache_dir,
                dtype="bfloat16",
                max_model_len=2048,
                tensor_parallel_size=n_gpus,
                enforce_eager=False,
                gpu_memory_utilization=0.9,
            )
        return self._llm

    def _response_format(self) -> GuidedDecodingParams:
        return GuidedDecodingParams(json=DecisionSchema.model_json_schema())

    def _prompt(self, a: Pair, b: Pair) -> str:
        return ("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are deciding whether two descriptor–explainer pairs, A and B, refer to the same concept.
            The descriptors describe a document, while the explainer provide additional context.
            
            *If the descriptor-explainer pairs are synonymous or near-synonymous in most contexts*
            1. Mark them as synonyms by setting is_synonym=true.
            2. Choose which of the two descriptor-explainer pairs to keep as the representative pair.
             - The representative pair should be the one with more concise, general, and commonly used phrasing that encompasses the meaning
            of both pairs.
            - The id of the representative pair should be given in the keep_id field and the other in the drop_id field.
            
            *If the descriptor-explainer pairs are not synonyms*
            1. Set is_synonym=false.
            2. Pick arbitrary keep/drop consistent with the input.
            
            Return JSON with these fields only: is_synonym (bool), keep_id (string), drop_id (string).
            IMPORTANT: Do not rewrite or invent any descriptors.
            Respond ONLY with JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>""" + f"""

            A.id: {a.id} A.descriptor: {a.descriptor} A.explainer: {a.explainer}
            B.id: {b.id} B.descriptor: {b.descriptor} B.explainer: {b.explainer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )

    def _cache_key(self, a: Pair, b: Pair) -> str:
        # Order-invariant key
        key = tuple(sorted([(a.id, a.descriptor, a.explainer), (b.id, b.descriptor, b.explainer)], key=lambda x: x[0]))
        return json.dumps(key, ensure_ascii=False)
    
    @staticmethod
    def _tokens_in_outputs(outputs):
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

    @log_execution_time
    def decide_batch(self, pairs: List[Tuple[Pair, Pair]]) -> List[Decision]:
        """Return one Decision per (a,b). Uses cache first, then queries LLM in full batches."""
        decisions: List[Decision] = [None] * len(pairs)

        # Cache sweep: fill any cached decisions; collect the indices we still need to query
        uncached_idxs: List[int] = []
        if self._cache is not None:
            for i, (a, b) in enumerate(pairs):
                ck = self._cache_key(a, b)
                if ck in self._cache:
                    try:
                        obj = json.loads(self._cache[ck])
                        decisions[i] = Decision(
                            is_synonym=bool(obj.get("is_synonym", False)),
                            keep_id=str(obj.get("keep_id", "")),
                            drop_id=str(obj.get("drop_id", "")),
                            reason=str(obj.get("reason", "cache_hit")),
                        )
                        continue
                    except Exception:
                        pass
                uncached_idxs.append(i)
        else:
            uncached_idxs = list(range(len(pairs)))

        if not uncached_idxs:
            logging.info("All candidates already cached. Skipping LLM.")
            return decisions  # all cached
        
        num_cached = len(pairs) - len(uncached_idxs)
        logging.info("%d candidates already cached. Sending %s to LLM", num_cached, len(uncached_idxs))

        # Sort uncached indices by approximate prompt length (short → long) to reduce padding waste
        def _plen(i: int) -> int:
            a, b = pairs[i]
            # cheap proxy for token count
            return len(a.descriptor) + len(a.explainer) + len(b.descriptor) + len(b.explainer)
        uncached_idxs.sort(key=_plen)

        # Generate in full batches built from uncached items
        for start in range(0, len(uncached_idxs), self.batch_size):
            chunk = uncached_idxs[start:start + self.batch_size]
            batch_pairs = [pairs[i] for i in chunk]
            batch_prompts = [self._prompt(a, b) for (a, b) in batch_pairs]

            params = SamplingParams(
                temperature=self.temperature,
                top_p=0.5,
                repetition_penalty=1.0,
                max_tokens=512,
                stop=["<|eot_id|>"],
                guided_decoding=self._response_format(),
                seed=42,
            )
            t0 = time.time()
            outputs = self.llm.generate(batch_prompts, sampling_params=params, use_tqdm=False)
            t1 = time.time()
            in_tok, gen_tok = self._tokens_in_outputs(outputs)
            self._log_throughput(in_tok, gen_tok, t1 - t0)

            parsed: List[Decision] = []
            for out_idx, out in enumerate(outputs):
                text = ""
                outs = getattr(out, "outputs", None) or []
                if outs:
                    text = (outs[0].text or "").strip(" `\n").removeprefix("json").strip()
                try:
                    obj = json_repair.loads(text)
                    parsed.append(Decision(
                        is_synonym=bool(obj.get("is_synonym", False)),
                        keep_id=str(obj.get("keep_id", "")),
                        drop_id=str(obj.get("drop_id", "")),
                        reason="LLM_decision",
                    ))
                except Exception as exc:
                    logging.warning("Falling back to non-merge due to parse error: %s", exc)
                    a, b = batch_pairs[out_idx]
                    parsed.append(Decision(
                        is_synonym=False,
                        keep_id=a.id,
                        drop_id=b.id,
                        reason="parse_error",
                    ))

            # Write results back to correct global positions + cache them
            for local_idx, global_idx in enumerate(chunk):
                d = parsed[local_idx]
                decisions[global_idx] = d
                if self._cache is not None:
                    a, b = pairs[global_idx]
                    ck = self._cache_key(a, b)
                    self._cache[ck] = json.dumps(d.__dict__, ensure_ascii=False)

        return decisions

# -----------------------------------------------------------------------------
# Merging & iteration
# -----------------------------------------------------------------------------

@dataclass
class MergeResult:
    kept: str
    dropped: str


def candidate_synonyms(
    neighbor_idx: np.ndarray,
    scores: np.ndarray,
    *,
    strategy: Literal["mutual", "unique"] = "mutual",
    min_similarity: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    Build deduplicated candidate (i, j, sim) pairs from FAISS neighbors.

    strategy="mutual": keep only mutual top-1 neighbors (i's best is j AND j's best is i).
    strategy="unique": consider all k neighbors for each i; deduplicate by unordered pair, keep the max sim.

    Returns:
        triples: [(i, j, sim)] sorted by descending sim, with i < j
        dropped: [(i, j, sim)] that were filtered out by min_similarity/NaN (ascending sim)
    """
    assert neighbor_idx.shape == scores.shape, "neighbor_idx and scores must have the same shape"
    n, k = neighbor_idx.shape

    # canonical containers
    kept: Dict[Tuple[int, int], float] = {}
    dropped_map: Dict[Tuple[int, int], float] = {}

    def _maybe_keep(a: int, b: int, sim: float) -> None:
        # order and screen
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            return
        i, j = (a, b) if a < b else (b, a)

        # thresholding / NaN handling
        if min_similarity is not None and (np.isnan(sim) or sim < min_similarity):
            prev = dropped_map.get((i, j))
            if prev is None or sim > prev:
                dropped_map[(i, j)] = sim
            return

        # keep highest similarity per unordered pair
        prev = kept.get((i, j))
        if (prev is None) or (sim > prev):
            kept[(i, j)] = sim

    if n == 0 or k == 0:
        return [], []

    if strategy == "mutual":
        # require mutual top-1; prefer the better of (i->j, j->i) sims
        for i in range(n):
            j = int(neighbor_idx[i, 0])
            if j < 0 or j == i or j >= n:
                continue
            jj0 = int(neighbor_idx[j, 0]) if j < n and k > 0 else -1
            if jj0 != i:
                continue
            sim_ij = float(scores[i, 0]) if scores.size else float("-inf")
            sim_ji = float(scores[j, 0]) if scores.size else float("-inf")
            _maybe_keep(i, j, max(sim_ij, sim_ji))
    else:
        # consider all-k neighbors for each i
        for i in range(n):
            for col in range(k):
                j = int(neighbor_idx[i, col])
                if j < 0 or j == i:
                    continue
                sim = float(scores[i, col]) if scores.size else float("-inf")
                _maybe_keep(i, j, sim)

    # Format outputs
    triples: List[Tuple[int, int, float]] = [(i, j, s) for (i, j), s in kept.items()]
    triples.sort(key=lambda t: t[2], reverse=True)

    dropped_list: List[Tuple[int, int, float]] = [(i, j, s) for (i, j), s in dropped_map.items()]
    dropped_list.sort(key=lambda t: (np.isnan(t[2]), t[2]))  # NaNs last

    if triples:
        sims = [t[2] for t in triples]
        logging.info("Pair similarity — avg: %.4f | max: %.4f | min: %.4f", sum(sims)/len(sims), max(sims), min(sims))
    logging.info(
        "Dropped %d candidate pairs due to threshold %s.",
        len(dropped_list),
        "None" if min_similarity is None else f"{min_similarity:.4f}",
    )
    return triples, dropped_list


def _apply_merges_greedy(pairs: List[Pair], decisions: List[Decision]) -> Tuple[List[Pair], List[MergeResult]]:
    id_to_pair: Dict[str, Pair] = {p.id: p for p in pairs}
    used: Set[str] = set()
    merges: List[MergeResult] = []
    
    for d in decisions:
        if not d.is_synonym:
            continue
        keep, drop = d.keep_id, d.drop_id
        if keep == drop or keep not in id_to_pair or drop not in id_to_pair:
            continue
        if keep in used or drop in used:
            continue # one merge per item per round

        used.add(keep); used.add(drop)
        merges.append(MergeResult(kept=keep, dropped=drop))

    dropped_ids = {m.dropped for m in merges}
    new_pairs = [id_to_pair[p.id] for p in pairs if p.id not in dropped_ids]
    return new_pairs, merges


@log_execution_time
def iterate_until_converged(
    pairs: List[Pair],
    embedder: StellaEmbedder,
    faiss_index: FaissIndex,
    combiner: SynonymCombiner,
    k: int = 1,
    max_iters: int = 5,
    min_similarity: Optional[float] = None,
    output_dir: Optional[Path] = None,
    candidate_strategy: Literal["mutual", "unique"] = "mutual",
) -> Tuple[List[Pair], List[List[MergeResult]]]:
    all_merges: List[List[MergeResult]] = []
    iteration = 0
    
    # Embed texts
    logging.info("Embedding descriptor-explainer pairs...")
    texts = [p.text for p in pairs]
    emb_all = embedder.embed_texts(texts)
    # Keep a mapping from current order to original row indices
    current = list(range(len(pairs)))

    while iteration < max_iters:
        iteration += 1
        logging.info("=== Iteration %d | %d descriptor-explainer pairs ===", iteration, len(pairs))
        if len(pairs) <= 1:
            break

        texts = [p.text for p in pairs]
        emb = emb_all[current] #slice current 

        logging.info("Building Faiss index...")
        # 2) Index
        index = faiss_index.build(emb)

        logging.info("Finding neighbors...")
        # 3) Neighbors
        sims, neigh = faiss_index.neighbors(index, emb, k=k)

        # 4) Find candidate synonym pairs
        triples, dropped = candidate_synonyms(
            neigh,
            sims,
            strategy=candidate_strategy,  # "mutual" or "unique"
            min_similarity=min_similarity,
        )
        
        if dropped:
            if output_dir:
                _log_dropped(pairs, dropped, output_dir / f"iteration_{iteration}_dropped.jsonl")
            else:
                _log_dropped(pairs, dropped, Path(f"iteration_{iteration}_dropped.jsonl"))
        if not triples:
            logging.info("No neighbor proposals. Stopping.")
            break
        
        logging.info("Found %d candidate synonyms", len(triples))

        logging.info("Prompting LLM...")
        # 5) Query LLM on candidates in order
        ordered_pairs: List[Tuple[Pair, Pair]] = []
        for i, j, _ in triples:
            ordered_pairs.append((pairs[i], pairs[j]))

        decisions = combiner.decide_batch(ordered_pairs)
        
        # Summarize decisions to log + write samples and optional full JSONL
        if output_dir is not None:
            _log_iteration_decisions(
                iteration=iteration,
                ordered_pairs=ordered_pairs,
                decisions=decisions,
                output_dir=output_dir,
                sample_n=10,             # number of examples to show
                write_full_jsonl=True,  # set False to skip the big JSONL
            )

        # 6) Apply merges — compute mask against the old pairs
        # Take copy of pairs from previous round
        prev_pairs = pairs
        # Drop the pairs that were merged in this round
        pairs, merges = _apply_merges_greedy(prev_pairs, decisions)
        # IDs of merged items
        all_merges.append(merges)

        # Build keep mask over prev_pairs, then filter pairs & current together
        if merges:
            dropped_ids = {m.dropped for m in merges}
            kept_mask = [p.id not in dropped_ids for p in prev_pairs]
            # Update mapping and pairs in lockstep
            current = [idx for idx, keep in zip(current, kept_mask) if keep]
            # (pairs is already the filtered list from _apply_merges_greedy)
        else:
            kept_mask = [True] * len(prev_pairs)
    
        logging.info("Iteration %d merged %d descriptor-explainer pairs; %d remain", iteration, len(merges), len(pairs))
        if output_dir is not None:
            write_jsonl(output_dir / f"checkpoint_iter_{iteration}.jsonl", pairs)

        # 7) Convergence check
        if len(merges) == 0:
            logging.info("No merges in this iteration. Converged.")
            break
        
        # Cleanup before next iteration to save memory
        try:
            del index, texts, emb, sims, neigh, triples, dropped, ordered_pairs, decisions
            gc.collect()
        except NameError:
            pass

    return pairs, all_merges


# -----------------------------------------------------------------------------
# Disjoint Set Union for grouping
# -----------------------------------------------------------------------------

class DSU:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path("../results/synonym_merges") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.jsonl"
    decision_cache_path = output_dir / (args.decision_cache or "decision_cache.sqlite")

    # HF cache dir (optional)
    if args.cache_dir:
        cache_dir: Optional[Path] = Path(args.cache_dir)
    else:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = Path(hf_home) if hf_home else None

    # Setup logging
    setup_logging(output_dir, args.verbose)
    
    # Log run settings
    with open(output_dir / f"{args.run_id}_settings.txt", "a") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f.write(f"{arg}: {value}\n")
        f.write("===========================\n")
    
    # Read input
    pairs = read_jsonl(input_path, sample_size=args.test)
    all_ids = {p.id for p in pairs}
    
    # Build embedder and index helpers
    embedder = StellaEmbedder(cache_dir=cache_dir, batch_size=args.batch_size)
    faiss_index = FaissIndex(nlist=args.faiss_nlist, nprobe=args.faiss_nprobe)
    
    # Initialise combiner
    combiner = SynonymCombiner(
        model_name=args.model,
        cache_dir=cache_dir,
        temperature=args.temperature,
        cache_db_path=Path(decision_cache_path),
        batch_size = args.llm_batch_size
    )
    
    logging.info("Starting iterations over data...")
    # Iterate until convergence / threshold
    final_pairs, all_merges = iterate_until_converged(
        pairs,
        embedder,
        faiss_index,
        combiner,
        k=args.k,
        max_iters=args.max_iters,
        min_similarity=args.min_similarity,
        output_dir=output_dir,
        candidate_strategy=args.candidate_strategy,
    )

    # Persist results
    write_jsonl(output_path, final_pairs)

    # Build transitive groups keyed by final kept IDs
    dsu = DSU()
    for round_merges in all_merges:
        for m in round_merges:
            dsu.union(m.kept, m.dropped)

    # Assign members to cluster roots
    clusters: Dict[str, Set[str]] = defaultdict(set)
    for pid in all_ids:
        clusters[dsu.find(pid)].add(pid)

    # Re-key clusters so that each final kept id maps to its members
    # This way we know what got merged into what
    final_ids = {p.id for p in final_pairs}
    keyed: Dict[str, List[str]] = {}
    for fid in final_ids:
        root = dsu.find(fid)
        members = sorted(clusters.get(root, {fid}))
        keyed[fid] = members

    groups_path = output_path.with_suffix(".groups.json")
    with groups_path.open("w", encoding="utf-8") as f:
        json.dump(keyed, f, ensure_ascii=False, indent=2)
    logging.info("Wrote groups -> %s", groups_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine synonym descriptors using FAISS + LLM.")
    parser.add_argument("--run-id", type=str, default="default", help="Run ID")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL with fields: id? (optional), descriptor, explainer")

    # Embeddings
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (optional)")
    parser.add_argument("--batch-size", type=int, default=32)

    # FAISS
    parser.add_argument("--faiss-nlist", type=int, default=100, help="#Voronoi cells the embedding space is divided into in Faiss")
    parser.add_argument("--faiss-nprobe", type=int, default=10, help="#Voronoi cells to visit when searching")
    parser.add_argument("--k", type=int, default=1, help="#neighbors to fetch per item (excluding self)")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help=(
            "Minimum similarity on pooled embeddings; if a candidate neighbor is less similar than this, "
            "skip sending that pair to the LLM"
        ),
    )

    # LLM
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name or local path")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--decision-cache", type=str, default="decision_cache.sqlite", help="Sqlite file to cache LLM decisions")
    parser.add_argument("--llm-batch-size", type=int, default=512, help="Batch size for LLM queries")

    # Loop
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--test", nargs="?", const="10000", default=None, type=int, #nargs="?" means 0 or 1 argument 
                        help="Run with small batch of data. Defaults to 10k sample. Give value if you want to change sample size.")
    parser.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")
    parser.add_argument("--candidate-strategy", choices=["unique", "mutual"], default="mutual")

    args = parser.parse_args()
    
    start_time = time.time()
    # Main process
    main(args)
    end_time = time.time()
    logging.info("========================================")
    logging.info(f"Finished. Process took {time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))}.")
    logging.info("========================================")