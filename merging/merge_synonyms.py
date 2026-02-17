#!/usr/bin/env python3
"""
What this script writes per run_id:
- <run_id>.jsonl                      : final kept pairs (id, descriptor, explainer)
- iteration_XX_llm_decisions.jsonl    : LLM-evaluated pairs (synonym? keep/drop + reason)
- iteration_XX_all_candidates.jsonl   : ALL considered pairs this iteration, including ones
                                        skipped by similarity threshold (evaluated=False)
- iteration_XX_lineage.jsonl          : MERGE events for this iteration only
- <run_id>_lineage.jsonl              : concatenation of all iteration_XX_lineage.jsonl (edges only)
- <run_id>_settings.txt               : run settings & metadata
- checkpoint_iter_XX.jsonl            : pairs snapshot after each iteration
- iteration_XX_dropped.jsonl          : (optional) raw list of neighbor candidates dropped by
                                        min_similarity / NaN (for debugging)
- <run_id>.groups.json                : mapping final kept_id -> list of member ids merged into it

Lineage event format:
{
  "event_type": "synonym_merge",          # only merges are edges
  "iteration": 2,                         # which iteration produced this merge
  "new_pair_id": "<kept_id>",
  "source_pair_ids": ["<dropped_id>"],
  "kept": {"id": "...", "descriptor": "...", "explainer": "..."},
  "dropped": {"id": "...", "descriptor": "...", "explainer": "..."},
  "similarity": 0.8735,                  # cosine on pooled embeddings (if available)
  "decision_reason": "LLM_decision|cache_hit|below_min_similarity|parse_error"
}

Non-merges are fully recorded in iteration_XX_all_candidates.jsonl, but they do not
emit edges in the lineage files.
"""
from __future__ import annotations

# ===== Standard library =====
import argparse
from collections import defaultdict
from dataclasses import dataclass
import functools
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Literal

# ===== Third-party =====
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
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_file = logging_dir / f"{logging_dir.name}.log"
    level = (
        logging.WARNING
        if verbosity <= 0
        else (logging.INFO if verbosity == 1 else logging.DEBUG)
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        t1 = time.time()
        logging.info(
            "Execution of %s took %s.",
            func.__name__,
            time.strftime("%H:%M:%S", time.gmtime(t1 - t0)),
        )
        return out

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
        d, e = self.descriptor.strip(), self.explainer.strip()
        return f"{d}; {e}" if d and e else (d or e)


@dataclass
class Decision:
    is_synonym: bool
    keep_id: str
    drop_id: str
    reason: str


class DecisionSchema(BaseModel):
    is_synonym: bool
    keep: Literal["A", "B"]
    drop: Literal["A", "B"]


@dataclass
class MergeResult:
    kept: str
    dropped: str
    iteration: int
    similarity: Optional[float]
    reason: str  # LLM_decision | cache_hit | below_min_similarity | parse_error


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


@log_execution_time
def read_jsonl(path: Path, sample_size: Optional[int] = None) -> List[Pair]:
    limit = sample_size
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            d = str(obj.get("descriptor")).strip()
            e = str(obj.get("explainer")).strip()
            if not d or not e:
                raise ValueError(
                    f"Missing descriptor or explainer at line {i+1} in {path}"
                )
            pid = obj.get("id")
            if not pid:
                raise ValueError(f"Missing id at line {i+1} in {path}")
            pairs.append(Pair(id=str(pid), descriptor=d, explainer=e))

            if limit is not None and len(pairs) >= limit:
                logging.info("Test mode: limiting to %d unique IDs", limit)
                break

    assert pairs, f"No valid pairs found in {path}"
    return pairs


def write_jsonl(path: Path, pairs: Sequence[Pair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "id": p.id,
                        "descriptor": p.descriptor,
                        "explainer": p.explainer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logging.info("Wrote %d pairs -> %s", len(pairs), path)


def _log_dropped(
    pairs: Sequence[Pair], dropped: List[Tuple[int, int, float]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, j, s in dropped:
            a, b = pairs[i], pairs[j]
            reason = "nan_similarity" if np.isnan(s) else "below_min_similarity"
            rec = {
                "i": i,
                "j": j,
                "a_id": a.id,
                "b_id": b.id,
                "a_descriptor": a.descriptor,
                "b_descriptor": b.descriptor,
                "a_explainer": a.explainer,
                "b_explainer": b.explainer,
                "similarity": None if np.isnan(s) else round(float(s), 6),
                "reason": reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_all_decisions_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
class StellaEmbedder:
    def __init__(
        self, cache_dir: Optional[Path], batch_size: int = 128, device: str = "cuda:0"
    ) -> None:
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        # Single-GPU (or CPU fallback)
        want_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device if want_cuda else "cpu")
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(cache_dir) if cache_dir else None,
                # dtype=torch_dtype,
            )
            .to(self.device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.batch_size = batch_size
        self.model_name = model_name  # for fingerprint/meta

    @staticmethod
    def _save_embeds(path: Path, emb: np.ndarray, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        np.savez_compressed(
            str(path), emb=emb, meta=np.frombuffer(meta_bytes, dtype=np.uint8)
        )

    @staticmethod
    def _load_embeds(path: Path) -> Tuple[np.ndarray, dict]:
        with np.load(str(path), allow_pickle=False) as z:
            emb = z["emb"].astype("float32", copy=False)
            meta_bytes = bytes(z["meta"].tolist())
            meta = json.loads(meta_bytes.decode("utf-8"))
            return emb, meta

    @staticmethod
    def _fingerprint_texts(texts: Sequence[str], model_name: str) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode("utf-8"))
        for t in texts:
            h.update(b"\x00")
            h.update((t or "").encode("utf-8", "ignore"))
        return h.hexdigest()

    @log_execution_time
    def embed_texts(
        self, texts: Sequence[str], cache_path: Optional[Path] = None
    ) -> np.ndarray:
        # 1) Load from cache if available & matches fingerprint
        fp = self._fingerprint_texts(texts, self.model_name)
        if cache_path and cache_path.exists():
            try:
                cached_emb, meta = self._load_embeds(cache_path)
                if (
                    meta.get("fingerprint") == fp
                    and meta.get("model_name") == self.model_name
                ):
                    logging.info("Loaded embeddings from cache: %s", cache_path)
                    return cached_emb
                else:
                    logging.info(
                        "Cache exists but fingerprint/model mismatch; re-embedding."
                    )
            except Exception as e:
                logging.warning("Failed to load cache (%s); re-embedding.", e)

        # 2) Compute embeddings
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(
                    self.device
                )  # single-device path

                outputs = self.model(**inputs)
                hidden = (
                    outputs[0]
                    if isinstance(outputs, (tuple, list))
                    else outputs.last_hidden_state
                )
                attn = inputs["attention_mask"]  # already on self.device
                masked = hidden.masked_fill(~attn[..., None].bool(), 0.0)
                pooled = masked.sum(dim=1) / attn.sum(dim=1)[..., None]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.detach().cpu().numpy().astype("float32"))

        embeddings = np.vstack(out) if out else np.zeros((0, 1024), dtype="float32")

        # 3) Save to cache
        if cache_path:
            meta = {
                "fingerprint": fp,
                "normalized": True,
                "dtype": embeddings.dtype.name,
                "shape": list(embeddings.shape),
                "model_name": self.model_name,
                "device": str(self.device),
            }
            self._save_embeds(cache_path, embeddings, meta)
            logging.info("Saved embeddings to %s", cache_path)

        return embeddings

    def drop_model(self) -> None:
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Neighbor search
# -----------------------------------------------------------------------------


@log_execution_time
def find_nn(embeddings: np.ndarray, k: int = 1, dtype=torch.float16):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    M = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    M = torch.nn.functional.normalize(M, p=2, dim=1)
    logging.info("Finding nearest neighbours. Matrix shape %s", tuple(M.shape))
    slice_size = 1000
    N = M.shape[0]
    vals: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, N, slice_size):
            row = M[i : i + slice_size]
            sim = row @ M.T
            diag_rows = torch.arange(i, min(i + slice_size, N), device=device)
            sim[torch.arange(sim.size(0), device=device), diag_rows] = float("-inf")
            topv, topi = torch.topk(sim, k=k, dim=1)
            vals.append(topv.cpu().numpy())
            idxs.append(topi.cpu().numpy())
    return np.vstack(vals), np.vstack(idxs)


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
        self.cache_dir = os.environ.get("HF_HOME") or (
            str(cache_dir) if cache_dir else None
        )
        self.temperature = temperature
        self._llm = None
        self.cache_db_path = cache_db_path
        self._cache = (
            SqliteDict(str(cache_db_path), autocommit=True) if cache_db_path else None
        )
        self.batch_size = batch_size

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPU available.")
            logging.info("Loading LLM, using %d GPU(s).", n_gpus)
            logging.info("Batch size per LLM call: %d", self.batch_size)
            self._llm = LLM(
                model=self.model_name,
                download_dir=self.cache_dir,
                dtype="bfloat16",
                max_model_len=2048,
                tensor_parallel_size=n_gpus,
                enforce_eager=False,
                gpu_memory_utilization=0.85,
            )
        return self._llm

    def _response_format(self) -> GuidedDecodingParams:
        return GuidedDecodingParams(json=DecisionSchema.model_json_schema())

    def _prompt(self, a: Pair, b: Pair) -> str:
        return (
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are deciding whether two descriptor–explainer pairs, A and B, refer to the same concept.
The descriptors describe a document, while the explainer provides additional context.

Treat them as synonyms (set is_synonym=true) if any of the following is true:
    - They refer to the same concept or entity despite different wording.
    - They deviate slightly in specificity but still cover the same core meaning.
    - They differ in phrasing but could be used interchangeably in most contexts.
    - They differ in word order, punctuation or spelling but mean the same thing.
    - They list similar things in different order.
    
Do NOT treat them as synonyms (set is_synonym=false) if any of the following is true:
    - They refer to different concepts/entities or mutually exclusive categories.
    - One is substantially broader/narrower in scope or specificity.
    - The explainer introduces a difference in meaning even when descriptors are similar.

In edge cases, prefer to NOT treat as synonyms.

If synonyms:
    - choose which to keep: "A" or "B" (prefer clearer, more general phrasing);
    - set keep to the chosen label and drop to the other.

If not synonyms:
    - Still return keep and drop as A and B (order doesn't matter).

Return ONLY JSON with keys: is_synonym (bool), keep ("A"|"B"), drop ("A"|"B").
Do not rewrite or invent descriptors.
<|eot_id|><|start_header_id|>user<|end_header_id|>"""
            + f"""
A:
descriptor: {a.descriptor}
explainer: {a.explainer}
==============
B:
descriptor: {b.descriptor}
explainer: {b.explainer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )

    def _cache_key(self, a: Pair, b: Pair) -> str:
        key = tuple(
            sorted(
                [(a.id, a.descriptor, a.explainer), (b.id, b.descriptor, b.explainer)],
                key=lambda x: x[0],
            )
        )
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
                (gen_tok / elapsed if gen_tok else 0),
                tot_tok,
                elapsed,
            )

    @log_execution_time
    def decide_batch(self, pairs: List[Tuple[Pair, Pair]]) -> List[Decision]:
        decisions: List[Decision] = [None] * len(pairs)  # type: ignore
        uncached_idxs: List[int] = []
        if self._cache is not None:
            for i, (a, b) in enumerate(pairs):
                ck = self._cache_key(a, b)
                if ck in self._cache:
                    try:
                        obj = json.loads(self._cache[ck])
                        decisions[i] = Decision(
                            bool(obj.get("is_synonym", False)),
                            str(obj.get("keep_id", "")),
                            str(obj.get("drop_id", "")),
                            reason=str(obj.get("reason", "cache_hit")),
                        )
                        continue
                    except Exception:
                        pass
                uncached_idxs.append(i)
        else:
            uncached_idxs = list(range(len(pairs)))

        if not uncached_idxs:
            logging.info("All candidates cached.")
            return decisions  # type: ignore

        def _plen(i: int) -> int:
            a, b = pairs[i]
            return (
                len(a.descriptor)
                + len(a.explainer)
                + len(b.descriptor)
                + len(b.explainer)
            )

        # Sort by prompt length (ascending) to improve batching efficiency
        uncached_idxs.sort(key=_plen)

        logging.info(
            "%d uncached candidate pairs to evaluate with LLM.", len(uncached_idxs)
        )

        for start in range(0, len(uncached_idxs), self.batch_size):
            logging.info(
                "Processing batch %d out of %d",
                start // self.batch_size + 1,
                (len(uncached_idxs) + self.batch_size - 1) // self.batch_size,
            )
            chunk = uncached_idxs[start : start + self.batch_size]
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

            # LLM call
            t0 = time.time()
            outputs = self.llm.generate(
                batch_prompts, sampling_params=params, use_tqdm=False
            )
            t1 = time.time()

            # Log throughput
            in_tok, gen_tok = self._tokens_in_outputs(outputs)
            self._log_throughput(in_tok, gen_tok, t1 - t0)

            parsed: List[Decision] = []
            for out_idx, out in enumerate(outputs):
                a, b = batch_pairs[out_idx]
                outs = getattr(out, "outputs", None) or []
                if not outs:
                    raise ValueError("LLM returned no outputs.")

                text = (outs[0].text or "").strip(" `\n").removeprefix("json").strip()
                try:
                    obj = json_repair.loads(text)
                    keep_lbl = str(obj.get("keep", "")).strip().upper()
                    drop_lbl = str(obj.get("drop", "")).strip().upper()
                    is_syn = bool(obj.get("is_synonym", False))

                    # Ensure IDs in output are valid
                    if (
                        keep_lbl not in {"A", "B"}
                        or drop_lbl not in {"A", "B"}
                        or keep_lbl == drop_lbl
                    ):
                        parsed.append(
                            Decision(False, a.id, b.id, reason="invalid_ids_in_output")
                        )
                        logging.warning(
                            "Invalid IDs in LLM output; treating as non-merge: %s", text
                        )
                        continue
                    # Map A and B to actual IDs
                    if keep_lbl == "A":
                        keep_id, drop_id = a.id, b.id
                    else:
                        keep_id, drop_id = b.id, a.id
                    parsed.append(
                        Decision(is_syn, keep_id, drop_id, reason="LLM_decision")
                    )

                except Exception as exc:
                    a, b = batch_pairs[out_idx]
                    logging.warning("Parse error; treating as non-merge: %s", exc)
                    parsed.append(Decision(False, a.id, b.id, reason="parse_error"))

            for local_idx, global_idx in enumerate(chunk):
                d = parsed[local_idx]
                decisions[global_idx] = d
                # Update cache if LLM-produced decision. Do not cache parse errors or other abnormalities.
                if (
                    self._cache is not None
                    and decisions[global_idx].reason == "LLM_decision"
                ):
                    a, b = pairs[global_idx]
                    ck = self._cache_key(a, b)
                    self._cache[ck] = json.dumps(d.__dict__, ensure_ascii=False)

        # Garbage collect after LLM use to free GPU memory
        del outputs
        del batch_prompts
        del batch_pairs
        torch.cuda.synchronize()  # Ensure all GPU work is done
        torch.cuda.empty_cache()
        gc.collect()

        return decisions


# -----------------------------------------------------------------------------
# Candidate construction
# -----------------------------------------------------------------------------


def candidate_synonyms(
    neighbor_idx: np.ndarray,
    scores: np.ndarray,
    *,
    strategy: Literal["mutual", "unique"] = "mutual",
    min_similarity: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    assert neighbor_idx.shape == scores.shape
    n, k = neighbor_idx.shape
    kept: Dict[Tuple[int, int], float] = {}
    dropped_map: Dict[Tuple[int, int], float] = {}

    def _maybe_keep(a: int, b: int, sim: float) -> None:
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            return
        i, j = (a, b) if a < b else (b, a)
        if min_similarity is not None and (
            np.isnan(sim) or sim < float(min_similarity)
        ):
            prev = dropped_map.get((i, j))
            if prev is None or sim > prev:
                dropped_map[(i, j)] = sim
            return
        prev = kept.get((i, j))
        if (prev is None) or (sim > prev):
            kept[(i, j)] = sim

    if n == 0 or k == 0:
        return [], []
    if strategy == "mutual":
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
        for i in range(n):
            for col in range(k):
                j = int(neighbor_idx[i, col])
                if j < 0 or j == i:
                    continue
                sim = float(scores[i, col]) if scores.size else float("-inf")
                _maybe_keep(i, j, sim)

    triples = [(i, j, s) for (i, j), s in kept.items()]
    triples.sort(key=lambda t: t[2], reverse=True)
    dropped_list = [(i, j, s) for (i, j), s in dropped_map.items()]
    dropped_list.sort(key=lambda t: (np.isnan(t[2]), t[2]))

    if triples:
        sims = [t[2] for t in triples]
        logging.info(
            "Pair similarity — avg: %.4f | max: %.4f | min: %.4f",
            sum(sims) / len(sims),
            max(sims),
            min(sims),
        )
    logging.info(
        "Dropped %d candidate pairs due to threshold %s.",
        len(dropped_list),
        "None" if min_similarity is None else f"{float(min_similarity):.4f}",
    )
    return triples, dropped_list


# -----------------------------------------------------------------------------
# Merge application + lineage writing
# -----------------------------------------------------------------------------


def apply_merges_write_lineage(
    pairs: List[Pair],
    decisions: List[Decision],
    triples: List[Tuple[int, int, float]],
    iteration: int,
    lineage_path: Path,
    all_candidates_path: Path,
) -> Tuple[List[Pair], List[MergeResult]]:
    id_to_pair: Dict[str, Pair] = {p.id: p for p in pairs}
    used: Set[str] = set()
    merges: List[MergeResult] = []

    # Build lookup from (a.id,b.id) by index to similarity for bookkeeping
    sim_map: Dict[Tuple[str, str], float] = {}
    for i, j, s in triples:
        a, b = pairs[i], pairs[j]
        key = tuple(sorted((a.id, b.id)))
        sim_map[key] = float(s)

    # Compose records for ALL candidates (evaluated=True) as we go
    all_records: List[dict] = []

    for i_j_s, d in zip(triples, decisions):
        i, j, s = i_j_s
        a, b = pairs[i], pairs[j]
        # Sanity: decision must reference these IDs only
        if {d.keep_id, d.drop_id} - {a.id, b.id}:
            d = Decision(False, a.id, b.id, reason=(d.reason or "invalid_ids"))
        # One merge per id per iteration
        if d.is_synonym:
            keep, drop = d.keep_id, d.drop_id
            if keep == drop or keep in used or drop in used:
                # Treat as non-merge in this iteration
                all_records.append(
                    {
                        "iteration": iteration,
                        "evaluated": True,
                        "is_synonym": False,
                        "a_id": a.id,
                        "b_id": b.id,
                        "similarity": float(s),
                        "reason": "already_used_or_self",
                    }
                )
                continue
            if keep not in id_to_pair or drop not in id_to_pair:
                all_records.append(
                    {
                        "iteration": iteration,
                        "evaluated": True,
                        "is_synonym": False,
                        "a_id": a.id,
                        "b_id": b.id,
                        "similarity": float(s),
                        "reason": "id_missing",
                    }
                )
                continue
            used.add(keep)
            used.add(drop)
            merges.append(
                MergeResult(
                    kept=keep,
                    dropped=drop,
                    iteration=iteration,
                    similarity=float(s),
                    reason=d.reason,
                )
            )
        # record decision
        all_records.append(
            {
                "iteration": iteration,
                "evaluated": True,
                "is_synonym": bool(d.is_synonym),
                "a_id": a.id,
                "b_id": b.id,
                "similarity": float(s),
                "keep_id": d.keep_id,
                "drop_id": d.drop_id,
                "reason": d.reason,
            }
        )

    # Write per-iteration lineage edges for merges
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        for m in merges:
            kept_p = id_to_pair[m.kept]
            dropped_p = id_to_pair[m.dropped]

            # ---- Write lineage event ----
            rec = {
                "event_type": "synonym_merge",
                "iteration": m.iteration,
                "new_pair_id": kept_p.id,
                "source_pair_ids": [dropped_p.id],
                "kept": {
                    "id": kept_p.id,
                    "descriptor": kept_p.descriptor,
                    "explainer": kept_p.explainer,
                },
                "dropped": {
                    "id": dropped_p.id,
                    "descriptor": dropped_p.descriptor,
                    "explainer": dropped_p.explainer,
                },
                "similarity": m.similarity,
                "decision_reason": m.reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write all candidates for this iteration (evaluated ones only; skipped by threshold are added by caller)
    _write_all_decisions_jsonl(all_candidates_path, all_records)

    # Drop merged items
    dropped_ids = {m.dropped for m in merges}
    new_pairs = [id_to_pair[p.id] for p in pairs if p.id not in dropped_ids]
    return new_pairs, merges


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


@log_execution_time
def iterate_until_converged(
    pairs: List[Pair],
    embedded_texts: np.ndarray,
    k: int,
    max_iters: int,
    min_similarity: Optional[float],
    output_dir: Optional[Path],
    combiner: SynonymCombiner,
    candidate_strategy: Literal["mutual", "unique"],
    *,
    start_iter: int = 1,
) -> Tuple[List[Pair], List[List[MergeResult]]]:

    all_merges: List[List[MergeResult]] = []
    iteration = start_iter - 1

    emb_all = embedded_texts
    current = list(range(len(pairs)))

    while iteration < max_iters:
        iteration += 1
        logging.info("=== Iteration %d | %d pairs ===", iteration, len(pairs))
        if len(pairs) <= 1:
            break

        emb = emb_all[current]
        logging.info("Finding neighbors...")
        sims, neigh = find_nn(emb, k=k)

        triples, dropped = candidate_synonyms(
            neigh, sims, strategy=candidate_strategy, min_similarity=min_similarity
        )
        # Log raw drops
        if output_dir:
            _log_dropped(
                pairs, dropped, output_dir / f"iteration_{iteration}_dropped.jsonl"
            )

        if not triples:
            logging.info("No candidate proposals. Stopping.")
            break

        logging.info("Found %d candidate pairs.", len(triples))
        ordered_pairs = [(pairs[i], pairs[j]) for i, j, _ in triples]
        decisions = combiner.decide_batch(ordered_pairs)

        # Also record candidates SKIPPED due to threshold in the all-candidates file
        skipped_records: List[dict] = []
        for i, j, s in dropped:
            a, b = pairs[i], pairs[j]
            skipped_records.append(
                {
                    "iteration": iteration,
                    "evaluated": False,
                    "is_synonym": False,
                    "a_id": a.id,
                    "b_id": b.id,
                    "similarity": None if np.isnan(s) else float(s),
                    "reason": (
                        "below_min_similarity" if not np.isnan(s) else "nan_similarity"
                    ),
                }
            )
        # Write them in a temp file; we'll append evaluated ones inside apply_merges_write_lineage
        if output_dir:
            tmp_all = output_dir / f"iteration_{iteration}_all_candidates.jsonl"
            _write_all_decisions_jsonl(tmp_all, skipped_records)

        # Apply merges + write per-iteration lineage and the evaluated candidates
        lineage_iter_path = (
            output_dir / f"iteration_{iteration}_lineage.jsonl"
            if output_dir
            else Path(f"iteration_{iteration}_lineage.jsonl")
        )
        all_cand_iter_path = (
            output_dir / f"iteration_{iteration}_all_candidates.jsonl"
            if output_dir
            else Path(f"iteration_{iteration}_all_candidates.jsonl")
        )
        new_pairs, merges = apply_merges_write_lineage(
            pairs, decisions, triples, iteration, lineage_iter_path, all_cand_iter_path
        )

        all_merges.append(merges)

        # Update current view after drops
        if merges:
            dropped_ids = {m.dropped for m in merges}
            kept_mask = [p.id not in dropped_ids for p in pairs]
            current = [idx for idx, keep in zip(current, kept_mask) if keep]
        pairs = new_pairs

        logging.info(
            "Iteration %d merged %d pairs; %d remain",
            iteration,
            len(merges),
            len(pairs),
        )
        if output_dir:
            write_jsonl(output_dir / f"checkpoint_iter_{iteration}.jsonl", pairs)

        if len(merges) == 0:
            logging.info("No merges this iteration. Converged.")
            break
        # Memory tidy
        try:
            del emb, sims, neigh, triples, ordered_pairs, decisions
            gc.collect()
        except NameError:
            pass

    return pairs, all_merges


# -----------------------------------------------------------------------------
# DSU (for groups)
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
# CLI entry
# -----------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_dir = Path("../results/synonym_merges") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.run_id}.jsonl"
    lineage_concat_path = out_dir / f"{args.run_id}_lineage.jsonl"
    decision_cache_path = out_dir / (args.decision_cache or "decision_cache.sqlite")

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else (Path(os.environ.get("HF_HOME")) if os.environ.get("HF_HOME") else None)
    )

    setup_logging(out_dir, args.verbose)
    with open(out_dir / f"{args.run_id}_settings.txt", "a", encoding="utf-8") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for k, v in sorted(vars(args).items()):
            logging.info(f"{k}: {v}")
            f.write(f"{k}: {v}\n")
        f.write("===========================\n")

    # Resume or fresh start
    if args.resume_from is not None:
        ckpt_path = out_dir / f"checkpoint_iter_{args.resume_from}.jsonl"
        if not ckpt_path.exists():
            raise SystemExit(
                f"--resume-from {args.resume_from} but {ckpt_path} not found."
            )
        logging.info("Resuming from %s", ckpt_path)
        pairs = read_jsonl(ckpt_path, sample_size=None)
        start_iter = args.resume_from + 1
    else:
        pairs = read_jsonl(input_path, sample_size=args.test)
        start_iter = 1

    all_ids = [p.id for p in pairs]

    logging.info("Starting embedding model...")
    embedder = StellaEmbedder(cache_dir=cache_dir, batch_size=args.batch_size)
    logging.info("Embedding %d pairs...", len(pairs))
    embedded_texts = embedder.embed_texts(
        [p.text for p in pairs], cache_path=Path(args.embedding_cache)
    )
    # Remove model and tokenizer to free GPU memory
    embedder.drop_model()

    combiner = SynonymCombiner(
        model_name=args.model,
        cache_dir=cache_dir,
        temperature=args.temperature,
        cache_db_path=Path(decision_cache_path),
        batch_size=args.llm_batch_size,
    )

    final_pairs, all_merges = iterate_until_converged(
        pairs=pairs,
        embedded_texts=embedded_texts,
        k=args.k,
        max_iters=args.max_iters,
        min_similarity=args.min_similarity,
        output_dir=out_dir,
        combiner=combiner,
        candidate_strategy=args.candidate_strategy,
        start_iter=start_iter,
    )

    # Persist results
    write_jsonl(output_path, final_pairs)

    def _iter_num(path: Path) -> int:
        m = re.search(r"iteration_(\d+)_lineage\.jsonl$", path.name)
        return int(m.group(1)) if m else -1

    # Write concatenated lineage (edges only) across iterations
    with lineage_concat_path.open("w", encoding="utf-8") as out:
        for p in sorted(out_dir.glob("iteration_*_lineage.jsonl"), key=_iter_num):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    logging.info("Wrote lineage -> %s", lineage_concat_path)

    # Build transitive groups keyed by final kept IDs
    dsu = DSU()
    for round_merges in all_merges:
        for m in round_merges:
            dsu.union(m.kept, m.dropped)

    clusters: Dict[str, Set[str]] = defaultdict(set)
    for pid in all_ids:
        clusters[dsu.find(pid)].add(pid)

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
    p = argparse.ArgumentParser(
        description="Combine synonym descriptor-explainer pairs with embeddings + LLM (with lineage)"
    )
    p.add_argument("--run-id", type=str, default="default", help="Run ID")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL with fields: id, descriptor, explainer",
    )
    p.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Resume from iteration N (loads checkpoint_iter_N.jsonl and continues with N+1)",
    )

    # Embeddings
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    p.add_argument(
        "--embedding-cache",
        type=str,
        default="../results/synonym_merges/embeddings.npz",
    )

    # Neighbor params (brute-force search)
    p.add_argument(
        "--k", type=int, default=1, help="#neighbors per item (excluding self)"
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum cosine sim to send pair to LLM",
    )
    p.add_argument(
        "--candidate-strategy", choices=["unique", "mutual"], default="mutual"
    )

    # LLM
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--decision-cache", type=str, default="decision_cache.sqlite")
    p.add_argument("--llm-batch-size", type=int, default=512)

    # Loop
    p.add_argument("--max-iters", type=int, default=50, help="Maximum iterations")
    p.add_argument(
        "--test",
        nargs="?",
        const=10000,
        default=None,
        type=int,
        help="Limit items in test mode (default 10k)",
    )
    p.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")

    args = p.parse_args()
    t0 = time.time()
    main(args)
    t1 = time.time()
    logging.info("========================================")
    logging.info("Finished. Took %s.", time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
    logging.info("========================================")
