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
"""
from __future__ import annotations

# Standard library imports
import argparse
from collections import defaultdict
from dataclasses import dataclass
import functools
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    representative_descriptor: str
    reason: str


# Pydantic schema for guided JSON decoding
class DecisionSchema(BaseModel):
    is_synonym: bool
    keep_id: str
    drop_id: str
    representative_descriptor: str
    reason: str


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
            item_id = obj.get("id") or f"row_{i:06d}"
            pairs.append(Pair(id=item_id, descriptor=d, explainer=e))
            if limit is not None and len(pairs) >= limit:
                logging.info("Test mode: limiting to %d items", limit)
                break
    logging.info("Loaded %d pairs", len(pairs))
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
    logging.info("Wrote %d pairs -> %s", len(pairs), path)


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
            # Build index with Inner-Produce quantizer
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train on sample of embeddings
            train_sample = embeddings[np.random.choice(n, size=min(100_000, n), replace=False)]
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
        # Find similar embeddings (0-1), higher is more similar
        sims, idxs = index.search(embeddings, k + 1)
        return sims[:, 1:], idxs[:, 1:]  # Drop self


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
                max_model_len=16384,
                tensor_parallel_size=n_gpus,
                enforce_eager=False,
                gpu_memory_utilization=0.9,
            )
        return self._llm

    def _response_format(self) -> GuidedDecodingParams:
        return GuidedDecodingParams(json=DecisionSchema.model_json_schema())

    def _prompt(self, a: Pair, b: Pair) -> str:
        return ("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are deciding whether two descriptor–explainer pairs refer to the same concept.
            If they are synonyms (or one is a more representative phrasing of the other), return is_synonym=true,
            choose which ID to keep, which to drop, and set representative_descriptor to the best descriptor.
            If they are not synonyms or describe different concepts, return is_synonym=false and leave other fields coherent.
            Prefer concise, general, and commonly used phrasing when picking the representative descriptor.
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
        decisions: List[Decision] = [None] * len(pairs)
        # Process in batches
        for batch_start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[batch_start:batch_start + self.batch_size]
            batch_prompts: List[str] = []
            batch_to_query: List[int] = []
            batch_decisions: List[Decision] = []
            # Check cache first
            for idx_in_batch, (a, b) in enumerate(batch_pairs):
                global_idx = batch_start + idx_in_batch
                if self._cache is not None:
                    ck = self._cache_key(a, b)
                    if ck in self._cache:
                        try:
                            obj = json.loads(self._cache[ck])
                            decisions[global_idx] = Decision(**obj)
                            continue
                        except Exception:
                            pass
                # Not cached
                batch_prompts.append(self._prompt(a, b))
                batch_to_query.append(idx_in_batch)
                batch_decisions.append(Decision(is_synonym=False, keep_id=a.id, drop_id=b.id, representative_descriptor=a.descriptor, reason="uncached"))
            if batch_prompts:
                params = SamplingParams(
                    temperature=self.temperature,
                    top_p=0.5,
                    repetition_penalty=1.0,
                    max_tokens=1_024,
                    stop=["<|eot_id|>"],
                    guided_decoding=self._response_format(),
                    seed=42,
                )
                start = time.time()
                outputs = self.llm.generate(batch_prompts, sampling_params=params, use_tqdm=False)
                end = time.time()
                in_tok, gen_tok = self._tokens_in_outputs(outputs)
                self._log_throughput(in_tok, gen_tok, end-start)
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
                            representative_descriptor=str(obj.get("representative_descriptor", "")),
                            reason=str(obj.get("reason", "")),
                        ))
                    except Exception as exc:
                        logging.warning("Falling back to non-merge due to parse error: %s", exc)
                        # Map back to the corresponding pair index
                        pair_idx = batch_to_query[out_idx]
                        a, b = batch_pairs[pair_idx]
                        parsed.append(Decision(
                            is_synonym=False,
                            keep_id=a.id,
                            drop_id=b.id,
                            representative_descriptor=a.descriptor,
                            reason="parse_error",
                        ))
                # Stitch parsed decisions into the full list in order
                p_iter = iter(parsed)
                for pos in batch_to_query:
                    d = next(p_iter)
                    global_pos = batch_start + pos
                    decisions[global_pos] = d
                # Write only uncached results back to cache
                if self._cache is not None:
                    for pos in batch_to_query:
                        a, b = batch_pairs[pos]
                        d = batch_decisions[pos]
                        ck = self._cache_key(a, b)
                        try:
                            self._cache[ck] = json.dumps(d.__dict__, ensure_ascii=False)
                        except Exception:
                            pass
        return decisions

# -----------------------------------------------------------------------------
# Merging & iteration
# -----------------------------------------------------------------------------

@dataclass
class MergeResult:
    kept: str
    dropped: str
    updated_descriptor: Optional[str]


def _unique_pairs_from_neighbors(
    ids: Sequence[str],
    neighbor_idx: np.ndarray,
    scores: np.ndarray,
    min_similarity: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """Return a deduplicated list of (i, j, dist) where i < j.

    Consider **all k neighbors** for each i, deduplicate by unordered pair, and
    keep the highest similarity when a pair appears multiple times.

    If `min_similarity` is set, discard candidates whose similarity is lower.
    """
    assert neighbor_idx.shape == scores.shape
    proposals: Dict[Tuple[int, int], float] = {}
    
    # n = items, k = neighbors per item
    n, k = neighbor_idx.shape
    for i in range(n):
        for col in range(k):
            j = int(neighbor_idx[i, col])
            if j < 0 or j == i:
                continue
            sim = float(scores[i, col]) if scores.size else float("-inf")
            
            # Drop neighbors with low similarity
            if min_similarity is not None and not np.isnan(sim) and sim > min_similarity:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) not in proposals or sim > proposals[(a, b)]:
                proposals[(a, b)] = sim
    triples = [(a, b, d) for (a, b), d in proposals.items()]
    triples.sort(key=lambda t: t[2], reverse=True)  # descending similairty
    return triples


def mutual_top1(
    ids: Sequence[str],
    neighbor_idx: np.ndarray,
    scores: np.ndarray,
    min_similarity: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    n, k = neighbor_idx.shape
    if k < 1 or n == 0:
        return []
    pairs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        j = int(neighbor_idx[i, 0])  # i’s best neighbor
        if j < 0 or j == i or j >= n:
            continue
        if int(neighbor_idx[j, 0]) != i:  # require mutual top-1
            continue
        # pick a single score to rank by (use the better of the two directions)
        sim_ij = float(scores[i, 0]) if scores.size else float("-inf")
        sim_ji = float(scores[j, 0]) if scores.size else float("-inf")
        sim = max(sim_ij, sim_ji)
        if min_similarity is not None and (np.isnan(sim) or sim < min_similarity):
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in pairs or sim > pairs[(a, b)]:
            pairs[(a, b)] = sim
    triples = [(a, b, s) for (a, b), s in pairs.items()]
    triples.sort(key=lambda t: t[2], reverse=True)  # descending similarity
    return triples


def _apply_merges_greedy(
    pairs: List[Pair],
    decisions: List[Decision],
) -> Tuple[List[Pair], List[MergeResult]]:
    id_to_pair: Dict[str, Pair] = {p.id: p for p in pairs}
    used: Set[str] = set()
    merges: List[MergeResult] = []

    for d in decisions:
        if not d.is_synonym:
            continue
        keep = d.keep_id
        drop = d.drop_id
        if keep == drop:
            continue
        if keep not in id_to_pair or drop not in id_to_pair:
            continue
        if keep in used or drop in used:
            continue  # already merged in this round

        keep_pair = id_to_pair[keep]

        # Update the descriptor of the kept pair to the representative descriptor (if provided)
        updated_descriptor = d.representative_descriptor.strip() or keep_pair.descriptor
        keep_pair = Pair(id=keep_pair.id, descriptor=updated_descriptor, explainer=keep_pair.explainer)
        id_to_pair[keep] = keep_pair

        # Mark drop as consumed
        used.add(keep)
        used.add(drop)
        merges.append(MergeResult(kept=keep, dropped=drop, updated_descriptor=updated_descriptor))

    # Build new list: keep everything not dropped; replaced kept items are already in id_to_pair
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
) -> Tuple[List[Pair], List[List[MergeResult]]]:
    all_merges: List[List[MergeResult]] = []
    iteration = 0

    while iteration < max_iters:
        iteration += 1
        logging.info("=== Iteration %d | %d pairs ===", iteration, len(pairs))
        if len(pairs) <= 1:
            break

        logging.info("Embedding descriptor-explainer pairs...")
        # 1) Embed
        texts = [p.text for p in pairs]
        emb = embedder.embed_texts(texts)

        logging.info("Building Faiss index...")
        # 2) Index
        index = faiss_index.build(emb)

        logging.info("Finding neighbors...")
        # 3) Neighbors
        sims, neigh = faiss_index.neighbors(index, emb, k=k)

        # 4) Build candidate pairs
        if args.candidate_strategy == "mutual":
            # Pairs have to be mutual nearest neighbors
            triples = mutual_top1([p.id for p in pairs], neigh, sims, min_similarity=min_similarity)
        else:
            # Find unique candidate pairs
            triples = _unique_pairs_from_neighbors([p.id for p in pairs], neigh, sims, min_similarity=min_similarity)
        if not triples:
            logging.info("No neighbor proposals. Stopping.")
            break

        logging.info("Prompting LLM...")
        # 5) Query LLM on candidates in order
        ordered_pairs: List[Tuple[Pair, Pair]] = []
        for i, j, _ in triples:
            ordered_pairs.append((pairs[i], pairs[j]))

        decisions = combiner.decide_batch(ordered_pairs)

        # 6) Greedy apply merges (skip items already merged this round)
        pairs, merges = _apply_merges_greedy(pairs, decisions)
        all_merges.append(merges)
        logging.info("Iteration %d merged %d pairs; %d remain", iteration, len(merges), len(pairs))

        # 7) Convergence check
        if len(merges) == 0:
            logging.info("No merges in this iteration. Converged.")
            break

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

    # Setup paths and logging
    input_path = Path(args.input)
    output_dir = Path("../results/synonym_merges") / args.run_id
    output_path = output_dir / f"{args.run_id}.jsonl"
    decision_cache_path = output_dir / (args.decision_cache or "decision_cache.sqlite")

    # HF cache dir (optional)
    if args.cache_dir:
        cache_dir: Optional[Path] = Path(args.cache_dir)
    else:
        hf_home = os.environ.get("HF_HOME")
        cache_dir = Path(hf_home) if hf_home else None

    setup_logging(output_dir, args.verbose)
    
    # Read input
    pairs = read_jsonl(input_path, sample_size=args.test)
    all_ids = {p.id for p in pairs}
    
    # Initialise combiner
    combiner = SynonymCombiner(
        model_name=args.model,
        cache_dir=cache_dir,
        temperature=args.temperature,
        cache_db_path=Path(decision_cache_path),
        batch_size = args.llm_batch_size
    )

    logging.info("Loading model...")
    # Access the LLM to trigger loading
    _ = combiner.llm
    
    # Embed and build index helpers
    embedder = StellaEmbedder(cache_dir=cache_dir, batch_size=args.batch_size)
    faiss_index = FaissIndex(nlist=args.faiss_nlist, nprobe=args.faiss_nprobe)
    
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
    parser.add_argument("--cache_dir", type=str, default=None, help="HF cache dir (optional)")
    parser.add_argument("--batch_size", type=int, default=32)

    # FAISS
    parser.add_argument("--faiss_nlist", type=int, default=100)
    parser.add_argument("--faiss_nprobe", type=int, default=10)
    parser.add_argument("--k", type=int, default=1, help="#neighbors to fetch per item (excluding self)")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help=(
            "Minimum similarity on pooled embeddings; if a candidate neighbor is less similar than this, "
            "skip sending that pair to the LLM"
        ),
    )

    # LLM
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name or local path")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--decision_cache", type=str, default="decision_cache.sqlite", help="Sqlite file to cache LLM decisions")
    parser.add_argument("--llm_batch_size", type=int, default=512, help="Batch size for LLM queries")

    # Loop
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--test", nargs="?", const="10000", default=None, type=int, #nargs="?" means 0 or 1 argument 
                        help="Run with small batch of data. Defaults to 10k sample. Give value if you want to change sample size.")
    parser.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")
    parser.add_argument("--candidate_strategy", choices=["unique", "mutual"], default="mutual")

    args = parser.parse_args()
    main(args)
