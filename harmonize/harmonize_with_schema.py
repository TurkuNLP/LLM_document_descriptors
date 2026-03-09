from __future__ import annotations

"""
Harmonizer for descriptor;explainer pairs using:
- Stella embeddings (Marqo/dunzhang-stella_en_400M_v5)
- vLLM with guided JSON decoding to choose the best synonym (or none)
"""

# Standard library
import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import hashlib

# Third‑party
import json_repair  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from pydantic import BaseModel  # type: ignore
from sqlitedict import SqliteDict  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore

# Local imports
from embed_and_rerank import (
    QwenReranker,
    QwenEmbedder,
    StellaEmbedder,
    save_embeds,
    load_embeds,
    embedding_fingerprint,
    find_nn,
)
from logging_utils import setup_logging, log_execution_time, log_gpu_memory_info  # type: ignore

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
    raw: Any  # original parsed JSON object or raw string
    pairs: List[Pair]  # descriptor/explainer items for this row


@dataclass
class Decision:
    chosen_id: Optional[str]  # schema id selected by LLM; None => drop


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
            obj = json.loads(line)
            pairs.append(
                Pair(
                    id=obj["id"],
                    descriptor=obj["descriptor"],
                    explainer=obj["explainer"],
                )
            )
    return pairs


def load_descriptors(path: Path, max_rows: Optional[int] = None) -> List[InputRow]:
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
            if obj.get("similarity"):
                best_idx = int(np.argmax(obj["similarity"]))
                desc_exp = obj["descriptors"][best_idx]
            else:
                desc_exp = obj["descriptors"][0]

            pairs_for_row: List[Pair] = []
            for desc_i, d_e in enumerate(desc_exp):
                d, e = _split_descriptor_explainer(d_e)
                pid = f"{doc_i}_{desc_i}"
                pairs_for_row.append(Pair(id=pid, descriptor=d, explainer=e))

            rows.append(InputRow(row_idx=doc_i, raw=obj, pairs=pairs_for_row))

            if max_rows is not None and len(rows) >= max_rows:
                logging.info(
                    f"Loaded {max_rows} rows from input; stopping early for testing."
                )
                break

    return rows

# -----------------------------------------------------------------------------
# vLLM selector
# -----------------------------------------------------------------------------
class Harmonizer:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[Path],
        temperature: float = 0.1,
        batch_size: int = 64,
        cache_db: Optional[Path] = None,
        min_rerank_score: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = os.environ.get("HF_HOME") or (
            str(cache_dir) if cache_dir else None
        )
        self.temperature = temperature
        self.batch_size = batch_size
        self._llm: Optional[LLM] = None
        self._cache: Optional[SqliteDict] = (
            SqliteDict(str(cache_db), autocommit=True) if cache_db else None
        )
        self.min_rerank_score = min_rerank_score

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
    def _parse_text_to_obj(text: str) -> Optional[dict]:
        s = (text or "").strip().strip("` ")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        try:
            return json_repair.loads(s)
        except Exception as exc:
            logging.warning("Failed to parse JSON verdict (%s): %r", exc, s[:200])
            return None

    @staticmethod
    def _cache_key(query: Pair, candidates: List[Pair]) -> str:
        # key = {
        #     "query": {"id": query.id, "d": query.descriptor, "e": query.explainer},
        #     "candidates": [
        #         {"id": c.id, "d": c.descriptor, "e": c.explainer} for c in candidates
        #     ],
        # }
        # return json.dumps(key, ensure_ascii=False, sort_keys=True)
        
        # Use hashlib for faster hashing than JSON serialization
        key_parts = [
            query.id,
            query.descriptor,
            query.explainer,
            *[f"{c.id}:{c.descriptor}:{c.explainer}" for c in candidates]
        ]
        return hashlib.sha256("|".join(key_parts).encode('utf-8')).hexdigest()
    
    def _check_cache(
        self, batch: List[Tuple[Pair, List[Pair]]], decisions: List[Optional[Decision]]
    ) -> List[Optional[Decision]]:
        
        cache_hits = 0
        for i, (q, cands) in enumerate(batch):
            ck = self._cache_key(q, cands)
            cached = self._cache.get(ck)
            if cached:
                try:
                    obj = json.loads(cached)
                    decisions[i] = Decision(
                        chosen_id=self._clean_chosen_id(obj.get("chosen_id"))
                    )
                except Exception as e:
                    logging.warning("Failed to parse cached decision: %s", e)
                cache_hits += 1
        
        if cache_hits > 0:
            logging.info("Cache hits: %d/%d (%.1f%%)",
                        cache_hits, len(batch),
                        100.0 * cache_hits / len(batch))
                
        return decisions

    @staticmethod
    def _clean_chosen_id(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        s = str(x).strip().strip("`\"'")
        if s.lower().startswith("id="):
            s = s[3:].lstrip()
        return s or None


class RerankerChooser(Harmonizer):
    def __init__(
        self,
        model_name: str = "qwen/Qwen3-Reranker-0.6B",
        cache_dir: Optional[Path] = None,
        batch_size: int = 64,
        cache_db: Optional[Path] = None,
        min_rerank_score: float = 0.5,
    ) -> None:
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
            cache_db=cache_db,
            min_rerank_score=min_rerank_score,
        )
        self.reranker = QwenReranker(model_name=model_name)

    def choose_with_reranker(
        self, batch: List[Tuple[Pair, List[Pair]]], decisions: List[Optional[Decision]]
    ) -> None:
        # Prepare batched inputs for reranking
        batched_queries = []
        batched_candidates = []
        batch_indices = []  # To map results back to original positions

        # Prepare batch
        for i, (q, cands) in enumerate(batch):
            if decisions[i] is None and cands:
                batched_queries.extend([q.text] * len(cands))
                batched_candidates.extend([c.text for c in cands])
                # Store original position and candidate count for this query
                batch_indices.append((i, len(cands)))

        num_synonyms = 0
        num_non_synonyms = 0
        best_scores = []
        if batched_queries:
            rerank_scores = self.reranker.rerank(batched_queries, batched_candidates)
            score_ptr = 0
            for pos, cand_count in batch_indices:
                # Extract scores for this query's candidates
                query_scores = rerank_scores[score_ptr : score_ptr + cand_count]
                score_ptr += cand_count

                best_idx = int(np.argmax(query_scores))
                best_scores.append(query_scores[best_idx])
                # If best score is above threshold, select that candidate; otherwise drop (None)
                if query_scores[best_idx] > self.min_rerank_score:
                    # Need to get the actual candidate ID from original batch
                    original_cands = batch[pos][1]
                    decisions[pos] = Decision(chosen_id=original_cands[best_idx].id)
                    num_synonyms += 1
                else:
                    decisions[pos] = Decision(chosen_id=None)
                    num_non_synonyms += 1
        
            logging.info(
                "Reranker decisions: %d synonyms, %d non-synonyms, threshold: %f",
                num_synonyms, num_non_synonyms, self.min_rerank_score
                )
            logging.info(
                "Best rerank score stats: mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
                np.mean(best_scores), np.median(best_scores), np.min(best_scores), np.max(best_scores)
                )
            
        else:
            logging.info("No queries to rerank, all decisions retrived from cache.")
            
        return decisions

    @log_execution_time
    def select_batch(self, batch: List[Tuple[Pair, List[Pair]]]) -> List[Decision]:
        decisions: List[Optional[Decision]] = [None] * len(batch)

        # First check cache to skip any already-seen queries
        if self._cache is not None:
            decisions = self._check_cache(batch, decisions)

        # Then use reranker for the rest
        decisions = self.choose_with_reranker(batch, decisions)

        # Finally, save new decisions to cache
        if self._cache is not None:
            for pos, (q, cands) in enumerate(batch):
                if decisions[pos] is not None:
                    ck = self._cache_key(q, cands)
                    try:
                        chosen_id = decisions[pos].chosen_id
                        self._cache[ck] = json.dumps(
                            {"chosen_id": chosen_id or ""}, ensure_ascii=False
                        )
                    except Exception:
                        logging.warning("Failed to save decision to cache")
                        pass

        return [d if d is not None else Decision(chosen_id=None) for d in decisions]


class LLMChooser(Harmonizer):
    def choose_with_llm(
        self, batch: List[Tuple[Pair, List[Pair]]], decisions: List[Optional[Decision]]
    ) -> None:
        prompts: List[str] = []
        out_positions: List[int] = []

        for i, (q, cands) in enumerate(batch):
            if decisions[i] is None:
                prompts.append(self._prompt(q, cands))
                out_positions.append(i)

        if prompts:
            params = SamplingParams(
                temperature=self.temperature,
                top_p=0.9,
                max_tokens=512,
                structured_outputs=StructuredOutputsParams(self._response_format()),
                seed=42,
            )
            t0 = time.time()
            outputs = self.llm.generate(prompts, sampling_params=params)
            t1 = time.time()
            in_tok, gen_tok = self._extract_tokens(outputs)
            tot = in_tok + gen_tok
            self._log_throughput(in_tok, gen_tok, t1 - t0)
            logging.info("LLM tokens: in=%d, gen=%d, total=%d", in_tok, gen_tok, tot)

            for pos, out in zip(out_positions, outputs):
                text = ""
                outs = getattr(out, "outputs", None) or []
                if outs:
                    text = outs[0].text or ""
                obj = self._parse_text_to_obj(text) or {}
                is_syn = bool(obj.get("is_synonym", False))
                chosen_id: Optional[str] = None
                if is_syn:
                    if "chosen_index" in obj:
                        try:
                            idx = int(obj["chosen_index"])
                        except Exception:
                            idx = -1
                        q, cands = batch[pos][0], batch[pos][1]
                        if 0 <= idx < len(cands):
                            chosen_id = cands[idx].id
                        else:
                            logging.warning(
                                "Chosen index %s out of range; defaulting to top-1.",
                                idx,
                            )
                            chosen_id = cands[0].id if cands else None
                    else:
                        chosen_id = self._clean_chosen_id(obj.get("chosen_id"))
                decisions[pos] = Decision(chosen_id=chosen_id)

    @log_execution_time
    def select_batch(self, batch: List[Tuple[Pair, List[Pair]]]) -> List[Decision]:
        prompts: List[str] = []
        out_positions: List[int] = []
        decisions: List[Optional[Decision]] = [None] * len(batch)

        if self._cache is not None:
            self._check_cache(batch, decisions)

        self.choose_with_llm(batch, decisions)


        if self._cache is not None:
            for pos, (q, cands) in enumerate(batch):
                if decisions[pos] is not None:
                    ck = self._cache_key(q, cands)
                    try:
                        chosen_id = decisions[pos].chosen_id
                        self._cache[ck] = json.dumps(
                            {"chosen_id": chosen_id or ""}, ensure_ascii=False
                        )
                    except Exception:
                        pass

        return [d if d is not None else Decision(chosen_id=None) for d in decisions]

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
        cand_lines = [f"[{i}] {c.text}" for i, c in enumerate(candidates)]
        candidates_block = "\n".join(cand_lines)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a careful ontology assistant.\n"
            "Task: Given a query pair in the form 'descriptor; explainer' and a list of schema candidates,\n"
            "pick the single candidate that is synonymous or near‑synonymous in most contexts to the query.\n"
            "If many candidates are synonymous, choose the one that is the closest match.\n"
            "If you find a suitable candidate, respond with is_synonym=true and set chosen_index to exactly the chosen candidate's index.\n"
            "If none are acceptable, return is_synonym=false.\n"
            "Output MUST be JSON only.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Query: {query.text}\n\n"
            f"Candidates:\n{candidates_block}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    @staticmethod
    def _response_format():
        class VerdictSchema(BaseModel):
            is_synonym: bool
            chosen_index: int

        return VerdictSchema.model_json_schema()


# -----------------------------------------------------------------------------
# Mocks for fast local testing
# -----------------------------------------------------------------------------
class RandomEmbedder:
    """Mock embedder: returns unit-normalized random vectors (deterministic per text)."""

    def __init__(self, dim: int = 1024, batch_size: int = 4096, seed: int = 0) -> None:
        self.model_name = f"Mock/random-normal-dim{dim}"
        self.dim = dim
        self.batch_size = batch_size
        self._base_rng = np.random.default_rng(seed)

    @log_execution_time
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.empty((n, self.dim), dtype=np.float32)
        # Deterministic per-text vectors via hash-based seeds (order-independent)
        for i, t in enumerate(texts):
            h = int.from_bytes(
                hashlib.sha256(str(t).encode("utf-8")).digest()[:8], "little"
            )
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            # Normalize
            norm = np.linalg.norm(v) + 1e-12
            out[i] = v / norm
        return out


class MockSynonymSelector(Harmonizer):
    """Mock LLM: randomly picks a candidate id with prob p_match; otherwise drops."""

    def __init__(
        self, p_match: float = 0.75, seed: int = 42, batch_size: int = 1024
    ) -> None:
        # No real LLM needed; call super for compatibility but we won't use it.
        super().__init__(
            model_name="mock/selector",
            cache_dir=None,
            temperature=0.0,
            batch_size=batch_size,
            cache_db=None,
        )
        self._rng = np.random.default_rng(seed)
        self.p_match = float(p_match)

    # Override to avoid vLLM entirely
    @log_execution_time
    def select_batch(self, batch: List[Tuple[Pair, List[Pair]]]) -> List[Decision]:
        out: List[Decision] = []
        for _q, cands in batch:
            if not cands or self._rng.random() > self.p_match:
                out.append(Decision(chosen_id=None))
            else:
                choice = cands[self._rng.integers(0, len(cands))]
                out.append(Decision(chosen_id=choice.id))
        return out


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
@log_execution_time
def run(args) -> None:
    results_dir = Path(f"../results/harmonized/{args.run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / f"{args.run_id}.log", verbosity=args.verbosity)

    logging.info("Loading input and schema JSONL ...")
    input_rows = load_descriptors(args.input, args.max_input_rows)
    # Flatten per-descriptor inputs for embed/search/LLM, keeping a map to regroup
    flat_inputs: List[Pair] = []
    owner_row: Dict[str, int] = {}  # pair.id -> row_idx
    for r in input_rows:
        for p in r.pairs:
            flat_inputs.append(p)
            owner_row[p.id] = r.row_idx

    # Load schema
    schema_pairs = load_schema(args.schema)
    if not schema_pairs:
        raise ValueError("Schema is empty; nothing to harmonize.")

    # Build lookup by id
    schema_by_id: Dict[str, Pair] = {p.id: p for p in schema_pairs}

    # Load embedder model or mock embedder
    if args.mock_run:
        logging.info("Using mock embedder and LLM for fast local testing.")
        embedder = RandomEmbedder(dim=1024, batch_size=args.batch_size_embed, seed=42)
        schema_fp = embedding_fingerprint(schema_pairs, model_name="mock-embedder")
    else:
        logging.info("Using embedder model '%s'.", args.embedder)
        if args.embedder == "qwen":
            embedder = QwenEmbedder(
                cache_dir=args.cache_dir, batch_size=args.batch_size_embed
            )
            model_name_for_fp = "Qwen/Qwen3-Embedding-0.6B"
        else:
            embedder = StellaEmbedder(
                cache_dir=args.cache_dir, batch_size=args.batch_size_embed
            )
            model_name_for_fp = "Marqo/dunzhang-stella_en_400M_v5"
        # Check if schema and embedding models match previously saved embeddings
        schema_fp = embedding_fingerprint(schema_pairs, model_name=model_name_for_fp)

    # Embed schema
    # First, check if we have cached schema embeddings matching the current schema and model
    schema_embeds = None
    if (
        (not args.rebuild_cache)
        and args.schema_embed_path
        and args.schema_embed_path.exists()
    ):
        logging.info("Trying to load schema embeddings from %s", args.schema_embed_path)
        try:
            cached_emb, meta = load_embeds(args.schema_embed_path)
            if meta.get("fingerprint") == schema_fp:
                logging.info(
                    "Loaded cached schema embeddings from %s", args.schema_embed_path
                )
                schema_embeds = cached_emb
            else:
                logging.info(
                    "Schema/model fingerprint changed; ignoring cached embeddings"
                )
        except Exception as exc:
            logging.warning(
                "Failed to load cached embeddings (%s); will recompute", exc
            )

    if schema_embeds is None:
        logging.info("Embedding schema (%d items)...", len(schema_pairs))
        schema_embeds = embedder.embed_texts([p.text for p in schema_pairs])
        if args.schema_embed_path and not args.mock_run:  # only save real embeddings
            save_embeds(
                args.schema_embed_path,
                schema_embeds,
                meta={
                    "fingerprint": schema_fp,
                    "normalized": True,
                    "dtype": "float32",
                    "shape": list(schema_embeds.shape),
                    "model": model_name_for_fp,
                },
            )
            logging.info("Saved schema embeddings to %s", args.schema_embed_path)

    # Embed input
    # First, check if we have cached input embeddings matching the current input and model
    query_embeds = None
    input_embed_path = results_dir / f"{args.run_id}_input_embeds.npz"
    if not args.rebuild_cache and input_embed_path.exists():
        logging.info("Trying to load input embeddings from %s", input_embed_path)
        try:
            cached_emb, meta = load_embeds(input_embed_path)
            if meta.get("fingerprint") == embedding_fingerprint(
                flat_inputs, model_name_for_fp
            ):
                logging.info("Loaded cached input embeddings from %s", input_embed_path)
                query_embeds = cached_emb
            else:
                logging.info(
                    "Input/model fingerprint changed; ignoring cached input embeddings"
                )
        except Exception as exc:
            logging.warning(
                "Failed to load cached input embeddings (%s); will recompute", exc
            )

    if query_embeds is None:
        logging.info("Embedding input descriptors (%d)...", len(flat_inputs))
        query_embeds = embedder.embed_texts([p.text for p in flat_inputs])

        # Save input embeddings for possible reuse
        if not args.mock_run:  # only save real embeddings
            input_embed_path = results_dir / f"{args.run_id}_input_embeds.npz"
            save_embeds(
                input_embed_path,
                query_embeds,
                meta={
                    "fingerprint": embedding_fingerprint(
                        flat_inputs, model_name_for_fp
                    ),
                    "normalized": True,
                    "dtype": "float32",
                    "shape": list(query_embeds.shape),
                    "model": model_name_for_fp,
                },
            )
            logging.info("Saved input embeddings to %s", input_embed_path)

    # We no longer need the embedder, and it may hold GPU memory, so we can delete it to free up memory.
    logging.info("Freeing embedder from memory to maximize GPU availability for vLLM...")
    del embedder  # free memory
    # Clear cache for all available GPUs
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch to GPU i
        torch.cuda.empty_cache()

    # Search for nearest K neighbors from schema
    logging.info(
        "Finding %s neighbours from schema for each input descriptor", args.topk
    )
    sims, idxs = find_nn(schema_embeds, query_embeds, k=args.topk)

    if args.mock_run:
        selector = MockSynonymSelector(
            p_match=0.75, seed=42, batch_size=args.batch_size_llm
        )
        logging.info("Using mock LLM selector for fast testing.")
    else:
        if args.use_reranker:
            logging.info("Using reranker (%s) for final selection.", args.reranker)
            selector = RerankerChooser(
                model_name=args.reranker,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size_llm,
                cache_db=args.cache_db,
                min_rerank_score=args.min_rerank_score,
            )
        else:
            logging.info("Using LLM (%s) for final selection.", args.llm_name)
            selector = LLMChooser(
                model_name=args.llm_name,
                cache_dir=args.cache_dir,
                temperature=0.1,
                batch_size=args.batch_size_llm,
                cache_db=args.cache_db,
            )

    kept = 0
    dropped = 0

    llm_jobs: List[Tuple[Pair, List[Pair], Dict[str, float]]] = []
    below_threshold_count = 0
    
    for qi, q in enumerate(flat_inputs):
        cand_ids = idxs[qi]
        cand_sims = sims[qi]
        cands: List[Pair] = []
        score_map: Dict[str, float] = {}
        for j, cid in enumerate(cand_ids):
            score = float(cand_sims[j])
            if score < args.min_embed_score:
                continue
            p = schema_pairs[int(cid)]
            cands.append(p)
            score_map[p.id] = score
        if not cands:
            below_threshold_count += 1
            # Fallback: always keep at least the single best candidate by similarity
            j_best = int(np.argmax(cand_sims))
            cid_best = int(cand_ids[j_best])
            p_best = schema_pairs[cid_best]
            cands = [p_best]
            score_map = {p_best.id: float(cand_sims[j_best])}
        llm_jobs.append((q, cands, score_map))
    
    if below_threshold_count > 0:
        logging.warning(
            "%d input descriptors had no candidates above the similarity threshold %.3f; falling back to best candidate.",
            below_threshold_count,
            args.min_embed_score,
        )

    # Iterate in micro‑batches for the LLM
    def _iter_batches(items, bs: int):
        for i in range(0, len(items), bs):
            yield items[i : i + bs]

    # Open file for logging how descriptors are harmonized
    decision_log_path = results_dir / "decision_log.jsonl"
    with decision_log_path.open("w", encoding="utf-8") as audit_f:
        harm_by_row: Dict[int, List[str]] = {}
        total_batches = (len(llm_jobs) + selector.batch_size - 1) // selector.batch_size
        for i, batch in enumerate(
            _iter_batches(llm_jobs, selector.batch_size), start=1
        ):
            logging.info("Processing batch %d/%d", i, total_batches)
            decisions = selector.select_batch(
                [(q, cands) for (q, cands, _scores) in batch]
            )

            for (q, cands, score_map), dec in zip(batch, decisions):
                if dec.chosen_id:
                    chosen = schema_by_id.get(dec.chosen_id)
                    if chosen is None:
                        logging.warning(
                            "Chosen id %s not in schema; defaulting to top-1 candidate.",
                            dec.chosen_id,
                        )
                        chosen = cands[0]
                    row_idx = owner_row[q.id]
                    harm_by_row.setdefault(row_idx, []).append(
                        f"{chosen.descriptor}; {chosen.explainer}"
                    )
                    kept += 1

                    embed_sim = score_map.get(chosen.id)
                    json.dump(
                        {
                            "input_id": q.id,
                            "input_descriptor": q.descriptor,
                            "input_explainer": q.explainer,
                            "is_synonym": True,
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
                    # Log negative verdict with full context
                    json.dump(
                        {
                            "input_id": q.id,
                            "input_descriptor": q.descriptor,
                            "input_explainer": q.explainer,
                            "is_synonym": False,
                            "candidates": [
                                {
                                    "id": c.id,
                                    "descriptor": c.descriptor,
                                    "explainer": c.explainer,
                                    "embed_sim": score_map.get(c.id),
                                }
                                for c in cands
                            ],
                        },
                        audit_f,
                        ensure_ascii=False,
                    )
                    audit_f.write("\n")

    # Save harmoinzed output JSONL, grouping back by original row and adding a "harmonized_descriptors" list with the chosen synonyms
    out_path = results_dir / f"{args.run_id}_harmonized.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in input_rows:
            out_obj: Dict[str, Any]
            if isinstance(r.raw, dict):
                out_obj = dict(r.raw)  # shallow copy
            else:
                # raw string input -> wrap into object
                out_obj = {"text": r.raw}
            harmonized_descriptors = harm_by_row.get(r.row_idx, [])

            if args.drop_duplicates:
                # Deduplicate while preserving order
                seen = set()
                unique_harm = []
                for hd in harmonized_descriptors:
                    if hd not in seen:
                        seen.add(hd)
                        unique_harm.append(hd)
                harmonized_descriptors = unique_harm

            out_obj["harmonized_descriptors"] = harmonized_descriptors
            json.dump(out_obj, f, ensure_ascii=False)
            f.write("\n")


    # Final logs
    logging.info("Results saved to %s and %s.", decision_log_path, out_path)
    if dropped > 0:
        logging.warning(
            "%d descriptors were dropped (no suitable synonym found); check %s for details.",
            dropped,
            decision_log_path,
        )
    if args.drop_duplicates:
        logging.info("Duplicate harmonized descriptor;explainer pairs were dropped, keeping only unique ones per document.")
    else:
        logging.info("Duplicate harmonized descriptor;explainer pairs were NOT dropped; all chosen synonyms are included in the output.")
    logging.info("Done. Kept=%d, Dropped=%d", kept, dropped)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replace descriptor;explainer pairs in INPUT with synonyms from SCHEMA"
    )

    # Required arguments
    p.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    p.add_argument("--schema", type=Path, required=True, help="Path to schema JSONL")
    p.add_argument("--run-id", type=str, required=True, help="Name for run")

    # Model arguments
    p.add_argument(
        "--llm",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="vLLM model name (ignored if --use-reranker or --mock-run is set)",
    )
    p.add_argument(
        "--reranker",
        type=str,
        default="qwen/Qwen3-Reranker-0.6B",
        help="Reranker model name (only used if --use-reranker)",
    )
    p.add_argument(
        "--embedder",
        choices=["stella", "qwen"],
        default="qwen",
        help="Embedding model to use for candidate retrieval",
    )
    p.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use a separate reranker model instead of LLM",
    )
    p.add_argument("--batch-size-embed", type=int, default=64)
    p.add_argument("--batch-size-llm", type=int, default=1024)
    p.add_argument(
        "--cache-dir", type=Path, default=None, help="HF cache dir for models"
    )

    # Testing/debugging arguments
    p.add_argument(
        "--mock-run",
        action="store_true",
        help="Use mock embedder and LLM for fast local testing (no real model calls)",
    )
    p.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="For testing: limit number of input rows to process (default: all)",
    )

    # Data filtering arguments
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of schema candidates per input for LLM",
    )
    p.add_argument(
        "--min-embed-score",
        type=float,
        default=0.0,
        help="Drop schema candidates below this cosine.",
    )
    p.add_argument(
        "--min-rerank-score",
        type=float,
        default=0.5,
        help="Minimum score for reranker to consider a candidate a match (only with --use-reranker)",
    )
    p.add_argument(
        "--drop-duplicates",
        action="store_true",
        default=True,
        help="If harmonization produce duplicate descriptor;explainer pairs for a document, keep only one.",
    )

    # Cache arguments
    p.add_argument(
        "--cache-db", type=Path, default=None, help="SQLite file to cache LLM decisions"
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force recomputation of schema embeddings/index, ignoring any cache",
    )
    p.add_argument(
        "--schema-embed-path",
        type=Path,
        default="../results/final_schema/schema_embeddings.npz",
        help="Path to save/load cached schema embeddings (.npz)",
    )

    # Logging arguments
    p.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])

    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
