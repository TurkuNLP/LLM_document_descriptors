#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Optional, Set
from collections import Counter, defaultdict
import logging
import re
import unicodedata
import time
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from rapidfuzz import fuzz  # type: ignore

# ---------- normalization ----------
_ws_re = re.compile(r"\s+")
_SPLIT_RE = re.compile(r";|；|:|：|\s\|\|\|\s")  # accept common separators

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip()
    # unify curly quotes/dashes
    s = (s.replace("“", "\"").replace("”", "\"")
           .replace("‘", "'").replace("’", "'")
           .replace("–", "-").replace("—", "-"))
    s = s.lower()
    s = _ws_re.sub(" ", s)
    # strip surrounding quotes/brackets
    s = s.strip('"\''"()[]{}")
    # drop trailing sentence punctuation
    s = re.sub(r"[.,;:!?…]+$", "", s)
    # collapse repeated punctuation
    s = re.sub(r"[^\w\s]{2,}", lambda m: m.group(0)[0], s)
    return s

# ---------- lineage: build text -> SET(member_ids) ----------
def build_text_to_ids_from_lineage(lineage_path: Path) -> Dict[Tuple[str, str], Set[str]]:
    """
    id_to_original_texts.jsonl rows:
      { "id": "...",
        "final": {"descriptor": "...", "explainer": "..."},
        "sources": [{"descriptor": "...", "explainer": "..."}, ...] }

    Build {(norm_desc, norm_expl) -> set(member_ids)} using BOTH sources and finals.
    """
    text2ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    n_rows = n_sources = n_finals = conflict_touches = 0

    with lineage_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rows += 1
            _id = (rec.get("id") or "").strip()
            if not _id:
                continue

            # sources
            for src in rec.get("sources", []):
                d = _norm(src.get("descriptor", ""))
                e = _norm(src.get("explainer", ""))
                if not d and not e:
                    continue
                key = (d, e)
                if text2ids[key] and _id not in text2ids[key]:
                    conflict_touches += 1
                text2ids[key].add(_id)
                n_sources += 1

            # finals
            fin = rec.get("final") or {}
            fd = _norm(fin.get("descriptor", ""))
            fe = _norm(fin.get("explainer", ""))
            if fd or fe:
                key = (fd, fe)
                if text2ids[key] and _id not in text2ids[key]:
                    conflict_touches += 1
                text2ids[key].add(_id)
                n_finals += 1

    logging.info(
        "Built text→IDs map from lineage: rows=%d | sources=%d | finals=%d | conflict_touches=%d",
        n_rows, n_sources, n_finals, conflict_touches
    )
    return text2ids

# ---------- groups: member -> root ----------
def load_member_to_root(groups_jsonl: Path) -> Dict[str, str]:
    """
    merged_groups.jsonl lines: {"ROOT_ID": ["member1", "member2", ...]}
    Returns: {member_id -> root_id}
    """
    m2r: Dict[str, str] = {}
    n_groups = 0
    with groups_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or not obj:
                continue
            (root, members), = obj.items()
            root = str(root)
            for mid in members:
                m2r[str(mid)] = root
            n_groups += 1
    logging.info("Loaded %d groups; %d members total", n_groups, len(m2r))
    return m2r

# ---------- resolve text key -> unique root (final id) ----------
def resolve_unique_root_for_key(
    key: Tuple[str, str],
    text2ids: Dict[Tuple[str, str], Set[str]],
    m2r: Dict[str, str],
) -> Optional[str]:
    mids = text2ids.get(key)
    if not mids:
        return None
    roots = { m2r.get(mid, mid) for mid in mids }
    if len(roots) == 1:
        return next(iter(roots))
    return None  # ambiguous across different roots; caller may try fuzzy/backoff

# ---------- argmax without numpy ----------
def argmax_idx(xs: List[float]) -> int:
    if not xs:
        return 0
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i

# ---------- parse docs ----------
def split_pair(item: str) -> Tuple[str, str]:
    if not isinstance(item, str):
        return "", ""
    parts = _SPLIT_RE.split(item, maxsplit=1)
    if len(parts) < 2:
        return "", ""
    left, right = parts[0], parts[1]
    return _norm(left), _norm(right)

def iter_selected_pairs(docs_jsonl: Path, all_lists: bool = False) -> Iterable[Tuple[str, str]]:
    dropped_no_sep = 0
    processed = 0
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            processed += 1
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue
            desc_lists = doc.get("descriptors")
            sims = doc.get("similarity")
            if not isinstance(desc_lists, list) or not desc_lists:
                continue

            lists_to_scan = range(len(desc_lists)) if all_lists else [argmax_idx(sims or [])]
            for li in lists_to_scan:
                if li < 0 or li >= len(desc_lists):
                    continue
                chosen = desc_lists[li]
                if not isinstance(chosen, list):
                    continue
                for item in chosen:
                    d, e = split_pair(item)
                    if not d and not e:
                        dropped_no_sep += 1
                        continue
                    yield (d, e)
    logging.info("Docs parsed: %d | dropped entries (no/invalid separator): %d", processed, dropped_no_sep)

# ---------- fuzzy (single-field) with SMALL index: prefix + length buckets ----------
SEP = " ||| "
_alnum_re = re.compile(r"[a-z0-9]+")

def _combo(d: str, e: str) -> str:
    return f"{d}{SEP}{e}"

def _prefix_key(desc: str, n: int = 3) -> str:
    toks = _alnum_re.findall(desc)
    joined = "".join(toks)
    return joined[:n]

def _len_bin(n: int, bin_size: int) -> int:
    return (n // bin_size) * bin_size

class SmallFuzzyIndex:
    """
    Memory-light candidate index over KEYS ONLY (list of (desc, expl) tuples):
      bucket by descriptor prefix + length bins of (desc, expl).
    """
    def __init__(self, keys: Iterable[Tuple[str, str]],
                 prefix_n: int = 3, bin_d: int = 10, bin_e: int = 20):
        self.keys: List[Tuple[str, str]] = list(keys)
        self.prefix_n = prefix_n
        self.bin_d = bin_d
        self.bin_e = bin_e
        self.desc_len = [len(d) for d, _ in self.keys]
        self.expl_len = [len(e) for _, e in self.keys]
        self.bucket = defaultdict(list)  # (prefix, bin_d, bin_e) -> [idx...]
        for i, (d, e) in enumerate(self.keys):
            bkey = ( _prefix_key(d, prefix_n),
                     _len_bin(self.desc_len[i], bin_d),
                     _len_bin(self.expl_len[i], bin_e) )
            self.bucket[bkey].append(i)

    def candidates(self, d: str, e: str) -> List[int]:
        bkey = ( _prefix_key(d, self.prefix_n),
                 _len_bin(len(d), self.bin_d),
                 _len_bin(len(e), self.bin_e) )
        return self.bucket.get(bkey, [])

def _rf_ratio(a: str, b: str, cutoff: float) -> float:
    return float(fuzz.ratio(a, b, score_cutoff=int(cutoff * 100))) / 100.0

def fuzzy_lookup_early_bucketed(
    d: str,
    e: str,
    *,
    index: SmallFuzzyIndex,
    topk: int = 80,
    cutoff: float = 0.92
) -> Tuple[Optional[int], float]:
    """
    Early-exit on first bucketed candidate with RF >= cutoff.
    Candidate order: small length deltas first, then stable index.
    """
    q = _combo(d, e)
    lq = len(q)
    cand = index.candidates(d, e)
    if not cand:
        return None, 0.0

    ordered = sorted(
        cand,
        key=lambda i: (abs((index.desc_len[i] + index.expl_len[i] + len(SEP)) - lq), i)
    )[:topk]

    for i in ordered:
        d2, e2 = index.keys[i]
        sc = _rf_ratio(q, _combo(d2, e2), cutoff)
        if sc >= cutoff:
            return i, sc  # EARLY EXIT
    return None, 0.0

# --- Globals for worker processes (filled via fork) ---
_G_INDEX: Optional[SmallFuzzyIndex] = None
_G_TEXT2IDS: Optional[Dict[Tuple[str, str], Set[str]]] = None
_G_M2R: Optional[Dict[str, str]] = None
_G_CUTOFF: float = 0.92
_G_TOPK: int = 80
_G_LOG_MATCHES: bool = False  # whether to collect match records in workers

def _resolve_root_from_index_key(idx_key: Tuple[str, str]) -> Optional[str]:
    """Resolve a (desc, expl) key from the index to a unique root via globals."""
    mids = _G_TEXT2IDS.get(idx_key) if _G_TEXT2IDS else None
    if not mids:
        return None
    roots = { _G_M2R.get(mid, mid) for mid in mids } if _G_M2R else set()
    if len(roots) == 1:
        return next(iter(roots))
    return None

def _worker_resolve_chunk(chunk: List[Tuple[str, str, int]]) -> Tuple[Counter, int, int, List[dict]]:
    """
    Resolve a chunk of (d,e,freq) using globals set in parent (via fork).
    Returns (final_counts_delta, hits_occ, hits_keys, match_records)
    """
    idx = _G_INDEX
    cutoff = _G_CUTOFF
    topk = _G_TOPK
    out: Counter = Counter()
    hits_occ = 0
    hits_keys = 0
    records: List[dict] = []

    for d, e, freq in chunk:
        j, sc = fuzzy_lookup_early_bucketed(d, e, index=idx, topk=topk, cutoff=cutoff)
        if j is not None:
            key2 = idx.keys[j]
            root = _resolve_root_from_index_key(key2)
            if root is not None:
                out[root] += freq
                hits_occ += freq
                hits_keys += 1
                if _G_LOG_MATCHES:
                    # include all member IDs for transparency
                    mids = sorted(list(_G_TEXT2IDS.get(key2, [])))
                    records.append({
                        "query": {"descriptor": d, "explainer": e, "count": freq},
                        "match": {"descriptor": key2[0], "explainer": key2[1]},
                        "member_ids": mids,
                        "root": root,
                        "score": round(sc, 4)
                    })
    return out, hits_occ, hits_keys, records

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate descriptor–explainer frequencies into final IDs (unique-root resolution, memory-light parallel fuzzy)."
    )
    ap.add_argument("--docs", required=True, help="Documents JSONL.")
    ap.add_argument("--lineage", required=True, help="id_to_original_texts.jsonl (per-ID lineage with 'sources').")
    ap.add_argument("--groups", required=True, help="merged_groups.jsonl (root → members).")
    ap.add_argument("--out-final", default="final_id_counts.jsonl",
                    help='Output JSONL: {"id": ..., "count": N}.')
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARN","ERROR"])
    ap.add_argument("--all-lists", action="store_true",
                    help="Count descriptor–explainers from ALL descriptor lists per doc (not just argmax).")

    # Fuzzy knobs (bucket-based)
    ap.add_argument("--fuzzy", action="store_true", help="Enable fuzzy fallback for unmapped pairs.")
    ap.add_argument("--fuzzy-thresh", type=float, default=0.65,
                    help="Minimum combined similarity (single-field) to accept fuzzy match (0..1).")
    ap.add_argument("--fuzzy-topk", type=int, default=2500,
                    help="Max candidates to inspect per unmapped query after bucketing.")
    ap.add_argument("--fuzzy-prefix-n", type=int, default=2,
                    help="Prefix length (alnum chars) used for descriptor buckets.")
    ap.add_argument("--fuzzy-bin-d", type=int, default=40,
                    help="Descriptor length bin size.")
    ap.add_argument("--fuzzy-bin-e", type=int, default=80,
                    help="Explainer length bin size.")
    ap.add_argument("--fuzzy-log", default="fuzzy_matches.jsonl",
                    help="JSONL file to record accepted fuzzy matches (query, match, member_ids, root, score).")

    # Parallel knobs
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes for fuzzy (0=auto; 1=serial).")
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="How many unique unmapped keys per task chunk.")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    docs_path = Path(args.docs)
    lineage_path = Path(args.lineage)
    groups_path = Path(args.groups)

    # Build maps
    text2ids = build_text_to_ids_from_lineage(lineage_path)  # {(d,e)->set(member_ids)}
    m2r = load_member_to_root(groups_path)                   # {member_id->root_id}

    # Small, memory-light fuzzy index (keys only)
    index = None
    if args.fuzzy:
        t0 = time.time()
        index = SmallFuzzyIndex(
            keys=text2ids.keys(),
            prefix_n=args.fuzzy_prefix_n,
            bin_d=args.fuzzy_bin_d,
            bin_e=args.fuzzy_bin_e
        )
        t1 = time.time()
        logging.info("Built small fuzzy index: %d buckets over %d keys in %s",
                     len(index.bucket), len(index.keys),
                     time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

    # Pass 1: exact mapping to ROOTS; collect unmapped counts for fuzzy
    final_counts: Counter[str] = Counter()
    unmapped_counts: Counter[Tuple[str, str]] = Counter()

    total_pairs = 0
    for d, e in iter_selected_pairs(docs_path, all_lists=args.all_lists):
        total_pairs += 1
        root = resolve_unique_root_for_key((d, e), text2ids, m2r)
        if root is None:
            if args.fuzzy:
                unmapped_counts[(d, e)] += 1
        else:
            final_counts[root] += 1

    logging.info(
        "Exact mapped occurrences (to roots): %d | unique unmapped keys: %d | total unmapped occurrences: %d",
        sum(final_counts.values()), len(unmapped_counts), sum(unmapped_counts.values())
    )

    # Pass 2: fuzzy resolve unique unmapped (parallel if workers>1)
    fuzzy_hits_occ = 0
    fuzzy_hits_keys = 0
    if args.fuzzy and unmapped_counts:
        # Prepare globals for workers (via fork these are shared)
        global _G_INDEX, _G_TEXT2IDS, _G_M2R, _G_CUTOFF, _G_TOPK, _G_LOG_MATCHES
        _G_INDEX = index
        _G_TEXT2IDS = text2ids
        _G_M2R = m2r
        _G_CUTOFF = args.fuzzy_thresh
        _G_TOPK = args.fuzzy_topk
        _G_LOG_MATCHES = bool(args.fuzzy_log)

        pairs = [(d, e, freq) for (d, e), freq in unmapped_counts.items()]
        if args.workers is None or args.workers <= 0:
            workers = max(1, min(os.cpu_count() or 1, 32))
        else:
            workers = max(1, args.workers)

        # open logger in parent
        flog = open(args.fuzzy_log, "w", encoding="utf-8") if args.fuzzy_log else None

        if workers == 1:
            # Serial
            t0 = time.time()
            delta, hits_occ, hits_keys, recs = _worker_resolve_chunk(pairs)
            final_counts.update(delta)
            fuzzy_hits_occ += hits_occ
            fuzzy_hits_keys += hits_keys
            if flog and recs:
                for r in recs:
                    flog.write(json.dumps(r, ensure_ascii=False) + "\n")
            t1 = time.time()
        else:
            # Parallel
            t0 = time.time()
            chunk = args.chunk_size
            chunks = [pairs[i:i+chunk] for i in range(0, len(pairs), chunk)]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_resolve_chunk, ch) for ch in chunks]
                for fu in as_completed(futs):
                    delta, hits_occ, hits_keys, recs = fu.result()
                    final_counts.update(delta)
                    fuzzy_hits_occ += hits_occ
                    fuzzy_hits_keys += hits_keys
                    if flog and recs:
                        for r in recs:
                            flog.write(json.dumps(r, ensure_ascii=False) + "\n")
            t1 = time.time()

        if flog:
            flog.close()

        logging.info(
            "Fuzzy phase: mapped %d occurrences across %d unique keys in %s (workers=%d). "
            "Matches logged to: %s",
            fuzzy_hits_occ, fuzzy_hits_keys, time.strftime("%H:%M:%S", time.gmtime(t1 - t0)),
            workers, args.fuzzy_log
        )

    logging.info(
        "Coverage — total pairs: %d | mapped(total to roots): %d | final roots hit: %d",
        total_pairs, sum(final_counts.values()), len(final_counts)
    )

    # Write outputs (final/root counts)
    out_final = Path(args.out_final)
    with out_final.open("w", encoding="utf-8") as f:
        for fid, cnt in final_counts.most_common():
            f.write(json.dumps({"id": fid, "count": int(cnt)}, ensure_ascii=False) + "\n")
    logging.info("Wrote final ID (root) counts -> %s (%d lines)", out_final, len(final_counts))

if __name__ == "__main__":
    main()
