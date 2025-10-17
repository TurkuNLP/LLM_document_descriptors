#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Optional
from collections import Counter, defaultdict
import logging
import re
import unicodedata
import time

from rapidfuzz import fuzz  # type: ignore

# ---------- normalization ----------
_ws_re = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = _ws_re.sub(" ", s)
    return s

# ---------- lineage: build text -> id map (sources + finals) ----------
def build_text_to_id_from_lineage(lineage_path: Path) -> Dict[Tuple[str, str], str]:
    text2id: Dict[Tuple[str, str], str] = {}
    n_rows = n_sources = n_finals = conflicts = 0

    with lineage_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rows += 1
            _id = (rec.get("id") or "").strip()
            if not _id:
                continue

            for src in rec.get("sources", []):
                d = _norm(src.get("descriptor", ""))
                e = _norm(src.get("explainer", ""))
                if not d and not e:
                    continue
                key = (d, e)
                if key in text2id and text2id[key] != _id:
                    conflicts += 1
                else:
                    if key not in text2id:
                        text2id[key] = _id
                        n_sources += 1

            fin = rec.get("final") or {}
            fd = _norm(fin.get("descriptor", ""))
            fe = _norm(fin.get("explainer", ""))
            if fd or fe:
                key = (fd, fe)
                if key in text2id and text2id[key] != _id:
                    conflicts += 1
                else:
                    text2id[key] = _id
                    n_finals += 1

    logging.info(
        "Built text→id map from lineage: rows=%d | sources=%d | finals=%d | conflicts=%d",
        n_rows, n_sources, n_finals, conflicts
    )
    return text2id

# ---------- groups: member -> root ----------
def load_member_to_root(groups_jsonl: Path) -> Dict[str, str]:
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
    if not isinstance(item, str) or ";" not in item:
        return "", ""
    left, right = item.split(";", 1)
    return _norm(left), _norm(right)

def iter_selected_pairs(docs_jsonl: Path) -> Iterable[Tuple[str, str]]:
    dropped_no_semicolon = 0
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
            if not isinstance(desc_lists, list) or not isinstance(sims, list) or not desc_lists:
                continue
            idx = argmax_idx(sims)
            if idx < 0 or idx >= len(desc_lists):
                idx = 0
            chosen = desc_lists[idx]
            if not isinstance(chosen, list):
                continue
            for item in chosen:
                d, e = split_pair(item)
                if not d and not e:
                    dropped_no_semicolon += 1
                    continue
                yield (d, e)
    logging.info("Docs parsed: %d | dropped entries (no/invalid ';'): %d", processed, dropped_no_semicolon)

# ---------- fuzzy (single-field): trigram index + early-exit RF ----------
SEP = " ||| "

def _combo(d: str, e: str) -> str:
    return f"{d}{SEP}{e}"

def _trigrams(s: str) -> set:
    s2 = f"  {s}  "
    return {s2[i:i+3] for i in range(len(s2)-2)}

def build_inverted_index_combined(text2id_keys: Iterable[Tuple[str, str]]):
    """
    Build trigram -> set(idx) over combined strings.
    Returns: keys (list[(d,e)]), combos (list[str]), combos_tri (list[set[str]]), combos_len (list[int]), idx_tri (dict[str, set[int]])
    """
    keys: List[Tuple[str, str]] = list(text2id_keys)
    combos: List[str] = [_combo(d, e) for (d, e) in keys]
    combos_tri: List[set] = []
    combos_len: List[int] = []
    idx_tri = defaultdict(set)
    for i, s in enumerate(combos):
        tri = _trigrams(s)
        combos_tri.append(tri)
        combos_len.append(len(s))
        for g in tri:
            idx_tri[g].add(i)
    return keys, combos, combos_tri, combos_len, idx_tri

def _rf_ratio(a: str, b: str, cutoff: float) -> float:
    return float(fuzz.ratio(a, b, score_cutoff=int(cutoff * 100))) / 100.0

def fuzzy_lookup_early_combined(
    d: str,
    e: str,
    *,
    combos: List[str],
    combos_tri: List[set],
    combos_len: List[int],
    idx_tri: Dict[str, set],
    topk: int = 120,
    cutoff: float = 0.94,
    min_overlap: int = 10,  # drop weak candidates before scoring
) -> Tuple[Optional[int], float]:
    """
    Early-exit on first candidate whose RapidFuzz ratio >= cutoff.
    Uses precomputed trigram sets/lengths to avoid recompute.
    """
    q = _combo(d, e)
    gq = _trigrams(q)
    lq = len(q)

    # candidate set = union of postings; require minimum trigram overlap
    cand = set()
    for g in gq:
        cand |= idx_tri.get(g, set())
    if not cand:
        return None, 0.0

    def overlap(i: int) -> int:
        return len(gq & combos_tri[i])

    # best-first order: overlap desc, small length delta, stable tiebreaker
    ordered = sorted(
        (i for i in cand if overlap(i) >= min_overlap),
        key=lambda i: (overlap(i), -abs(combos_len[i] - lq), -i),
        reverse=True
    )[:topk]

    for i in ordered:
        sc = _rf_ratio(q, combos[i], cutoff)
        if sc >= cutoff:
            return i, sc  # EARLY EXIT
    return None, 0.0

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate descriptor–explainer frequencies into final IDs (lineage-backed, fast fuzzy fallback)."
    )
    ap.add_argument("--docs", required=True, help="Documents JSONL.")
    ap.add_argument("--lineage", required=True, help="id_to_original_texts.jsonl (per-ID lineage with 'sources').")
    ap.add_argument("--groups", required=True, help="merged_groups.jsonl (root → members).")
    ap.add_argument("--out-final", default="final_id_counts.jsonl",
                    help='Output JSONL: {"id": ..., "count": N}.')
    ap.add_argument("--out-members", default=None, help="Optional per-member counts JSONL.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARN","ERROR"])
    # Fuzzy knobs
    ap.add_argument("--fuzzy", action="store_true", help="Enable fuzzy fallback for unmapped pairs.")
    ap.add_argument("--fuzzy-thresh", type=float, default=0.90,
                    help="Minimum combined similarity (single-field) to accept fuzzy match (0..1).")
    ap.add_argument("--fuzzy-topk", type=int, default=100,
                    help="Max candidates to inspect per unmapped query after pruning.")
    ap.add_argument("--fuzzy-min-overlap", type=int, default=10,
                    help="Minimum shared trigrams required before scoring.")
    ap.add_argument("--fuzzy-log", default=None, help="Optional JSONL to record accepted fuzzy matches.")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    docs_path = Path(args.docs)
    lineage_path = Path(args.lineage)
    groups_path = Path(args.groups)

    # Build maps
    text2id = build_text_to_id_from_lineage(lineage_path)  # {(d,e)->member_id}
    m2r = load_member_to_root(groups_path)                 # {member_id->root_id}

    keys = combos = combos_tri = combos_len = idx_tri = None
    if args.fuzzy:
        keys, combos, combos_tri, combos_len, idx_tri = build_inverted_index_combined(text2id.keys())

    # Pass 1: exact mapping; collect unmapped counts
    member_counts: Counter[str] = Counter()
    unmapped_counts: Counter[Tuple[str, str]] = Counter()

    total_pairs = 0
    for d, e in iter_selected_pairs(docs_path):
        total_pairs += 1
        mid = text2id.get((d, e))
        if mid is None:
            if args.fuzzy:
                unmapped_counts[(d, e)] += 1
        else:
            member_counts[mid] += 1

    logging.info("Exact mapped occurrences: %d | unique unmapped keys: %d | total unmapped occurrences: %d",
                 sum(member_counts.values()), len(unmapped_counts), sum(unmapped_counts.values()))

    # Pass 2: fuzzy resolve unique unmapped (if enabled)
    fuzzy_hits_occ = 0
    fuzzy_hits_keys = 0
    flog = None
    if args.fuzzy and unmapped_counts:
        if args.fuzzy_log:
            flog = open(args.fuzzy_log, "w", encoding="utf-8")

        t0 = time.time()
        for (d, e), freq in unmapped_counts.items():
            logging.info("Finding a fuzzy match for %d; %e", d, e)
            idx, sc = fuzzy_lookup_early_combined(
                d, e,
                combos=combos, combos_tri=combos_tri, combos_len=combos_len, idx_tri=idx_tri,
                topk=args.fuzzy_topk, cutoff=args.fuzzy_thresh, min_overlap=args.fuzzy_min_overlap
            )
            if idx is not None:
                logging.info("Found fuzzy match %d; %e", d2, e2)
                d2, e2 = keys[idx]
                mid = text2id[(d2, e2)]
                member_counts[mid] += freq
                fuzzy_hits_occ += freq
                fuzzy_hits_keys += 1
                if flog:
                    flog.write(json.dumps({
                        "query": {"descriptor": d, "explainer": e, "count": freq},
                        "match": {"descriptor": d2, "explainer": e2, "id": mid},
                        "score": round(sc, 4)
                    }, ensure_ascii=False) + "\n")
        t1 = time.time()
        logging.info("Fuzzy phase: mapped %d occurrences across %d unique keys in %s",
                     fuzzy_hits_occ, fuzzy_hits_keys, time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

    if flog:
        flog.close()

    # Roll up to roots
    final_counts: Counter[str] = Counter()
    for mid, c in member_counts.items():
        root = m2r.get(mid, mid)  # singleton if not in groups
        final_counts[root] += c

    logging.info(
        "Coverage — total pairs: %d | mapped(total): %d | roots: %d",
        total_pairs, sum(member_counts.values()), len(final_counts)
    )

    # Write outputs
    out_final = Path(args.out_final)
    with out_final.open("w", encoding="utf-8") as f:
        for fid, cnt in final_counts.most_common():
            f.write(json.dumps({"id": fid, "count": int(cnt)}, ensure_ascii=False) + "\n")
    logging.info("Wrote final ID counts -> %s (%d lines)", out_final, len(final_counts))

    if args.out_members:
        out_members = Path(args.out_members)
        with out_members.open("w", encoding="utf-8") as f:
            for mid, cnt in member_counts.most_common():
                f.write(json.dumps({"id": mid, "count": int(cnt)}, ensure_ascii=False) + "\n")
        logging.info("Wrote member ID counts -> %s (%d lines)", out_members, len(member_counts))

if __name__ == "__main__":
    main()
