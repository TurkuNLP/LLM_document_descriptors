#!/usr/bin/env python3
"""
Backward lineage tracer for descriptor;explainer merges WITH dead-end diagnostics.

Strategy (depth-first, mass-conserving):
- Load the *original* input and compute per-original pair mass (including duplicates).
- For each run r=1..N, load lineage edges and build a reverse map: tgt -> [src,...],
  and the set of outputs (IDs that persisted/ended the run).
- For each final ID in run N, recursively trace *backwards*:
    * If the node has parents in run r: sum contributions from all parents, recurse to r-1.
    * If the node has no parents in run r: allow identity passthrough only if the node existed
      in outputs of run r-1 (i.e., persisted unchanged); then recurse with the same node to r-1.
    * Stop at r < 0: count only IDs present in the original set and add their mass.
- Memoize (r, node) results so it’s fast even with millions of edges.

Dead-end diagnostics:
- A "dead-end branch" is any (r, node) where:
    * r >= 1 and the node has no parents in round r and is NOT in outputs of round r-1; OR
    * r == 0 and the node is not in the original set.
- Counted uniquely (set of (r, node)) and also summarized per round.

Outputs:
- JSONL: one row per final (from the last run) with fields:
    id, descriptor, explainer, contribution_count, (optional) original_counts
- Sorted by descending contribution_count, then id.

Validation (--validate):
- Coverage vs union(run1 sources, run1 outputs).
- Finals with/without weight.
- Dead-end counts: total & per-round breakdown.
- Weight conservation: sum(original masses) == sum(final masses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Set

import numpy as np  # type: ignore


# -------------------------- ID & normalization ---------------------------

def _normalize_descriptor(s: str) -> str:
    return re.sub(r"[_\s]+", " ", (s or "")).strip().lower()


def pair_id(descriptor: str, explainer: str, *, length: int = 12) -> str:
    key = f"{_normalize_descriptor(descriptor)}\u241f{explainer}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]


# -------------------------- Load originals ------------------------------

def _split_pair_raw(text: str) -> Tuple[str, str]:
    """Split raw 'descriptor;explainer' string (matches extractor logic)."""
    try:
        d, e = text.split(";", 1)
        return _normalize_descriptor(d), e.strip()
    except ValueError:
        return "", ""


def load_original_counts_from_file(path: Path) -> Dict[str, int]:
    """Return counts of *original* (possibly duplicated) pairs keyed by pair_id.

    Auto-detects line format:
      - RAW extractor lines: keys {"similarity", "descriptors"}
      - PROCESSED lines: keys {"descriptor", "explainer"}
    """
    counts: Dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            if "similarity" in row and "descriptors" in row:
                best_idx = int(np.argmax(row["similarity"]))
                descriptors: Iterable[str] = row["descriptors"][best_idx]
                for desc_exp in descriptors:
                    d, e = _split_pair_raw(desc_exp)
                    if not d or not e:
                        continue
                    counts[pair_id(d, e)] += 1
            else:
                d = (row.get("descriptor") or "").strip().lower()
                e = (row.get("explainer") or "").strip()
                if not d or not e:
                    continue
                counts[pair_id(d, e)] += 1

    return counts


# -------------------------- Run artifacts -------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_lineage_edges(run_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (source_pair_id, new_pair_id) edges for a run."""
    candidates = [
        run_dir / "checkpoints" / "full_lineage.jsonl",
        run_dir / "full_lineage.jsonl",
        run_dir / "all_merges_full_lineage.jsonl",
        run_dir / "all_mergesfull_lineage.jsonl",
    ]
    lineage = next((p for p in candidates if p.exists()), None)

    edges: List[Tuple[str, str]] = []
    if not lineage:
        return edges

    for ev in read_jsonl(lineage):
        tgt = ev.get("new_pair_id")
        srcs = ev.get("source_pair_ids") or []
        if not tgt:
            continue
        for s in srcs:
            if s:
                edges.append((s, tgt))
    return edges


def load_final_pairs(run_dir: Path, run_id: str = "") -> Dict[str, Tuple[str, str]]:
    """Return map id -> (descriptor, explainer) from the run's final output.

    Prefer all_merges_disambig.jsonl; else try <run_id>_disambig.jsonl; else any *_disambig.jsonl.
    Keep rows even if descriptor/explainer are empty so lookups never fail.
    """
    preferred = run_dir / "all_merges_disambig.jsonl"
    candidates: List[Path]
    if preferred.exists():
        candidates = [preferred]
    else:
        rid = run_dir / f"{run_id}_disambig.jsonl" if run_id else None
        candidates = [p for p in [rid] if p is not None and p.exists()]  # type: ignore
        if not candidates:
            candidates = sorted(run_dir.glob("*_disambig.jsonl"))

    final_map: Dict[str, Tuple[str, str]] = {}
    for out_path in candidates:
        for obj in read_jsonl(out_path):
            pid = obj.get("id")
            d = (obj.get("descriptor") or "").strip()
            e = (obj.get("explainer") or "").strip()
            if pid:
                final_map[pid] = (d, e)
    return final_map


# -------------------------- Backward tracer ------------------------------

def compute_contributions_by_backtrace(
    original_counts: Dict[str, int],
    run_dirs: List[Path],
    finals: Iterable[str],
    include_breakdown: bool = False,
) -> Tuple[Dict[str, int], Optional[Dict[str, Dict[str, int]]], Dict[str, int]]:
    """
    Trace *backwards* from each final ID to all original IDs (depth-first).
    For each round r (from last to first):
      - parents = reverse_lineage[r][node] (could be many due to merges)
      - if no parents for 'node' in round r, carry the node unchanged only if it
        exists in outputs of the *previous* round (identity passthrough).
    Stop at r < 0 and sum only original IDs from `original_counts`.

    Returns:
      final_counts, ancestry, dead_end_stats
    where dead_end_stats = {
        "unique_dead_ends": int,            # number of unique (r,node) dead-end branches
        "base_no_original": int,            # of those at r==0 where node is not in originals
        "per_round": {r:int, ...}           # r indexes: 1..N dead-ends (no parents & no passthrough)
    }
    """
    # Preload reverse maps and outputs per round
    rev_maps: List[Dict[str, List[str]]] = []
    outputs_sets: List[Set[str]] = []

    for run_dir in run_dirs:
        rev = defaultdict(list)
        for s, t in load_lineage_edges(run_dir):
            rev[t].append(s)
        rev_maps.append(rev)
        outputs_sets.append(set(load_final_pairs(run_dir, "").keys()))

    # Dead-end tracking (unique)
    dead_end_nodes: Set[Tuple[int, str]] = set()
    per_round_dead: Dict[int, int] = defaultdict(int)  # r>=1 only
    base_no_original = 0

    # Cache contribution dicts as tuples (hashable for lru_cache)
    @lru_cache(maxsize=None)
    def back_contrib(r: int, node: str) -> Tuple[Tuple[str, int], ...]:
        """
        Returns an immutable (sorted) tuple of (original_id, mass) pairs
        representing contributions of 'node' when tracing back to the original set
        through rounds [r..0]. Also records dead-ends in outer sets.
        """
        nonlocal base_no_original

        # Base: before the first run -> only originals count
        if r < 0:
            # We shouldn't actually reach r < 0 because r==0 checks originals, but keep safe.
            c = original_counts.get(node, 0)
            if not c:
                # Treat as base dead-end
                if (r, node) not in dead_end_nodes:
                    dead_end_nodes.add((r, node))
                    base_no_original += 1
            return ((node, c),) if c else tuple()

        rev = rev_maps[r]
        parents = rev.get(node, [])

        if parents:
            acc: Dict[str, int] = defaultdict(int)
            for p in parents:
                for oid, c in back_contrib(r - 1, p):
                    if c:
                        acc[oid] += c
            return tuple(sorted(acc.items()))

        # No parents recorded this round -> identity passthrough only if node existed in previous outputs
        if r - 1 >= 0:
            prev_outputs = outputs_sets[r - 1]
            if node in prev_outputs:
                return back_contrib(r - 1, node)
            else:
                # Dead-end at round r (no parents, no passthrough)
                if (r, node) not in dead_end_nodes:
                    dead_end_nodes.add((r, node))
                    per_round_dead[r] += 1
                return tuple()
        else:
            # r == 0 and no parents -> check original at base
            c = original_counts.get(node, 0)
            if not c:
                if (r, node) not in dead_end_nodes:
                    dead_end_nodes.add((r, node))
                    base_no_original += 1
            return ((node, c),) if c else tuple()

    final_counts: Dict[str, int] = {}
    ancestry: Optional[Dict[str, Dict[str, int]]] = {} if include_breakdown else None

    last_idx = len(run_dirs) - 1
    for fid in finals:
        contrib = dict(back_contrib(last_idx, fid))
        total = sum(contrib.values())
        final_counts[fid] = total
        if include_breakdown and ancestry is not None:
            ancestry[fid] = contrib

    dead_stats = {
        "unique_dead_ends": len(dead_end_nodes),
        "base_no_original": base_no_original,
        "per_round": {r: per_round_dead[r] for r in sorted(per_round_dead)},
    }
    return final_counts, ancestry, dead_stats


# -------------------------- Validation ----------------------------------

def validate_coverage(
    original_counts: Dict[str, int],
    run_dirs: List[Path],
    final_counts: Dict[str, int],
    id_to_pair: Dict[str, Tuple[str, str]],
    dead_stats: Optional[Dict[str, int]] = None,
) -> None:
    """Print concise lineage coverage + conservation + dead-end counts."""
    print("[validate] Per-run lineage stats:")
    for idx, run_dir in enumerate(run_dirs, 1):
        edges = load_lineage_edges(run_dir)
        srcs = {s for s, _ in edges}
        tgts = {t for _, t in edges}
        print(
            f"[validate] run#{idx} '{run_dir}': "
            f"edges={len(edges)} unique_srcs={len(srcs)} unique_tgts={len(tgts)}"
        )

    originals_total = len(original_counts)
    run1_srcs = set()
    run1_outputs = set()
    if run_dirs:
        run1_edges = load_lineage_edges(run_dirs[0])
        run1_srcs = {s for s, _ in run1_edges}
        run1_outputs = set(load_final_pairs(run_dirs[0], "").keys())
    run1_union = run1_srcs | run1_outputs

    print("[validate] Global coverage:")
    if originals_total > 0:
        originals_ids = set(original_counts.keys())
        overlap = len(run1_union & originals_ids)
        only_in_run1 = len(run1_union - originals_ids)
        only_in_originals = len(originals_ids - run1_union)
        pct = overlap / originals_total if originals_total else 0.0
        print(
            f"[validate] originals: {originals_total} | "
            f"originals_seen_in_run1_lineage_or_outputs: {overlap} ({pct:.2%} coverage)"
        )
        print(
            f"[validate] run1_(sources∪outputs)_not_in_originals: {only_in_run1} | "
            f"originals_not_in_run1_(sources∪outputs): {only_in_originals}"
        )
    else:
        print("[validate] originals: 0 (no originals loaded) — check --original-input path and format)")

    finals_total = len(id_to_pair)
    finals_set = set(id_to_pair.keys())

    finals_with_weight = sum(1 for fid in finals_set if final_counts.get(fid, 0) > 0)
    finals_zero = finals_total - finals_with_weight
    lineage_orphans = len({k for k, v in final_counts.items() if v > 0} - finals_set)

    print(
        f"[validate] finals: {finals_total} | "
        f"finals_with_weight: {finals_with_weight} | finals_zero: {finals_zero}"
    )
    print(f"[validate] propagated_ids_not_in_finals (orphans): {lineage_orphans}")

    # Dead-end stats (if provided)
    if dead_stats is not None:
        print(f"[validate] dead-end branches (unique): {dead_stats.get('unique_dead_ends', 0)}")
        per_round = dead_stats.get("per_round", {})
        if per_round:
            parts = [f"r{r}={per_round[r]}" for r in sorted(per_round)]
            print(f"[validate]   per-round: " + " | ".join(parts))
        base = dead_stats.get("base_no_original", 0)
        print(f"[validate]   base-no-original (r=0): {base}")

    # Weight conservation
    sum_originals = sum(original_counts.values())
    sum_finals = sum(final_counts.values())
    delta = sum_finals - sum_originals
    if sum_originals == 0:
        print("[validate] weight conservation: originals_sum=0 — cannot compare.")
    else:
        status = "OK" if delta == 0 else "MISMATCH"
        rel = (abs(delta) / sum_originals) if sum_originals else 0.0
        print(
            f"[validate] weight conservation: "
            f"originals_sum={sum_originals} | finals_sum={sum_finals} | delta={delta} "
            f"(rel {rel:.6%}) [{status}]"
        )


# -------------------------- Main CLI ------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backwards tracer: from final pairs to originals (depth-first, mass-conserving) with dead-end diagnostics."
    )

    ap.add_argument(
        "--original-input",
        type=Path,
        required=True,
        help="Path to the original JSONL used before run #1 (raw or processed).",
    )
    ap.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help=("One or more run dirs, oldest->newest (e.g., .../round_1/results ... /round_5/results)"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default="../results/disambiguate_merges/traced_lineages/backtrace_enriched_counts.jsonl",
        help="Where to write the final enriched JSONL.",
    )
    ap.add_argument(
        "--include-breakdown",
        action="store_true",
        help="Include 'original_counts' object for each final row.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Print per-run stats, coverage, zero-weight finals, orphans, dead-ends, and conservation.",
    )

    args = ap.parse_args()

    # 1) originals
    original_counts = load_original_counts_from_file(args.original_input)

    # 2) finals from last run
    last_run_dir = args.runs[-1]
    id_to_pair = load_final_pairs(last_run_dir)
    finals_list = list(id_to_pair.keys())

    # 3) backtrace and sum contributions (with dead-end stats)
    final_counts, ancestry, dead_stats = compute_contributions_by_backtrace(
        original_counts,
        list(args.runs),
        finals_list,
        include_breakdown=args.include_breakdown,
    )

    # 4) optional validation
    if args.validate:
        validate_coverage(original_counts, list(args.runs), final_counts, id_to_pair, dead_stats=dead_stats)

    # 5) write output (sorted by descending contribution_count, then id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid in finals_list:
        d, e = id_to_pair.get(fid, ("", ""))
        total = int(final_counts.get(fid, 0))
        row = {
            "id": fid,
            "descriptor": d,
            "explainer": e,
            "contribution_count": total,
        }
        if args.include_breakdown and ancestry is not None and fid in ancestry:
            row["original_counts"] = {k: int(v) for k, v in sorted(ancestry[fid].items())}
        rows.append(row)

    rows.sort(key=lambda r: (-r["contribution_count"], r["id"]))

    with args.out.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Diagnostics
    print(f"Wrote {args.out}")
    print(f"[info] finals: {len(id_to_pair)}  counted: {sum(1 for fid in id_to_pair if final_counts.get(fid, 0) > 0)}")
    print(f"[info] finals with zero contribution: {sum(1 for fid in id_to_pair if final_counts.get(fid, 0) == 0)}")


if __name__ == "__main__":
    main()
