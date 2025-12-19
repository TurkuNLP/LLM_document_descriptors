#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from typing import Dict, Set, List, Any, Tuple, Iterable, Optional
import numpy as np
import os

import sys
sys.path.insert(0, "../disambiguation")  # for input_processing
from input_processing import normalize_descriptor, split_pair, generate_stable_id


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_many_jsonl(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load and concatenate many JSONL files."""
    events: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"WARNING: lineage file not found: {p}")
            continue
        events.extend(load_jsonl(p))
    return events


def build_lineage_graph(lineage_events: List[Dict[str, Any]]):
    """
    Build a directed lineage graph from events across ALL steps.
    Crucial difference: if the same new_pair_id appears in multiple events,
    we **accumulate** (union) its sources instead of overwriting.

    Returns:
        sources_map: dict[new_id] -> set(source_ids)
        target_map:  dict[source_id] -> list(new_id)
    """
    sources_map: Dict[str, Set[str]] = {}
    target_map: Dict[str, List[str]] = defaultdict(list)

    for ev in lineage_events:
        # tolerate different event payloads as long as these keys exist
        if "new_pair_id" not in ev or "source_pair_ids" not in ev:
            # skip malformed events silently (or print a warning)
            # print(f"Skipping malformed lineage event: {ev}")
            continue

        new_id = ev["new_pair_id"]
        srcs = ev["source_pair_ids"]
        if not isinstance(srcs, list):
            # normalize scalar to list
            srcs = [srcs]

        # ACCUMULATE sources for the same new_id
        if new_id not in sources_map:
            sources_map[new_id] = set()
        for sid in srcs:
            sid = str(sid)
            sources_map[new_id].add(sid)
            target_map[sid].append(new_id)

    return sources_map, target_map


def find_root_inputs(
    id_to_trace: str,
    sources_map: Dict[str, Set[str]],
    memo: Optional[Dict[str, Set[str]]] = None,
    path_guard: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Recursively find all root input IDs that contributed to the given ID.
    Uses memoization and a simple cycle guard.
    """
    if memo is None:
        memo = {}
    if path_guard is None:
        path_guard = set()

    if id_to_trace in memo:
        return memo[id_to_trace]

    # cycle guard
    if id_to_trace in path_guard:
        # Cycle detected; treat this node as a leaf to avoid infinite recursion.
        # You may also choose to log/raise. We degrade gracefully here.
        memo[id_to_trace] = {id_to_trace}
        return memo[id_to_trace]
    path_guard.add(id_to_trace)

    # If this ID doesn't have sources in our map, it is a root input
    if id_to_trace not in sources_map or not sources_map[id_to_trace]:
        memo[id_to_trace] = {id_to_trace}
        path_guard.remove(id_to_trace)
        return memo[id_to_trace]

    # Recursively trace back all sources
    all_roots = set()
    for source_id in sources_map[id_to_trace]:
        all_roots.update(find_root_inputs(source_id, sources_map, memo, path_guard))

    memo[id_to_trace] = all_roots
    path_guard.remove(id_to_trace)
    return all_roots


def count_root_occurrences(root_ids: Set[str], original_ids: List[str]) -> Dict[str, int]:
    """
    Count the occurrences of each root ID in the original data.
    This handles cases where IDs were deduplicated during extraction
    but originally appeared multiple times.
    """
    id_counts = Counter(original_ids)

    # Make sure every root ID has at least a count of 1
    for root_id in root_ids:
        if root_id not in id_counts:
            id_counts[root_id] = 1
            print(f"Warning: root ID {root_id} not found in original data, defaulting to count=1")

    return id_counts


def collect_original_ids(original_data_path: Path) -> List[str]:
    """Collect all explainer IDs from the original data (pre-pipeline stable IDs)."""
    original_ids: List[str] = []
    with original_data_path.open("r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            best_idx = int(np.argmax(doc["similarity"]))
            descriptors: Iterable[str] = doc["descriptors"][best_idx]
            for desc_exp in descriptors:
                d, e = split_pair(desc_exp)
                if d and e:
                    d = normalize_descriptor(d)
                    e = e.strip()
                    pid = generate_stable_id(d, e)
                    original_ids.append(pid)
    return original_ids


def analyze_lineage(
    final_results_path: Path,
    lineage_paths: List[Path],
    original_data_path: Path,
    output_path: Path
) -> bool:
    """
    Analyze lineage across all three pipeline stages.

    - `final_results_path`: the very final results after force merging
    - `lineage_paths`: one or more lineage JSONL files from *all* stages
      (e.g., disambiguation full_lineage.jsonl, synonym merges lineage, force merges lineage)
    - `original_data_path`: the raw original inputs used to generate stable IDs
    """
    print(f"Loading final results from {final_results_path}")
    final_results = load_jsonl(final_results_path)

    print(f"Loading lineage from {len(lineage_paths)} file(s)")
    lineage_events = load_many_jsonl(lineage_paths)

    # Build the consolidated lineage graph with ACCUMULATED sources
    sources_map, target_map = build_lineage_graph(lineage_events)

    # Get all original IDs from the input data
    print(f"Collecting original IDs from {original_data_path}")
    original_ids: List[str] = collect_original_ids(original_data_path)
    unique_original_ids = set(original_ids)
    print(f"Found {len(unique_original_ids)} original IDs")

    # Process each final result
    results: List[Dict[str, Any]] = []
    memo: Dict[str, Set[str]] = {}
    checks_passed = True
    original_to_final: Dict[str, List[str]] = defaultdict(list)

    print(f"Analyzing {len(final_results)} final pairs...")

    # Compute consolidated duplicate-source-set check AFTER consolidation
    consolidated_source_sets: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for new_id, srcs in sources_map.items():
        key = tuple(sorted(srcs))
        consolidated_source_sets[key].append(new_id)
    collisions = [(srcs, ids) for srcs, ids in consolidated_source_sets.items() if len(ids) > 1]
    if collisions:
        checks_passed = False
        print(f"\nFound {len(collisions)} consolidated source-set collisions (same sources -> multiple new IDs).")
        for i, (srcs, ids) in enumerate(collisions[:5]):
            print(f"  {i+1}. Sources {list(srcs)[:3]}... -> Finals: {ids[:3]}{'...' if len(ids) > 3 else ''}")

    # Trace each final to roots
    for i, result in enumerate(final_results, 1):
        if i % 1000 == 0:
            print(f"Processed {i} pairs...")

        final_id = result.get("id")
        descriptor = result.get("descriptor", "")
        explainer = result.get("explainer", "")

        root_ids = find_root_inputs(final_id, sources_map, memo)

        # Fallbacks for pass-through cases not explicitly recorded in lineage
        if not root_ids:
            if "source_pair_ids" in result and result["source_pair_ids"]:
                root_ids = set(map(str, result["source_pair_ids"]))
            elif final_id in unique_original_ids:
                root_ids = {final_id}

        for rid in root_ids:
            original_to_final[rid].append(final_id)

        results.append({
            "final_id": final_id,
            "descriptor": descriptor,
            "explainer": explainer,
            "root_ids": sorted(root_ids),
            "root_count": len(root_ids),
        })

    # Weighting by original frequency
    all_root_ids: Set[str] = set()
    for r in results:
        all_root_ids.update(r["root_ids"])
    print(f"Found {len(all_root_ids)} unique root IDs")

    print("Counting occurrences in original data.")
    id_counts = count_root_occurrences(all_root_ids, original_ids)

    for r in results:
        r["weight"] = sum(id_counts[root_id] for root_id in r["root_ids"])

    # Sort by weight desc
    results.sort(key=lambda x: x["weight"], reverse=True)

    # Save analysis
    print(f"Saving analysis to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total_weight = sum(r["weight"] for r in results)
    print(f"Analysis complete. Total weight: {total_weight}")
    print("Top 5 weighted pairs:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"  {r['descriptor']}: weight={r['weight']}, roots={r['root_count']}")

    # SANITY CHECK 1: Each final pair is traceable to at least one original input
    no_roots = [r for r in results if not r["root_ids"]]
    if no_roots:
        checks_passed = False
        print("\nWARNING: Found final pairs with no traceable root inputs!")
        print(f"Count: {len(no_roots)}")
        for i, r in enumerate(no_roots[:5]):
            print(f"  {i+1}. {r['final_id']}: {r['descriptor']}")
        if len(no_roots) > 5:
            print(f"  ... and {len(no_roots) - 5} more")
    else:
        print("\nSanity check passed: All final pairs are traceable to at least one original input.")

    # SANITY CHECK 2: All original inputs are traceable to exactly one final pair
    traced_roots = set(original_to_final.keys())
    untraceable = unique_original_ids - traced_roots
    if untraceable:
        checks_passed = False
        print("\nWARNING: Found original inputs that aren't traceable to any final pair!")
        print(f"Count: {len(untraceable)}")
        print(f"Examples: {list(untraceable)[:5]}")

    duplicated = [oid for oid, finals in original_to_final.items() if len(set(finals)) > 1]
    if duplicated:
        checks_passed = False
        print("\nWARNING: Found original inputs that trace to multiple final pairs!")
        print(f"Count: {len(duplicated)}")
        for i, oid in enumerate(duplicated[:5]):
            print(f"  {i+1}. {oid} -> {original_to_final[oid]}")
        if len(duplicated) > 5:
            print(f"  ... and {len(duplicated) - 5} more")

    if not untraceable and not duplicated:
        print("Sanity check passed: All original inputs trace to exactly one final pair.")

    return checks_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze lineage end-to-end (disambiguation + synonym merges + force merges)"
    )
    parser.add_argument("--final", type=Path, required=True,
                        help="Path to FINAL results JSONL (after force merge).")
    parser.add_argument("--lineage", type=Path, nargs='+', required=True,
                        help="One or more lineage JSONL files from ALL stages (order does not matter).")
    parser.add_argument("--original", type=Path, required=True,
                        help="Path to original data.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for consolidated lineage analysis JSONL.")
    args = parser.parse_args()

    checks_passed = analyze_lineage(args.final, args.lineage, args.original, args.output)

    slurm_id = os.getenv("SLURM_JOB_ID", "")
    if not checks_passed:
        with open(args.output.with_suffix('.failed'), 'w') as f:
            f.write("Lineage analysis failed sanity checks.\n")
            if slurm_id:
                f.write(f"See log file with SLURM_ID {slurm_id} for details.\n")
    else:
        with open(args.output.with_suffix('.passed'), 'w') as f:
            f.write("Lineage analysis passed all sanity checks.\n")
            if slurm_id:
                f.write(f"See log file with SLURM_ID: {slurm_id} for details.\n")
