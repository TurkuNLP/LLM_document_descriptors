#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from typing import Dict, Set, List, Any, Tuple, Iterable
import numpy as np
import os

from input_processing import normalize_descriptor, split_pair, generate_stable_id


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def build_lineage_graph(lineage_events: List[Dict[str, Any]]):
    """Build a directed graph of ID -> source IDs from lineage events."""
    # Map from ID to its immediate sources
    sources_map = {}
    # Map from source ID to IDs it feeds into
    target_map = defaultdict(list)
    
    for event in lineage_events:
        new_id = event["new_pair_id"]
        source_ids = event["source_pair_ids"]
        
        sources_map[new_id] = source_ids
        for source_id in source_ids:
            target_map[source_id].append(new_id)
    
    return sources_map, target_map


def find_root_inputs(id_to_trace: str, sources_map: Dict[str, List[str]], 
                     memo: Dict[str, Set[str]] = None) -> Set[str]:
    """
    Recursively find all root input IDs that contributed to the given ID.
    Uses memoization to avoid redundant computation.
    """
    if memo is None:
        memo = {}
    
    if id_to_trace in memo:
        return memo[id_to_trace]
    
    # If this ID doesn't have sources in our map, it is a root input
    if id_to_trace not in sources_map:
        memo[id_to_trace] = {id_to_trace}
        return memo[id_to_trace]
    
    # Recursively trace back all sources
    all_roots = set()
    for source_id in sources_map[id_to_trace]:
        all_roots.update(find_root_inputs(source_id, sources_map, memo))
    
    memo[id_to_trace] = all_roots
    return all_roots


def count_root_occurrences(root_ids: Set[str], original_ids) -> Dict[str, int]:
    """
    Count the occurrences of each root ID in the original data.
    This handles cases where IDs were deduplicated during extraction but originally appeared multiple times.
    """
    # Counts frequencies of all original IDs
    id_counts = Counter(original_ids)

    # Make sure every root ID has at least a count of 1
    for root_id in root_ids:
        if root_id not in id_counts:
            id_counts[root_id] = 1
            print(f"Warning: root ID {root_id} not found in original data, defaulting to count=1")
    
    return id_counts


def collect_original_ids(original_data_path: Path) -> List[str]:
    """Collect all explainer IDs from the original data."""
    original_ids = []
    with open(original_data_path, "r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            best_idx = int(np.argmax(doc["similarity"]))
            descriptors: Iterable[str] = doc["descriptors"][best_idx]
            for desc_exp in descriptors:
                # Split pair by ";"
                d, e = split_pair(desc_exp)
                # At this stage, we silently ignore malformed pairs
                if d and e:
                    # Normalize descriptor and strip explainer
                    d = normalize_descriptor(d)
                    e = e.strip()
                    pid = generate_stable_id(d, e)
                    original_ids.append(pid)

    return original_ids

def analyze_lineage(
    final_results_path: Path,
    lineage_path: Path,
    original_data_path: Path,
    output_path: Path
    ) -> bool:
    """
    Analyze the lineage of final descriptor-explainer pairs.
    
    For each final pair, find:
    1. All root input IDs that contributed to it
    2. The weight (total occurrences of these root IDs in original data)
    
    Also performs sanity checks:
    1. Each final pair is traceable to at least one original input
    2. All original inputs are traceable to exactly one final pair
    """
    print(f"Loading final results from {final_results_path}")
    final_results = load_jsonl(final_results_path)
    
    print(f"Loading lineage from {lineage_path}")
    lineage_events = load_jsonl(lineage_path)
    
    # Build the lineage graph
    sources_map, target_map = build_lineage_graph(lineage_events)
    
    # Set to keep track of all root IDs we find
    all_root_ids = set()
    
    # Get all original IDs from the input data
    print(f"Collecting original IDs from {original_data_path}")
    original_ids: List[str] = collect_original_ids(original_data_path)
    unique_original_ids = set(original_ids)
    print(f"Found {len(unique_original_ids)} original IDs")
    
    # Process each final result
    results = []
    memo = {}  # Memoization for find_root_inputs to avoid redundant computation
    
    # Flip to False if any sanity checks fail
    checks_passed = True
    
    # Track which final ID each original ID traces to
    original_to_final = defaultdict(list)
    
    print(f"Analyzing {len(final_results)} final pairs...")
    
    # Find identical source ID sets pointing to different final IDs
    source_sets = {}
    duplicates = []

    for event in lineage_events:
        new_id = event["new_pair_id"]
        source_ids = tuple(sorted(event["source_pair_ids"]))  # Convert to sorted tuple for hashing
        
        if source_ids in source_sets:
            # Found duplicate - same source IDs, different final ID
            existing_id = source_sets[source_ids]
            duplicates.append((source_ids, existing_id, new_id))
        else:
            source_sets[source_ids] = new_id

    if duplicates:
        checks_passed = False
        print(f"\nFound {len(duplicates)} cases of identical source ID sets pointing to different final IDs")
        for i, (source_ids, id1, id2) in enumerate(duplicates[:5]):
            print(f"  {i+1}. Sources {source_ids[:3]}... -> Finals: {id1} and {id2}")
        
        # Get descriptor/explainer for these final IDs
        final_id_info = {r["final_id"]: (r["descriptor"], r["explainer"]) for r in results}
        
        for i, (_, id1, id2) in enumerate(duplicates[:5]):
            if id1 in final_id_info and id2 in final_id_info:
                desc1, exp1 = final_id_info[id1]
                desc2, exp2 = final_id_info[id2]
                print(f"\nDuplicate {i+1}:")
                print(f"  ID1 {id1}: {desc1} | {exp1[:50]}...")
                print(f"  ID2 {id2}: {desc2} | {exp2[:50]}...")
    
    
    
    
    for i, result in enumerate(final_results):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} pairs...")
            
        final_id = result["id"]
        descriptor = result["descriptor"]
        explainer = result["explainer"]
        
        # Find all root inputs for this final ID
        root_ids = find_root_inputs(final_id, sources_map, memo)
        
        # Handle singletons (pass-through pairs) that don't appear in lineage
        if not root_ids:
            # Check if it has source_pair_ids field
            if "source_pair_ids" in result and result["source_pair_ids"]:
                root_ids = set(result["source_pair_ids"])
            else:
                # It might be an original ID that passed through unchanged
                if final_id in unique_original_ids:
                    root_ids = {final_id}
        
        all_root_ids.update(root_ids)
        
        # Update the mapping from original IDs to final IDs
        for root_id in root_ids:
            original_to_final[root_id].append(final_id)
        
        results.append({
            "final_id": final_id,
            "descriptor": descriptor,
            "explainer": explainer,
            "root_ids": sorted(list(root_ids)),
            "root_count": len(root_ids)
        })
    
    print(f"Found {len(all_root_ids)} unique root IDs")
    
    # Count occurrences of root IDs in original data
    print(f"Counting occurrences in original data.")
    id_counts = count_root_occurrences(all_root_ids, original_ids)
    
    # Calculate weights
    for result in results:
        weight = sum(id_counts[root_id] for root_id in result["root_ids"])
        result["weight"] = weight
    
    # Sort by weight in descending order
    results.sort(key=lambda x: x["weight"], reverse=True)
    
    # Save results
    print(f"Saving analysis to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Print summary
    total_weight = sum(result["weight"] for result in results)
    print(f"Analysis complete. Total weight: {total_weight}")
    print(f"Top 5 weighted pairs:")
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
    untraceable = unique_original_ids - all_root_ids
    if untraceable:
        checks_passed = False
        print("\nWARNING: Found original inputs that aren't traceable to any final pair!")
        print(f"Count: {len(untraceable)}")
        print(f"Examples: {list(untraceable)[:5]}")
    
    duplicated = [oid for oid, finals in original_to_final.items() if len(finals) > 1]
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
    parser = argparse.ArgumentParser(description="Analyze lineage of descriptor-explainer pairs")
    parser.add_argument("--final", type=Path, required=True, 
                        help="Path to final disambiguation results (disambig.jsonl)")
    parser.add_argument("--lineage", type=Path, required=True,
                        help="Path to lineage events (full_lineage.jsonl)")
    parser.add_argument("--original", type=Path, required=True,
                        help="Path to original data (grouped_descriptors_with_ids.jsonl)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for analysis")
    
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
