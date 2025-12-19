#!/usr/bin/env python3
"""
Trace post-disambiguation pairs through synonym + force merges to final outputs.

Inputs:
  --intermediate : JSONL from the first tracer (one per line):
                   {"final_id","descriptor","explainer","root_ids":[...],"root_count", "weight"}
  --lineage      : one or more JSONL lineage files with events:
                   {"new_pair_id": "...", "source_pair_ids": ["..."], ...}
  --final        : JSONL of final pairs after all merges:
                   {"id","descriptor","explainer"}
  --output       : JSONL to write enriched final rows:
                   {"final_id","descriptor","explainer","root_ids","root_count","weight","members"}

Behavior:
  * Includes the kept node’s own intermediate roots even when it has sources.
  * Aggregates roots/weights across all merged members.
  * members = set of intermediate (post-disambiguation) IDs rolled into each final.
  * Sorts by weight desc, tie-break by final_id for determinism.
  * Strong asserts at key points. Crashes on unexpected conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import sys

# ---------------- I/O ----------------

def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            rows.append(json.loads(line))
    return rows

# ---------------- Data checks ----------------

def load_intermediate_map(path: Path) -> Dict[str, dict]:
    """Map intermediate_id -> {root_ids: set[str], weight: int, descriptor, explainer}"""
    rows = load_jsonl(path)
    assert rows, f"No rows in intermediate file: {path}"
    seen: Set[str] = set()
    out: Dict[str, dict] = {}
    for r in rows:
        fid = r.get("final_id")
        assert isinstance(fid, str) and fid, "Each intermediate row must have non-empty 'final_id'."
        assert fid not in seen, f"Duplicate intermediate final_id encountered: {fid}"
        seen.add(fid)

        desc = r.get("descriptor", "")
        exp = r.get("explainer", "")
        roots = r.get("root_ids")
        w = r.get("weight")

        assert isinstance(desc, str) and desc.strip(), f"Intermediate({fid}) missing/empty descriptor."
        assert isinstance(exp, str), f"Intermediate({fid}) missing explainer (must be str, can be empty)."
        assert isinstance(roots, list) and len(roots) > 0, f"Intermediate({fid}) must have non-empty root_ids list."
        assert all(isinstance(x, str) and x for x in roots), f"Intermediate({fid}) has invalid root_ids."
        assert isinstance(w, int) and w >= 1, f"Intermediate({fid}) invalid weight: {w}"

        out[fid] = {
            "root_ids": set(roots),
            "weight": int(w),
            "descriptor": desc,
            "explainer": exp,
        }
    return out

def load_final_pairs(path: Path) -> Dict[str, dict]:
    """Map final_id -> {descriptor, explainer}"""
    rows = load_jsonl(path)
    assert rows, f"No rows in final file: {path}"
    out: Dict[str, dict] = {}
    for r in rows:
        fid = r.get("id")
        assert isinstance(fid, str) and fid, "Final row missing 'id'."
        assert fid not in out, f"Duplicate final id encountered: {fid}"
        desc = r.get("descriptor", "")
        exp = r.get("explainer", "")
        assert isinstance(desc, str) and desc.strip(), f"Final({fid}) missing/empty descriptor."
        assert isinstance(exp, str), f"Final({fid}) explainer must be a string."
        out[fid] = {"descriptor": desc, "explainer": exp}
    return out

def load_lineage_events(paths: List[Path]) -> List[dict]:
    events: List[dict] = []
    for p in paths:
        rows = load_jsonl(p)
        for ev in rows:
            nid = ev.get("new_pair_id")
            srcs = ev.get("source_pair_ids")
            assert isinstance(nid, str) and nid, f"Lineage event missing valid new_pair_id in {p}"
            assert isinstance(srcs, list) and srcs and all(isinstance(s, str) and s for s in srcs), \
                f"Lineage event({nid}) has invalid source_pair_ids in {p}"
            events.append({"new_pair_id": nid, "source_pair_ids": list(srcs)})
    assert events, f"No lineage events loaded from: {paths}"
    return events

# ---------------- Graph ----------------

def build_sources_map(events: List[dict]) -> Dict[str, List[str]]:
    """
    Map kept/new ID -> list of source IDs (accumulated across events).
    Multiple events with same new_pair_id are concatenated.
    """
    sources_map: Dict[str, List[str]] = defaultdict(list)
    count = 0
    for ev in events:
        nid = ev["new_pair_id"]
        srcs = ev["source_pair_ids"]
        sources_map[nid].extend(srcs)
        count += len(srcs)
    # Simple consistency assertion: we didn't “lose” sources
    total_accumulated = sum(len(v) for v in sources_map.values())
    assert total_accumulated == count, "Mismatch accumulating lineage sources."
    return sources_map

# ---------------- Tracing ----------------

def trace_final(
    final_id: str,
    sources_map: Dict[str, List[str]],
    interm: Dict[str, dict],
    memo: Dict[str, Tuple[Set[str], int, Set[str]]],
    stack: Set[str],
) -> Tuple[Set[str], int, Set[str]]:
    """
    Return (combined_original_root_ids, combined_weight, intermediate_members)
    DFS with cycle detection; includes the kept node's own intermediate roots if present.
    """
    if final_id in memo:
        return memo[final_id]

    if final_id in stack:
        # Cycle detected -> crash loudly
        raise AssertionError(f"Cycle detected in lineage at id={final_id}")

    stack.add(final_id)

    combined_roots: Set[str] = set()
    combined_members: Set[str] = set()
    combined_weight = 0

    # Include the kept node's own intermediate roots/weight, if it existed after disambiguation
    if final_id in interm:
        combined_roots |= interm[final_id]["root_ids"]
        combined_weight += int(interm[final_id]["weight"])
        combined_members.add(final_id)

    # Recurse into sources (dropped ids merged into this id)
    for src in sources_map.get(final_id, []):
        roots, w, members = trace_final(src, sources_map, interm, memo, stack)
        combined_roots |= roots
        combined_weight += w
        combined_members |= members

    stack.remove(final_id)
    memo[final_id] = (combined_roots, combined_weight, combined_members)
    return memo[final_id]

# ---------------- Audit ----------------

def audit_mappings(
    final_rows: List[dict],
    intermediate_ids: Set[str],
    members_per_final: Dict[str, Set[str]],
) -> None:
    """
    Crash on violations:
      - A final has no members or no roots.
      - Any intermediate id maps to 0 or >1 finals.
    """
    # 1) Per-final assertions already checked below, but double-check here too
    for r in final_rows:
        assert r["root_ids"], f"Final({r['final_id']}) has empty root_ids after tracing."
        assert r["members"], f"Final({r['final_id']}) has empty members after tracing."
        assert r["weight"] >= 1, f"Final({r['final_id']}) has non-positive weight."

    # 2) Intermediate coverage and uniqueness
    owner: Dict[str, str] = {}
    for fid, mset in members_per_final.items():
        for mid in mset:
            if mid in owner and owner[mid] != fid:
                raise AssertionError(
                    f"Intermediate id {mid} appears under two finals: {owner[mid]} and {fid}"
                )
            owner[mid] = fid

    uncovered = intermediate_ids - set(owner.keys())
    if uncovered:
        # Hard fail: every post-disambig id must end up in exactly one final
        raise AssertionError(
            f"{len(uncovered)} intermediate ids are not accounted for by any final."
        )

# ---------------- Orchestration ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Trace final merges from intermediate roots with strong audits.")
    ap.add_argument("--intermediate", type=Path, required=True,
                    help="JSONL: rows from the disambiguation tracer with fields final_id, root_ids, weight, descriptor, explainer")
    ap.add_argument("--lineage", type=Path, nargs="+", required=True,
                    help="One or more lineage JSONL files from synonym/force merges (events with new_pair_id, source_pair_ids)")
    ap.add_argument("--final", type=Path, required=True,
                    help="JSONL: final pairs after all merges (id, descriptor, explainer)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write enriched final rows as JSONL")
    args = ap.parse_args()

    # Load inputs
    interm = load_intermediate_map(args.intermediate)
    intermediate_ids = set(interm.keys())

    events = load_lineage_events(args.lineage)
    sources_map = build_sources_map(events)

    final_pairs = load_final_pairs(args.final)

    # Compute results
    memo: Dict[str, Tuple[Set[str], int, Set[str]]] = {}
    final_rows: List[dict] = []
    members_per_final: Dict[str, Set[str]] = {}

    for fid, meta in final_pairs.items():
        roots, weight, members = trace_final(fid, sources_map, interm, memo, set())
        # If this final id never appeared in lineage AND wasn't in intermediate,
        # that's unexpected: there is no path to any original roots.
        if not roots:
            # Attempt pass-through rescue if it equals an intermediate id (should already be covered)
            if fid in interm:
                roots = set(interm[fid]["root_ids"])
                weight = int(interm[fid]["weight"])
                members = {fid}

        assert roots, f"Final({fid}) could not be traced to any intermediate roots."
        assert isinstance(weight, int) and weight >= 1, f"Final({fid}) has invalid weight: {weight}"
        assert members, f"Final({fid}) produced empty members set."

        row = {
            "final_id": fid,
            "descriptor": meta["descriptor"],
            "explainer": meta["explainer"],
            "root_ids": sorted(roots),
            "root_count": len(roots),
            "weight": int(weight),
            "members": sorted(members),
        }
        final_rows.append(row)
        members_per_final[fid] = set(members)

    # Global audit: uniqueness & coverage of intermediate ids
    audit_mappings(final_rows, intermediate_ids, members_per_final)

    # Sort by weight desc, tie-break by final_id for determinism
    final_rows.sort(key=lambda r: (-r["weight"], r["final_id"]))

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Basic summary to stderr
    total_weight = sum(r["weight"] for r in final_rows)
    print(f"OK: wrote {len(final_rows)} rows. Total weight={total_weight}.", file=sys.stderr)

if __name__ == "__main__":
    main()
