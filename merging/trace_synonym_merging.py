#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple
import re

# ---------- DSU ----------
class DSU:
    def __init__(self):
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

# ---------- Helpers ----------
_ITER_RX = re.compile(r"iteration_(\d+)_llm_decisions\.jsonl$")

def list_iteration_files(run_dir: Path) -> List[Tuple[int, Path]]:
    """Return [(iter_number, path), ...] sorted by iteration number."""
    files = []
    for p in run_dir.glob("iteration_*_llm_decisions.jsonl"):
        m = _ITER_RX.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        files.append((it, p))
    files.sort(key=lambda t: t[0])
    return files

def load_all_ids_from_jsonl(jsonl_path: Path, field: str = "id") -> Set[str]:
    ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                _id = str(rec.get(field, "") or "")
                if _id:
                    ids.add(_id)
            except Exception:
                continue
    return ids

def replay_iteration_jsonl(dsu: DSU, jsonl_path: Path, all_ids: Set[str],
                           edges_out: List[Dict], run_label: str, iter_num: int) -> int:
    """Apply merges from one iteration decisions file, honoring per-iteration 'used' constraint."""
    used: Set[str] = set()
    merges = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # collect ids even if never merged
            a_id = str(rec.get("a_id", "") or "")
            b_id = str(rec.get("b_id", "") or "")
            if a_id: all_ids.add(a_id)
            if b_id: all_ids.add(b_id)

            if not rec.get("is_synonym", False):
                continue
            keep = str(rec.get("keep_id", "")).strip()
            drop = str(rec.get("drop_id", "")).strip()
            if not keep or not drop or keep == drop:
                continue
            if keep in used or drop in used:
                continue  # would have been skipped by _apply_merges_greedy
            used.add(keep); used.add(drop)
            dsu.union(keep, drop)
            edges_out.append({"keep": keep, "drop": drop, "run": run_label, "iter": iter_num})
            merges += 1
    return merges

def build_groups(dsu: DSU, all_ids: Set[str]) -> Dict[str, List[str]]:
    clusters: Dict[str, Set[str]] = {}
    for x in sorted(all_ids):
        root = dsu.find(x)
        clusters.setdefault(root, set()).add(x)
    # Sort members; keep DSU root as key (stable because DSU is deterministic here)
    return {root: sorted(list(members)) for root, members in sorted(clusters.items(), key=lambda kv: kv[0])}

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct end-to-end lineage across multiple runs of synonym merging."
    )
    p.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        required=True,
        help="Add a run with a label and its directory containing iteration_*_llm_decisions.jsonl. "
             "Add in chronological order: five earlier runs first, then the final run.",
    )
    p.add_argument(
        "--all-ids",
        action="append",
        default=[],
        metavar="JSONL",
        help="Optional: JSONL file(s) with original 'id' fields to ensure all IDs, even if never in any decision, appear as singletons. "
             "Repeat flag to add multiple files.",
    )
    p.add_argument(
        "--out-groups",
        type=str,
        default="merged_groups.jsonl",
        help="Output JSONL: each line is {\"ROOT_ID\": [\"member1\", ...]}",
    )
    p.add_argument(
        "--out-edges",
        type=str,
        default="merge_edges.jsonl",
        help="Output JSONL: each line is {\"keep\":\"ID\", \"drop\":\"ID\", \"run\":\"label\", \"iter\":N}",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Collect all IDs from provided input JSONLs (optional but recommended)
    all_ids: Set[str] = set()
    for jp in args.all_ids:
        ids = load_all_ids_from_jsonl(Path(jp), field="id")
        all_ids.update(ids)

    dsu = DSU()
    edges: List[Dict] = []
    total_merges = 0

    # Process runs in the order provided
    for label, dirpath in args.run:
        run_dir = Path(dirpath)
        files = list_iteration_files(run_dir)
        if not files:
            raise FileNotFoundError(f"No iteration_*_llm_decisions.jsonl in {run_dir}")
        for iter_num, fp in files:
            m = replay_iteration_jsonl(dsu, fp, all_ids, edges, label, iter_num)
            total_merges += m
            print(f"[{label}] iter {iter_num}: {m} merges")

    # Build groups including singletons
    groups = build_groups(dsu, all_ids if all_ids else set(dsu.parent.keys()))

    # Write groups.jsonl
    out_groups = Path(args.out_groups)
    with out_groups.open("w", encoding="utf-8") as f:
        for root, members in groups.items():
            f.write(json.dumps({root: members}, ensure_ascii=False) + "\n")

    # Write edges.jsonl (provenance)
    out_edges = Path(args.out_edges)
    with out_edges.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(groups)} groups to {out_groups} and {len(edges)} edges to {out_edges}. "
          f"Total unions applied: {total_merges}")

if __name__ == "__main__":
    main()
