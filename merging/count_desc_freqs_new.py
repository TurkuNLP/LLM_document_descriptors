#!/usr/bin/env python3
"""
Trace lineage using *pair hash IDs* across many shards, with per-final *pair* counting.

Changes vs v1:
- final_descriptor_counts.csv now counts by (final_descriptor, final_explainer)
  so duplicate descriptors with different explainers appear on separate rows.
- final_descriptor_pair_counts.csv now includes `final_explainer` in addition to
  the original pair breakdown.

CLI examples
------------
python trace_lineage_with_ids_multi_v2.py \
  --source ../results/new_descriptors/all_descriptors_new.jsonl \
  --lineage-roots ../runA/checkpoints ../runB/checkpoints \
  --finals-roots ../runA ../runB \
  --out-dir ../merged_trace
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, DefaultDict
import re
from collections import Counter, defaultdict
import hashlib

# -------------------- Normalization + ID --------------------
SEP = "\u241f"  # must match extractor/disambiguator


def select_true_finals(adj, final_ids):
    """
    Return only those final_ids that are true sinks in the lineage graph:
    nodes with no outgoing edges to a *different* node (ignoring self-loops).
    """
    sinks = set()
    for fid in final_ids:
        outs = adj.get(fid, set())
        outs_wo_self = {o for o in outs if o != fid}
        if len(outs_wo_self) == 0:
            sinks.add(fid)
    return sinks


def normalize_descriptor(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\s]+", " ", s)
    return s


def pair_id(descriptor_normalized: str, explainer_raw: str, length: int = 12) -> str:
    key = f"{descriptor_normalized}{SEP}{explainer_raw}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]


# -------------------- IO helpers --------------------


def load_jsonl(path: Path) -> List[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {e}") from e
    return data


# -------------------- Source extractor --------------------


def split_pair(text: str) -> Tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    try:
        d, e = text.split(";", 1)
    except ValueError:
        return "", ""
    d_n = normalize_descriptor(d)
    e = (e or "").strip()
    if not d_n or not e:
        return "", ""
    return d_n, e


def extract_pairs_from_doc(doc: dict) -> List[Tuple[str, str]]:
    src = doc.get("descriptors")
    if src is None:
        src = doc.get("descriptor_list")
    if src is None:
        return []

    if isinstance(src, list) and all(isinstance(x, str) for x in src):
        pairs = [split_pair(x) for x in src]
        return [(d, e) for d, e in pairs if d and e]

    if isinstance(src, list) and all(isinstance(x, list) for x in src):
        sims = doc.get("similarity")
        chosen_lists: List[List[str]]
        if isinstance(sims, list) and len(sims) == len(src) and len(sims) > 0:
            try:
                import numpy as _np  # type: ignore

                best_idx = int(_np.argmax(_np.array(sims)))
                chosen_lists = [src[best_idx]]
            except Exception:
                chosen_lists = src
        else:
            chosen_lists = src
        out: List[Tuple[str, str]] = []
        for lst in chosen_lists:
            for item in lst:
                d, e = split_pair(item)
                if d and e:
                    out.append((d, e))
        return out
    return []


# -------------------- Lineage graph --------------------


def iter_lineage_files(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_file():
            if p.name == "full_lineage.jsonl":
                out.append(p)
        elif p.is_dir():
            out.extend(p.rglob("full_lineage.jsonl"))
    return sorted(set(out))


def build_graph_from_lineage(lineage_paths: List[Path]) -> DefaultDict[str, List[str]]:
    adj: DefaultDict[str, List[str]] = defaultdict(list)
    for p in lineage_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tgt = obj.get("new_pair_id")
                srcs = obj.get("source_pair_ids") or []
                if not tgt or not isinstance(srcs, list) or len(srcs) == 0:
                    continue
                tgt = str(tgt)
                for s in srcs:
                    s = str(s)
                    adj[s].append(tgt)
    return adj


# -------------------- Finals --------------------


def iter_disambig_files(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".jsonl":
            out.append(p)
        elif p.is_dir():
            for pat in ["*_disambig.jsonl", "*.jsonl"]:
                out.extend([q for q in p.glob(pat)])
    return sorted(set(out))


def load_finals_multi(paths: List[Path]) -> Tuple[Set[str], Dict[str, Tuple[str, str]]]:
    """Return (final_ids, id2pair) where id2pair[id] = (normalized_descriptor, explainer_raw)."""
    final_ids: Set[str] = set()
    id2pair: Dict[str, Tuple[str, str]] = {}
    for p in paths:
        rows = load_jsonl(p)
        for r in rows:
            pid = r.get("id")
            desc = r.get("descriptor") or r.get("final_descriptor") or r.get("name")
            expl = r.get("explainer") or r.get("group_explainer") or ""
            if isinstance(pid, str) and isinstance(desc, str):
                final_ids.add(pid)
                nd = normalize_descriptor(desc)
                if pid in id2pair and id2pair[pid] != (nd, expl):
                    print(
                        f"[warn] conflicting final pair for id {pid}: {id2pair[pid]} vs {(nd, expl)} from {p}"
                    )
                else:
                    id2pair[pid] = (nd, expl)
    return final_ids, id2pair


# -------------------- Reachability + counting --------------------


def reachable_finals(
    start_id: str,
    adj: Dict[str, List[str]],
    final_ids: Set[str],
    cache: Dict[str, Set[str]],
) -> Set[str]:
    if start_id in cache:
        return cache[start_id]
    seen: Set[str] = set()
    out: Set[str] = set()
    stack = [start_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in final_ids:
            out.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)
    cache[start_id] = out
    return out


def count_contributions(
    docs: Iterable[dict],
    adj: Dict[str, List[str]],
    final_ids: Set[str],
    id2pair: Dict[str, Tuple[str, str]],
):
    """
    Returns:
      totals: Dict[(final_descriptor, final_explainer), total_count]
      pair_breakdown: Dict[(final_descriptor, final_explainer, original_descriptor, original_explainer), count]
      resolved_pairs_records: list of diagnostic dicts per original pair (with final ids + pairs)
      unresolved_pairs: list of (original_descriptor, original_explainer)
    """
    totals: Counter = Counter()
    pair_breakdown: Counter = Counter()
    unresolved_pairs: List[Tuple[str, str]] = []
    resolved_pairs_records: List[dict] = []

    pair_occurrences: Counter = Counter()  # original id -> count
    id2orig: Dict[str, Tuple[str, str]] = {}

    for doc in docs:
        # compute original ids
        src = doc.get("descriptors")
        if src is None:
            src = doc.get("descriptor_list")
        if src is None:
            continue

        lists: List[str] = []
        if isinstance(src, list) and all(isinstance(x, str) for x in src):
            lists = src
        elif isinstance(src, list) and all(isinstance(x, list) for x in src):
            sims = doc.get("similarity")
            if isinstance(sims, list) and len(sims) == len(src) and len(sims) > 0:
                try:
                    import numpy as _np  # type: ignore

                    best_idx = int(_np.argmax(_np.array(sims)))
                    lists = src[best_idx]
                except Exception:
                    for lst in src:
                        lists.extend(lst)
            else:
                for lst in src:
                    lists.extend(lst)
        else:
            continue

        for item in lists:
            d, e = split_pair(item)
            if not d or not e:
                continue
            pid = pair_id(d, e)
            pair_occurrences[pid] += 1
            id2orig.setdefault(pid, (d, e))

    cache: Dict[str, Set[str]] = {}
    for pid, occ in pair_occurrences.items():
        finals = reachable_finals(pid, adj, final_ids, cache)
        if not finals:
            unresolved_pairs.append(id2orig.get(pid, ("", "")))
            continue
        fin_pairs = [id2pair[fid] for fid in finals]  # (desc_norm, expl_raw)
        fin_pairs_sorted = sorted(set(fin_pairs), key=lambda t: (t[0], t[1]))
        resolved_pairs_records.append(
            {
                "original_id": pid,
                "original_descriptor": id2orig[pid][0],
                "original_explainer": id2orig[pid][1],
                "final_ids": sorted(list(finals)),
                "final_pairs": [
                    {"descriptor": d, "explainer": e} for (d, e) in fin_pairs_sorted
                ],
                "occurrences": int(occ),
            }
        )
        for fid in finals:
            fdesc, fexp = id2pair[fid]
            totals[(fdesc, fexp)] += occ
            pair_breakdown[(fdesc, fexp, id2orig[pid][0], id2orig[pid][1])] += occ

    return dict(totals), dict(pair_breakdown), resolved_pairs_records, unresolved_pairs


# -------------------- Writers --------------------


def write_counts_csv(path: Path, counts: Dict[Tuple[str, str], int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["final_descriptor", "final_explainer", "count"])
        for (desc, expl), c in sorted(
            counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
        ):
            w.writerow([desc, expl, c])


def write_pair_breakdown_csv(
    path: Path, breakdown: Dict[Tuple[str, str, str, str], int]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "final_descriptor",
                "final_explainer",
                "original_descriptor",
                "original_explainer",
                "count",
            ]
        )
        for (fdesc, fexp, od, oe), c in sorted(
            breakdown.items(), key=lambda x: (-x[1], x[0][0], x[0][1], x[0][2])
        ):
            w.writerow([fdesc, fexp, od, oe, c])


def write_resolved_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_unresolved_txt(path: Path, pairs: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d, e in pairs:
            f.write(f"{d}; {e}\n")


# -------------------- CLI --------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", type=Path, required=True, help="Path to all_descriptors_new.jsonl"
    )

    lg = ap.add_mutually_exclusive_group(required=True)
    lg.add_argument(
        "--lineage-roots",
        nargs="+",
        type=Path,
        help="Directories/files to scan for full_lineage.jsonl",
    )
    lg.add_argument(
        "--lineage",
        nargs="+",
        type=Path,
        help="Explicit lineage files (full_lineage.jsonl)",
    )

    fg = ap.add_mutually_exclusive_group(required=True)
    fg.add_argument(
        "--finals-roots",
        nargs="+",
        type=Path,
        help="Directories/files to scan for disambig jsonl files",
    )
    fg.add_argument(
        "--finals", nargs="+", type=Path, help="Explicit disambig jsonl files"
    )

    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")

    args = ap.parse_args()

    # Lineage
    lineage_paths = iter_lineage_files(
        args.lineage if args.lineage else args.lineage_roots
    )
    if not lineage_paths:
        raise SystemExit("No full_lineage.jsonl files found.")
    adj = build_graph_from_lineage(lineage_paths)

    # Finals
    finals_inputs = args.finals if args.finals else args.finals_roots
    disambig_files = []
    for p in finals_inputs:
        if Path(p).is_dir():
            disambig_files.extend(list(Path(p).rglob("*_disambig.jsonl")))
        else:
            disambig_files.append(Path(p))
    disambig_files = sorted(set(disambig_files))
    if not disambig_files:
        raise SystemExit("No disambig jsonl files found in finals inputs.")
    final_ids, id2pair = load_finals_multi(disambig_files)

    # Source
    docs = load_jsonl(args.source)

    # Count
    totals, breakdown, resolved_rows, unresolved = count_contributions(
        docs, adj, final_ids, id2pair
    )

    # Output
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_counts_csv(out_dir / "final_descriptor_counts.csv", totals)
    write_pair_breakdown_csv(out_dir / "final_descriptor_pair_counts.csv", breakdown)
    write_resolved_jsonl(out_dir / "resolved_pairs.jsonl", resolved_rows)
    write_unresolved_txt(out_dir / "unresolved_pairs.txt", unresolved)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
