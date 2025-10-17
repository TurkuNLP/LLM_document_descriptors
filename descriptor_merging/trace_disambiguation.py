#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, List, Iterable
import re
import hashlib
import glob
import logging

# ---------- light normalization & fingerprints ----------
_ws_re = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = _ws_re.sub(" ", s)
    return s

def fp_pair(descriptor: str, explainer: str) -> str:
    # fingerprint over normalized text
    d = _norm(descriptor); e = _norm(explainer)
    blob = json.dumps({"d": d, "e": e}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

# ---------- loaders ----------
def load_id_texts(id_jsonl: Path) -> Dict[str, Tuple[str,str]]:
    """id -> (descriptor, explainer) (normalized)"""
    id_to_text: Dict[str, Tuple[str,str]] = {}
    with id_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            _id = str(rec.get("id", "")).strip()
            if not _id: continue
            d = _norm(str(rec.get("descriptor", "") or ""))
            e = _norm(str(rec.get("explainer", "") or ""))
            id_to_text[_id] = (d, e)
    return id_to_text

def expand_files(specs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in specs:
        matches = glob.glob(s)
        if matches:
            out += [Path(m) for m in matches]
        else:
            out.append(Path(s))
    return out

def load_transform_logs_staged(
    stages: List[Tuple[str, List[Path]]]
) -> Tuple[Dict[str, Set[str]], Dict[str, Tuple[str,str]], Dict[str, Set[str]]]:
    """
    Build reverse rewrite graph across multiple labeled stages.

    Returns:
      child_to_parents: child_key -> set(parent_keys)   (self-loops dropped)
      key_to_text:      key -> (descriptor, explainer)  (normalized) for rendering
      source_stages:    key -> set(stage_labels)        (where key appeared as a *parent* anywhere)
    """
    child_to_parents: Dict[str, Set[str]] = {}
    key_to_text: Dict[str, Tuple[str,str]] = {}
    source_stages: Dict[str, Set[str]] = {}

    def remember(desc: str, expl: str) -> str:
        k = fp_pair(desc, expl)
        if k not in key_to_text:
            key_to_text[k] = (_norm(desc), _norm(expl))
        return k

    n_files = n_lines = n_edges = n_self = n_bad = 0

    for label, paths in stages:
        files = [p for p in paths if p.exists()]
        for path in files:
            n_files += 1
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    n_lines += 1
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        n_bad += 1
                        continue

                    od = str(rec.get("original_descriptor", "") or "")
                    nd = str(rec.get("new_descriptor", "") or "")
                    ne = str(rec.get("new_explainer", "") or "")

                    orig_exps = rec.get("original_explainers", [])
                    if isinstance(orig_exps, str):
                        orig_exps = [orig_exps]
                    elif not isinstance(orig_exps, list):
                        n_bad += 1
                        continue

                    child_key = remember(nd, ne)
                    parents = child_to_parents.setdefault(child_key, set())

                    for oe in orig_exps:
                        parent_key = remember(od, str(oe or ""))
                        if parent_key == child_key:
                            n_self += 1
                            continue
                        if parent_key not in parents:
                            parents.add(parent_key)
                            n_edges += 1
                        source_stages.setdefault(parent_key, set()).add(label)

    logging.info("Loaded %d files, %d lines; edges=%d, self_loops=%d, bad=%d, nodes=%d",
                 n_files, n_lines, n_edges, n_self, n_bad, len(key_to_text))
    return child_to_parents, key_to_text, source_stages

# ---------- cycle-safe, non-recursive closure ----------
def build_ancestor_closure(child_to_parents: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    # drop explicit self-loops again, just in case
    for c, ps in list(child_to_parents.items()):
        if c in ps:
            ps.discard(c)
        if not ps:
            child_to_parents[c] = set()

    memo: Dict[str, Set[str]] = {}
    visiting: Set[str] = set()

    all_nodes: Set[str] = set(child_to_parents.keys())
    for ps in child_to_parents.values():
        all_nodes.update(ps)

    for start in all_nodes:
        if start in memo:
            continue
        stack: List[Tuple[str,int]] = [(start, 0)]  # 0=enter,1=exit
        while stack:
            node, state = stack.pop()
            if state == 0:
                if node in memo: continue
                if node in visiting:
                    # back-edge: seed to itself so we terminate
                    if node not in memo:
                        memo[node] = {node}
                    continue
                visiting.add(node)
                stack.append((node, 1))
                ps = child_to_parents.get(node)
                if ps:
                    for p in ps:
                        if p not in memo:
                            stack.append((p, 0))
                else:
                    memo[node] = {node}
            else:
                ps = child_to_parents.get(node)
                if ps:
                    acc: Set[str] = set()
                    for p in ps:
                        if p not in memo:
                            memo[p] = {p}
                        acc |= memo[p]
                    memo[node] = acc
                visiting.discard(node)
    return memo

# ---------- cluster aggregation ----------
def aggregate_clusters(
    groups_jsonl: Path,
    id_to_text: Dict[str, Tuple[str,str]],
    id_lineage: Dict[str, List[Tuple[str,str, List[str]]]],
    out_path: Path
) -> None:
    with groups_jsonl.open("r", encoding="utf-8") as gf, out_path.open("w", encoding="utf-8") as out:
        for line in gf:
            if not line.strip(): continue
            g = json.loads(line)
            [(root, members)] = g.items()

            # union of ORIGINAL sources across members (include stage provenance)
            seen: Set[Tuple[str,str,Tuple[str,...]]] = set()
            for mid in members:
                for d, e, stages in id_lineage.get(mid, []):
                    seen.add((d, e, tuple(stages)))

            sources = [
                {"descriptor": d, "explainer": e, "stages": list(stages)}
                for (d, e, stages) in sorted(seen, key=lambda t: (t[0], t[1], t[2]))
            ]

            # include each member's final text
            finals = []
            for mid in members:
                de = id_to_text.get(mid)
                if de:
                    d, e = de
                    finals.append({"id": mid, "descriptor": d, "explainer": e})
                else:
                    finals.append({"id": mid, "descriptor": "", "explainer": ""})

            rec = {"root": root, "members": members, "finals": finals, "sources": sources}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Trace final IDs back to original descriptor/explainer texts across multiple staged runs; optional cluster aggregation."
    )
    ap.add_argument("--id-map", required=True,
                    help="merge_array_concat_merged_ids.jsonl (fields: id, descriptor, explainer).")
    ap.add_argument("--stage", action="append", nargs="+", metavar=("LABEL","FILE_OR_GLOB"),
                    help="Add a stage label followed by one or more files/globs of rewrite logs for that run. "
                         "Example: --stage split1 logs/s1_*.jsonl --stage split2 logs/s2_*.jsonl ...")
    ap.add_argument("--out", default="id_to_original_texts.jsonl",
                    help="Per-ID lineage JSONL (id -> final + sources with stages).")
    ap.add_argument("--groups-jsonl", default=None,
                    help="Optional groups JSONL (each line {\"ROOT_ID\":[\"member\",...]}); if set, also writes --out-clusters.")
    ap.add_argument("--out-clusters", default="cluster_lineage.jsonl",
                    help="Cluster lineage JSONL (root, members, finals, sources with stages).")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARN","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.stage:
        raise SystemExit("Provide at least one --stage LABEL FILES...")

    # Parse stages: each entry is [LABEL, FILE1, FILE2, ...]
    stages: List[Tuple[str, List[Path]]] = []
    for chunk in args.stage:
        if len(chunk) < 2:
            raise SystemExit("Each --stage must have a LABEL and at least one FILE/GLOB.")
        label, *specs = chunk
        paths = expand_files(specs)
        stages.append((label, paths))

    id_to_text = load_id_texts(Path(args.id_map))
    child_to_parents, key_to_text, source_stages = load_transform_logs_staged(stages)
    closure = build_ancestor_closure(child_to_parents)

    # Per-ID lineage (include stage provenance for each source key)
    id_lineage: Dict[str, List[Tuple[str,str, List[str]]]] = {}
    outp = Path(args.out)
    wrote = 0
    with outp.open("w", encoding="utf-8") as f:
        for _id, (d_final, e_final) in id_to_text.items():
            final_key = fp_pair(d_final, e_final)
            src_keys = closure.get(final_key, {final_key})
            sources = []
            for k in sorted(src_keys):
                d_src, e_src = key_to_text.get(k, ("", ""))
                stages = sorted(source_stages.get(k, []))
                sources.append({"descriptor": d_src, "explainer": e_src, "stages": stages})
            id_lineage[_id] = [(s["descriptor"], s["explainer"], s["stages"]) for s in sources]
            rec = {"id": _id, "final": {"descriptor": d_final, "explainer": e_final}, "sources": sources}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    print(f"Wrote {wrote} per-ID lineage rows -> {outp}")

    if args.groups_jsonl:
        aggregate_clusters(Path(args.groups_jsonl), id_to_text, id_lineage, Path(args.out_clusters))
        print(f"Wrote cluster lineage -> {args.out_clusters}")

if __name__ == "__main__":
    main()
