import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Iterable
import re
import hashlib
import argparse


def split_pair(text: str) -> Tuple[str, str]:
    """Splits a string into a descriptor and explanation.
    The descriptor and explanation are separated by a semicolon.
    Normalizes the descriptor by replacing runs of underscores/spaces with a single space,
    trimming whitespace, and converting to lowercase.
    If malformed, returns a tuple of empty strings.
    """

    def normalize_descriptor(s: str) -> str:
        # replace runs of underscores/spaces with a single space, trim, lowercase
        return re.sub(r"[_\s]+", " ", (s or "")).strip().lower()

    try:
        d, e = text.split(";", 1)
        d = normalize_descriptor(d)
        return d, e.strip()
    except ValueError:
        return "", ""


def load_descriptors(file_path: Path, format: str) -> List[Tuple[str, str]]:
    descriptor_list: List[Tuple[str, str]] = []
    if format == "raw":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                doc = json.loads(line)
                best_idx = int(np.argmax(doc["similarity"]))
                descriptors: Iterable[str] = doc["descriptors"][best_idx]
                for desc_exp in descriptors:
                    d, e = split_pair(desc_exp)
                    if d and e:
                        descriptor_list.append((d, e))

        return descriptor_list
    
    elif format == "processed":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                d = row.get("descriptor", "").strip().lower()
                e = row.get("explainer", "").strip()
                if d and e:
                    descriptor_list.append((d, e))
        return descriptor_list
    
    else:
        raise ValueError(f"Unknown format: {format}. Supported formats are 'raw' and 'processed'.")


def group_descriptors(descriptor_list: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for d, e in descriptor_list:
        if d not in grouped:
            grouped[d] = []
        grouped[d].append(e)
    return grouped


def deduplicate_groups(grouped: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # Keep explainers unique per descriptor, stable-ish order (by appearance)
    deduped: Dict[str, List[str]] = {}
    for d, explainers in grouped.items():
        seen = set()
        ordered_unique: List[str] = []
        for e in explainers:
            if e not in seen:
                seen.add(e)
                ordered_unique.append(e)
        deduped[d] = ordered_unique
    return deduped


def pair_id(descriptor: str, explainer: str, *, length: int = 12) -> str:
    """Deterministic, stable ID for a descriptor–explainer pair.
    Uses SHA-1 over the normalized descriptor + raw explainer string.
    Truncated to `length` hex characters (default 12).
    """
    key = f"{descriptor}\u241f{explainer}"  # use an unlikely separator
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return digest[:length]


def attach_ids(deduped: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    """Convert {descriptor: [explainer,...]} into {descriptor: [{id, explainer}, ...]}"""
    with_ids: Dict[str, List[Dict[str, str]]] = {}
    for d, explainers in deduped.items():
        with_ids[d] = [{"id": pair_id(d, e), "explainer": e} for e in explainers]
    return with_ids


def write_stats(file_path: Path, deduped: Dict[str, List[str]]) -> None:
    total_descriptors = len(deduped)
    total_explainers = sum(len(explainers) for explainers in deduped.values())
    avg_explainers = total_explainers / total_descriptors if total_descriptors > 0 else 0.0
    singletons = sum(1 for explainers in deduped.values() if len(explainers) == 1)

    with file_path.open("w", encoding="utf-8") as stats_file:
        stats_file.write(f"Unique Descriptors: {total_descriptors}\n")
        stats_file.write(f"Average Explainers per Descriptor: {avg_explainers:.2f}\n")
        stats_file.write(f"Singleton Descriptors (just one explainer): {singletons}\n")
        stats_file.write(f"Total Unique Pairs: {total_explainers}\n")


def write_grouped_with_ids(file_path: Path, grouped_with_ids: Dict[str, List[Dict[str, str]]]) -> None:
    """Write one JSON line per descriptor with its explainer objects including IDs.
    Example line: {"descriptor": "foo", "pairs": [{"id": "abc123", "explainer": "..."}, ...]}
    """
    with file_path.open("w", encoding="utf-8") as out_file:
        for descriptor, pairs in grouped_with_ids.items():
            out_file.write(
                json.dumps({"descriptor": descriptor, "pairs": pairs}, ensure_ascii=False) + "\n"
            )


def write_flat_pairs(file_path: Path, grouped_with_ids: Dict[str, List[Dict[str, str]]]) -> None:
    """Optional: write a flat file with one line per unique pair for easy joins.
    Example line: {"id": "abc123", "descriptor": "foo", "explainer": "..."}
    """
    with file_path.open("w", encoding="utf-8") as out_file:
        for descriptor, pairs in grouped_with_ids.items():
            for p in pairs:
                out_file.write(
                    json.dumps(
                        {"id": p["id"], "descriptor": descriptor, "explainer": p["explainer"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract and group descriptor-explainer pairs with IDs.")
    p.add_argument("--input", type=Path, required=True,
                   help="Path to input JSONL file with descriptors.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory where results will be stored.")
    p.add_argument("--input-format", type=str, choices=["raw", "processed"], default="raw",
                   help="Format of the input file. Raw means unprocessed descriptors straight from generation step."
                   " Processed means descriptors have already been through one or more disambiguations steps.")
    args = p.parse_args()
    
    
    input_path = Path(args.input)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_out_path = Path(out_dir / "grouped_descriptors_with_ids.jsonl")
    flat_out_path = Path(out_dir / "descriptor_explainer_pairs.jsonl")
    stats_out_path = Path(out_dir / "descriptor_stats.txt")

    descriptors = load_descriptors(input_path, format=args.input_format)
    grouped = group_descriptors(descriptors)
    deduped = deduplicate_groups(grouped)

    # Write stats (now includes total unique pairs)
    write_stats(stats_out_path, deduped)

    # Attach IDs and write outputs
    grouped_with_ids = attach_ids(deduped)
    write_grouped_with_ids(grouped_out_path, grouped_with_ids)
    write_flat_pairs(flat_out_path, grouped_with_ids)

