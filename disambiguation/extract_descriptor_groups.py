import json
from pathlib import Path
import numpy as np # type: ignore
from typing import List, Dict, Tuple, Iterable
import argparse
import uuid

# Local imports
from input_processing import normalize_descriptor, split_pair, generate_stable_id
import split_large_files


def load_descriptors(file_path: Path) -> Tuple[List[Tuple[str, str]], List[str], str]:
    descriptor_list: List[Dict[str, str]] = []
    malformed_entries: List[str] = []
    format = ""
    # First, load one line to get the format
    # Raw means descriptors right after initial generation
    # Processed means the data has already gone through one or more rounds of disambgiguation.
    # Or at the very least, has been normalized, ridden of malformed entries, and assigned IDs.
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            break  # only need the first line
        if "descriptors" in sample and "similarity" in sample:
            format = "raw"
        elif "descriptor" in sample and "explainer" in sample:
            format = "processed"
        else:
            raise ValueError("Invalid data. Cannot determine format.")

    if format == "raw":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                doc = json.loads(line)
                best_idx = int(np.argmax(doc["similarity"]))
                descriptors: Iterable[str] = doc["descriptors"][best_idx]
                for desc_exp in descriptors:
                    # Split pair by ";"
                    d, e = split_pair(desc_exp)
                    # Malformed entries yield empty strings and are silently dropped
                    # This is fine at this stage: we want to get rid of bad entries.
                    # After this, all entries should be well-formed and no silent drops should occur
                    # Still, let's log malformed entries for future examination
                    if d and e:
                        # Normalize descriptor and strip explainer
                        d = normalize_descriptor(d)
                        e = e.strip()
                        # Generate deterministic ID
                        # On first round, we use deterministic IDs to ensure stability across runs
                        # We don't mind duplicate IDs at this stage, as deduplication will happen later
                        id = generate_stable_id(d, e)
                        descriptor_list.append(
                            {"descriptor": d, "explainer": e, "id": id}
                        )
                    else:
                        malformed_entries.append(desc_exp)
        if malformed_entries:
            print(
                f"Warning: {len(malformed_entries)} malformed entries were found and skipped."
            )

    elif format == "processed":
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                d = row.get("descriptor", "")
                e = row.get("explainer", "")
                id = row.get("id", "")
                # Raise if malformed entry with no descriptor, explainer, or id.
                # There really should be no such entries at this point.
                # They indicate a bug upstream.
                if not d or not e or not id:
                    raise ValueError(
                        "Malformed entry (descriptor/explainer/id missing)."
                    )
                else:
                    d = normalize_descriptor(d)
                    e = e.strip()
                    descriptor_list.append({"descriptor": d, "explainer": e, "id": id})
    else:
        raise ValueError(
            f"Unknown format: {format}. Supported formats are 'raw' and 'processed'."
        )

    return descriptor_list, malformed_entries, format


def make_groups(
    descriptor_list: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    # Group explainers by descriptor

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for obj in descriptor_list:
        d = obj["descriptor"]
        e = obj["explainer"]
        id = obj["id"]
        if d not in grouped:
            grouped[d] = []

        grouped[d].append({"explainer": e, "id": id})

    return grouped


def deduplicate_groups(
    grouped: Dict[str, List[Dict[str, str]]],
) -> Dict[str, List[Dict[str, str]]]:
    # Keep explainers unique per descriptor, stable-ish order (by appearance)
    deduped: Dict[str, List[Dict[str, str]]] = {}
    for d, explainers in grouped.items():
        # explainers is a list of dicts with "explainer" and "id"
        seen = set()
        ordered_unique: List[str] = []
        for e in explainers:
            exp = e["explainer"]
            id = e["id"]
            if exp not in seen:
                seen.add(exp)
                ordered_unique.append({"explainer": exp, "id": id})
        deduped[d] = ordered_unique
    return deduped


def generate_uuid_id() -> str:
    """Random UUID4-based ID. This will ensure uniqueness across runs.
    However, IDs will not be stable across runs
    (identical descriptor-exlainer pairs will get different IDs each time).
    This is not an issue as long as we make sure we never lose the ID to pair mappings.
    """
    return str(uuid.uuid4())


def ensure_unique_ids(grouped: Dict[str, List[Dict[str, str]]]) -> None:
    """Ensure all descriptor-explainer pairs have unique IDs.
    If duplicate IDs are found, raise ValueError because it indicates but upstream.
    We cannot simply generate new IDs here, because that would break the lineage mappings.
    """
    seen_ids = set()
    for d, explainers in grouped.items():
        for e in explainers:
            id = e["id"]
            if id in seen_ids:
                raise ValueError(
                    f"Duplicate ID found: {id}. IDs should be unique across all pairs."
                )
            else:
                seen_ids.add(id)
    print(f"All {len(seen_ids)} descriptor-explainer pairs have unique IDs.")


def log_stats(grouped: Dict[str, List[Dict[str, str]]]) -> None:
    """Log some statistics about the grouped descriptors."""
    num_descriptors = len(grouped)
    num_explainers = sum(len(expls) for expls in grouped.values())
    singletons = sum(1 for expls in grouped.values() if len(expls) == 1)
    multis = sum(1 for expls in grouped.values() if len(expls) > 1)
    avg_explainers_per_descriptor = num_explainers / num_descriptors
    largest_groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"Total unique descriptors: {num_descriptors}")
    print(f"Total descriptor-explainer pairs: {num_explainers}")
    print(f"Descriptors with single explainer: {singletons}")
    print(f"Descriptors with multiple explainers: {multis}")
    print(f"Average explainers per descriptor: {avg_explainers_per_descriptor:.3f}")
    print("Top 5 largest descriptor groups:")
    for d, expls in largest_groups:
        print(f"  Descriptor: '{d}' - {len(expls)} explainers")


def write_descriptors_with_ids(
    descriptor_list: List[Dict[str, str]], file_path: Path
) -> None:
    """Write one JSON line per descriptor with its explainer and ID."""
    with file_path.open("w", encoding="utf-8") as out_file:
        for obj in descriptor_list:
            out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_grouped_with_ids(
    file_path: Path, grouped_with_ids: Dict[str, List[Dict[str, str]]]
) -> None:
    """Write one JSON line per descriptor with its explainer objects including IDs.
    Example line: {"descriptor": "foo", "explainers": [{"id": "abc123", "explainer": "..."}, ...]}
    """
    with file_path.open("w", encoding="utf-8") as out_file:
        for descriptor, expl in grouped_with_ids.items():
            out_file.write(
                json.dumps(
                    {"descriptor": descriptor, "explainers": expl}, ensure_ascii=False
                )
                + "\n"
            )


def write_flat_pairs(
    file_path: Path, grouped_with_ids: Dict[str, List[Dict[str, str]]]
) -> None:
    """Optional: write a flat file with one line per unique pair for easy joins.
    Example line: {"id": "abc123", "descriptor": "foo", "explainer": "..."}
    """
    with file_path.open("w", encoding="utf-8") as out_file:
        for descriptor, pairs in grouped_with_ids.items():
            for p in pairs:
                out_file.write(
                    json.dumps(
                        {
                            "id": p["id"],
                            "descriptor": descriptor,
                            "explainer": p["explainer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def write_malformed_entries(file_path: Path, malformed_entries: List[str]) -> None:
    """Write malformed entries to a separate file for examination."""
    with file_path.open("w", encoding="utf-8") as out_file:
        for entry in malformed_entries:
            out_file.write(entry + "\n")


def group_descriptors(input_path: Path, out_dir: Path, num_splits: int) -> None:
    # Main function to group descriptors and write outputs

    out_dir.mkdir(parents=True, exist_ok=True)
    descriptors_with_ids_path = Path(out_dir / "descriptors_with_ids.jsonl")
    grouped_out_path = Path(out_dir / "grouped_descriptors_with_ids.jsonl")
    flat_out_path = Path(out_dir / "descriptor_explainer_pairs.jsonl")
    malformed_out_path = Path(out_dir / "malformed_entries.txt")

    descriptors, malformed, format = load_descriptors(input_path)
    grouped = make_groups(descriptors)
    if format == "raw":
        # IDs are deterministic, we can deduplicate safely
        final_groups = deduplicate_groups(grouped)
    else:
        # We do not want ot deduplicate anymore with processed data
        # That would lead to data loss
        final_groups = grouped

    # Ensure unique IDs across all pairs
    # If input data waw "raw", there should be no duplicates, because of deduplication above
    # If input data was "processed", duplicates also should not exist because of UUID
    # This is just a sanity check
    ensure_unique_ids(final_groups)
    log_stats(final_groups)

    write_descriptors_with_ids(descriptors, descriptors_with_ids_path)
    write_grouped_with_ids(grouped_out_path, final_groups)
    write_flat_pairs(flat_out_path, final_groups)
    if malformed:
        write_malformed_entries(malformed_out_path, malformed)

    if not isinstance(num_splits, int):
        print(f"Invalid num_splits value: {num_splits}. Must be an integer.")
        print("Skipping splitting step.")
    else:
        if num_splits > 1:
            print(f"Splitting grouped output into {num_splits} files...")
            # Split the grouped output into smaller files
            splits_out_dir = out_dir / "splits"
            splits_out_dir.mkdir(parents=True, exist_ok=True)
            split_large_files.split_jsonl_files(
                input=grouped_out_path,
                output_dir=splits_out_dir,
                split_count=num_splits,
                shuffle=True,
                seed=42,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract and group descriptor-explainer pairs with IDs."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file with descriptors.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where results will be stored.",
    )
    p.add_argument(
        "--num-splits",
        type=int,
        default=1,
        help="Number of splits to create for the output files.",
    )
    args = p.parse_args()

    group_descriptors(args.input, args.out_dir, args.num_splits)
