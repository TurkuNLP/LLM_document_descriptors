from __future__ import annotations

# -- standard library --
import argparse
from collections import defaultdict, Counter
import functools
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple, Optional, Set
import uuid

# -- third-party --
import json_repair  # type: ignore
from pydantic import BaseModel  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# -- local imports --
import disambiguation_prompt  # type: ignore
from input_processing import normalize_descriptor  # type: ignore

# ----------------------- Logging setup -----------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def configure_logging(log_file: Path) -> None:
    # reset handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    file_h = logging.FileHandler(log_file)
    file_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_h)
    logging.root.addHandler(stream_h)

    logging.info("=" * 40)
    logging.info("Logging configured to: %s", log_file)
    logging.info("SLURM Job ID: %s", os.environ.get("SLURM_JOB_ID", "N/A"))


def log_execution_time(func):
    """Decorator that logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(
            f"Execution of {func.__name__} took "
            f"{time.strftime('%H:%M:%S', time.gmtime(execution_time))}."
        )
        return result

    return wrapper


# ----------------------- I/O helpers -----------------------
def generate_uuid_id() -> str:
    """Random UUID4-based ID. This will ensure uniqueness across runs.
    However, IDs will not be stable across runs
    (identical descriptor-exlainer pairs will get different IDs each time).
    This is not an issue as long as we make sure we never lose the ID to pair mappings.
    """
    return str(uuid.uuid4())


def read_grouped_descriptors_file(
    path_to_file: Path, test_size: int = 0
) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Read groups from JSONL. Each line: {descriptor: str, explainers: [{id, explainer}, ...]}.
    Returns list of (descriptor, explainers[ {id, explainer}, ... ]).
    """
    groups: List[Tuple[str, List[Dict[str, str]]]] = []
    with path_to_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            desc = normalize_descriptor(obj.get("descriptor") or "")
            if not desc:
                raise ValueError("Missing descriptor in input data.")

            expls = []
            for p in obj["explainers"]:
                pid = p.get("id")
                txt = (p.get("explainer") or "").strip()
                # Raise errors if data is malformed.
                # We really should not have empty explainers or missing IDs here.
                if not txt:
                    raise ValueError(
                        f"Missing explainer text for descriptor={desc!r} in pairs."
                    )
                if not pid:
                    raise ValueError(
                        f"Missing ID for descriptor={desc!r}, explainer={txt!r} in pairs."
                    )
                expls.append({"id": pid, "explainer": txt})
            if expls:
                groups.append((desc, expls))

            if test_size > 0 and len(groups) >= test_size:
                break

    # Validate that each input ID is unique across the entire input set
    all_ids: Set[str] = set()
    for desc, expls in groups:
        for e in expls:
            pid = e["id"]
            if pid in all_ids:
                raise ValueError(f"Duplicate input ID found across groups: {pid!r}")
            all_ids.add(pid)

    return groups


def load_original_input_ids(path: Path, test: bool) -> Set[str]:
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            for e in obj.get("explainers", []):
                pid = e.get("id")
                if pid:
                    if pid in ids:
                        raise RuntimeError(f"Duplicate input ID in raw data: {pid}")
                    ids.add(pid)
            if test and i + 1 >= 10_000:
                break
    return ids


def save_results(save_path: Path, pairs: List[Dict[str, Any]]) -> None:
    """Write final singletons with stable IDs and lineage.
    Each line: {"id": <pair_id>, "descriptor": str, "explainer": str, "source_pair_ids": [..]}
    """
    logging.info("Saving final results to %s...", save_path)
    with save_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def save_checkpoint_full(
    checkpoint_dir: Path,
    step_idx: int,
    final_groups: List[Dict[str, Any]],
    pending_groups: List[Dict[str, Any]],
    lineage_events: List[Dict[str, Any]],
    processed_ids: Set[str] | None = None,
    final: bool = False,
) -> None:
    """Save a checkpoint with:
    - final singletons (descriptor/explainer *objects* with IDs)
    - pending multi groups (explainers as {id, explainer})
    - lineage events mapping source IDs to new pair IDs
    """
    final_path = checkpoint_dir / f"iter_{step_idx:08d}_final.jsonl"
    pending_path = checkpoint_dir / f"iter_{step_idx:08d}_pending.jsonl"
    if not final:
        lineage_path = checkpoint_dir / f"iter_{step_idx:08d}_lineage.jsonl"
    else:
        lineage_path = checkpoint_dir / "full_lineage.jsonl"

    if not final:
        # Skip saving these in the final saving pass
        logging.info(
            "[CHKPT %08d] Saving %d final groups -> %s",
            step_idx,
            len(final_groups),
            final_path.name,
        )
        with final_path.open("w", encoding="utf-8") as f:
            for g in final_groups:
                # singletons guaranteed to have one explainer object
                expl_obj = (
                    g["explainers"][0]
                    if g.get("explainers")
                    else {"id": None, "explainer": ""}
                )
                out = {
                    "id": expl_obj.get("id"),
                    "descriptor": g.get("descriptor"),
                    "explainer": expl_obj.get("explainer", ""),
                    "source_pair_ids": expl_obj.get("source_pair_ids", []),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        logging.info(
            "[CHKPT %08d] Saving %d pending groups -> %s",
            step_idx,
            len(pending_groups),
            pending_path.name,
        )
        with pending_path.open("w", encoding="utf-8") as f:
            for g in pending_groups:
                out = {
                    "descriptor": g.get("descriptor"),
                    "explainers": g.get("explainers", []),  # list of {id, explainer}
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        if processed_ids:
            processed_ids_path = (
                checkpoint_dir / f"iter_{step_idx:08d}_processed_ids.json"
            )
            with processed_ids_path.open("w") as f:
                json.dump(list(processed_ids), f)

    logging.info(
        "[CHKPT %08d] Saving %d lineage events -> %s",
        step_idx,
        len(lineage_events),
        lineage_path.name,
    )
    with lineage_path.open("w", encoding="utf-8") as f:
        for ev in lineage_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def find_latest_checkpoint(base_dir) -> Optional[Tuple[Path, int]]:
    checkpoint_dir = base_dir / "checkpoints"
    pattern = re.compile(r"^iter_(\d+)_final\.jsonl$")
    latest: Optional[Tuple[Path, int]] = None

    if not checkpoint_dir.exists():
        return None

    for p in checkpoint_dir.glob("iter_*_final.jsonl"):
        m = pattern.match(p.name)
        if not m:
            continue
        it = int(m.group(1))
        if latest is None or it > latest[1]:
            latest = (p, it)

    return latest


def load_final_groups(path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            desc = normalize_descriptor(obj.get("descriptor") or "")
            pid = obj.get("id")
            expl = (obj.get("explainer") or "").strip()
            if not desc:
                raise ValueError("Missing descriptor in final groups.")
            if not pid:
                raise ValueError(f"Missing ID for descriptor={desc!r} in final groups.")
            groups.append(
                {
                    "descriptor": desc,
                    "explainers": [
                        {
                            "id": pid,
                            "explainer": expl,
                            "source_pair_ids": obj.get("source_pair_ids", []),
                        }
                    ],
                }
            )
    return groups


def load_pending_groups(path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not path.exists():
        return groups
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            desc = normalize_descriptor(obj.get("descriptor") or "")
            expls = obj.get("explainers") or []
            fixed = []
            for e in expls:
                pid = e.get("id")
                exp = (e.get("explainer") or "").strip()
                spi = e.get("source_pair_ids", [])
                if not exp:
                    raise ValueError(
                        f"Missing explainer text for descriptor={desc!r} in pending groups."
                    )
                if not pid:
                    raise ValueError(
                        f"Missing ID for descriptor={desc!r}, explainer={exp!r} in pending groups."
                    )
                fixed.append({"id": pid, "explainer": exp, "source_pair_ids": spi})
            groups.append({"descriptor": desc, "explainers": fixed})
    return groups


def load_lineage_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            events.append(obj)
    return events


# -------------------Descriptor Merger-------------------------


class DescriptorMerger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cache_dir: str = os.path.expanduser(
            args.cache_dir or os.environ.get("HF_HOME", "~/.cache")
        )
        self.model_name: str = args.model
        self.temperature: float = args.temperature
        self.llm = None
        self.use_mock = args.mock_llm if hasattr(args, "mock_llm") else False

    @log_execution_time
    def llm_setup(self) -> LLM | None:
        if self.use_mock:
            logging.info("Using mock LLM for debugging (no model loaded)")
            return None

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPU available.")
        else:
            logging.info(f"Using {n_gpus} GPU(s).")
        self.llm = LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=16_384,
            tensor_parallel_size=n_gpus,
            enforce_eager=False,
            gpu_memory_utilization=0.9,
        )
        return self.llm

    def generate(self, prompts: List[str]) -> List[Dict]:
        if self.use_mock:
            return mock_llm_generator(prompts)

        params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            repetition_penalty=1.0,
            max_tokens=10_000,
            stop=["<|eot_id|>"],
            guided_decoding=self._response_format(),
            seed=42,
        )
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        elapsed = time.perf_counter() - start if start else 0.0

        response_texts: List[str] = []
        in_tok = 0
        gen_tok = 0
        for o in outputs:
            if getattr(o, "prompt_token_ids", None):
                in_tok += len(o.prompt_token_ids)

            outs = getattr(o, "outputs", None) or []
            if outs:
                cand = outs[0]
                if getattr(cand, "token_ids", None):
                    gen_tok += len(cand.token_ids)
                txt = getattr(cand, "text", "") or ""
                response_texts.append(txt.strip(" `\n").removeprefix("json"))
            else:
                response_texts.append("{}")

        tot_tok = gen_tok + in_tok
        if elapsed > 0 and tot_tok > 0:
            logging.info(
                "LLM throughput: %.1f tok/s (%.1f gen tok/s) — %s tokens in %.2fs",
                tot_tok / elapsed,
                gen_tok / elapsed if gen_tok else 0,
                tot_tok,
                elapsed,
            )

        parsed_responses = [self.validate_json(text) for text in response_texts]

        return parsed_responses

    @staticmethod
    def validate_completeness(
        outputs: List[Dict[str, Any]], batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate that each *input explainer ID* is present exactly once in the output groups.
        If missing, inject it into the closest group by lexical overlap of explainer text.
        """
        new_outputs = []
        for output, in_group in zip(outputs, batch):
            output = dict(output)
            output["original_descriptor"] = in_group["descriptor"]
            out_groups = output.get("groups")
            assert out_groups is not None, "Missing 'groups' in output."

            # Gather all IDs in output
            ids_in_output: List[str] = []
            for g in out_groups:
                ids_in_output.extend(g.get("original_explainer_ids_in_this_group"))

            # Input IDs -> text map
            id2text = {e["id"]: e["explainer"] for e in in_group["explainers"]}
            input_ids = list(id2text.keys())

            # Remove duplicate IDs across groups (keep first occurrence)
            seen = set()
            for g in out_groups:
                ids = g.get("original_explainer_ids_in_this_group", [])
                dedup = []
                for pid in ids:
                    if pid not in seen:
                        dedup.append(pid)
                        seen.add(pid)
                g["original_explainer_ids_in_this_group"] = dedup
            # Remove empty groups
            out_groups = [
                g for g in out_groups if g.get("original_explainer_ids_in_this_group")
            ]
            output["groups"] = out_groups

            # Inject missing IDs
            missing = [pid for pid in input_ids if pid not in seen]
            if missing:
                # crude lexical-overlap target selection
                for pid in missing:
                    missing_text = (id2text.get(pid)).lower()
                    missing_tokens = set(missing_text.split())
                    best_idx, best_ov = (0, -1)
                    for i, g in enumerate(out_groups or []):
                        gexp = (g.get("group_explainer") or "").lower()
                        ov = len(set(gexp.split()) & missing_tokens)
                        if ov > best_ov:
                            best_idx, best_ov = i, ov
                    if not out_groups:
                        out_groups = [
                            {
                                "group_descriptor": in_group["descriptor"],
                                "group_explainer": id2text.get(pid) or "",
                                "original_explainer_ids_in_this_group": [],
                            }
                        ]
                        output["groups"] = out_groups
                        best_idx = 0
                    out_groups[best_idx].setdefault(
                        "original_explainer_ids_in_this_group", []
                    ).append(pid)

            # Check that output is now complete
            assert (
                len(output.get("groups", [])) > 0
            ), "Output must have at least one group."
            assert all(
                len(g.get("original_explainer_ids_in_this_group", [])) > 0
                for g in output.get("groups", [])
            ), "All groups must have at least one explainer ID."

            # Gather all IDs in output again after duplicate removal and injection
            ids_in_output: List[str] = []
            for g in out_groups:
                ids_in_output.extend(g.get("original_explainer_ids_in_this_group"))
            # Each input ID must be present exactly once in output
            if sorted(input_ids) != sorted(ids_in_output):
                missing = set(input_ids) - set(ids_in_output)
                extra = set(ids_in_output) - set(input_ids)
                if missing and extra:
                    raise ValueError(
                        f"Output validation failed! Missing and extra explainer IDs in output. Missing: {missing}, Extra: {extra}"
                    )
                if missing:
                    raise ValueError(
                        f"Output validation failed! Missing explainer IDs in output: {missing}"
                    )
                if extra:
                    raise ValueError(
                        f"Output validation failed! Extra explainer IDs in output: {extra}"
                    )

            new_outputs.append(output)
        return new_outputs

    @staticmethod
    def validate_json(text: str) -> Dict[str, Any]:
        text = text.strip().strip("`").lstrip().removeprefix("json").lstrip()
        try:
            obj = json_repair.loads(text)
            return obj if isinstance(obj, dict) else {}
        except (json.JSONDecodeError, ValueError):
            logging.warning("JSON parse failed.")
            return {}

    @staticmethod
    def _response_format() -> GuidedDecodingParams:
        class G(BaseModel):  # type: ignore[misc]
            group_descriptor: str
            group_explainer: str
            original_explainer_indices_in_this_group: List[int]

        class R(BaseModel):  # type: ignore[misc]
            original_descriptor: str
            groups: List[G]

        return GuidedDecodingParams(json=R.model_json_schema())


# ------------------- Mock LLM for debugging -------------------------
def mock_llm_generator(prompts: List[str]) -> List[Dict]:
    """
    Simulates LLM output for debugging without loading a real model.
    Returns valid JSON responses with random but valid groupings.

    Args:
        prompts: List of prompt strings

    Returns:
        List of dictionaries mimicking LLM JSON output
    """
    import random
    import re

    responses = []

    for prompt in prompts:
        # Extract the descriptor from the prompt
        descriptor_match = re.search(
            r"Descriptor: (.+?)\nExplainers", prompt, re.DOTALL
        )
        descriptor = (
            descriptor_match.group(1).strip() if descriptor_match else "Unknown"
        )

        # Count how many explainer indices are in the prompt
        explainer_indices = re.findall(r"\[(\d+)\]", prompt)
        indices = [int(idx) for idx in explainer_indices]

        if not indices:
            # Fallback if we can't find indices
            indices = list(range(5))

        # Decide how many groups to create (between 1 and 3, or fewer if we have few indices)
        num_groups = min(max(1, len(indices) // 4), 3)

        # Randomly assign indices to groups
        random.shuffle(indices)
        groups_indices = [[] for _ in range(num_groups)]

        # Distribute indices among groups
        for i, idx in enumerate(indices):
            group_num = i % num_groups
            groups_indices[group_num].append(idx)

        # Generate a response with these groups
        groups = []
        for i, group_indices in enumerate(groups_indices):
            if not group_indices:  # Skip empty groups
                continue

            group = {
                "group_descriptor": f"Category {chr(65+i)}",  # A, B, C, etc.
                "group_explainer": f"This is an automatically generated explainer for group {chr(65+i)} of descriptor '{descriptor}'.",
                "original_explainer_indices_in_this_group": sorted(group_indices),
            }
            groups.append(group)

        # Create the complete response
        response = {"original_descriptor": descriptor, "groups": groups}

        responses.append(response)

    return responses


# ------------------- Helper functions -------------------------


def map_indices_to_ids(
    output: Dict[str, Any], in_group: Dict[str, Any]
) -> Dict[str, Any]:
    """Translate model-produced indices to trusted IDs for a single output object.
    Robust to empty/malformed JSON and out-of-range indices."""
    total = len(in_group["explainers"])
    idx2id = {i: e["id"] for i, e in enumerate(in_group["explainers"])}

    groups = output.get("groups")
    if not isinstance(groups, list) or not groups:
        # Auto-repair: one group containing all indices, with a safe explainer
        logging.warning(
            "LLM output missing/empty 'groups'; auto-repairing to a single all-indices group."
        )
        groups = [
            {
                "group_descriptor": in_group["descriptor"],
                "group_explainer": in_group["explainers"][0]["explainer"],
                "original_explainer_indices_in_this_group": list(range(total)),
            }
        ]

    for g in groups:
        idxs = g.get("original_explainer_indices_in_this_group", [])
        invalid = [i for i in idxs if i not in idx2id]
        if invalid:
            logging.error(
                "LLM returned invalid indices %s; they will be ignored.", invalid
            )
        ids = [idx2id[i] for i in idxs if i in idx2id]
        g["original_explainer_ids_in_this_group"] = ids
        g.pop("original_explainer_indices_in_this_group", None)

    output["groups"] = groups
    return output


def group_pairs_with_ids(
    pairs: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate a list of {descriptor, explainer, id, source_pair_ids} by descriptor."""
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        desc = p["descriptor"]
        d[desc].append(
            {
                "id": p["id"],
                "explainer": p["explainer"],
                "source_pair_ids": p.get("source_pair_ids", []),
            }
        )
    return d


def flatten_and_id_output(merged_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten model outputs into canonical pairs with lineage.
    Returns a list of dicts: {id, descriptor, explainer, source_pair_ids}.
    """
    out: List[Dict[str, Any]] = []
    for output in merged_groups:
        orig_desc = normalize_descriptor(output.get("original_descriptor"))
        if not orig_desc:
            raise ValueError("Missing original descriptor in merge output.")
        for g in output.get("groups"):
            gd = normalize_descriptor(g.get("group_descriptor"))
            ge = (g.get("group_explainer") or "").strip()
            if not gd or not ge:
                raise ValueError("Invalid group in merge events.")
            canonical_id = generate_uuid_id()
            source_ids = [
                sid for sid in (g.get("original_explainer_ids_in_this_group"))
            ]
            out.append(
                {
                    "id": canonical_id,
                    "descriptor": gd,
                    "explainer": ge,
                    "source_pair_ids": source_ids,
                }
            )
    return out


def make_prompts(batch: List[Dict[str, Any]]) -> List[str]:
    prompts: List[str] = []
    for group in batch:
        descriptor = group["descriptor"]
        explainers = group["explainers"]  # list of {id, explainer}
        prompt = disambiguation_prompt.merge_descriptors_prompt(descriptor, explainers)
        prompts.append(prompt)
    return prompts


def batch_and_process(
    pending_groups: List[Dict[str, Any]],
    llm_processor: callable,
    output_validator: callable,
    processed_ids: Set[str],
    max_explainers_per_prompt=20,
    max_prompts_per_call=512,
    max_chars_per_call=2_000_000,
):
    """
    Process all pending groups through the LLM with efficient batching.
    Results are aligned by index to pending_groups.
    """
    # One bucket per group, aligned by index
    results_by_group: List[List[Dict[str, Any]]] = [[] for _ in pending_groups]

    # STEP 1: Prepare all prompts with their metadata
    all_prompts = []
    for cohort_idx, group in enumerate(pending_groups):
        descriptor = group["descriptor"]

        # Filter out already-processed originals
        filtered_ids = [
            e["id"] for e in group["explainers"] if e["id"] in processed_ids
        ]
        if filtered_ids:
            logging.info(
                f"Filtered out {len(filtered_ids)} already processed IDs from cohort index {cohort_idx}"
            )

        remaining_explainers = [
            e for e in group["explainers"] if e["id"] not in processed_ids
        ]

        # Split large groups into chunks
        for i in range(0, len(remaining_explainers), max_explainers_per_prompt):
            chunk = remaining_explainers[i : i + max_explainers_per_prompt]
            prompt_text = disambiguation_prompt.merge_descriptors_prompt(
                descriptor, chunk
            )
            all_prompts.append(
                {
                    "prompt": prompt_text,
                    "chars": len(prompt_text),
                    "cohort_idx": cohort_idx,
                    "explainers": chunk,
                    "descriptor": descriptor,
                }
            )

    # STEP 2: Pack prompts into batches
    prompt_batches = []
    current_batch, current_chars = [], 0
    all_prompts.sort(key=lambda p: p["chars"])
    for item in all_prompts:
        if current_batch and (
            len(current_batch) + 1 > max_prompts_per_call
            or current_chars + item["chars"] > max_chars_per_call
        ):
            prompt_batches.append(current_batch)
            current_batch, current_chars = [], 0
        current_batch.append(item)
        current_chars += item["chars"]
    if current_batch:
        prompt_batches.append(current_batch)

    # STEP 3: Process each batch
    for batch_idx, batch in enumerate(prompt_batches):
        logging.info(
            f"Processing batch {batch_idx + 1}/{len(prompt_batches)} with {len(batch)} prompts"
        )
        prompt_texts = [it["prompt"] for it in batch]
        outputs = llm_processor(prompt_texts)

        for item, output in zip(batch, outputs):
            cohort_idx = item["cohort_idx"]

            processed_output = map_indices_to_ids(output, item)
            validated_output = output_validator([processed_output], [item])[0]

            results_by_group[cohort_idx].append(validated_output)

            # Verify completeness: inputs in chunk == IDs in validated output
            input_ids = {e["id"] for e in item["explainers"]}
            output_ids = set()
            for g in validated_output.get("groups", []):
                output_ids.update(g.get("original_explainer_ids_in_this_group", []))

            if input_ids != output_ids:
                missing = input_ids - output_ids
                extra = output_ids - input_ids
                logging.error(
                    f"Validation failed for cohort index {cohort_idx}: missing IDs: {missing}, extra IDs: {extra}"
                )
                raise ValueError("Batch validation failed")

    return results_by_group


def make_descriptor_explainer_groups(
    pairs: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    return [{"descriptor": desc, "explainers": expls} for desc, expls in pairs]


# ------------------- Trace helpers -------------------------


def build_merge_events(
    merged_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Produce structured merge events mapping source_pair_ids -> new_pair_id."""
    events: List[Dict[str, Any]] = []
    for group in merged_groups:
        id = group.get("id")
        desc = group.get("descriptor")
        expl = group.get("explainer")
        source_ids = group.get("source_pair_ids") or []
        if (
            not id
            or not desc
            or not expl
            or not isinstance(source_ids, list)
            or not source_ids
        ):
            raise ValueError("Invalid group in merge events.")
        events.append(
            {
                "new_descriptor": desc,
                "new_explainer": expl,
                "new_pair_id": id,
                "source_pair_ids": source_ids,
            }
        )
    return events


def audit_results_completeness(
    path_to_input: Path,
    path_to_output: Path,
    path_to_lineage: Path,
) -> bool:
    """Graph-aware audit:

    - Pass-through inputs (singletons) are allowed to appear 0 times in lineage and must be in final outputs.
    - Merged inputs must appear exactly once as sources in lineage.
    - Sinks (new IDs that never reappear as sources) must match final outputs minus pass-throughs.
    - No unknown IDs referenced; no duplicate new IDs; no stray outputs.
    """

    # --- load inputs ---
    input_ids: list[str] = []
    with path_to_input.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for e in obj.get("explainers") or []:
                pid = e.get("id")
                if not pid:
                    raise RuntimeError("Input contains an explainer without 'id'.")
                input_ids.append(pid)
    input_set = set(input_ids)

    # --- load outputs ---
    output_ids: list[str] = []
    with path_to_output.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("id")
            if not pid:
                raise RuntimeError("Output contains a pair without 'id'.")
            output_ids.append(pid)
    output_set = set(output_ids)

    # --- load lineage ---
    sources_multiset: list[str] = []
    new_ids: list[str] = []

    with path_to_lineage.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            nid = obj.get("new_pair_id")
            srcs = obj.get("source_pair_ids")
            if not nid or not isinstance(srcs, list) or not srcs:
                raise RuntimeError(f"Malformed lineage event: {obj!r}")
            new_ids.append(nid)
            sources_multiset.extend(srcs)

    new_set = set(new_ids)
    src_set = set(sources_multiset)

    # --- compute graph roles ---
    pass_through = input_set & output_set  # inputs that became outputs untouched
    merged_inputs = input_set - pass_through  # inputs that must be consumed by lineage
    leaves = src_set - new_set  # sources that are not produced by earlier merges
    sinks = new_set - src_set  # produced IDs that never feed into later merges
    expected_sinks = (
        output_set - pass_through
    )  # merges must account for all non-pass-through outputs

    problems: list[str] = []

    # 1) Leaves must match merged inputs (by identity)
    if leaves != merged_inputs:
        missing = list(merged_inputs - leaves)
        unknown = list(leaves - merged_inputs)
        problems.append(
            f"Leaf mismatch. Missing leaves={missing[:50]} (and {max(0,len(missing)-50)} more), "
            f"Unknown leaves={unknown[:50]} (and {max(0,len(unknown)-50)} more)"
        )

    # 2) Multiplicity: merged inputs appear exactly once as sources; no input appears >1 time
    src_counts = Counter(sources_multiset)
    wrong_mult = {
        pid: c for pid, c in src_counts.items() if pid in merged_inputs and c != 1
    }
    if wrong_mult:
        problems.append(
            f"Merged inputs must appear exactly once as sources. Offenders={list(wrong_mult.items())[:50]}"
        )
    overused_inputs = {
        pid: c for pid, c in src_counts.items() if pid in input_set and c > 1
    }
    if overused_inputs:
        problems.append(
            f"Input IDs used more than once as sources: {list(overused_inputs.items())[:50]}"
        )

    # 3) Sinks must match all non-pass-through outputs
    if sinks != expected_sinks:
        missing = list(expected_sinks - sinks)
        unknown = list(sinks - expected_sinks)
        problems.append(
            f"Sink mismatch. Missing sinks={missing[:50]} ; Unknown sinks={unknown[:50]}"
        )

    # 4) Every output is either a sink or a pass-through; nothing else
    stray_outputs = output_set - (sinks | pass_through)
    if stray_outputs:
        problems.append(
            f"Stray outputs not accounted for by lineage or pass-through: {list(stray_outputs)[:50]}"
        )

    # 5) No unknown sources (must be an input or a previously produced new id)
    unknown_sources = [
        sid for sid in src_set if sid not in input_set and sid not in new_set
    ]
    if unknown_sources:
        problems.append(
            f"Unknown source IDs referenced in lineage: {unknown_sources[:50]}"
        )

    # 6) No duplicate creation of new IDs
    dup_new = [nid for nid, c in Counter(new_ids).items() if c != 1]
    if dup_new:
        problems.append(
            f"Each new_pair_id must be created exactly once. Duplicates={dup_new[:50]}"
        )

    # 7) Ensure merges don’t reuse input IDs as new IDs (policy: new IDs are UUIDs)
    reused = list(new_set & input_set)
    if reused:
        problems.append(
            f"new_pair_id collided with an input ID (should be UUID-only): {reused[:50]}"
        )

    if problems:
        for p in problems:
            logging.error(p)
            logging.error("-" * 20)
            logging.error(
                "AUDIT FAILED. There is something fishy in the results. See errors above."
            )
            logging.error("-" * 20)
        return False
    else:
        logging.info(
            "Audit passed: pass-through + merged inputs fully accounted, sinks match outputs, no unknowns."
        )
        return True


def debug_duplicate_ids(cohort, results_by_group):
    """Real duplicates only; results_by_group is a list aligned to cohort."""
    in_locs = defaultdict(set)
    out_locs = defaultdict(set)

    for idx, group in enumerate(cohort):
        desc = group["descriptor"]

        for explainer in group["explainers"]:
            in_locs[explainer["id"]].add(f"Input cohort#{idx} ({desc})")

        for o_idx, output in enumerate(results_by_group[idx]):
            for g_idx, g in enumerate(output.get("groups", [])):
                for pid in g.get("original_explainer_ids_in_this_group", []):
                    out_locs[pid].add(f"Output cohort#{idx}-{o_idx}-{g_idx}")

    dup_inputs = {pid: sorted(locs) for pid, locs in in_locs.items() if len(locs) > 1}
    dup_outputs = {pid: sorted(locs) for pid, locs in out_locs.items() if len(locs) > 1}

    if dup_inputs or dup_outputs:
        logging.error(
            "Found real duplicates: %d in inputs, %d in outputs",
            len(dup_inputs),
            len(dup_outputs),
        )
        for i, (pid, locs) in enumerate(list(dup_inputs.items())[:10]):
            logging.error("Input-duplicate ID %d: %s", i + 1, pid)
            logging.error("  Appears in inputs: %s", locs)
        for i, (pid, locs) in enumerate(list(dup_outputs.items())[:10]):
            logging.error("Output-duplicate ID %d: %s", i + 1, pid)
            logging.error("  Appears in outputs: %s", locs)

    return {"input": dup_inputs, "output": dup_outputs}


# ------------------- Main -------------------------


def main(args):
    results_dir = Path(f"../results/disambiguate_merges/{args.run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(Path(results_dir / f"{args.run_id}.log"))
    print(f"Starting run {args.run_id}", flush=True)
    with open(results_dir / f"{args.run_id}_settings.txt", "w", encoding="utf-8") as f:
        slurm_id = os.environ.get("SLURM_JOB_ID", "N/A")
        f.write(f"SLURM_JOB_ID: {slurm_id}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    dm = DescriptorMerger(args)

    if not args.mock_llm:
        logging.info("Loading LLM...")
        dm.llm_setup()
        logging.info("LLM loaded.")
    else:
        logging.info("Using mock LLM; skipping model load.")

    # Prepare loop state -----------------------------------------------
    iter_start = time.time()

    final_groups: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    lineage_events: List[Dict[str, Any]] = []

    # Get a set of all input IDs
    input_ids = load_original_input_ids(Path(args.data_path), args.test)
    # Keep track of all processed IDs globally to avoid duplicates
    global_processed_ids = set()

    # Resume logic ------------------------------------------------------
    latest = find_latest_checkpoint(results_dir) if args.resume else None
    if latest:
        step_idx = latest[1]
        final_path = checkpoint_dir / f"iter_{step_idx:08d}_final.jsonl"
        pending_path = checkpoint_dir / f"iter_{step_idx:08d}_pending.jsonl"
        lineage_path = checkpoint_dir / f"iter_{step_idx:08d}_lineage.jsonl"
        final_groups = load_final_groups(final_path)
        pending = load_pending_groups(pending_path)
        lineage_events = load_lineage_events(lineage_path)

        processed_from_lineage = set()
        for ev in lineage_events:
            for sid in ev.get("source_pair_ids", []):
                if sid in input_ids:  # keep originals only
                    processed_from_lineage.add(sid)

        # Prefer lineage; if a processed_ids JSON exists, intersect it
        if (checkpoint_dir / f"iter_{step_idx:08d}_processed_ids.json").exists():
            with (checkpoint_dir / f"iter_{step_idx:08d}_processed_ids.json").open(
                "r"
            ) as f:
                previously_saved = set(json.load(f))
            previously_saved &= input_ids
            global_processed_ids = processed_from_lineage | previously_saved
        else:
            global_processed_ids = processed_from_lineage

        logging.info(
            "Resuming from checkpoint #%d: %d final, %d pending.",
            step_idx,
            len(final_groups),
            len(pending),
        )
        logging.info(
            "Resume: %d original input IDs already processed.",
            len(global_processed_ids),
        )

    else:
        descriptor_explainers = read_grouped_descriptors_file(
            Path(args.data_path), test_size=10_000 if args.test else 0
        )
        groups = make_descriptor_explainer_groups(descriptor_explainers)
        final_groups = [g for g in groups if len(g["explainers"]) == 1]
        pending = [g for g in groups if len(g["explainers"]) > 1]
        logging.info(
            "Fresh start: %d singles, %d multis.", len(final_groups), len(pending)
        )

    if not pending:
        logging.info("No multi-explainer groups remaining.")
        pairs = [
            {
                "id": g["explainers"][0]["id"],
                "descriptor": g["descriptor"],
                "explainer": g["explainers"][0].get("explainer", ""),
                "source_pair_ids": g["explainers"][0].get("source_pair_ids", []),
            }
            for g in final_groups
            if g.get("explainers")
        ]
        save_results(results_dir / f"{args.run_id}_disambig.jsonl", pairs)
        logging.info("Results saved.")
        return

    CHECKPOINT_EVERY = args.checkpoint_every

    pass_num = 0
    while pending:
        pass_num += 1
        start_time = time.time()

        # Take a cohort of groups to process this pass
        cohort = []
        if args.cohort_size > 0:
            for _ in range(args.cohort_size):
                if not pending:
                    break
                cohort.append(pending.pop())
        elif args.cohort_size == 0:
            # Cohort size 0 means process all pending in one pass.
            cohort = pending
            pending = []
        else:
            raise ValueError("Cohort size must be non-negative.")

        logging.info("=" * 40)
        logging.info("Starting pass %d on cohort of %d groups.", pass_num, len(cohort))

        # Process all groups in the cohort
        results_by_group = batch_and_process(
            pending_groups=cohort,
            llm_processor=dm.generate,
            output_validator=dm.validate_completeness,
            processed_ids=global_processed_ids,
            max_explainers_per_prompt=20,
            max_prompts_per_call=args.max_prompts_per_call,
            max_chars_per_call=args.max_chars_per_call,
        )

        # Debugging: check for duplicate IDs in this cohort
        debug_duplicate_ids(cohort, results_by_group)

        # Validate no duplicate processing of IDs across the cohort
        cohort_processed_ids = set()
        for i, group in enumerate(cohort):
            # Extract IDs we've processed for this group
            group_processed_ids = set()
            for output in results_by_group[i]:
                for g in output.get("groups", []):
                    group_processed_ids.update(
                        g.get("original_explainer_ids_in_this_group", [])
                    )

            orig_ids_this_group = {
                pid for pid in group_processed_ids if pid in input_ids
            }
            # Ensure no ID duplication across groups
            if orig_ids_this_group & cohort_processed_ids:
                duplicate_ids = orig_ids_this_group & cohort_processed_ids
                raise ValueError(f"Found duplicate IDs across groups: {duplicate_ids}")
            cohort_processed_ids.update(orig_ids_this_group)
            # Also update global processed IDs
            global_processed_ids.update(orig_ids_this_group)

        next_round_multis = []
        for i, group in enumerate(cohort):
            # Groups that went through LLM
            merged_groups = results_by_group[i]
            if not merged_groups:
                logging.info(
                    "Skipping cohort index %d: no outputs (all explainers for this group were already processed).",
                    i,
                )
                continue

            # Flatten -> canonical pairs with lineage
            # flat_pairs: [{id, descriptor, explainer, source_pair_ids}, ...]
            flat_pairs = flatten_and_id_output(merged_groups)

            # Sanity check and dedup within this parent group
            seen = set()
            dedup = []
            for p in flat_pairs:
                key = tuple(sorted(p["source_pair_ids"]))
                if key in seen:
                    logging.warning(
                        "Dedup: duplicate source set within one parent; dropping"
                    )
                    continue
                seen.add(key)
                dedup.append(p)
            flat_pairs = dedup

            assert flat_pairs, "No pairs produced from merge."
            # Log lineage events
            lineage_events.extend(build_merge_events(flat_pairs))

            # Re-group by descriptor for the next pass
            aggregated = group_pairs_with_ids(
                flat_pairs
            )  # {descriptor: [{id, expl, sources},...], ...}
            aggregated_pairs = list(
                aggregated.items()
            )  # [(desc, [{id, expl, sources},...]), ...]
            sub_groups = [
                {"descriptor": desc, "explainers": expls}
                for (desc, expls) in aggregated_pairs
            ]

            # Partition and collect
            new_multis = [g for g in sub_groups if len(g["explainers"]) > 1]
            new_singles = [g for g in sub_groups if len(g["explainers"]) == 1]
            final_groups.extend(new_singles)
            next_round_multis.extend(new_multis)

        pending.extend(next_round_multis)

        if CHECKPOINT_EVERY and (pass_num % CHECKPOINT_EVERY == 0) and pass_num > 0:
            step_idx = pass_num
            save_checkpoint_full(
                checkpoint_dir,
                step_idx,
                final_groups,
                pending,
                lineage_events,
                global_processed_ids,
            )

        end_time = time.time()
        pass_time = end_time - start_time
        logging.info(
            f"Pass with {len(cohort)} groups done in "
            f"{time.strftime('%H:%M:%S', time.gmtime(pass_time))}. "
            f"{len(pending)} pending remain."
        )

    # --------------------------- Final save and sanity checks -----------------------------------
    # Sanity check: all final groups must be singletons
    bad = [g for g in final_groups if len(g.get("explainers", [])) != 1]
    if bad:
        raise ValueError(
            f"Expected singleton groups only at final save; found {len(bad)} non-singletons."
        )

    pairs = []
    for g in final_groups:
        ex = g["explainers"][0]
        pairs.append(
            {
                "id": ex["id"],
                "descriptor": g["descriptor"],
                "explainer": ex.get("explainer", ""),
                "source_pair_ids": ex.get("source_pair_ids", []),
            }
        )
    out_path = results_dir / f"{args.run_id}_disambig.jsonl"
    save_results(out_path, pairs)
    # Save final checkpoint in results_dir
    save_checkpoint_full(
        results_dir, pass_num, final_groups, pending, lineage_events, final=True
    )

    # End-to-end audit
    audit_passed = audit_results_completeness(
        Path(args.data_path),
        out_path,
        results_dir / "full_lineage.jsonl",
    )
    if not audit_passed:
        # If audit failed, create a marker file so its hard to miss
        with open(results_dir / "AUDIT_FAILED.txt", "w", encoding="utf-8") as f:
            f.write("Audit FAILED. See log for details.\n")

    exec_time = time.time() - iter_start
    logging.info(
        f"Disambiguation done in {time.strftime('%H:%M:%S', time.gmtime(exec_time))}. "
        f"Saved {len(pairs)} final pairs."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to JSONL file with objects {descriptor, pairs:[{id, explainer}]} or legacy {descriptor, explainers}.",
    )

    # Batching parameters
    p.add_argument(
        "--cohort-size",
        dest="cohort_size",
        type=int,
        default=100,
        help="How many groups to process together at once. Set to 0 to process all in one pass (might cause OOMs).",
    )
    p.add_argument(
        "--max-prompts-per-call",
        dest="max_prompts_per_call",
        type=int,
        default=512,
        help="Cap on prompts per vLLM.generate call.",
    )
    p.add_argument(
        "--max-chars-per-call",
        dest="max_chars_per_call",
        type=int,
        default=2_000_000,
        help="Cap on total characters across all prompts in one call.",
    )

    # LLM parameters
    p.add_argument(
        "--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Model name"
    )
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", "~/.cache"),
        help="Cache directory for model files",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature. Default 0.1. Between 0 and 1.",
    )
    p.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use a mock LLM for debugging (no model loading)",
    )

    # Control parameters
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in run directory.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save a full checkpoint after every N passes(cohorts). 0 disables.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 10,000 descriptors for quick test runs.",
    )
    args = p.parse_args()

    main(args)

    logging.info("Done.")
