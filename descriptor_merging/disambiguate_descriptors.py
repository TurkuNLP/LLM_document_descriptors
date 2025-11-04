from __future__ import annotations

# -- standard library --
import argparse
from collections import defaultdict
import functools
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Iterator, Tuple, Optional
import itertools
import hashlib

# -- third-party --
import json_repair  # type: ignore
import numpy as np  # type: ignore
from pydantic import BaseModel  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# -- local imports --
import LLM_document_descriptors.descriptor_merging.disambiguation_prompt as disambiguation_prompt  # type: ignore

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
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"))

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"))

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


def _normalize_descriptor(s: str) -> str:
    return re.sub(r'[_\s]+', ' ', (s or '')).strip().lower()


def _pair_id(descriptor: str, explainer: str, *, length: int = 12) -> str:
    """Deterministic ID for a descriptor–explainer pair.
    Must match the extractor's scheme so IDs are stable across the pipeline.
    """
    key = f"{_normalize_descriptor(descriptor)}\u241f{explainer}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]


def read_grouped_descriptors_file(path_to_file: Path, test_size: int = 0) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Read groups from JSONL.
    Supports *new* format with IDs:
      {"descriptor": str, "pairs": [{"id": str, "explainer": str}, ...]}
    and *legacy* format:
      {"descriptor": str, "explainers": [str, ...]}
    Returns list of (descriptor, explainers[ {id, explainer}, ... ]).
    """
    groups: List[Tuple[str, List[Dict[str, str]]]] = []
    with path_to_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            desc = _normalize_descriptor(obj.get("descriptor") or "")
            if not desc:
                continue

            if "pairs" in obj and isinstance(obj["pairs"], list):
                expls = []
                for p in obj["pairs"]:
                    pid = p.get("id")
                    txt = (p.get("explainer") or "").strip()
                    if not txt:
                        continue
                    if not pid:
                        pid = _pair_id(desc, txt)
                    expls.append({"id": pid, "explainer": txt})
            else:
                # legacy
                items = obj.get("explainers") or []
                expls = [{"id": _pair_id(desc, str(t)), "explainer": str(t)} for t in items if str(t).strip()]

            if expls:
                groups.append((desc, expls))

            if test_size > 0 and len(groups) >= test_size:
                break

    return groups


def save_results(save_path: Path, pairs: List[Dict[str, Any]]) -> None:
    """Write final singletons with stable IDs and lineage.
    Each line: {"id": <pair_id>, "descriptor": str, "explainer": str, "source_pair_ids": [..]}
    """
    logging.info("Saving final results to %s...", save_path)
    with save_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def save_checkpoint_full(checkpoint_dir: Path,
                         step_idx: int,
                         final_groups: List[Dict[str, Any]],
                         pending_groups: List[Dict[str, Any]],
                         lineage_events: List[Dict[str, Any]],
                         final: bool = False) -> None:
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
        logging.info("[CHKPT %08d] Saving %d final groups -> %s",
                    step_idx, len(final_groups), final_path.name)
        with final_path.open("w", encoding="utf-8") as f:
            for g in final_groups:
                # singletons guaranteed to have one explainer object
                expl_obj = g["explainers"][0] if g.get("explainers") else {"id": None, "explainer": ""}
                out = {
                    "gid": g.get("gid"),
                    "id": expl_obj.get("id"),
                    "descriptor": g.get("descriptor"),
                    "explainer": expl_obj.get("explainer", ""),
                    "source_pair_ids": expl_obj.get("source_pair_ids", []),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        logging.info("[CHKPT %08d] Saving %d pending groups -> %s",
                    step_idx, len(pending_groups), pending_path.name)
        with pending_path.open("w", encoding="utf-8") as f:
            for g in pending_groups:
                out = {
                    "gid": g.get("gid"),
                    "descriptor": g.get("descriptor"),
                    "explainers": g.get("explainers", []),  # list of {id, explainer}
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logging.info("[CHKPT %08d] Saving %d lineage events -> %s",
                 step_idx, len(lineage_events), lineage_path.name)
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
    if not path.exists():
        return groups
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            desc = _normalize_descriptor(obj.get("descriptor") or "")
            pid = obj.get("id")
            expl = (obj.get("explainer") or "").strip()
            if not desc:
                continue
            if not pid:
                pid = _pair_id(desc, expl)
            groups.append({
                "gid": obj.get("gid"),
                "descriptor": desc,
                "explainers": [{"id": pid, "explainer": expl, "source_pair_ids": obj.get("source_pair_ids", [])}],
            })
    return groups


def load_pending_groups(path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not path.exists():
        return groups
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            desc = _normalize_descriptor(obj.get("descriptor") or "")
            expls = obj.get("explainers") or []
            fixed = []
            for e in expls:
                if isinstance(e, dict):
                    pid = e.get("id") or _pair_id(desc, e.get("explainer") or "")
                    fixed.append({"id": pid, "explainer": (e.get("explainer") or "").strip(), "source_pair_ids": e.get("source_pair_ids", [])})
                else:
                    txt = str(e)
                    fixed.append({"id": _pair_id(desc, txt), "explainer": txt, "source_pair_ids": []})
            if desc and fixed:
                groups.append({"gid": obj.get("gid"), "descriptor": desc, "explainers": fixed})
    return groups


def load_lineage_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            events.append(obj)
    return events


# -------------------Descriptor Merger-------------------------

class DescriptorMerger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cache_dir: str = os.path.expanduser(
            args.cache_dir or os.environ.get("HF_HOME", "~/.cache"))
        self.model_name: str = args.model
        self.temperature: float = args.temperature
        self.llm = None

    @log_execution_time
    def llm_setup(self) -> LLM:
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
        outputs = self.llm.generate(
            prompts, sampling_params=params, use_tqdm=False)
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

        parsed_responses = [self.validate_json(
            text) for text in response_texts]

        return parsed_responses

    @staticmethod
    def validate_completeness(outputs: List[Dict[str, Any]], batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that each *input explainer ID* is present exactly once in the output groups.
        If missing, inject it into the closest group by lexical overlap of explainer text.
        """
        new_outputs = []
        for output, in_group in zip(outputs, batch):
            output = dict(output)
            output["original_descriptor"] = in_group["descriptor"]
            out_groups = output.get("groups") or []

            # Gather all IDs in output
            ids_in_output: List[str] = []
            for g in out_groups:
                ids_in_output.extend(g.get("original_explainer_ids_in_this_group", []) )

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
            out_groups = [g for g in out_groups if g.get("original_explainer_ids_in_this_group")]
            output["groups"] = out_groups

            # Inject missing IDs
            missing = [pid for pid in input_ids if pid not in seen]
            if missing:
                # crude lexical-overlap target selection
                for pid in missing:
                    missing_text = (id2text.get(pid) or "").lower()
                    missing_tokens = set(missing_text.split())
                    best_idx, best_ov = (0, -1)
                    for i, g in enumerate(out_groups or []):
                        gexp = (g.get("group_explainer") or "").lower()
                        ov = len(set(gexp.split()) & missing_tokens)
                        if ov > best_ov:
                            best_idx, best_ov = i, ov
                    if not out_groups:
                        out_groups = [{
                            "group_descriptor": in_group["descriptor"],
                            "group_explainer": id2text.get(pid) or "",
                            "original_explainer_ids_in_this_group": [],
                        }]
                        output["groups"] = out_groups
                        best_idx = 0
                    out_groups[best_idx].setdefault("original_explainer_ids_in_this_group", []).append(pid)
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


# ------------------- Helper functions -------------------------


def map_indices_to_ids(output: Dict[str, Any], in_group: Dict[str, Any]) -> Dict[str, Any]:
    """Translate model-produced indices to trusted IDs in-place for a single output object."""
    out = dict(output or {})
    groups = out.get("groups") or []
    idx2id = {i: e["id"] for i, e in enumerate(in_group["explainers"])}
    for g in groups:
        idxs = g.get("original_explainer_indices_in_this_group") or []
        ids = [idx2id[i] for i in idxs if i in idx2id]
        g["original_explainer_ids_in_this_group"] = ids
        g.pop("original_explainer_indices_in_this_group", None)
    out["groups"] = groups
    return out


def group_pairs_with_ids(pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate a list of {descriptor, explainer, id, source_pair_ids} by descriptor.
    For the *next* pass, the explainer set is the canonical group explainers we just produced.
    """
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        desc = p["descriptor"]
        if desc and p.get("explainer"):
            d[desc].append({
                "id": p["id"],
                "explainer": p["explainer"],
                "source_pair_ids": p.get("source_pair_ids", []),
            })
        else:
            logging.warning("Dropping empty pair for descriptor=%r", desc)
    return d


def parse_output(merged_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten model outputs into canonical pairs with lineage.
    Returns a list of dicts: {id, descriptor, explainer, source_pair_ids}
    where `id` is the deterministic hash of (descriptor, group_explainer).
    """
    out: List[Dict[str, Any]] = []
    for output in merged_groups:
        orig_desc = _normalize_descriptor(output.get("original_descriptor") or "")
        for g in (output.get("groups") or []):
            gd = _normalize_descriptor(g.get("group_descriptor") or orig_desc)
            ge = (g.get("group_explainer") or "").strip()
            if not gd or not ge:
                continue
            canonical_id = _pair_id(gd, ge)
            source_ids = [sid for sid in (g.get("original_explainer_ids_in_this_group") or []) if sid]
            out.append({
                "id": canonical_id,
                "descriptor": gd,
                "explainer": ge,
                "source_pair_ids": source_ids,
            })
    return out


def make_prompts(batch: List[Dict[str, Any]]) -> List[str]:
    prompts: List[str] = []
    for group in batch:
        descriptor = group["descriptor"]
        explainers = group["explainers"]      # list of {id, explainer}
        prompt = disambiguation_prompt.merge_descriptors_prompt(
            descriptor, explainers)
        prompts.append(prompt)
    return prompts


def make_batches(
    group: Dict[str, Any],
    max_chars: int,
    max_items: int = 20,
) -> Iterator[List[Dict[str, List[Dict[str, str]]]]]:
    """Yield batches without exceeding max_chars. Keeps explainer objects with IDs."""
    def split(desc: str, expls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        return [
            {"descriptor": desc,
             "explainers": expls[i:i+max_items]} for i in range(0, len(expls), max_items)
        ]

    normalized = split(group["descriptor"], group["explainers"]) if len(
        group["explainers"]) > max_items else [group]

    batch: List[Dict[str, Any]] = []
    cur_chars = 0

    for g in normalized:
        prompt = disambiguation_prompt.merge_descriptors_prompt(
            g["descriptor"], g["explainers"])
        plen = len(prompt)
        if batch and (cur_chars + plen > max_chars):
            yield batch
            batch, cur_chars = [], 0
        batch.append(g)
        cur_chars += plen

    if batch:
        yield batch


def coalesce_prompts(
    groups_for_pass,
    chars_per_batch: int,
    max_prompts_per_call: int,
    max_chars_per_call: int,
):
    per_group_subbatches = {}
    for g in groups_for_pass:
        subbatches = list(make_batches(g, max_chars=chars_per_batch))
        per_group_subbatches[g["gid"]] = subbatches

    flat = []  # entries: (gid, local_batch_idx, subbatch_item)
    for g in groups_for_pass:
        gid = g["gid"]
        for i, subb in enumerate(per_group_subbatches[gid]):
            for item in subb:  # {descriptor, explainers:[{id, explainer}]}
                flat.append((gid, i, item))

    cur_prompts, cur_map = [], []
    cur_chars = 0
    for gid, i, item in flat:
        prompt = disambiguation_prompt.merge_descriptors_prompt(
            item["descriptor"], item["explainers"]
        )
        plen = len(prompt)
        would_exceed_prompts = (len(cur_prompts) + 1) > max_prompts_per_call
        would_exceed_chars = (cur_chars + plen) > max_chars_per_call
        if cur_prompts and (would_exceed_prompts or would_exceed_chars):
            yield cur_prompts, cur_map
            cur_prompts, cur_map, cur_chars = [], [], 0
        cur_prompts.append(prompt)
        cur_map.append((gid, i, item))
        cur_chars += plen

    if cur_prompts:
        yield cur_prompts, cur_map


def make_descriptor_explainer_groups(pairs: List[Tuple[str, List[Dict[str, str]]]], start_id: int = 0) -> List[Dict[str, Any]]:
    groups = []
    gid_counter = itertools.count(start=start_id)
    for desc, expls in pairs:
        groups.append({"gid": next(gid_counter),
                      "descriptor": desc, "explainers": expls})
    return groups


# ------------------- Trace helpers -------------------------

def build_merge_events(
    merged_groups: List[Dict[str, Any]],
    parent_gid: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Produce structured merge events mapping source_pair_ids -> new_pair_id."""
    events: List[Dict[str, Any]] = []
    for out in (merged_groups or []):
        orig_desc = _normalize_descriptor(out.get("original_descriptor") or "")
        for g in (out.get("groups") or []):
            gd = _normalize_descriptor(g.get("group_descriptor") or orig_desc)
            ge = (g.get("group_explainer") or "").strip()
            if not gd or not ge:
                continue
            new_pid = _pair_id(gd, ge)
            source_ids = [sid for sid in (g.get("original_explainer_ids_in_this_group") or []) if sid]
            events.append({
                "original_gid": parent_gid,
                "original_descriptor": orig_desc,
                "descriptor": gd,
                "group_explainer": ge,
                "new_pair_id": new_pid,
                "source_pair_ids": source_ids,
            })
    return events


# ------------------- Main -------------------------


def main(args):
    results_dir = Path(f"../results/disambiguate_merges/{args.run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(Path(results_dir / f"{args.run_id}_log.txt"))
    print(f"Starting run {args.run_id}", flush=True)
    with open(results_dir / f"{args.run_id}_settings.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    dm = DescriptorMerger(args)
    logging.info("Loading LLM...")
    dm.llm_setup()

    # Prepare loop state -----------------------------------------------
    iter_start = time.time()

    final_groups: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    lineage_events: List[Dict[str, Any]] = []

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
        logging.info("Resumed from checkpoint #%d: %d final, %d pending.",
                     step_idx, len(final_groups), len(pending))
    else:
        descriptor_explainers = read_grouped_descriptors_file(
            Path(args.data_path), test_size=10_000 if args.test else 0)
        groups = make_descriptor_explainer_groups(
            descriptor_explainers, start_id=0)
        final_groups = [g for g in groups if len(g["explainers"]) <= 1]
        pending = [g for g in groups if len(g["explainers"]) > 1]
        logging.info("Fresh start: %d singles, %d multis.",
                     len(final_groups), len(pending))

    if not pending:
        logging.info("No multi-explainer groups remaining.")
        pairs = [{
            "id": g["explainers"][0]["id"],
            "descriptor": g["descriptor"],
            "explainer": g["explainers"][0].get("explainer", ""),
            "source_pair_ids": g["explainers"][0].get("source_pair_ids", []),
        } for g in final_groups if g.get("explainers")]
        save_results(results_dir / f"{args.run_id}_disambig.jsonl", pairs)
        logging.info("Results saved.")
        return

    MAX_ITERS_PER_GROUP = args.max_passes_per_group
    CHECKPOINT_EVERY = args.checkpoint_every

    # next_gid counter seeded from existing gids
    all_gids = [g.get("gid") for g in (final_groups + pending)
                if g.get("gid") is not None]
    start_next_gid = (max(all_gids) + 1) if all_gids else 0
    next_gid = itertools.count(start=start_next_gid)


    pass_num = 0
    while pending:
        pass_num += 1
        start_time = time.time()
        cohort = [pending.pop()]
        while len(cohort) < max(1, args.groups_per_llm_call) and pending:
            cohort.append(pending.pop())

        logging.info("=" * 40)
        logging.info("Starting pass %d on cohort of %d groups.",
                     pass_num, len(cohort))

        gid_to_outputs = {g["gid"]: [] for g in cohort}

        for prompts, index_map in coalesce_prompts(
            groups_for_pass=cohort,
            chars_per_batch=args.chars_per_batch,
            max_prompts_per_call=args.max_prompts_per_call,
            max_chars_per_call=args.max_chars_per_call,
        ):
            outputs = dm.generate(prompts)
            for out, (gid, _local_idx, item_dict) in zip(outputs, index_map):
                # map indices -> ids before any validation
                out = map_indices_to_ids(out, item_dict)
                validated = dm.validate_completeness([out], [item_dict])[0]
                gid_to_outputs[gid].append(validated)

        next_round_multis = []
        for group in cohort:
            merged_groups = gid_to_outputs[group["gid"]]

            # Flatten -> canonical pairs with lineage
            flat_pairs = parse_output(merged_groups)

            # Aggregate by descriptor for the next pass
            aggregated = group_pairs_with_ids(flat_pairs)
            aggregated_pairs = list(aggregated.items())  # [(desc, [expl_obj,...]), ...]

            # new gids for sub-groups
            sub_groups = make_descriptor_explainer_groups(
                aggregated_pairs, start_id=next(next_gid)
            )
            for _ in range(len(sub_groups) - 1):
                next(next_gid)

            # lineage
            lineage_events.extend(build_merge_events(merged_groups, parent_gid=group.get("gid")))

            # Partition and collect
            new_multis = [g for g in sub_groups if len(g["explainers"]) > 1]
            new_singles = [g for g in sub_groups if len(g["explainers"]) <= 1]
            final_groups.extend(new_singles)
            next_round_multis.extend(new_multis)

        pending.extend(next_round_multis)

        if CHECKPOINT_EVERY and (pass_num % CHECKPOINT_EVERY == 0) and pass_num > 0:
            step_idx = pass_num
            save_checkpoint_full(checkpoint_dir, step_idx, final_groups, pending, lineage_events)

        end_time = time.time()
        pass_time = end_time - start_time
        logging.info(f"Pass with {len(cohort)} groups done in "
                     f"{time.strftime('%H:%M:%S', time.gmtime(pass_time))}. "
                     f"{len(pending)} pending remain.")

    # Final save ---------------------------------------------------------
    pairs = [{
        "id": g["explainers"][0]["id"],
        "descriptor": g["descriptor"],
        "explainer": g["explainers"][0].get("explainer", ""),
        "source_pair_ids": g["explainers"][0].get("source_pair_ids", []),
    } for g in final_groups if g.get("explainers")]
    out_path = results_dir / f"{args.run_id}_disambig.jsonl"
    save_results(out_path, pairs)
    # Save final checkpoint in results_dir
    save_checkpoint_full(results_dir, pass_num,
                        final_groups, pending, lineage_events, final=True)

    logging.info("Disambiguation done in %.2fs. Saved %d final pairs.",
                time.time() - iter_start, len(pairs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-path", required=True, type=Path,
                   help="Path to JSONL file with objects {descriptor, pairs:[{id, explainer}]} or legacy {descriptor, explainers}.")

    # Batching parameters
    p.add_argument("--chars-per-batch", type=int, default=500_000,
                   help="Max prompt characters per batch.")
    p.add_argument("--groups-per-llm-call", dest="groups_per_llm_call", type=int, default=100,
               help="How many groups to process together per vLLM call (cohort size).")
    p.add_argument("--max-prompts-per-call", dest="max_prompts_per_call", type=int, default=512,
                help="Cap on prompts per vLLM.generate call.")
    p.add_argument("--max-chars-per-call", dest="max_chars_per_call", type=int, default=2_000_000,
                help="Cap on total characters across all prompts in one call.")

    # LLM parameters
    p.add_argument(
        "--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME",
                   "~/.cache"), help="Cache directory for model files")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM temperature. Default 0.1. Between 0 and 1.")

    # Control parameters
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in run directory.")
    p.add_argument("--checkpoint-every", type=int, default=20,
                   help="Save a full checkpoint after every N passes(cohorts). 0 disables.")
    p.add_argument("--max-passes-per-group", type=int, default=10,
                   help="Safety valve to stop infinite loops when a group refuses to change.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: limit to 10,000 descriptors for quick test runs.")
    args = p.parse_args()

    main(args)

    logging.info("Done.")
