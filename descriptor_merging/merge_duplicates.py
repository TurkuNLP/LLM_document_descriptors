"""Pipeline for merging semantically‑similar descriptor–explainer pairs
using an LLM, while keeping a full mapping from *every* original pair to
its final canonical version and an audit‑trail of each merge step. It
also rewrites the source documents with the new pairs when finished.
"""

from __future__ import annotations

# ‑‑ standard library ‑‑
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import functools
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Iterator, Tuple

# ‑‑ third‑party ‑‑
import json_repair  # type: ignore
import numpy as np  # type: ignore
from pydantic import BaseModel  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# ‑‑ local imports ‑‑
import merge_prompts  # type: ignore

# ---------------------------------------------------------------------------
# environment configuration
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def log_execution_time(func):
    """Decorator that logs the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution of {func.__name__} took {time.strftime('%H:%M:%S', time.gmtime(execution_time))}.")
        return result
    return wrapper

# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------
class DescriptorMerger:
    """Merge duplicates and keep full provenance plus checkpointing."""

    # ────────────────────────────────────────────────────────────────────
    # construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(self, args: argparse.Namespace) -> None:
        # CLI params -----------------------------------------------------
        self.run_id: str = args.run_id
        self.cache_dir: str = args.cache_dir or os.environ.get("HF_HOME", "~/.cache")
        self.model_name: str = args.model
        self.batch_size: int = args.batch_size
        self.temperature: float = args.temperature
        self.resume: bool = args.resume
        self.data_path: Path = Path(args.data_path)

        # storage --------------------------------------------------------
        self.base_dir: Path = Path("..") / "results" / "LLM_merges" / self.run_id
        self.checkpoint_file: Path = self.base_dir / "checkpoint.json"
        
        # Set up logging
        self._configure_logging()

        # audit structures
        # Tracks all mappings from original pairs to their current merged versions
        self.global_map: Dict[str, str] = {}
        
        # Records each merge decision for audit trail purposes
        self.history: List[Dict[str, Any]] = []

        # to be initialised later in pipeline()
        self.llm: LLM | None = None
        
    
    def _configure_logging(self) -> None:
        """Configure logging to write to the run-specific log file."""
        log_file = self.base_dir / f"{self.run_id}.log"
        
        # Remove any existing handlers (from the basicConfig above)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Add a new file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        
        # Add a console handler to still see output in the terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        
        # Configure the root logger
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
        # Log the configuration change
        logging.info("="*40)
        logging.info(f"Logging configured to: {log_file}")
        # Also log the slurm job ID if available
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
        logging.info(f"SLURM Job ID: {slurm_job_id}")

    @log_execution_time
    def llm_setup(self) -> LLM:
        logging.info("Loading LLM model…")
        return LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=128_000,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=False,
            gpu_memory_utilization=0.8,
        )

    @log_execution_time
    def generate(self, prompts: List[str]) -> List[Dict]:
        params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            repetition_penalty=1.0,
            max_tokens=10_000,
            stop=["<|eot_id|>"],
            guided_decoding=self._response_format(),
        )
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - start if start else 0.0
        
        # Log throughput
        gen_tok = sum(len(o.outputs[0].token_ids) for o in outputs if getattr(o.outputs[0], "token_ids", None))
        in_tok = sum(len(o.prompt_token_ids) for o in outputs if getattr(o, "prompt_token_ids", None))
        tot_tok = gen_tok + in_tok
        if elapsed > 0 and tot_tok > 0:
            logging.info("LLM throughput: %.1f tok/s (%.1f gen tok/s) — %s tokens in %.2fs", tot_tok / elapsed, gen_tok / elapsed if gen_tok else 0, tot_tok, elapsed)
        
        response_texts = [out.outputs[0].text.strip(" `\n").removeprefix("json") for out in outputs]
    
        # Parallel validation if there are many responses
        if len(response_texts) > 10:
            try:
                with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
                    parsed_responses = list(executor.map(validate_json, response_texts))
            except Exception as e:
                logging.error(f"Error in parallel JSON validation, falling back to sequential processing. Error: {e}")
                # Fallback to sequential processing
                parsed_responses = [validate_json(text) for text in response_texts]
        else:
            parsed_responses = [validate_json(text) for text in response_texts]
        
        return parsed_responses

    def _save_checkpoint(self, groups: Dict[str, List[str]], iteration: int, 
                        batch_index: int = 0) -> None:
        state = {
            "iteration": iteration,
            "batch_index": batch_index,
            "batch_size": self.batch_size,
            "groups": groups,
            "global_map": self.global_map,
            "history": self.history,
        }
        # Create a temporary file first for safer atomic writes
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(state, ensure_ascii=False))
        temp_file.replace(self.checkpoint_file)  # Atomic replacement
        
        logging.info(f"Saved checkpoint @ iteration {iteration}, batch {batch_index}")

    def _load_checkpoint(self) -> Tuple[Dict[str, List[str]], int, int]:
        data = json.loads(self.checkpoint_file.read_text())
        self.global_map = data.get("global_map", {})
        self.history = data.get("history", [])
        iteration = data.get("iteration", 0)
        batch_index = data.get("batch_index", 0)
        groups = {k: v for k, v in data.get("groups", {}).items()}
        
        # Check for saved batch size and override if necessary
        saved_batch_size = data.get("batch_size")
        if saved_batch_size is not None and saved_batch_size != self.batch_size:
            logging.warning(f"Overriding provided batch size ({self.batch_size}) with checkpoint batch size ({saved_batch_size})")
            self.batch_size = saved_batch_size
            
        logging.info("Loaded checkpoint with iteration=%s, batch=%s, |groups|=%s, batch_size=%s", 
                    iteration, batch_index, len(groups), self.batch_size)
        return groups, iteration, batch_index

    @staticmethod
    def _response_format() -> GuidedDecodingParams:
        class G(BaseModel):  # type: ignore[misc]
            group_descriptor: str
            group_explainer: str
            original_explainers_in_this_group: List[str]

        class R(BaseModel):  # type: ignore[misc]
            original_descriptor: str
            groups: List[G]

        return GuidedDecodingParams(json=R.model_json_schema())

    @staticmethod
    def _batch_dicts(
        groups: Dict[str, List[str]],
        max_items: int = 50,
        max_dicts: int = 200,
    ) -> Iterator[List[Dict[str, List[str]]]]:
        def split(key: str, lst: List[str]) -> List[Dict[str, List[str]]]:
            return [{key: lst[i : i + max_items]} for i in range(0, len(lst), max_items)]

        normalised: List[Dict[str, List[str]]] = []
        for k, v in groups.items():
            normalised.extend(split(k, v) if len(v) > max_items else [{k: v}])
        for i in range(0, len(normalised), max_dicts):
            yield normalised[i : i + max_dicts]

    @staticmethod
    def _split_pair(text: str) -> Tuple[str, str]:
        try:
            d, e = text.split(";", 1)
            return d.lower().strip(), e.strip()
        except ValueError:
            return text.lower().strip(), ""

    @staticmethod
    def _group_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        d: Dict[str, List[str]] = defaultdict(list)
        for desc, expl in pairs:
            d[desc].append(expl)
        return d

    def _collapse_map(self) -> Dict[str, str]:
        def find_final_mapping(key: str, cache: Dict[str, str], path: set) -> str:
            if key not in self.global_map or key == self.global_map[key]:
                return key
                
            if key in cache:
                return cache[key]
                
            if key in path:  # Detect cycle
                logging.warning(f"Circular reference detected in _collapse_map() for key: {key}")
                return key  # Break the cycle
                
            path.add(key)
            result = find_final_mapping(self.global_map[key], cache, path)
            path.remove(key)
            cache[key] = result
            return result
        
        out: Dict[str, str] = {}
        cache: Dict[str, str] = {}
        for src in self.global_map:
            out[src] = find_final_mapping(src, cache, set())
        return out

    def _save_artifacts(self, groups: Dict[str, List[str]]) -> None:
        (self.base_dir / f"final_merged_{self.run_id}.json").write_text(
            json.dumps(groups, ensure_ascii=False, indent=2)
        )
        self.final_mapping = self._collapse_map()
        (self.base_dir / f"{self.run_id}_pair_map.json").write_text(
            json.dumps(self.final_mapping, ensure_ascii=False, indent=2)
        )
        (self.base_dir / f"{self.run_id}_merge_history.json").write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2)
        )
        logging.info("Artifacts written to %s", self.base_dir)

    @staticmethod
    def _load_raw_descriptor_strings(path: Path) -> List[str]:
        logging.info("Loading descriptor strings from %s", path)
        out: List[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                best = int(np.argmax(doc["similarity"]))
                out.extend(doc["descriptors"][best])
                #out.extend(doc["descriptors"])
        return out

    # ---------------------------------------------------------------------
    # main driver
    # ---------------------------------------------------------------------
    def pipeline(self) -> None:
        self.llm = self.llm_setup()

        # ────────────────────────────────
        # 1. (Re-)initialise state
        # ────────────────────────────────
        if self.resume:
            if not self.checkpoint_file.exists():
                raise FileNotFoundError(
                    f"--resume was given but no checkpoint found at {self.checkpoint_file}"
                )

            ckpt = json.loads(self.checkpoint_file.read_text())
            self.global_map = ckpt.get("global_map", {})
            self.history = ckpt.get("history", [])
            groups = ckpt.get("groups", {})
            iteration = ckpt.get("iteration", 1)
            batch_index = ckpt.get("batch_index", 0)
            saved_bs = ckpt.get("batch_size", self.batch_size)

            if saved_bs != self.batch_size:
                logging.warning(
                    "Overriding --batch-size %s with the value stored in the checkpoint (%s)",
                    self.batch_size,
                    saved_bs,
                )
                self.batch_size = saved_bs

            logging.info(
                "Resuming from iteration=%s, next batch=%s, groups=%s",
                iteration,
                batch_index,
                len(groups),
            )

        else:
            raw    = self._load_raw_descriptor_strings(self.data_path)
            pairs  = [self._split_pair(s) for s in raw]
            groups = self._group_pairs(pairs)
            iteration   = 1
            batch_index = 0
            self._save_checkpoint(groups, iteration, batch_index)
            logging.info("New run started with groups=%s", len(groups))

        # ────────────────────────────────
        # 2. Iterative merge loop
        # ────────────────────────────────
        while True:
            logging.info("── Iteration %s │ groups=%s", iteration, len(groups))

            # split once per iteration (groups mutates in-place afterwards)
            singletons = {k: v for k, v in groups.items() if len(v) == 1}
            multis     = {k: v for k, v in groups.items() if len(v) > 1}

            if not multis:
                logging.info("No multi-explainer groups left → finished")
                break

            batches = list(self._batch_dicts(multis, max_dicts=self.batch_size))
            if batch_index >= len(batches):
                # we somehow completed all batches the last time we ran
                iteration   += 1
                batch_index  = 0
                self._save_checkpoint(groups, iteration, batch_index)
                continue

            changed   = False
            iter_map  = {}

            # ───────────────────────
            #  per-batch processing
            # ───────────────────────
            for i in range(batch_index, len(batches)):
                batch = batches[i]
                logging.info("---> Processing batch %s/%s", i + 1, len(batches))

                prompts          = [
                    merge_prompts.merge_descriptors_prompt(desc, expls)
                    for d in batch
                    for desc, expls in d.items()
                ]
                parsed_responses = self.generate(prompts)

                try:
                    with ProcessPoolExecutor(
                        max_workers=min(32, os.cpu_count() or 4)
                    ) as ex:
                        results = list(ex.map(create_mapping, parsed_responses))
                except Exception as exc:
                    logging.error("Fallback to sequential mapping: %s", exc)
                    results = [create_mapping(r) for r in parsed_responses]

                # ────────────────────────────
                #  apply mappings in-place
                # ────────────────────────────
                for mapping, resp in results:
                    # audit log
                    with (self.base_dir / f"iteration_{iteration}_response.jsonl").open(
                        "a", encoding="utf-8"
                    ) as fh:
                        fh.write(json.dumps(resp, ensure_ascii=False) + "\n")

                    # for each new merged descriptor…
                    for g in resp.get("groups", []):
                        new_desc = g["group_descriptor"].lower().strip()
                        new_expl = g["group_explainer"].strip()
                        groups.setdefault(new_desc, []).append(new_expl)

                    # global + iteration maps
                    for old, new in mapping.items():
                        iter_map[old] = new
                        root = self.global_map.get(old, old)
                        self.global_map[root] = new
                        self.global_map[old] = new

                    # drop old descriptors that were merged in this response
                    for old in mapping.keys():
                        old_desc, _ = self._split_pair(old)
                        groups.pop(old_desc, None)

                    changed = changed or bool(mapping)

                # ────────────────────────────
                #  checkpoint after every batch
                # ────────────────────────────
                self._save_checkpoint(groups, iteration, i + 1)

            # ─────────────────────────
            #  post-iteration housekeeping
            # ─────────────────────────
            self.history.append({"iteration": iteration, "mapping": iter_map})
            iteration   += 1
            batch_index  = 0
            self._save_checkpoint(groups, iteration, batch_index)

            if not changed:
                logging.info("Nothing changed during iteration %s → stopping", iteration - 1)
                break

        # ────────────────────────────────
        # 3. finalise artefacts
        # ────────────────────────────────
        self._save_artifacts(groups)
        self.checkpoint_file.unlink(missing_ok=True)


def validate_json(text: str) -> Dict[str, Any]:
    try:
        obj = json_repair.loads(text)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("JSON parse failed: %s", exc)
        return {}


def create_mapping(resp: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    mapping: Dict[str, str] = {}
    base = resp.get("original_descriptor", "")
    for g in resp.get("groups", []):
        new_pair = f"{g.get('group_descriptor', 'Error in descriptor merging')}; {g.get('group_explainer', 'Error in descriptor merging')}"
        for old in g.get("original_explainers_in_this_group", []):
            mapping[f"{base}; {old}"] = new_pair
    return mapping, resp  # Return both mapping and original response


def replace_pairs_in_documents(
    source_jsonl: Path,
    target_jsonl: Path,
    pair_map: Dict[str, str],
) -> None:
    with source_jsonl.open(encoding="utf-8") as fin, target_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            best = int(np.argmax(doc["similarity"]))
            doc["descriptors"] = [pair_map.get(s, s) for s in doc["descriptors"][best]]
            #doc["descriptors"] = [pair_map.get(s, s) for s in doc["descriptors"]]
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logging.info("Rewrote descriptors → %s", target_jsonl)

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge descriptor‑explainer pairs with an LLM")

    parser.add_argument("--run-id", required=True, help="ID for this run, e.g. run1")
    parser.add_argument("--data-path", required=True, help="Original documents JSONL")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for prompts")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if present")
 
    args = parser.parse_args()

    # ensure dirs
    Path("../logs").mkdir(exist_ok=True)
    Path("../results").mkdir(exist_ok=True)
    Path("../results/LLM_merges").mkdir(exist_ok=True)

    # log run settings
    run_dir = Path("../results/LLM_merges") /  args.run_id
    run_dir.mkdir(exist_ok=True)
    with (run_dir / f"{args.run_id}_settings.txt").open("a", encoding="utf‑8") as f_settings:
        f_settings.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f_settings.write(f"{arg}: {value}\n")
        f_settings.write("===========================\n")

    # run pipeline
    dm = DescriptorMerger(args)
    dm.pipeline()

    # rewrite documents with canonical pairs
    replace_pairs_in_documents(
        Path(args.data_path),
        dm.base_dir / f"{args.run_id}_merged.jsonl",
        dm.final_mapping,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
