from __future__ import annotations

# -- standard library --
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import functools
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Iterator, Tuple, Optional

# -- third-party --
import json_repair  # type: ignore
import numpy as np  # type: ignore
from pydantic import BaseModel  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# -- local imports --
import merge_prompts  # type: ignore

#Pipeline:
#1. Read data
#2. Identify duplicate descriptors
#3. If there are more than 50 duplicates in group, split into even slices of max 50
#4. For each slice, run LLM to generate merged descriptor(s) and canonical explainer(s)
#5. Write merged descriptors to file
#6. Repeat until no more duplicates are found

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
        logging.info(
            f"Execution of {func.__name__} took "
            f"{time.strftime('%H:%M:%S', time.gmtime(execution_time))}."
        )
        return result
    return wrapper


class DescriptorMerger:
    def __init__(self, args: argparse.Namespace) -> None:
        # CLI params -----------------------------------------------------
        self.run_id: str = args.run_id
        self.cache_dir: str = os.path.expanduser(args.cache_dir or os.environ.get("HF_HOME", "~/.cache"))
        self.model_name: str = args.model
        self.batch_size: int = args.batch_size
        self.temperature: float = args.temperature
        self.resume: bool = args.resume
        self.data_path: Path = Path(args.data_path)
        self.test: bool = args.test

        # storage --------------------------------------------------------
        self.base_dir: Path = Path("..") / "results" / "LLM_merges" / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.base_dir / "merged_descriptors.jsonl"

        # Set up logging
        self._configure_logging()

        # to be initialised later in pipeline()
        self.llm: LLM | None = None

    def _configure_logging(self) -> None:
        log_file = self.base_dir / f"{self.run_id}.log"

        # reset handlers
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        file_h = logging.FileHandler(log_file)
        file_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        stream_h = logging.StreamHandler()  # goes to stdout -> Slurm captures it
        stream_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_h)
        logging.root.addHandler(stream_h)

        logging.info("=" * 40)
        logging.info("Logging configured to: %s", log_file)
        logging.info("SLURM Job ID: %s", os.environ.get("SLURM_JOB_ID", "N/A"))

    @log_execution_time
    def llm_setup(self) -> LLM:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPU available.")
        else:
            logging.info(f"Using {n_gpus} GPU(s).")
        return LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=128_000,
            tensor_parallel_size=n_gpus,
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
                # keep slot alignment; an empty dict will be ignored later
                response_texts.append("{}")
        
        # Log throughput
        tot_tok = gen_tok + in_tok
        if elapsed > 0 and tot_tok > 0:
            logging.info(
                "LLM throughput: %.1f tok/s (%.1f gen tok/s) — %s tokens in %.2fs",
                tot_tok / elapsed,
                gen_tok / elapsed if gen_tok else 0,
                tot_tok,
                elapsed,
            )

        # Parallel validation if there are many responses
        if len(response_texts) > 10:
            try:
                with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
                    parsed_responses = list(executor.map(validate_json, response_texts))
            except Exception as e:
                logging.error(
                    f"Error in parallel JSON validation, falling back to sequential processing. Error: {e}"
                )
                parsed_responses = [validate_json(text) for text in response_texts]
        else:
            parsed_responses = [validate_json(text) for text in response_texts]

        return parsed_responses
        
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
        """Split descriptor-explainer groups into max size of max_items.
        Yield batches of size max_dicts.

        Args:
            groups (Dict[str, List[str]]): _description_
            max_items (int, optional): _description_. Defaults to 50.
            max_dicts (int, optional): _description_. Defaults to 200.

        Yields:
            Iterator[List[Dict[str, List[str]]]]: _description_
        """
        def split(key: str, lst: List[str]) -> List[Dict[str, List[str]]]:
            return [{key: lst[i : i + max_items]} for i in range(0, len(lst), max_items)]

        normalised: List[Dict[str, List[str]]] = []
        for descriptor, explainers in groups.items():
            normalised.extend(
                split(descriptor, explainers) if len(explainers) > max_items else [{descriptor: explainers}]
                )

        logging.info(f"Multi-explainer groups split into {len(normalised)} sub-groups with max 50 explainers.")
        logging.info(f"Yielding batches of {max_dicts} groups at a time, for a total of {len(normalised) // max_dicts + 1} batches.")
        
        for i in range(0, len(normalised), max_dicts):
            yield normalised[i : i + max_dicts]

    @staticmethod
    def _split_pair(text: str) -> Tuple[str, str]:
        try:
            d, e = text.split(";", 1)
            d = DescriptorMerger._normalize_descriptor(d)
            return d, e.strip()
        except ValueError:
            return text.lower().strip(), ""

    @staticmethod
    def _group_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        d: Dict[str, List[str]] = defaultdict(list)
        for desc, expl in pairs:
            d[desc].append(expl)
        return d

    def _load_initial_pairs(self, path: Path) -> List[Tuple[str, str]]:
        logging.info("Loading descriptor strings from %s", path)

        # Limit to a fixed number of descriptors if in test mode
        TEST_SIZE=10_000
        
        descriptor_explainer_strings: List[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                best = int(np.argmax(doc["similarity"]))
                descriptor_explainer_strings.extend(doc["descriptors"][best])
                if self.test and len(descriptor_explainer_strings) >= TEST_SIZE:
                    logging.info(f"Test mode: limiting to {TEST_SIZE} descriptors")
                    break
        
        # Split desc;exp strings into (desc, exp) tuples
        pairs: List[Tuple[str, str]] = [self._split_pair(s) for s in descriptor_explainer_strings]
        return pairs

    def _process_batch(self, batch: List[Dict[str, List[str]]]) -> List[Dict[str, Any]]:
        prompts = [
                    merge_prompts.merge_descriptors_prompt(desc, expls)
                    for group in batch
                    for desc, expls in group.items()
                ]
        
        return self.generate(prompts)
    
    @staticmethod
    def _normalize_descriptor(s: str) -> str:
        # replace runs of underscores/spaces with a single space, trim, lowercase
        return re.sub(r'[_\s]+', ' ', (s or '')).strip().lower()
        
    @staticmethod
    def _save_checkpoint(pairs: List[Tuple[str, str]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for desc, expl in pairs:
                d = {"descriptor": desc, "explainer": expl}
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_output(merged_groups: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for output in merged_groups:
            if not isinstance(output, dict):
                continue
            for g in (output.get("groups") or []):
                gd = DescriptorMerger._normalize_descriptor(g.get("group_descriptor") or "")
                ge = (g.get("group_explainer") or "").strip()
                # SDrop empty descriptor or explainer
                if not gd or not ge:
                    logging.warning(f"Dropping model output with empty field(s): {g}")
                    continue
                pairs.append((gd.lower(), ge))  # keep descriptor case-consistent
        return pairs

    def _find_latest_checkpoint(self) -> Optional[Tuple[Path, int]]:
        """
        Find the newest checkpoint named like 'iter_<N>.jsonl' in self.base_dir.
        Returns (path, iteration_number) or None if no checkpoints exist.
        """
        pattern = re.compile(r"^iter_(\d+)\.jsonl$")
        latest: Optional[Tuple[Path, int]] = None

        if not self.base_dir.exists():
            return None

        for p in self.base_dir.glob("iter_*.jsonl"):
            m = pattern.match(p.name)
            if not m:
                continue
            it = int(m.group(1))
            if latest is None or it > latest[1]:
                latest = (p, it)

        return latest

    @staticmethod
    def _load_checkpoint(path: Path) -> List[Tuple[str, str]]:
        """
        Load a checkpoint JSONL file with one object per line:
        {"descriptor": "<str>", "explainer": "<str>"}
        Returns a list of (descriptor, explainer) tuples.
        """
        pairs: List[Tuple[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logging.warning("Skipping malformed JSON on line %d in %s: %s", lineno, path.name, e)
                    continue

                desc = DescriptorMerger._normalize_descriptor(obj.get("descriptor") or "")
                expl = (obj.get("explainer") or "").strip()
                if not desc:
                    logging.warning("Skipping line %d in %s: missing 'descriptor'", lineno, path.name)
                    continue

                # Keep descriptor case consistent with the rest of the pipeline
                pairs.append((desc.lower(), expl))
        return pairs
    
    def _save_output(self, merged_groups: List[Dict[str, Any]], iteration: int) -> None:
        path = self.base_dir / f"merge_log_iter_{iteration}.jsonl"
        with path.open("w") as f:
            for output in merged_groups:
                for group in output["groups"]:
                    d = {
                        "original_descriptor": output.get("original_descriptor", ""),
                        "new_descriptor": group.get("group_descriptor", ""),
                        "original_explainers": group.get("original_explainers_in_this_group", []),
                        "new_explainer": group.get("group_explainer", "")
                    }
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline(self):
        logging.info("Loading LLM...")
        self.llm = self.llm_setup()
        
        # Either start fresh or continue from checkpoint
        src_path = self.data_path
        if not self.resume:
            pairs = self._load_initial_pairs(src_path)
            iteration = 1
        else:
            latest = self._find_latest_checkpoint()
            if latest is None:
                logging.info("Resume requested, but no checkpoints found in %s. Starting fresh.", self.base_dir)
                pairs = self._load_initial_pairs(src_path)
                iteration = 1
            else:
                ckpt_path, last_iter = latest
                logging.info("Resuming from %s (iteration %d).", ckpt_path, last_iter)
                pairs = self._load_checkpoint(ckpt_path)
                if not pairs:
                    logging.warning("Latest checkpoint %s is empty or unreadable. Starting fresh.", ckpt_path)
                    pairs = self._load_initial_pairs(src_path)
                    iteration = 1
                else:
                    # Make the checkpoint the new 'source' for subsequent saves/logs
                    src_path = ckpt_path
                    self.save_path = ckpt_path
                    iteration = last_iter + 1
        
        # While-loop ends, when no more multi-explainer groups remain
        while True:
            iter_start = time.time()
            logging.info(
                f"++++++++++ Starting iteration {iteration} "
                f"with {len(pairs)} descriptor-explainer pairs ++++++++++"
                )
            
            # Group by descriptors like this {descriptor: [explainer, explainer, explainer,...]}
            groups = self._group_pairs(pairs)
            
            # Process only descriptors that have >1 explainer
            multi_map: Dict[str, List[str]] = {
                descriptor: explainers for descriptor, explainers in groups.items() if len(explainers) > 1
                }
            singletons = {
                descriptor: explainers for descriptor, explainers in groups.items() if len(explainers) == 1
                }

            if not multi_map:
                logging.info("No multi-explainer groups remaining. Process finished!")
                break

            logging.info("There are %d multi-explainer groups and %d singletons", len(multi_map), len(singletons))

            # Push through LLM in batches
            merged_groups: List[Dict[str, Any]] = []
            for idx, batch in enumerate(self._batch_dicts(multi_map, max_dicts=self.batch_size)):
                logging.info("====>  Processing batch %d with %d group(s)…", idx, len(batch))
                merged_groups.extend(self._process_batch(batch))

            # Save output to log what gets merged
            self._save_output(merged_groups, iteration)
            
            # Parse LLM output into list of (descriptor,explainer) tuples
            new_pairs = self._parse_output(merged_groups)
            
            # Append singleton pairs to new pairs
            singleton_pairs = [(d, e) for d, expls in singletons.items() for e in expls]
            all_pairs = new_pairs + singleton_pairs
            
            # Save results as JSONL like this:
            # {descriptor: "descriptor", explainer: "explainer"}
            out_path = self.base_dir / f"iter_{iteration}.jsonl"
            logging.info("Saving results to %s…", out_path)
            self._save_checkpoint(all_pairs, out_path)
            logging.info("Results saved.")
            
            # Update paths and pairs for next iteration
            src_path = out_path
            self.save_path = out_path
            pairs = all_pairs
            iteration += 1
            
            iter_end = time.time()
            logging.info(f"Iteration took {time.strftime('%H:%M:%S', time.gmtime(iter_end-iter_start))}.")
            

# Keep this outside of DescriptorMerger for pickling reasons
def validate_json(text: str) -> Dict[str, Any]:
    try:
        obj = json_repair.loads(text)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("JSON parse failed: %s", exc)
        return {}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "~/.cache"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    
    print(f"Starting run {args.run_id}", flush=True)

    dm = DescriptorMerger(args)
    dm.pipeline()
    logging.info("Done.")