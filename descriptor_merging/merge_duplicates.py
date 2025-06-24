"""Pipeline for merging semantically‑similar descriptor–explainer pairs
using an LLM, while keeping a full mapping from *every* original pair to
its final canonical version and an audit‑trail of each merge step. It
also rewrites the source documents with the new pairs when finished.
"""

from __future__ import annotations

# ‑‑ standard library ‑‑
import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple

# ‑‑ third‑party ‑‑
import json_repair  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from pydantic import BaseModel  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# ‑‑ local imports ‑‑
import merge_prompts  # type: ignore

# ---------------------------------------------------------------------------
# logging configuration
# ---------------------------------------------------------------------------
slurm_job_id = os.environ.get("SLURM_JOB_ID", "default_id")
logging.basicConfig(
    filename=f"../logs/{slurm_job_id}.log",
    filemode="a",
    format="%(asctime)s ‑ %(levelname)s ‑ %(message)s",
    level=logging.INFO,
)

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

        # audit structures ----------------------------------------------
        self.global_map: Dict[str, str] = {}
        self.history: List[Dict[str, Any]] = []

        # to be initialised later
        self.llm: LLM | None = None

    # ────────────────────────────────────────────────────────────────────
    # LLM helpers
    # ────────────────────────────────────────────────────────────────────
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

    def generate(self, prompts: List[str]) -> List[str]:
        params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            repetition_penalty=1.0,
            max_tokens=1024,
            stop=["<|eot_id|>"],
            guided_decoding=self._response_format(),
        )
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)  # type: ignore[arg-type]

        return [out.outputs[0].text.strip(" `\n").removeprefix("json") for out in outputs]

    @staticmethod
    def validate_output(text: str) -> Dict[str, Any]:
        try:
            obj = json_repair.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logging.warning("JSON parse failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # checkpoint helpers ------------------------------------------------
    # ------------------------------------------------------------------
    def _save_checkpoint(self, groups: Dict[str, List[str]], iteration: int) -> None:
        state = {
            "iteration": iteration,
            "groups": groups,
            "global_map": self.global_map,
            "history": self.history,
        }
        self.checkpoint_file.write_text(json.dumps(state, ensure_ascii=False))
        logging.info("Saved checkpoint @ iteration %s → %s", iteration, self.checkpoint_file)

    def _load_checkpoint(self) -> Tuple[Dict[str, List[str]], int]:
        data = json.loads(self.checkpoint_file.read_text())
        self.global_map = data.get("global_map", {})
        self.history = data.get("history", [])
        iteration = data.get("iteration", 0)
        groups = {k: v for k, v in data.get("groups", {}).items()}
        logging.info("Loaded checkpoint with iteration=%s, |groups|=%s", iteration, len(groups))
        return groups, iteration

    # ------------------------------------------------------------------
    # schema helpers ----------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # batching ----------------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # util helpers ------------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # create mapping ----------------------------------------------------
    # ------------------------------------------------------------------
    def _create_mapping(self, resp: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        base = resp.get("original_descriptor", "")
        for g in resp.get("groups", []):
            new_pair = f"{g['group_descriptor']}; {g['group_explainer']}"
            for old in g.get("original_explainers_in_this_group", []):
                mapping[f"{base}; {old}"] = new_pair
        return mapping

    # ------------------------------------------------------------------
    # mapping collapse & save ------------------------------------------
    # ------------------------------------------------------------------
    def _collapse_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for src in self.global_map:
            dst = self.global_map[src]
            while dst in self.global_map and dst != self.global_map[dst]:
                dst = self.global_map[dst]
            out[src] = dst
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
        
    # ------------------------------------------------------------------
    # raw data load helper ---------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _load_raw_descriptor_strings(path: Path) -> List[str]:
        logging.info("Loading descriptor strings from %s", path)
        out: List[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                best = int(np.argmax(doc["similarity"]))
                out.extend(doc["descriptors"][best])
        return out

    # ------------------------------------------------------------------
    # pipeline ----------------------------------------------------------
    # ------------------------------------------------------------------
    def pipeline(self) -> None:
        self.llm = self.llm_setup()

        # ----- load / resume -------------------------------------------
        if self.resume and self.checkpoint_file.exists():
            groups, iteration = self._load_checkpoint()
            iteration += 1  # resume *after* the checkpointed iteration
        else:
            raw = self._load_raw_descriptor_strings(self.data_path)
            pairs = [self._split_pair(s) for s in raw]
            groups = self._group_pairs(pairs)
            iteration = 1
            logging.info("Starting fresh run with %s groups", len(groups))

        # ----- main loop ----------------------------------------------
        while True:
            logging.info("Iteration %s: |groups|=%s", iteration, len(groups))
            
            # split into singletons and multis
            # (singletons are not processed, they are left untouched)
            singletons = {k: v for k, v in groups.items() if len(v) == 1}
            multis      = {k: v for k, v in groups.items() if len(v) > 1}
            if not multis:
                logging.info("No multi‑explainer groups left → done")
                break
            
            logging.info("Singleton groups: %s | Multi-explainer groups: %s", len(singletons), len(multis))
            
            merged: Dict[str, List[str]] = defaultdict(list)
            iter_map: Dict[str, str] = {}
            changed = False

            for batch in self._batch_dicts(multis, max_dicts=self.batch_size):
                prompts = [merge_prompts.merge_descriptors_prompt(k, v) for d in batch for k, v in d.items()]
                resp_texts = self.generate(prompts)
                parsed = [self.validate_output(t) for t in resp_texts]

                for resp in parsed:
                    mapping = self._create_mapping(resp)

                    for g in resp.get("groups", []):
                        merged[g["group_descriptor"].lower().strip()].append(g["group_explainer"].strip())
                        changed = True

                    for old, new in mapping.items():
                        iter_map[old] = new
                        root = self.global_map.get(old, old)
                        self.global_map[root] = new
                        self.global_map[old] = new

            groups = self._group_pairs([(d, e) for d, lst in merged.items() for e in lst])
            groups.update(singletons)  # re‑attach untouched singletons
            
            self.history.append({"iteration": iteration, "mapping": iter_map})
            self._save_checkpoint(groups, iteration)

            if not changed:
                logging.info("No changes in iteration %s → finishing", iteration)
                break
            iteration += 1

        # ----- final save ---------------------------------------------
        self._save_artifacts(groups)
        self.checkpoint_file.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# document rewrite helper
# ---------------------------------------------------------------------------

def replace_pairs_in_documents(
    source_jsonl: Path,
    target_jsonl: Path,
    pair_map: Dict[str, str],
) -> None:
    with source_jsonl.open(encoding="utf-8") as fin, target_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            doc["descriptors"] = [pair_map.get(s, s) for s in doc.get("descriptors", [])]
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
