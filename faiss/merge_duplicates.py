# Standard libraries
import argparse
from collections import Counter, defaultdict
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
from random import shuffle
import shutil
import time
import warnings

# Third party imports
import json_repair  # type: ignore
import pandas as pd  # type: ignore
from pydantic import BaseModel, RootModel  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from sklearn.cluster import AgglomerativeClustering  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# Local imports
from embed import StellaEmbedder
import prompts_with_explainers
from utils import (
    load_documents,
    save_descriptors,
    initialise_descriptor_vocab,
    init_results,
    save_results,
    save_synonym_dict,
    log_execution_time,
    get_best_results,
    get_best_descriptors,
)


# Configure logging
slurm_job_id = os.environ.get("SLURM_JOB_ID", "default_id")
logging.basicConfig(
    filename=f"../logs/{slurm_job_id}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Suppress sentence_transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Suppress transformers logging (used internally by sentence_transformers)
logging.getLogger("transformers").setLevel(logging.WARNING)
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DescriptorMerger:
    def __init__(self, args):
        self.cache_dir = args.cache_dir or os.environ["HF_HOME"]
        self.model = args.model
        self.start_index = args.start_index
        self.num_batches = args.num_batches
        self.run_id = args.run_id
        self.num_rewrites = args.num_rewrites
        self.batch_size = args.batch_size
        self.max_vocab = args.max_vocab
        self.synonym_threshold = args.synonym_threshold
        self.data_source = args.data_source
        self.temperature = args.temperature
        self.checkpoint_interval = args.checkpoint_interval
        self.base_dir = Path("..") / "results" / self.run_id
        self.embedder = StellaEmbedder(self.cache_dir)

    @log_execution_time
    def LLM_setup(self):
        return LLM(
            model=self.model,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=128_000,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=False,
            gpu_memory_utilization=0.8,
        )

    def generate(self, input, stage):
        response_schema = self.get_response_format(stage)
        max_tokens = {
            "initial": 2000,
            "rewrite": 4000,
            "revise": 5000,
            "synonyms": 2000,
        }
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            repetition_penalty=1,  # 1 = no penalty, >1 penalty
            max_tokens=max_tokens.get(stage, 5000),  # max tokens to generate
            guided_decoding=response_schema,
        )

        outputs = self.llm.generate(
            input, sampling_params=sampling_params, use_tqdm=False
        )

        return [out.outputs[0].text.strip(" `\n").removeprefix("json") for out in outputs]

    def validate_output(self, output):
        validated = json_repair.loads(output)
        return validated if isinstance(validated, dict) else {}

    def batched(self, data, batch_size=None, start_index=None):
        if not batch_size:
            batch_size = self.batch_size
        if not start_index:
            start_index = self.start_index

        batch = []
        for i, doc in enumerate(data):
            if i < start_index:
                continue
            batch.append(doc)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @log_execution_time
    def initial_stage(self, documents, vocab):
        if len(vocab) == 0:
            vocab = "The list of general descriptors is currently empty."
        else:
            vocab = "\n".join(vocab)

        stage = "initial"
        prompts = [
            self.format_prompt(stage=stage, original=document, vocab=vocab)
            for document in documents
        ]
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output) for output in batched_output
        ]

        return validated_outputs


    def update_descriptor_vocab(self, descriptor_path):
        desc_counts = Counter()
        with open(self.base_dir / f"descriptors_{self.run_id}.jsonl", "r") as f:
            for line in f.readlines():
                doc = json.loads(line.strip())
                for descriptor_list in doc["descriptors"]:
                    descriptors_without_explanations = self.remove_explanations(descriptor_list)
                    desc_counts.update(descriptors_without_explanations)

        save_descriptors(desc_counts, descriptor_path)
        self.log_descriptor_growth(desc_counts)

        # Keep max_vocab most common general descriptors. These will be given to the model as possible options.
        descriptor_vocab = (
            desc_counts.most_common(self.max_vocab)
            if self.max_vocab != -1
            else desc_counts.most_common()
        )
        descriptor_vocab = [item[0] for item in descriptor_vocab]
        # Shuffle the list of descriptors to avoid ordering bias
        shuffle(descriptor_vocab)

        return descriptor_vocab

    def log_descriptor_growth(self, vocab):
        with open(
            self.base_dir / f"descriptor_count_growth_{self.run_id}.txt", "a"
        ) as f:
            f.write(f"{len(vocab)}\n")

    def format_prompt(
        self,
        stage,
        original=None,
        rewritten=None,
        descriptors=None,
        specifics=None,
        vocab=None,
        group_name=None,
        synonyms=None,
    ):
        # Truncate original document to max length
        if original:
            original = self.tokenize_and_truncate(original, 100_000)

        if stage == "initial":
            return prompts_with_explainers.initial_prompt_one_descriptor_type(original, vocab)
        elif stage == "rewrite":
            return prompts_with_explainers.rewrite_prompt_one_descriptor_type(descriptors, specifics)
        elif stage == "revise":
            return prompts_with_explainers.revise_keyphrases_prompt_one_descriptor_type(
                original, rewritten, descriptors, specifics, vocab
            )
        elif stage == "synonyms":
            return prompts_with_explainers.review_synonyms(group_name, synonyms)
        elif stage == "fix":
            return prompts_with_explainers.reformat_output_prompt(original)

    def get_response_format(self, stage):
        if stage == "initial":

            class ResponseFormat(BaseModel):
                descriptors: list[str]
                specifics: list[str]

        elif stage == "rewrite":

            class ResponseFormat(BaseModel):
                document: str

        elif stage == "revise":

            class ResponseFormat(BaseModel):
                differences: str
                descriptors: list[str]
                specifics: list[str]

        elif stage == "synonyms":

            class ResponseFormat(RootModel):
                root: dict[str, list[str]]

        json_schema = ResponseFormat.model_json_schema()
        return GuidedDecodingParams(json=json_schema)


                
    def split_descriptors_and_explainers(self, d_and_e):
        desc_and_exp = e_and_e.split(":",1)
        desc = desc_and_exp[0].lower()
        exp = desc_and_exp[1].lower()

        return desc, exp

    def pipeline(self):

        descriptors_and_explainers = defauldict(list)
        data = self.load_descriptors()
        with open(self.data_path, "r") as f:
            for line in f:
                doc = json.loads(line)
                desc, exp = split_descriptors_and_explainers(doc["descriptors"])
                descriptors_and_explainers[desc].append(exp)


        prompts = []
        for desc, exp in descriptors_and_explainers.items():
            if len(exp) > 1:
                prompts.append(self.format_prompt(desc, exp))
        
        duplicates_remain = True

        validated_outputs = []
        while duplicates_remain:
            for batch in batched(prompts):
                model_responses = self.generate(batch)
                validated_outputs.append([self.validate_output(output) for output in model_responses])
                
                        

def main(args):

    dg = DescriptorMerger(args)
    dg.pipeline()
    logging.info("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A script for getting document descriptors with LLMs."
    )

    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Path to cache directory, where model is or will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of model to use.",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Index of first document to analyse."
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=-1,
        help="Number of batches of size batch-size to process. Set to -1 to process all data.",
    )
    parser.add_argument(
        "--num-rewrites",
        type=int,
        default=0,
        help="How many rewriting cycles the script should go through.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Model temperature."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents given to the model at one time.",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=-1,
        help="Max number of descriptors given in the prompt. Give -1 to use all descriptors.",
    )
    parser.add_argument(
        "--synonym-threshold",
        type=float,
        default=0.2,
        help="""Distance threshold for when two descriptors should count as synonyms.
        Smaller value means words are less likely to count as synonyms.""",
    )
    parser.add_argument(
        "--data-source", type=str, default="original", help="Which data set to process."
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=50,
        help="Number of batches after which all results so far will be saved into a checkpoint."
    )

    args = parser.parse_args()

    # Create required directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    os.makedirs(f"../results/{args.run_id}", exist_ok=True)

    # Log the run settings
    with open(f"../results/{args.run_id}/{args.run_id}_settings.txt", "a") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f.write(f"{arg}: {value}\n")
        f.write("===========================\n")

    main(args)
