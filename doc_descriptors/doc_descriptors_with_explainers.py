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


class DescriptorGenerator:
    def __init__(self, args):
        self.cache_dir = args.cache_dir or os.environ["HF_HOME"]
        self.model = args.model
        self.start_index = args.start_index
        self.num_batches = args.num_batches
        self.run_id = args.run_id
        self.num_rewrites = args.num_rewrites
        self.batch_size = args.batch_size
        self.data_source = args.data_source
        self.temperature = args.temperature
        self.checkpoint_interval = args.checkpoint_interval
        self.base_dir = Path("..") / "results" / self.run_id
        self.embedder = StellaEmbedder(self.cache_dir)
        self.text_column = args.text_column

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

    @staticmethod
    def validate_output(output):
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
    def initial_stage(self, documents):
        stage = "initial"
        prompts = [
            self.format_prompt(stage=stage, original=document)
            for document in documents
        ]
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output) for output in batched_output
        ]

        return validated_outputs

    @log_execution_time
    def rewrite_stage(self, general, specific):
        """
        Processes lists of descriptors through the rewrite generation stage.

        Args:
            stage (str): The stage of the pipeline.
            general (list of list): A list of lists of general descriptors.
            specific (list of list): A list of lists of specific descriptors.
            llm (object): The language model used for generating responses.

        Returns:
            list: A list of validated and possibly reformatted outputs in JSON format.
        """
        stage = "rewrite"
        prompts = []
        for g, s in zip(general, specific):
            prompts.append(self.format_prompt(stage=stage, descriptors=g, specifics=s))
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output) for output in batched_output
        ]
        
        return validated_outputs

    @log_execution_time
    def revise_stage(self, document, rewritten, general, specific):
        """
        Processes lists of descriptors and rewrites through the revise generation stage.

        Args:
            stage (str): The current stage of the pipeline.
            document (list of str): The original documents.
            rewritten (list of str): The rewritten documents.
            general (list of list): General descriptors for the documents.
            specific (list of list): Specific descriptors for the documents.
            llm (object): The language model used for generating outputs.

        Returns:
            list of dict: A list of validated outputs, each containing general and specific descriptors.
        """
        stage = "revise"
        prompts = []
        for d, r, g, s in zip(document, rewritten, general, specific):
            prompts.append(
                self.format_prompt(
                    stage=stage,
                    original=d,
                    rewritten=r,
                    descriptors=g,
                    specifics=s,
                )
            )
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
    ):
        # Truncate original document to max length
        if original:
            original = self.tokenize_and_truncate(original, 100_000)

        if stage == "initial":
            return prompts_with_explainers.initial_prompt_one_descriptor_type(original)
        elif stage == "rewrite":
            return prompts_with_explainers.rewrite_prompt_one_descriptor_type(descriptors, specifics)
        elif stage == "revise":
            return prompts_with_explainers.revise_keyphrases_prompt_one_descriptor_type(
                original, rewritten, descriptors, specifics)

    @staticmethod
    def get_response_format(stage):
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

        json_schema = ResponseFormat.model_json_schema()
        return GuidedDecodingParams(json=json_schema)

    def tokenize_and_truncate(self, document, max_input_len):
        tokenizer = self.llm.get_tokenizer()
        prompt_token_ids = tokenizer.encode(document)
        if len(prompt_token_ids) > max_input_len:
            logging.warning(
                f"Document is too long: ({len(prompt_token_ids)} tokens). Truncating..."
            )
            prompt_token_ids = prompt_token_ids[:max_input_len]

        return tokenizer.decode(prompt_token_ids)

    @staticmethod
    def update_results(results, general=None, specific=None, rewrites=None):
        """Updates results dictionary with new descriptors or rewrites."""
        for index in results:
            if general:
                results[index]["descriptors"].append(general[index])
            if specific:
                results[index]["specifics"].append(specific[index])
            if rewrites:
                results[index]["rewrite"].append(rewrites[index])

    def generate_final_rewrites(self):
        filepath = self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl"
        
        # Calculate the number of docs we have processed for logging purpses
        with open(filepath, 'rb') as f:
            num_docs= sum(buf.count(b'\n') for buf in iter(lambda: f.read(1024 * 1024), b''))
        
        with open(filepath, "r") as f:
            batch_num = 0
            results = {}
            idx = 0
            for line in f:
                doc = json.loads(line.strip())
                results[idx] = doc
                idx += 1
                
                # Process in batches of batch_size
                
                # TO FIX: if document do not neatly divide into batch_size
                # last documents will not be processed!
                if idx == self.batch_size:
                    batch_num += 1
                    logging.info(f"Processing batch {batch_num} out of "
                                 f"{math.ceil(num_docs/self.batch_size)}.")
                    general_descriptors = []
                    specific_descriptors = []
                    for index in results:
                        general_descriptors.append(results[index]["descriptors"])
                        specific_descriptors.append(results[index]["specifics"])

                    model_outputs = self.rewrite_stage(
                        general_descriptors, specific_descriptors
                    )
                    rewrites = [
                        output.get("document", "Generation failed.") for output in model_outputs
                    ]
                    for i, index in enumerate(results):
                        results[index]["rewrite"].append(rewrites[i])

                    for index in results:
                        similarities = self.embedder.calculate_similarity(
                            results[index]["document"], results[index]["rewrite"]
                        )
                        results[index]["similarity"].extend(similarities)

                    with open(
                        self.base_dir / f"descriptors_{self.run_id}_final.jsonl", "a"
                    ) as f:
                        for doc in results.values():
                            json_line = json.dumps(doc, ensure_ascii=False)
                            f.write(json_line + "\n")
                            
                    idx = 0
                    results = {}

    def make_checkpoint(self):
        # Calculate how many documents we have processed so far
        with open(self.base_dir / f"descriptor_count_growth_{self.run_id}.txt", "r") as f:
            num_batches = len(f.readlines())
        docs_processed = num_batches * self.batch_size
        checkpoint_dir = self.base_dir / f"checkpoint_{docs_processed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through the files in the directory
        for item_path in self.base_dir.iterdir():
            if item_path.is_file():
                # Destination path in the checkpoint directory
                destination_path = checkpoint_dir / item_path.name
                # Copy the file
                shutil.copy2(item_path, destination_path)
               
    @staticmethod 
    def remove_explanations(list_of_descriptors):
        return [
            descriptor.split(";")[0] if ";" in descriptor else descriptor 
            for descriptor in list_of_descriptors
            ]

    def pipeline(self):

        self.llm = self.LLM_setup()
        logging.info("Model loaded.")
        data = load_documents(self.data_source, self.cache_dir)
        logging.info("Data loaded.")
        
        # The text needs to be in a column called "text".
        # If not, give the column name in --text-column arg and it will be renamed to "text"
        if self.text_column != "text":
            data = data.rename_column(self.text_column, "text")

        descriptor_path = self.base_dir / f"descriptor_vocab_{self.run_id}.tsv"

        for batch_num, batch in enumerate(self.batched(data)):
            if self.num_batches == 0:
                break
            logging.info(f"===============New Batch: {batch_num}===============")
            start = time.time()
            
            # Initialise empty results dictionary
            results = init_results(batch)

            # Generate initial descriptors for document.
            documents = [doc["text"] for doc in batch]
            logging.info("Stage: initial.")
            model_outputs = self.initial_stage(documents)

            # Extract output and append to results.
            general_descriptors = [
                output.get("descriptors", "Generation failed.") for output in model_outputs
            ]
            specific_descriptors = [
                output.get("specifics", "Generation failed.") for output in model_outputs
            ]
            self.update_results(results, general=general_descriptors, specific=specific_descriptors)

            # Generate rewrites of the document based on descriptors.
            # After the rewrite, we revise the descriptors to create an even better rewrite.
            for round_num in range(self.num_rewrites):
                # Remove explanations from descriptors for rewriting.
                general_without_explanations = [
                    self.remove_explanations(g) for g in general_descriptors
                ]
                specific_without_explanations = [
                    self.remove_explanations(s) for s in specific_descriptors
                ]
                # Rewrite doc based on the descriptors.
                logging.info(f"Stage: rewrite {round_num+1}.")
                model_outputs = self.rewrite_stage(
                    general_without_explanations, specific_without_explanations
                )

                # Extract output and append to results.
                rewrites = [
                    output.get("document", "Generation failed.")
                    for output in model_outputs
                ]
                self.update_results(results, rewrites=rewrites)

                if not round_num == self.num_rewrites - 1:
                    # Evaluate rewrite and revise descriptors.
                    # This stage is skipped on the last round: since we do not do another rewrite
                    # we do not need another set of descriptors.
                    logging.info(f"Stage: revise {round_num+1}.")
                    model_outputs = self.revise_stage(
                        documents,
                        rewrites,
                        general_descriptors,
                        specific_descriptors,
                    )

                    # Extract output and append to results.
                    general_descriptors = [
                        output.get("descriptors", "Generation failed.")
                        for output in model_outputs
                    ]
                    specific_descriptors = [
                        output.get("specifics", "Generation failed.")
                        for output in model_outputs
                    ]
                    self.update_results(results, general=general_descriptors, specific=specific_descriptors)

            # Calculate similarity between rewrites and original.
            # Append to results.
            for index in results:
                similarities = self.embedder.calculate_similarity(
                    results[index]["document"], results[index]["rewrite"]
                )
                results[index]["similarity"].extend(similarities)

            # Save all results so far
            save_results(results, self.base_dir, self.run_id, only_best=False)
            logging.info("Results saved.")
            
            # Update the descriptor vocabulary with new descriptors
            self.update_descriptor_vocab(descriptor_path)
            
            # Log execution time
            end = time.time()
            execution_time = end - start
            logging.info(f"Processing batch took {time.strftime('%H:%M:%S', time.gmtime(execution_time))}.")
            
            if self.checkpoint_interval > 0:
                if batch_num > 0 and (batch_num + 1) % self.checkpoint_interval == 0:
                    self.make_checkpoint()
                    logging.info(f"Checkpoint created after {batch_num + 1} batches.")
            
            # Stop iterating through new data after num_batches batches have been processed.
            if self.num_batches == -1:
                continue
            elif batch_num + 1 >= self.num_batches:
                break

def main(args):

    dg = DescriptorGenerator(args)
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
        default=3,
        help="How many rewriting cycles the script should go through.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Model temperature."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of documents given to the model at one time.",
    )
    parser.add_argument(
        "--data-source", type=str, default="fineweb", help="Which data set to process."
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Name of the text column in the dataset."
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help="Number of batches after which all results so far will be saved into a checkpoint. "
        "If 0, no checkpoints are created. Default: 0 (no checkpoints)."
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
