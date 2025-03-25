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
import re
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
import prompts
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
        self.max_vocab = args.max_vocab
        self.synonym_threshold = args.synonym_threshold
        self.data_source = args.data_source
        self.temperature = args.temperature
        self.base_dir = Path(f"../results") / self.run_id
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

    def fix_json(self, message):
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            max_tokens=4_000,  # max tokens to generate
        )

        output = self.llm.generate(
            message, sampling_params=sampling_params, use_tqdm=False
        )

        return output[0].outputs[0].text.strip(" `\n").removeprefix("json")

    def generate(self, input, stage):
        if stage != "fix":
            response_schema = self.get_response_format(stage)
            max_tokens = {
                "initial": 500,
                "rewrite": 3000,
                "revise": 4000,
                "synonyms": 1000,
            }
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=0.5,
                repetition_penalty=1,  # 1 = no penalty, >1 penalty
                max_tokens=max_tokens[stage],  # max tokens to generate
                guided_decoding=response_schema,
            )

            outputs = self.llm.generate(
                input, sampling_params=sampling_params, use_tqdm=False
            )

            return [out.outputs[0].text.strip(" `\n").removeprefix("json") for out in outputs]

        else:
            return self.fix_json(input)

    def reformat_output(self, output):
        logging.warning("Fixing JSON formatting.")
        stage = "fix"
        for i in range(2):
            # Remove newlines and spaces from start and end
            output = output.strip(" \n")
            # Remove any text outside curly brackets
            json_start = output.find("{")
            json_end = output.find("}")
            if json_start != -1 and json_end != -1:
                output = output[json_start : json_end + 1]  # Include the '}'
            # Remove trailing commas
            output = re.sub(r",\s*([\]}])", r"\1", output)
            # Add double quotes around keys (if missing)
            output = re.sub(
                r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', output
            )
            # Add double quotes around string values (if missing)
            output = re.sub(r':\s*([^"\s\d\[{].*?)(?=[,}\]])', r':"\1"', output)
            valid_json = self.validate_output(output)
            if valid_json:
                logging.warning(f"Fixed JSON formatting with {i} LLM call(s).")
                return json.loads(output, strict=False)

            # If fixing JSON with regex does not work,
            # we try giving it to the model to fix.
            # This takes quite a lot of time (~1-2 min/attempt), so consider lowering
            # number of attempts to speed up the process.
            prompt = self.format_prompt(stage, original=output)
            output = self.generate(prompt, stage)
            valid_json = self.validate_output(output)
            if valid_json:
                logging.warning(f"Fixed JSON formatting with {i+1} LLM call(s).")
                return json.loads(output, strict=False)

        # If fixing does not work, save the malformed JSON to disk for later inspection.
        # Return "FAIL"
        logging.warning("Failed to fix JSON formatting.")
        with open("../results/malformed_JSON_output.txt", "a") as f:
            f.write(f"{output}\n======================\n")
        return "FAIL"

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
            prompts.append(self.format_prompt(stage=stage, general=g, specific=s))
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output) for output in batched_output
        ]
        
        return validated_outputs

    @log_execution_time
    def revise_stage(self, document, rewritten, general, specific, vocab):
        """
        Processes lists of descriptors and rewrites through the revise generation stage.

        Args:
            stage (str): The current stage of the pipeline.
            document (list of str): The original documents.
            rewritten (list of str): The rewritten documents.
            general (list of list): General descriptors for the documents.
            specific (list of list): Specific descriptors for the documents.
            vocab (list of str): The current general descriptor vocabulary.
            llm (object): The language model used for generating outputs.

        Returns:
            list of dict: A list of validated outputs, each containing general and specific descriptors.
        """
        stage = "revise"
        vocab = "\n".join(vocab)
        prompts = []
        for d, r, g, s in zip(document, rewritten, general, specific):
            prompts.append(
                self.format_prompt(
                    stage=stage,
                    original=d,
                    rewritten=r,
                    general=g,
                    specific=s,
                    vocab=vocab,
                )
            )
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output) for output in batched_output
        ]
        
        return validated_outputs

    @log_execution_time
    def synonym_stage(
        self,
        best_results,
        best_descriptors,
        synonym_threshold,
    ):
        # Load full vocabulary (if it exists) and append it to this round of descriptors
        stage = "synonyms"
        try:
            with open(self.base_dir / f"descriptor_vocab_{self.run_id}.tsv", "r") as f:
                file = f.readlines()
                descriptors = [line.split("\t")[0] for line in file] + best_descriptors
        except FileNotFoundError:
            descriptors = best_descriptors

        # Embed best descriptors
        embeddings = self.embedder.embed_descriptors(descriptors)

        # Group similar descriptors
        synonyms = self.find_synonyms(
            descriptors, embeddings, synonym_threshold, save_groups=False
        )

        # Use LLM to evaluate and form final synonyms
        prompts = [
            self.format_prompt(stage=stage, group_name=group_name, synonyms=syns)
            for group_name, syns in synonyms.items()
        ]

        # Since the number of synonym groups can grow quite large,
        # we split the prompts into batches
        validated_outputs = []
        for prompt_batch in self.batched(prompts, batch_size=200, start_index=0):
            batched_output = self.generate(prompt_batch, stage)
            validated_outputs.extend([self.validate_output(output) for output in batched_output])

        # Make dictionary from LLM outputs
        synonyms = defaultdict(list)
        for d in validated_outputs:
            for key, value in d.items():
                synonyms[key].extend(value)

        # Replace the original descriptors
        self.replace_synonyms(synonyms, best_results)
        save_synonym_dict(synonyms, self.base_dir, self.run_id)

        # Also update the descriptors of previously processed and saved documents
        try:
            with open(
                self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl", "r"
            ) as f:
                prev_results = {}
                file = [json.loads(line.strip()) for line in f.readlines()]
                for idx, doc in enumerate(file):
                    prev_results[idx] = doc

            self.replace_synonyms(synonyms, prev_results)
            with open(
                self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl", "w"
            ) as f:
                for doc in prev_results.values():
                    json_line = json.dumps(doc, ensure_ascii=False)
                    f.write(json_line + "\n")
        except FileNotFoundError:
            pass

        return best_results

    def update_descriptor_vocab(self, results, descriptor_path):
        results = pd.read_json(
            self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl",
            lines=True,
            orient="records",
        )
        desc_counts = Counter()
        for descs in results["general"].to_list():
            desc_counts.update(descs)

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
        general=None,
        specific=None,
        vocab=None,
        group_name=None,
        synonyms=None,
    ):
        # Truncate original document to max length
        if original:
            original = self.tokenize_and_truncate(original, 100_000)

        if stage == "initial":
            return prompts.initial_prompt(original, vocab)
        elif stage == "rewrite":
            return prompts.rewrite_prompt(general, specific)
        elif stage == "revise":
            return prompts.revise_keyphrases_prompt(
                original, rewritten, general, specific, vocab
            )
        elif stage == "synonyms":
            return prompts.review_synonyms(group_name, synonyms)
        elif stage == "fix":
            return prompts.reformat_output_prompt(original)

    def get_response_format(self, stage):
        if stage == "initial":

            class ResponseFormat(BaseModel):
                general: list[str]
                specific: list[str]

        elif stage == "rewrite":

            class ResponseFormat(BaseModel):
                document: str

        elif stage == "revise":

            class ResponseFormat(BaseModel):
                differences: str
                general: list[str]
                specific: list[str]

        elif stage == "synonyms":

            class ResponseFormat(RootModel):
                root: dict[str, list[str]]

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

    def find_synonyms(
        self, descriptors, embeddings, distance_threshold, save_groups=False
    ):
        # Convert embeddings to NumPy array if needed
        embeddings = np.array(embeddings)

        # Identify and remove zero vectors
        valid_indices = [
            i for i, vec in enumerate(embeddings) if np.linalg.norm(vec) > 0
        ]

        descriptors = [descriptors[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Group words by cluster labels
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)

        # Find the medoid for each group
        group_dict = {}
        for label, indices in groups.items():
            group_vectors = embeddings[indices]
            distance_matrix = cdist(group_vectors, group_vectors, metric="cosine")
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid_word = descriptors[indices[medoid_index]]

            # Store the group with the medoid as the key
            # Remove duplicates in each group
            group_dict[medoid_word] = list(set([descriptors[idx] for idx in indices]))

        if save_groups:
            # Save groups for later inspection
            with open("../results/synonyms.jsonl", "a") as f:
                f.write(json.dumps(group_dict, ensure_ascii=False) + "\n")

        return group_dict

    def replace_synonyms(self, synonyms, results):
        # Create a mapping from descriptor to its synonym for fast lookup
        synonym_map = {
            syn: group for group, members in synonyms.items() for syn in members
        }
            
        # Replace synonyms and remove possible duplicates
        for doc in results.values():
            replaced = []
            for descriptor in doc["general"]:
                if descriptor in synonym_map:
                    if descriptor not in replaced:
                        replaced.append(synonym_map[descriptor])
                else:
                    synonyms[descriptor] = [descriptor]
                    if descriptor not in replaced:
                        replaced.append(descriptor)
            doc["general"] = replaced

    def update_results(self, results, general=None, specific=None, rewrites=None):
        """Updates results dictionary with new descriptors or rewrites."""
        for index in results:
            if general:
                results[index]["general"].append(general[index])
            if specific:
                results[index]["specific"].append(specific[index])
            if rewrites:
                results[index]["rewrite"].append(rewrites[index])

    def generate_final_rewrites(self):
        with open(
            self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl", "r"
        ) as f:
            results = {}
            file = [json.loads(line.strip()) for line in f.readlines()]
            for idx, doc in enumerate(file):
                results[idx] = doc

        # Process in batches of batch_size
        for batch_num, batch in enumerate(
            self.batched(results, self.batch_size, start_index=0)
        ):
            logging.info(
                f"Batch {batch_num+1} out of {math.ceil(len(results)/self.batch_size)}"
            )
            general_descriptors = []
            specific_descriptors = []
            for index in batch:
                general_descriptors.append(results[index]["general"])
                specific_descriptors.append(results[index]["specific"])

            model_outputs = self.rewrite_stage(
                general_descriptors, specific_descriptors
            )
            rewrites = [
                output.get("document", "Generation failed.") for output in model_outputs
            ]
            for i, index in enumerate(batch):
                results[index]["rewrite"].append(rewrites[i])

            for index in batch:
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

    def pipeline(self):

        self.llm = self.LLM_setup()
        logging.info("Model loaded.")
        data = load_documents(self.data_source, self.cache_dir)
        logging.info("Data loaded.")

        # Load previous descriptors or create an empty dictionary
        descriptor_path = self.base_dir / f"descriptor_vocab_{self.run_id}.tsv"
        descriptor_counts = initialise_descriptor_vocab(descriptor_path)
        # Keep the top max_vocab general descriptors. These will be given to the model as possible options.
        descriptor_vocab = (
            descriptor_counts.most_common(self.max_vocab)
            if self.max_vocab != -1
            else descriptor_counts.most_common()
        )
        descriptor_vocab = [item[0] for item in descriptor_vocab]
        # Shuffle the list of descriptors to avoid ordering bias
        shuffle(descriptor_vocab)

        for batch_num, batch in enumerate(self.batched(data)):
            if self.num_batches == 0:
                break
            start_time = time.time()
            logging.info("================New Batch================")
            
            # Initialise empty results dictionary
            results = init_results(batch)

            # Generate initial descriptors for document.
            documents = [doc["text"] for doc in batch]
            logging.info("Stage: initial.")
            model_outputs = self.initial_stage(documents, descriptor_vocab)

            # Extract output and append to results.
            general_descriptors = [
                output.get("general", "Generation failed.") for output in model_outputs
            ]
            specific_descriptors = [
                output.get("specific", "Generation failed.") for output in model_outputs
            ]
            self.update_results(results, general=general_descriptors, specific=specific_descriptors)

            # Generate rewrites of the document based on descriptors.
            # After the rewrite, we revise the descriptors to create an even better rewrite.
            for round_num in range(self.num_rewrites):
                # Rewrite doc based on the descriptors.
                logging.info(f"Stage: rewrite {round_num+1}.")
                model_outputs = self.rewrite_stage(
                    general_descriptors, specific_descriptors
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
                        descriptor_vocab,
                    )

                    # Extract output and append to results.
                    general_descriptors = [
                        output.get("general", "Generation failed.")
                        for output in model_outputs
                    ]
                    specific_descriptors = [
                        output.get("specific", "Generation failed.")
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

            # Get the descriptors that produced the best rewrite
            best_descriptors = get_best_descriptors(results)
            # Get results from the round that produced the best rewrite
            best_results = get_best_results(results)

            # For best descriptors, find and combine synonyms
            # This will limit the number of unique descriptors
            logging.info("Stage: synonyms")
            best_results = self.synonym_stage(
                best_results, best_descriptors, self.synonym_threshold
            )

            # Save the new results with the new descriptors to a separate file.
            # Empty "rewrite" and "similarity" since they refer to text and scores
            # generated with the old descriptors.
            for index in best_results:
                best_results[index]["rewrite"] = []
                best_results[index]["similarity"] = []
            save_results(
                best_results,
                self.base_dir,
                run_id=self.run_id + "_syn_replaced",
                only_best=False,
            )

            # Update the descriptor vocabulary with new descriptors
            descriptor_vocab = self.update_descriptor_vocab(
                best_results, descriptor_path
            )

            end_time = time.time()

            logging.info(
                f"Processed {len(results)} documents in "
                f"{time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))}."
            )
            logging.info(
                f"Processed a total of {(batch_num+1)*self.batch_size} documents."
            )

            # Stop iterating through new data after num_batches batches have been processed.
            if self.num_batches == -1:
                continue
            elif batch_num + 1 >= self.num_batches:
                break

        logging.info("==============================================")
        logging.info("Generating rewrites with final descriptor set.")
        # When we have processed all batches or all data,
        # we do one more rewrite to see how much has changed due to synonym replacement.
        self.generate_final_rewrites()
        logging.info("Final results saved.")


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
