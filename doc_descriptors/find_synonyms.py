# Standard libraries
import argparse
from collections import defaultdict
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import warnings

# Third party imports
import json_repair  # type: ignore
import pandas as pd  # type: ignore
from pydantic import BaseModel, RootModel  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from sklearn.cluster import AgglomerativeClustering, HDBSCAN  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# Local imports
from embed import StellaEmbedder
import prompts
from utils import (
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


class SynonymFinder:
    
    def __init__(self, args):
        for attr in [
            "data_path",
            "model",
            "start_index",
            "run_id",
            "batch_size",
            "synonym_threshold",
            "temperature",
            "clustering_algorithm",
            ]:
            setattr(self, attr, getattr(args, attr))
        self.cache_dir = args.cache_dir or os.environ["HF_HOME"]
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
            "rewrite": 4000,
            "synonyms": 4000,
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
    
    def find_synonyms(
        self, descriptors, embeddings, distance_threshold, save_groups=False
    ):
        # Convert embeddings to NumPy array if needed
        embeddings = np.array(embeddings)

        if self.clustering_algorithm == "agglomerative":
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
        
        elif self.clustering_algorithm =="hdbscan" or self.clustering_algorithm =="hbdscan":
            clustering = HDBSCAN(min_cluster_size=2)
            labels = clustering.fit_predict(embeddings)

            groups = {}
            for idx, label in enumerate(labels):
                if label == -1:  # -1 is noise
                    continue
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
            
    def format_prompt(
        self,
        stage,
        general=None,
        specific=None,
        group_name=None,
        synonyms=None,
    ):
        if stage == "rewrite":
            return prompts.rewrite_prompt(general, specific)
        elif stage == "synonyms":
            return prompts.review_synonyms(group_name, synonyms)


    def get_response_format(self, stage):

        if stage == "rewrite":

            class ResponseFormat(BaseModel):
                document: str

        elif stage == "synonyms":

            class ResponseFormat(RootModel):
                root: dict[str, list[str]]

        json_schema = ResponseFormat.model_json_schema()
        return GuidedDecodingParams(json=json_schema)
    
    def load_data(self):
        data = {}
        with open(self.data_path, "r") as f:
            for idx, line in enumerate(f):
                line = json.loads(line, strict=False)
                data[idx] = line
                
        return data

    def synonym_stage(self, descriptors, best_results):
        stage = "synonyms"
        # Embed best descriptors
        embeddings = self.embedder.embed_descriptors(descriptors)

        # Group similar descriptors with agglomerative clustering
        synonyms = self.find_synonyms(
            descriptors, embeddings, self.synonym_threshold, save_groups=False
        )

        # Format synonym groups into LLM prompts
        prompts = [
            self.format_prompt(stage=stage, group_name=group_name, synonyms=syns)
            for group_name, syns in synonyms.items()
        ]
    
        # Use LLM to evaluate and form final synonyms
        # Since the number of synonym groups can grow quite large,
        # we split the prompts into batches
        validated_outputs = []
        prompt_batch = []
        for idx, prompt in enumerate(prompts):
            prompt_batch.append(prompt)
            if len(prompt_batch) == 200 or idx+1 == len(prompts):    
                batched_output = self.generate(prompt_batch)
                validated_outputs.extend([self.validate_output(output) for output in batched_output])
                prompt_batch = []
                
        # Make dictionary from LLM outputs
        synonyms = defaultdict(list)
        for d in validated_outputs:
            for key, value in d.items():
                synonyms[key].extend(value)

        # Replace the original descriptors
        self.replace_synonyms(synonyms, best_results)
        
        save_synonym_dict(synonyms, self.base_dir, self.run_id)

        return best_results
    
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
    
    def generate_final_rewrites(self, data):
        num_docs = len(data)
        
        batch_num = 0
        results = {}
        idx = 0
        for doc in data:
            idx += 1
            results[idx] = doc
            if idx == self.batch_size:
                batch_num += 1
                logging.info(f"Processing batch {batch_num} out of "
                                f"{math.ceil(num_docs/self.batch_size)}.")
                general_descriptors = []
                specific_descriptors = []
                for index in results:
                    general_descriptors.append(results[index]["general"])
                    specific_descriptors.append(results[index]["specific"])

                # Generate rewrites
                model_outputs = self.rewrite_stage(
                    general_descriptors, specific_descriptors
                )
                rewrites = [
                    output.get("document", "Generation failed.") for output in model_outputs
                ]
                
                # Add to results
                for i, index in enumerate(results):
                    results[index]["rewrite"] = [rewrites[i]]

                # Calculate cosine similarity
                for index in results:
                    similarities = self.embedder.calculate_similarity(
                        results[index]["document"], results[index]["rewrite"]
                    )
                    results[index]["similarity"] = [similarities]
                
                # Save batch
                with open(
                    self.base_dir / f"descriptors_{self.run_id}_final.jsonl", "a"
                    ) as f:
                        for doc in results.values():
                            json_line = json.dumps(doc, ensure_ascii=False)
                            f.write(json_line + "\n")
                idx = 0
                results = {}

    def synonym_pipeline(self):
        self.llm = self.LLM_setup()
        data = self.load_data()
        best_results = get_best_results(data)
        best_descriptors = get_best_descriptors(data)
        syn_replaced = self.synonym_stage(best_descriptors, best_results)
        with open(
            self.base_dir / f"descriptors_{self.run_id}_syn_replaced.jsonl", "a") as f:
                for doc in syn_replaced.values():
                    json_line = json.dumps(doc, ensure_ascii=False)
                    f.write(json_line + "\n")
        self.generate_final_rewrites(syn_replaced)
        
def main(args):
    sf = SynonymFinder(args)
    sf.synonym_pipeline()
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="A script for getting document descriptors with LLMs."
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Which data set to process."
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
        "--temperature", type=float, default=0, help="Model temperature."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents given to the model at one time.",
    )
    parser.add_argument(
        "--synonym-threshold",
        type=float,
        default=0.2,
        help="""Distance threshold for when two descriptors should count as synonyms.
        Smaller value means words are less likely to count as synonyms.""",
    )
    parser.add_argument(
        "--clustering-algorithm",
        type=str,
        default="agglomerative",
        help="""Which clustering algorithm to use. Choose either 'agglomerative' or 'hdbscan'.""",
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
