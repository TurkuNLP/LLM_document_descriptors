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
from vllm_document_descriptors import DescriptorGenerator


def rewrite_and_get_similarity(dg, general, specific, batch):
    model_outputs = dg.rewrite_stage(general, specific)

    rewrites = [
        output.get("document", "Generation failed.")
        for output in model_outputs
    ]

    similarities = []
    for doc, rewrite in zip(batch, rewrites):
        similarities.extend(dg.embedder.calculate_similarity(
        doc["document"], rewrite)
        )

    return similarities

def main(args):

    dg = DescriptorGenerator(args)
    dg.llm = dg.LLM_setup()
    
    data = []
    
    # Load the documents and descriptors
    file_path = Path("../results/final_zero_vocab/descriptors_final_zero_vocab_final.jsonl")
    with file_path.open("r") as f:
        for line in f:
            doc = json.loads(line, strict=False)
            data_entry = {
                "document": doc["document"],
                "rewrite": doc["rewrite"][0],
                "similarity": doc["similarity"][0],
                "general": doc["general"],
                "specific": doc["specific"]
            }
            data.append(data_entry)
            if len(data) == 5_000:
                break
    
    similarities_with_all_descriptors = [doc["similarity"] for doc in data]
    similarities_without_specific = []
    similarities_without_general = []
    
    for batch in dg.batched(data):
        general = [doc["general"] for doc in batch]
        specific = [doc["specific"] for doc in batch]
        empty_general = [[] for _ in general]
        empty_specific = [[] for _ in specific]
        
        # Get the rewrites and similarities without specific descriptors
        similarities = rewrite_and_get_similarity(
            dg, general, empty_specific, batch
        )
        similarities_without_specific.extend(similarities)
        
        # Get the rewrites and similarities without general descriptors
        similarities = rewrite_and_get_similarity(
            dg, empty_general, specific, batch
        )
        similarities_without_general.extend(similarities)
        
        
    with open("../results/evaluations/similarity_scores_general_vs_specific.txt", "w") as f:
        f.write("Average similarity scores with all descriptors:\n")
        f.write(f"{np.mean(similarities_with_all_descriptors)}\n")
        f.write("Average similarity scores without specific descriptors:\n")
        f.write(f"{np.mean(similarities_without_specific)}\n")
        f.write("Average similarity scores without general descriptors:\n")
        f.write(f"{np.mean(similarities_without_general)}\n")



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
        required=False,
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

    main(args)