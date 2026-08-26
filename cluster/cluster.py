#!/usr/bin/env python3
"""
Cluster descriptors and auto-label clusters using vLLM.

Usage:
    python cluster_descriptors_vllm.py --input input.jsonl --output clusters.jsonl --llm-model Qwen/Qwen3-Next-80B-A3B-Instruct
"""

import argparse
import json
import logging
from logging import config
from pathlib import Path
from typing import List, Dict, Any
import gc

import torch
import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from vllm import LLM, SamplingParams
import random
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


random.seed(42)  # For reproducibility


def load_descriptors(input_path: Path) -> List[str]:
    """Load descriptors from JSONL file."""
    descriptors = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if data.get("similarity", None) is not None:
                    best_idx = np.argmax(data["similarity"])
                    descriptors.extend(data["descriptors"][best_idx])
                else:
                    descriptors.extend(data.get("descriptors", []))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")

    logger.info(f"Loaded {len(descriptors)} descriptors from {input_path}")

    # Deduplicate descriptors while preserving order
    unique_descriptors = []
    seen = set()
    for d in descriptors:
        if d not in seen:
            seen.add(d)
            unique_descriptors.append(d)
    descriptors = unique_descriptors
    logger.info(f"After deduplication, {len(descriptors)} unique descriptors remain.")

    return descriptors


def embed_descriptors(
    descriptors: List[str], model_name: str, cache_dir: Path | None = None
) -> np.ndarray:
    """Embed descriptors using SentenceTransformer with PyTorch attention."""

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device="cuda",
        cache_folder=str(cache_dir) if cache_dir else None,
        config_kwargs={
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        },
    )

    embeddings = model.encode(
        descriptors,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )

    # Clear GPU memory after embedding and garbace collection
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings


def cluster_embeddings(embeddings: np.ndarray) -> hdbscan.HDBSCAN:
    """Cluster embeddings with HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusterer.fit(embeddings)
    return clusterer


def prompt_template(descriptors: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are an expert at categorizing document descriptors. The descriptors describe the contents of web documents.\n"
            "You are given examples from a cluster of descriptors. Your task is to generate a concise label to describe the cluster.\n"
            "Answer with a single word or phrase that best captures the common theme of the descriptors. Do not provide any additional explanation or preamble.\n",
        },
        {
            "role": "user",
            "content": f"Examples:\n{descriptors}",
        },
    ]


def build_prompt(examples: List[str]) -> str:
    """Format the prompt for LLM labeling."""
    examples_str = "\n".join(f"- {ex}" for ex in examples)
    prompt = prompt_template(examples_str)
    return prompt


def generate_llm_labels(
    clusters: Dict[int, Dict[str, Any]],
    llm: LLM,
    max_tokens: int = 20,
) -> Dict[int, str]:
    """Generate labels for clusters using vLLM."""
    prompts = []
    cluster_ids = []
    for cluster_id, data in clusters.items():
        if cluster_id == -1:  # Skip noise
            continue
        prompt = build_prompt(data["sample_descriptors"])
        prompts.append(prompt)
        cluster_ids.append(cluster_id)

    # Batch inference
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens)
    outputs = llm.chat(prompts, sampling_params)

    # Map outputs to clusters
    labels = {}
    for cluster_id, output in zip(cluster_ids, outputs):
        label = output.outputs[0].text.strip().replace('"', "")
        labels[cluster_id] = label
    return labels


def setup_llm(
    model: str,
    max_model_len: int,
    cache_dir: Path | None = None,
) -> LLM:
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "tensor_parallel_size": max(1, torch.cuda.device_count()),
        "enforce_eager": False,
        "gpu_memory_utilization": 0.9,
    }

    if cache_dir:
        llm_kwargs["download_dir"] = str(cache_dir) if cache_dir else None

    return LLM(**llm_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of descriptors to sample from each cluster for labeling",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="NovaSearch/stella_en_400M_v5",
        help="Sentence embedding model",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="vLLM model",
    )
    parser.add_argument("--cache-dir", type=Path, help="Cache directory for models")
    args = parser.parse_args()
    
    if not Path("clustered_descriptors.csv").exists():

        # Step 1: Load descriptors
        logger.info("Loading descriptors...")
        descriptors = load_descriptors(args.input)
        if not descriptors:
            raise ValueError("No descriptors found.")

        # Step 2: Embed descriptors
        logger.info("Embedding descriptors...")
        embeddings = embed_descriptors(
            descriptors, args.embedding_model, cache_dir=args.cache_dir
        )
        # Normalize embeddings to unit length for cosine similarity
        embeddings = normalize(embeddings)

        # Step 3: Reduce dimensions
        logger.info("Reducing dimensions with UMAP...")
        reducer = UMAP(n_components=10, metric="euclidean", random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Step 4: Cluster
        logger.info("Clustering embeddings...")
        clusterer = cluster_embeddings(reduced_embeddings)
        logger.info(f"Found {len(set(clusterer.labels_)) - 1} clusters.")

        # Step 5: Prepare clusters for LLM labeling
        df = pd.DataFrame({"text": descriptors, "cluster": clusterer.labels_})
        clusters = {}
        for cluster_id in df["cluster"].unique():
            if cluster_id == -1:
                continue
            texts = df[df["cluster"] == cluster_id]["text"].tolist()
            clusters[cluster_id] = {
                "size": len(texts),
                "sample_descriptors": random.sample(
                    texts, min(args.sample_size, len(texts))
                ),
            }

        # Save dataframe to CSV for inspection
        df.to_csv("clustered_descriptors.csv", index=False)
        
    else:
        logger.info("Loading clustered descriptors from CSV...")
        df = pd.read_csv("clustered_descriptors.csv")
        descriptors = df["text"].tolist()
        clusterer = hdbscan.HDBSCAN()
        clusterer.labels_ = df["cluster"].to_numpy()

        clusters = {}
        for cluster_id in df["cluster"].unique():
            if cluster_id == -1:
                continue
            texts = df[df["cluster"] == cluster_id]["text"].tolist()
            clusters[cluster_id] = {
                "size": len(texts),
                "sample_descriptors": random.sample(
                    texts, min(args.sample_size, len(texts))
                ),
            }

    # Step 6: Load LLM and generate labels
    logger.info("Loading LLM for labeling...")
    llm = setup_llm(model=args.llm_model, max_model_len=2048, cache_dir=args.cache_dir)
    logger.info("Generating labels...")
    cluster_labels = generate_llm_labels(clusters, llm)

    # Step 7: Save results
    logger.info("Saving results...")
    with args.output.open("w", encoding="utf-8") as f:
        for descriptor, label in zip(descriptors, clusterer.labels_):
            result = {
                "descriptor": descriptor,
                "cluster_id": int(label),
                "cluster_label": cluster_labels.get(label, "noise"),
            }
            f.write(json.dumps(result) + "\n")

    logger.info("Cluster labels:")
    # Order the output by cluster size for better readability
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)
    for cluster_id, data in sorted_clusters:
        label = cluster_labels.get(cluster_id, "noise")
        logger.info(f"  Cluster {cluster_id}: {label} (n={data['size']})")


if __name__ == "__main__":
    main()
