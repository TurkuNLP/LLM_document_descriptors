from faiss_index import FaissIndex  # type: ignore
from embed import StellaEmbedder
from document import Document, Descriptor  # type: ignore

import argparse
import os
import json
import numpy as np  # type: ignore
from collections import defaultdict
from typing import Any

"""
This script provides functionality to build a Faiss index from a collection of documents with associated descriptors,
and to perform similarity search on that index using query strings.
It supports loading precomputed embeddings, saving and loading the Faiss index,
and handling metadata for retrieved results.

On first run, use --build-index to create the index from the input data.
This will process the documents, extract descriptors, compute embeddings, and save the index and metadata.

Use --query to perform similarity search on the built index. You can specify multiple queries separated by commas.
On subsequent runs, you can skip --build-index if the index file already exists, and just use --query to search.
For querying, you need to decide how many top results to return with --top-k.
Optionally, you can set a maximum distance threshold with --max-distance to filter results.
Setting a max distance can lead to more or fewer results than specified with top-k, depending on the distribution of distances in the index.
A good max distance to start with is typically between 200 and 300, but this can vary based on the data and embedding model.

To force rebuilding the index (e.g., if the underlying data has changed),
use --force-rebuild to ignore existing index files and create a new one.
To force re-embedding descriptors (e.g., if the embedding model has changed),
use --force-reembed to ignore existing embeddings and recompute them.
If you re-embed the data, you should also rebuild the index
to ensure the index is consistent with the new embeddings.

Base usage:
1. Build index:
python search.py --data-path path/to/documents.jsonl
                 --cache-dir path/to/cache
                 --build-index
                 --index-type IndexIVFFlat
                 --nlist 100
                 --dimension 1024
2. Search:
python search.py --query "example query","another query"
                 --cache-dir path/to/cache
                 --index-path path/to/index.faiss
                 --embeddings-path path/to/embeddings.npy
                 --top-k 5
                 --nprobe 10
                 --output-path path/to/results.jsonl
"""


def load_documents(args) -> list[Document]:
    documents: list[Document] = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        if args.descriptor_type == "raw":
            print(
                "Using 'raw' descriptors. "
                "Set --descriptor-type to 'harmonized' to use harmonized descriptors, if available.",
                flush=True,
            )
        for i, line in enumerate(f):
            obj = json.loads(line.strip())

            # Get text content, trying "text" first and falling back to "document" if "text" is not present.
            text = obj.get("text") or obj.get("document")

            # If no explicit doc_id is provided, use the line number as a fallback ID.
            doc_id = obj.get("doc_id") or str(i)

            # Use "harmonized_descriptors" if available and descriptor_type is "harmonized",
            # otherwise use "descriptors".
            if args.descriptor_type == "raw":
                descriptor_explainers = obj.get("descriptors", [])
            else:
                if "harmonized_descriptors" not in obj:
                    print(
                        f"Warning: --descriptor-type set to 'harmonized' but 'harmonized_descriptors' not found for document {i}, doc_id: {doc_id}. "
                        "Falling back to 'descriptors'.",
                        flush=True,
                    )
                descriptor_explainers = obj.get("harmonized_descriptors") or obj.get(
                    "descriptors", []
                )

            if not descriptor_explainers:
                print(
                    f"Warning: No descriptors found for document {i}, doc_id: {doc_id}. Skipping.",
                    flush=True,
                )
                continue
            # If "similarity" scores are present and descriptor_explainers is a list of lists
            # select the best set of descriptors based on the highest similarity score.
            if (
                "similarity" in obj
                and isinstance(descriptor_explainers, list)
                and any(isinstance(d, list) for d in descriptor_explainers)
            ):
                best_idx = np.argmax(obj["similarity"])
                descriptor_explainers = descriptor_explainers[best_idx]
            descriptors = []
            for j, d_e in enumerate(descriptor_explainers):
                assert isinstance(
                    d_e, str
                ), f"Expected descriptor to be a string but got {type(d_e)} in document {doc_id}"
                if ";" in d_e:
                    descriptor, explainer = d_e.split(";", 1)
                    descriptor = descriptor.strip()
                    explainer = explainer.strip()
                else:
                    descriptor = d_e.strip()
                    explainer = ""
                descriptors.append(
                    Descriptor(
                        descriptor_id=doc_id + "_" + str(j),
                        descriptor=descriptor,
                        explainer=explainer,
                    )
                )
            documents.append(
                Document(doc_id=doc_id, text=text, descriptors=descriptors)
            )
            if args.max_docs and len(documents) >= args.max_docs:
                break

    return documents


def resolve_metadata_path(args):
    if args.metadata_path:
        return args.metadata_path
    if args.index_path:
        return args.index_path + ".meta.json"
    raise ValueError(
        "Provide --metadata-path or --index-path so metadata can be saved/loaded."
    )


def build_descriptor_map(documents: list[Document]) -> dict[str, list[dict[str, str]]]:
    descriptor_map: dict[str, list[dict[str, str]]] = defaultdict(list)

    for doc in documents:
        seen_in_doc: set[str] = set()

        for descriptor in doc.descriptors:
            descriptor_text = descriptor.text

            # Avoid storing the same descriptor more than once for a single document.
            if descriptor_text in seen_in_doc:
                continue
            seen_in_doc.add(descriptor_text)

            descriptor_map[descriptor_text].append(
                {
                    "doc_id": doc.doc_id,
                    "descriptor_id": descriptor.descriptor_id,
                    "text": doc.text,
                }
            )

    return dict(descriptor_map)


def add_precomputed_embeddings(
    index: FaissIndex,
    descriptors: list[str],
    descriptor_map: dict[str, list[dict[str, str]]],
    embeddings: np.ndarray,
) -> None:
    for descriptor_text, embedding in zip(descriptors, embeddings):
        index.add(
            embedding,
            data={
                "descriptor": descriptor_text,
                "documents": descriptor_map[descriptor_text],
            },
        )


def search(
    index: FaissIndex,
    embedder: StellaEmbedder,
    query: str,
    top_k: int = 5,
    nprobe: int = 10,
):
    query_embedding = embedder.embed_descriptors([query])[0]
    return index.search(query_embedding, top_k=top_k, nprobe=nprobe)


def save_results(
    output_path: str,
    query: str,
    distances: np.ndarray,
    indices: np.ndarray,
    index: FaissIndex,
) -> None:
    result = {"query": query, "results": []}

    for distance, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx == -1:
            continue

        hit = index.id_to_data.get(idx)
        if hit is None:
            continue

        result["results"].append(
            {
                "distance": float(distance),
                "descriptor": hit["descriptor"],
                "documents": hit["documents"],
            }
        )

    if not output_path.endswith(".jsonl"):
        output_path += ".jsonl"
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")


def print_results(
    query: str, distances: np.ndarray, indices: np.ndarray, index: FaissIndex
) -> None:
    print(f"Query: {query}")

    found_any = False
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        idx = int(idx)
        if idx == -1:
            continue

        hit = index.id_to_data.get(idx)
        if hit is None:
            continue

        found_any = True
        print(f"[{rank}] Distance: {float(distance):.6f}")
        print(f"Descriptor: {hit['descriptor']}")
        print("Documents:")

        for doc in hit["documents"]:
            preview = doc["text"].replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  - doc_id={doc['doc_id']} " f"text={preview}")
        print()

    if not found_any:
        print("No matches found.")

    print("-" * 80)


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Build and query a Faiss index with Stella embeddings."
    )
    # Data arguments
    parser.add_argument("--data-path", type=str, help="Path to the document data.")
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to load for building the index. Mainly for testing.",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=["raw", "harmonized"],
        default="raw",
        help="Whether to use 'descriptors' or 'harmonized_descriptors' in the input data.",
    )

    # Search arguments
    parser.add_argument(
        "--output-path", type=str, help="Path to save the search results."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query string to search in the index. Separate multiple queries with comma.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top results to return for the query.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum distance threshold for search results. A reasonable value is typically between 200 and 300."
        "Note that setting max-distance might give more or fewer results than specified with top-k.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=10,
        help="Number of clusters to search over for the Faiss index.",
    )

    # Index arguments
    parser.add_argument(
        "--index-type",
        type=str,
        default="IndexIVFFlat",
        help="Type of the Faiss index.",
    )
    parser.add_argument(
        "--nlist", type=int, default=100, help="Number of clusters for the Faiss index."
    )
    parser.add_argument(
        "--index-path", type=str, help="Path to save/load the Faiss index."
    )
    parser.add_argument(
        "--build-index", action="store_true", help="Whether to build the index."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the index even if an existing index file is found.",
    )

    # Embedding arguments
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory for caching model files.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Dimension of the embedding vectors.",
    )
    parser.add_argument(
        "--embeddings-path", type=str, help="Path to save/load precomputed embeddings."
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Force re-embedding descriptors even if precomputed embeddings are found.",
    )

    return parser.parse_args()


def build_index(args):
    documents: list[Document] | None = None
    embeddings: np.ndarray | None = None

    print("Building index... Index type:", args.index_type)
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        if args.force_reembed:
            print(
                f"--force-reembed specified. Ignoring existing embeddings at {args.embeddings_path} and recomputing.",
                flush=True,
            )
            os.remove(args.embeddings_path)
        else:
            print(
                f"Loading precomputed embeddings from {args.embeddings_path}",
                flush=True,
            )
            embeddings = np.load(args.embeddings_path).astype(np.float32)

    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading documents from {args.data_path}", flush=True)
        documents = load_documents(args)
    else:
        raise FileNotFoundError(
            f"Data file not found ({args.data_path}). Please provide a valid data path to build the index."
        )

    descriptor_map = build_descriptor_map(documents)
    descriptors = list(descriptor_map.keys())

    if not descriptors:
        raise ValueError("No descriptors found after processing the input documents.")

    embedder = StellaEmbedder(args.cache_dir)
    index = FaissIndex(args.dimension, index_type=args.index_type, nlist=args.nlist)

    if embeddings is None:
        print("Computing embeddings for descriptors...", flush=True)
        embeddings = embedder.embed_descriptors(descriptors).astype(np.float32)

    if len(embeddings) != len(descriptors):
        raise ValueError(
            "Number of embeddings does not match number of unique descriptors. "
            "Precomputed embeddings must correspond exactly to the descriptor list."
        )

    if args.index_type == "IndexIVFFlat":
        print("Training Faiss index with descriptor embeddings...", flush=True)
        train_size = min(len(embeddings), max(args.nlist, len(embeddings) // 10, 10))
        train_vectors = embeddings[:train_size]
        index.train(train_vectors)

    print("Indexing descriptors into Faiss...", flush=True)
    add_precomputed_embeddings(index, descriptors, descriptor_map, embeddings)

    if args.index_path:
        index.save_index(args.index_path)
        print(f"Index saved to {args.index_path}", flush=True)

    if args.embeddings_path and not os.path.exists(args.embeddings_path):
        np.save(args.embeddings_path, embeddings)
        print(f"Embeddings saved to {args.embeddings_path}", flush=True)

    return index, embedder


def build_or_load_index(args):
    if args.build_index or args.force_rebuild:
        if (
            args.index_path
            and os.path.exists(args.index_path)
            and not args.force_rebuild
        ):
            print(
                f"--build-index specified but index file {args.index_path} already exists."
                f" Loading existing index instead of rebuilding."
                f" Use --force-rebuild to ignore the existing index and build a new one.",
                flush=True,
            )
            index = FaissIndex(
                args.dimension, index_type=args.index_type, nlist=args.nlist
            )
            index.load_index(args.index_path)
            embedder = StellaEmbedder(args.cache_dir)
        elif args.index_path and os.path.exists(args.index_path) and args.force_rebuild:
            print(f"--force-rebuild specified. Rebuilding the index.", flush=True)
            os.remove(args.index_path)
            if os.path.exists(args.index_path + ".meta.json"):
                os.remove(args.index_path + ".meta.json")
            index, embedder = build_index(args)
        else:
            index, embedder = build_index(args)
    else:
        print(
            "No --build-index flag specified. Attempting to load existing index.",
            flush=True,
        )
        embedder = StellaEmbedder(args.cache_dir)
        index = FaissIndex(args.dimension, index_type=args.index_type, nlist=args.nlist)

        if args.index_path and os.path.exists(args.index_path):
            index.load_index(args.index_path)
        else:
            raise FileNotFoundError(
                f"Index file not found ({args.index_path}). "
                f"Please build the index first or provide valid index path."
            )

    return index, embedder


def main(args):
    if not args.build_index and not args.query:
        print(
            "No action specified. Use --build-index to build the index or --query to search.",
            flush=True,
        )
        return

    index, embedder = build_or_load_index(args)

    if args.query:
        print("Searching...")
        queries = [query.strip() for query in args.query.split(",") if query.strip()]
        if args.max_distance is not None:
            print(
                f"Maximum distance threshold is set. You might get more or fewer results than specified with --top-k ({args.top_k})",
                flush=True,
            )
        for query in queries:
            attempts = 0
            while True:
                top_k = args.top_k * (
                    2**attempts
                )  # Exponentially increase top_k with each attempt
                attempts += 1
                distances, indices = search(
                    index,
                    embedder,
                    query,
                    top_k=top_k,
                    nprobe=args.nprobe,
                )
                max_distance = args.max_distance
                # if greatest distance in results below the threshold, we increase top_k and try again to get more results
                # we keep going until we have at least one result above the distance threshold, or we have tried 10 times
                if (
                    max_distance is None
                    or (
                        len(distances[0]) > 0 and float(distances[0][-1]) > max_distance
                    )
                    or attempts >= 10
                ):
                    break
                else:
                    print(
                        f"All results below distance threshold {max_distance}. Expanding search and trying again...",
                        flush=True,
                    )
            # Remove results that are above the distance threshold (if specified)
            if max_distance is not None:
                filtered_results = [
                    (d, idx)
                    for d, idx in zip(distances[0], indices[0])
                    if float(d) <= max_distance
                ]
                if not filtered_results:
                    print(
                        f"No results found within the distance threshold of {max_distance} for query: {query}",
                        flush=True,
                    )
                    continue
                filtered_distances, filtered_indices = zip(*filtered_results)
                distances = np.array([filtered_distances], dtype=np.float32)
                indices = np.array([filtered_indices], dtype=np.int64)
            # Print results to console
            print_results(query, distances, indices, index)
            # Also save results to output file if specified
            if args.output_path:
                save_results(args.output_path, query, distances, indices, index)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    print("Done.")
