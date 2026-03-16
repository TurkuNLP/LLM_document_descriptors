from document import Document, Descriptor
from faiss_index import FaissIndex
from stella_embedder import StellaEmbedder
import numpy as np #type: ignore
import os
import json
from collections import defaultdict
from typing import Any

def build_index(args):
    documents: list[Document] | None = None
    embeddings: np.ndarray | None = None

    if args.embeddings_path and os.path.exists(args.embeddings_path):
        embeddings = np.load(args.embeddings_path).astype(np.float32)

    if args.data_path and os.path.exists(args.data_path):
        documents = load_documents(args.data_path, max_docs=args.max_docs)
    elif embeddings is None:
        raise ValueError(
            "No data found. Provide --data-path to build the descriptor metadata needed for retrieval."
        )

    if not documents:
        raise ValueError("No documents loaded. Cannot build descriptor index.")

    descriptor_map = build_descriptor_map(documents)
    descriptors = list(descriptor_map.keys())

    if not descriptors:
        raise ValueError("No descriptors found after processing the input documents.")
    
    embedder = StellaEmbedder(args.cache_dir)
    index = FaissIndex(args.dimension, index_type=args.index_type, nlist=args.nlist)

    if embeddings is None:
        embeddings = embedder.embed_descriptors(descriptors).astype(np.float32)

    if len(embeddings) != len(descriptors):
        raise ValueError(
            "Number of embeddings does not match number of unique descriptors. "
            "Precomputed embeddings must correspond exactly to the descriptor list."
        )

    if args.index_type == "IndexIVFFlat":
        train_size = min(len(embeddings), max(args.nlist, len(embeddings) // 10, 10))
        train_vectors = embeddings[:train_size]
        index.train(train_vectors)

    add_precomputed_embeddings(index, descriptors, descriptor_map, embeddings)

    if args.index_path:
        index.save_index(args.index_path)

    if args.embeddings_path and not os.path.exists(args.embeddings_path):
        np.save(args.embeddings_path, embeddings)

    return index, embedder