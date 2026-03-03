from pyexpat import model

import torch  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
import sys
import os
from sentence_transformers import SentenceTransformer  # type: ignore

# Script to embed text and calculate similarity scores for STS tasks.
# Use either Stella for English or Qwen for multilingual applications.
# Used as part of the descriptor generation process to evaluate the semantic similarity of rewrites against original text.
# Can also be used standalone in small scale to compute similarity scores  by providing texts as command-line arguments.


class StellaEmbedder:
    def __init__(self, cache_dir, batch_size=32):
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        self.model = (
            AutoModel.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=cache_dir
            )
            .cuda()
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.batch_size = batch_size

    def embed_descriptors(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                last_hidden_state = self.model(**inputs)[0]
                attention_mask = inputs["attention_mask"]
                last_hidden = last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )
                embeddings = (
                    last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                )
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def embed_for_sts(self, docs):
        with torch.no_grad():
            input_data = self.tokenizer(
                docs,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = self.model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            return normalize(vectors.cpu().numpy())

    def calculate_similarity(self, original, rewrites, use_prompt=False):
        sts_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "

        if use_prompt:
            original = [sts_prompt + original]
        else:
            original = [original]
        rewrites = [rewrites] if isinstance(rewrites, str) else rewrites

        # Embed original with prompt
        original_embeddings = self.embed_for_sts(original).reshape(
            1, -1
        )  # Ensure (1, D)
        rewrite_embeddings = self.embed_for_sts(rewrites).reshape(
            len(rewrites), -1
        )  # Ensure (N, D)

        similarities = (original_embeddings @ rewrite_embeddings.T).astype(np.float32)

        return [round(float(sim), 4) for sim in similarities[0]]


class QwenEmbedder:
    def __init__(self, cache_dir, batch_size=16):
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", cache_folder=cache_dir
        ).cuda().eval().half()
        self.batch_size = batch_size

    def calculate_similarity(self, original, rewrites, use_prompt=False):
        if isinstance(original, str):
            original = [original]
        if isinstance(rewrites, str):
            rewrites = [rewrites]

        query_embeddings = self.model.encode(original)
        document_embeddings = self.model.encode(rewrites)

        # Compute the (cosine) similarity between the query and document embeddings
        similarities = self.model.similarity(query_embeddings, document_embeddings)
        # Cosine similarity output is a tensor, convert to numpy and ensure it's a list of floats
        similarities = similarities.cpu().numpy().tolist()

        return [round(float(sim), 4) for sim in similarities[0]]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python embed.py <original_text> <semi-colon_separated_rewrite_texts>"
        )
        sys.exit(1)

    original_text = sys.argv[1]
    rewrites_str = sys.argv[2]
    rewrites = rewrites_str.split(";")

    # embedder = StellaEmbedder(cache_dir=os.environ["HF_HUB_CACHE"])
    embedder = QwenEmbedder(cache_dir=os.environ["HF_HUB_CACHE"])
    
    similarities = embedder.calculate_similarity(original_text, rewrites)

    # Print results
    print("Original Text:", original_text)
    for rewrite, sim in zip(rewrites, similarities):
        print(f"Rewrite: {rewrite}\nSimilarity: {sim}\n")
