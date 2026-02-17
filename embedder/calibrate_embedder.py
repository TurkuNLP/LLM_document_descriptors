from embed import StellaEmbedder
from datasets import load_dataset  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore


data = load_dataset("ryanbaker/synonyms_1K")
embedder = StellaEmbedder()


min_similarities = []
max_similarities = []
mean_similarities = []
for syn_list in data["train"]["synonyms"]:
    embeddings = embedder.embed_descriptors(syn_list)
    similarity_matrix = cosine_similarity(embeddings)
    # Mask the diagonal elements
    masked_similarity_matrix = np.ma.masked_where(
        np.eye(similarity_matrix.shape[0]), similarity_matrix
    )

    # Calculate average, minimum, and maximum cosine similarity ignoring identity matches
    mean_similarities.append(np.mean(masked_similarity_matrix))
    min_similarities.append(np.min(masked_similarity_matrix))
    max_similarities.append(np.max(masked_similarity_matrix))

print("=" * 20)
print(f"Average Cosine Similarity: {np.mean(mean_similarities):.4f}")
print(f"Minimum Cosine Similarity: {np.mean(min_similarities):.4f}")
print(f"Maximum Cosine Similarity: {np.mean(max_similarities):.4f}")
print("=" * 20)
