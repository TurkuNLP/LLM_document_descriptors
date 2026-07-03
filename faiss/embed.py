import numpy as np
from sentence_transformers import SentenceTransformer
import os


class StellaEmbedder:
    def __init__(self, cache_dir, batch_size=64, max_length=512):
        model_name = "NovaSearch/stella_en_400M_v5"

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            trust_remote_code=True,
            config_kwargs={
                "use_memory_efficient_attention": False,
                "unpad_inputs": False,
            },
        )
        self.model.eval()
        self.model.max_seq_length = max_length

        self.batch_size = batch_size
        self.max_length = max_length

    def embed_descriptors(self, texts):
        texts = ["" if t is None else str(t) for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)


if __name__ == "__main__":
    argv = os.sys.argv
    if len(argv) > 1:
        texts = argv[1:]
    else:
        texts = ["Hello world!", "This is a test.", "Embedding text with Stella."]

    cache_dir = os.getenv("HF_HOME", "./hf_cache")
    embedder = StellaEmbedder(cache_dir=cache_dir)
    embeddings = embedder.embed_descriptors(texts)
    print(embeddings.shape)
