import torch  # type: ignore
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.preprocessing import normalize  # type: ignore


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

    def calculate_similarity(self, original, rewrites):
        sts_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "

        original = [sts_prompt + original]
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
