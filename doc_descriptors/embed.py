import torch # type: ignore
from transformers import AutoModel, AutoTokenizer
import numpy as np


class StellaEmbedder:
    def __init__(self, batch_size=32):
        self.model = (
            AutoModel.from_pretrained(
                "Marqo/dunzhang-stella_en_400M_v5",
                trust_remote_code=True,
                cache_dir = "../.cache"
            )
            .cuda()
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Marqo/dunzhang-stella_en_400M_v5",
            trust_remote_code=True,
            cache_dir = "../.cache"
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