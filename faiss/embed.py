from transformers import AutoModel, AutoTokenizer  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from tqdm import tqdm  # type: ignore


class StellaEmbedder:
    def __init__(self, cache_dir, batch_size=64):
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModel.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=cache_dir
            )
            .to(device)
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.batch_size = batch_size

    def embed_descriptors(self, texts):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i : i + self.batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.model.device)
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
    
