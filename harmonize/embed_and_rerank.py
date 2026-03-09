# Standard library imports
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, List, Sequence, Tuple

# Third party imports
import numpy as np  # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.inputs.data import TokensPrompt  # type: ignore
from logging_utils import log_execution_time  # type: ignore


class QwenReranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        number_of_gpu = torch.cuda.device_count()
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=number_of_gpu,
            max_model_len=10000,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.7,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length = 8192
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

    def format_instruction(self, instruction, query, doc):
        text = [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]
        return text

    def process_inputs(self, pairs, instruction, max_length, suffix_tokens):
        messages = [
            self.format_instruction(instruction, query, doc) for query, doc in pairs
        ]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:max_length] + suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages

    def compute_logits(self, messages, sampling_params, true_token, false_token):
        outputs = self.model.generate(messages, sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            token_count = len(outputs[i].outputs[0].token_ids)
            if true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[true_token].logprob
            if false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores

    def rerank(self, queries, documents):
        instruct = 'Determine whether the Query is synonymous or near-synonymous to the Document. Answer "yes" if they are synonymous or near-synonymous, and answer "no" otherwise.'

        pairs = list(zip(queries, documents))
        inputs = self.process_inputs(
            pairs,
            instruct,
            self.max_length - len(self.suffix_tokens),
            self.suffix_tokens,
        )
        scores = self.compute_logits(
            inputs, self.sampling_params, self.true_token, self.false_token
        )
        return scores


# -----------------------------------------------------------------------------
# Embeddings (Stella)
# -----------------------------------------------------------------------------
class StellaEmbedder:
    """Minimal pooled embedding wrapper around Marqo/dunzhang-stella_en_400M_v5."""

    def __init__(
        self, cache_dir: Optional[Path], batch_size: int = 32, device: str = "cuda:0"
    ) -> None:
        cache_dir = os.environ.get("HF_HOME") or (cache_dir if cache_dir else None)
        model_name = "Marqo/dunzhang-stella_en_400M_v5"
        self.device = torch.device(device)
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            .to(self.device)
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.batch_size = batch_size

    @log_execution_time
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[i : i + self.batch_size])
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)[0]  # [B, T, H]
                attn = inputs["attention_mask"]  # [B, T]
                masked = outputs.masked_fill(~attn[..., None].bool(), 0.0)
                pooled = masked.sum(dim=1) / attn.sum(dim=1)[..., None]

                # Normalize to unit length for distance measuring
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.detach().cpu().numpy().astype("float32"))
        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.zeros((0, 1024), dtype="float32")
        )


class QwenEmbedder:
    def __init__(self, cache_dir, batch_size=32, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model = (
            SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                cache_folder=cache_dir,
                tokenizer_kwargs={"padding_side": "left"},
            )
            .to(self.device)
            .eval()
            .half()
        )
        self.batch_size = batch_size

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def embed_texts(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch_texts, convert_to_tensor=True, show_progress_bar=False
                )
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


def save_embeds(path: Path, emb: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    np.savez_compressed(
        str(path), emb=emb, meta=np.frombuffer(meta_bytes, dtype=np.uint8)
    )


def load_embeds(path: Path) -> Tuple[np.ndarray, dict]:
    with np.load(str(path), allow_pickle=False) as z:
        emb = z["emb"].astype("float32", copy=False)
        meta_bytes = bytes(z["meta"].tolist())
        meta = json.loads(meta_bytes.decode("utf-8"))
        return emb, meta


def embedding_fingerprint(pairs, model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for p in pairs:
        # include ids + text so any change invalidates cache
        h.update(b"\x00")
        h.update(p.id.encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((p.descriptor or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((p.explainer or "").encode("utf-8", "ignore"))
    return h.hexdigest()


def find_nn(embeddings: np.ndarray, query: np.ndarray, k: int = 1, dtype=torch.float16):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    M = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    M = torch.nn.functional.normalize(M, p=2, dim=1)
    N = M.shape[0]

    q = torch.as_tensor(query, device=device, dtype=dtype)
    if q.ndim == 1:
        q = q.unsqueeze(0)  # (1, D)
    q = torch.nn.functional.normalize(q, p=2, dim=1)  # (Q, D)

    chunk_size = 2048
    with torch.no_grad():
        vals, idxs = [], []
        for s in range(0, q.size(0), chunk_size):
            sims = q[s : s + chunk_size] @ M.T
            topv, topi = torch.topk(sims, k=min(k, N), dim=1)
            vals.append(topv.cpu().numpy())
            idxs.append(topi.cpu().numpy())

    return np.vstack(vals), np.vstack(idxs)
