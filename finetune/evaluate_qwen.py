#!/usr/bin/env python3
"""
Evaluate a fine-tuned generative LLM that produces document descriptors.

Metrics:
- Exact descriptor precision / recall / F1
- Soft token-level descriptor overlap
- ROUGE-L / ROUGE-1 / ROUGE-2, optional
- BERTScore, optional
- JSONL output with predictions

Install:
    pip install torch transformers datasets evaluate rouge_score bert_score tqdm

Example:
    python eval_descriptors.py \
        --model ./my-finetuned-model \
        --test_file ./test.jsonl \
        --output_file ./eval_predictions.jsonl \
        --max_new_tokens 256 \
        --batch_size 4
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch # type: ignore
from tqdm import tqdm # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore


DESCRIPTOR_SPLIT_RE = re.compile(r"\s*(?:,|;|\n|\|)\s*")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    return text.strip(" .,;:-")


def parse_descriptors(text_or_list: Any) -> List[str]:
    """
    Converts either a descriptor list or generated text into normalized descriptors.
    Handles comma-separated, semicolon-separated, newline-separated, and bullet-ish output.
    """
    if isinstance(text_or_list, list):
        raw_items = [str(x) for x in text_or_list]
    else:
        text = str(text_or_list).strip()

        # Remove common assistant framing.
        text = re.sub(
            r"^(descriptors?|keywords?|tags?)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        raw_items = DESCRIPTOR_SPLIT_RE.split(text)

    items = []
    seen = set()
    for item in raw_items:
        item = normalize_text(item)
        if item and item not in seen:
            seen.add(item)
            items.append(item)

    return items


def descriptor_set_scores(pred: List[str], ref: List[str]) -> Dict[str, float]:
    pred_set = set(pred)
    ref_set = set(ref)

    if not pred_set and not ref_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & ref_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(ref_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {"precision": precision, "recall": recall, "f1": f1}


def token_overlap_f1(pred: List[str], ref: List[str]) -> float:
    """
    A forgiving metric: compares tokens across all descriptors.
    Useful when the model says "academic tone" and the label says "academic style".
    """
    pred_tokens = set(" ".join(pred).split())
    ref_tokens = set(" ".join(ref).split())

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    tp = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def extract_prompt_and_reference(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Any]:
    """
    Supports two common formats:

    1. Chat format:
       {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    2. Document/descriptor format:
       {"document": "...", "descriptors": [...]}
    """
    if "messages" in row:
        messages = row["messages"]

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            raise ValueError("Chat row has no assistant message containing reference descriptors.")

        reference = assistant_msgs[-1]["content"]

        # Prompt excludes assistant answer.
        prompt_messages = []
        for m in messages:
            if m.get("role") == "assistant":
                break
            prompt_messages.append({"role": m["role"], "content": m["content"]})

        return prompt_messages, reference

    if "document" in row and "descriptors" in row:
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Generate a concise list of words and phrases that describe the "
                    "document's main meaning, tone, style, genre, and salient topics.\n\n"
                    f"Document:\n{row['document']}"
                ),
            }
        ]
        return prompt_messages, row["descriptors"]

    raise ValueError("Unsupported row format. Expected `messages` or `document` + `descriptors`.")


def build_inputs(tokenizer, batch_messages: List[List[Dict[str, str]]], device: str):
    """
    Uses the model tokenizer's chat template when available.
    Falls back to a simple prompt format if no chat template exists.
    """
    prompts = []

    for messages in batch_messages:
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for base models without chat templates.
            user_text = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            prompt = f"{user_text}\n\nDescriptors:"

        prompts.append(prompt)

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    return encoded, prompts


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
) -> List[Dict[str, Any]]:
    results = []

    for start in tqdm(range(0, len(rows), batch_size), desc="Generating"):
        batch_rows = rows[start : start + batch_size]

        batch_messages = []
        references = []

        for row in batch_rows:
            messages, reference = extract_prompt_and_reference(row)
            batch_messages.append(messages)
            references.append(reference)

        encoded, prompts = build_inputs(tokenizer, batch_messages, device)

        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only newly generated tokens, not the prompt.
        input_lengths = encoded["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_lengths:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for row, prompt, ref, pred_text in zip(batch_rows, prompts, references, decoded):
            ref_desc = parse_descriptors(ref)
            pred_desc = parse_descriptors(pred_text)

            set_scores = descriptor_set_scores(pred_desc, ref_desc)
            overlap_f1 = token_overlap_f1(pred_desc, ref_desc)

            results.append(
                {
                    "reference_text": ref if isinstance(ref, str) else ", ".join(map(str, ref)),
                    "prediction_text": pred_text.strip(),
                    "reference_descriptors": ref_desc,
                    "prediction_descriptors": pred_desc,
                    "exact_precision": set_scores["precision"],
                    "exact_recall": set_scores["recall"],
                    "exact_f1": set_scores["f1"],
                    "token_overlap_f1": overlap_f1,
                }
            )

    return results


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def add_optional_text_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Adds ROUGE and BERTScore if the required packages are installed.
    Hugging Face's `evaluate` library is the current home for metrics;
    older `datasets.load_metric` usage is deprecated. 
    """
    metrics = {}

    predictions = [", ".join(r["prediction_descriptors"]) for r in results]
    references = [", ".join(r["reference_descriptors"]) for r in results]

    try:
        import evaluate # type: ignore

        try:
            rouge = evaluate.load("rouge")
            rouge_result = rouge.compute(predictions=predictions, references=references)
            for k, v in rouge_result.items():
                metrics[k] = float(v)
        except Exception as e:
            metrics["rouge_error"] = str(e)

        try:
            bertscore = evaluate.load("bertscore")
            bert_result = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
            )
            metrics["bertscore_precision"] = mean(bert_result["precision"])
            metrics["bertscore_recall"] = mean(bert_result["recall"])
            metrics["bertscore_f1"] = mean(bert_result["f1"])
        except Exception as e:
            metrics["bertscore_error"] = str(e)

    except ImportError:
        metrics["evaluate_error"] = "Package `evaluate` is not installed."

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or Hugging Face model ID.")
    parser.add_argument("--test_file", required=True, help="JSONL test file.")
    parser.add_argument("--output_file", default="eval_predictions.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_text_metrics", action="store_true")
    args = parser.parse_args()

    do_sample = args.temperature > 0

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }

    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if not args.load_in_8bit:
        model.to(device)

    model.eval()

    rows = load_jsonl(args.test_file)

    results = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        device=device,
    )

    summary = {
        "n_examples": len(results),
        "exact_precision": mean([r["exact_precision"] for r in results]),
        "exact_recall": mean([r["exact_recall"] for r in results]),
        "exact_f1": mean([r["exact_f1"] for r in results]),
        "token_overlap_f1": mean([r["token_overlap_f1"] for r in results]),
    }

    if not args.skip_text_metrics:
        summary.update(add_optional_text_metrics(results))

    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Evaluation summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote per-example predictions to: {output_path}")


if __name__ == "__main__":
    main()