import argparse
import glob
import json
import os
import random
from typing import Iterator

from datasets import Dataset, Features, Value, Sequence  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch  # type: ignore


SYSTEM_PROMPT = """Annotate documents with a comprehensive list of descriptors — words or phrases that distill the meaning, tone, style, genre, topics, and other characteristics of the document.
Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, language, quality, sentiment etc. Describe anything that you believe is essential for understanding, summarizing, or rewriting the document.
The descriptors should be in English, even if the document is in another language.
Respond with a JSON object containing a single key "descriptors" whose value is a list of descriptors."""

USER_PROMPT = """<start_of_document>
{content}
<end_of_document>"""

CHARS_PER_TOKEN = 3  # Rough heuristic for quick truncation

HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")

documents_truncated = 0


def parse_descriptor_list(raw_descriptors: list[str]) -> list[str]:
    cleaned: list[str] = []
    for desc in raw_descriptors:
        if ";" not in desc:
            continue
        value = desc.split(";", 1)[0].strip()
        if value:
            cleaned.append(value)
    return cleaned


def drop_long_document(document: str, max_tokens: int, tokenizer, quick: bool) -> str:
    global documents_truncated

    if quick:
        char_limit = max_tokens * CHARS_PER_TOKEN
        if len(document) <= char_limit:
            return document
        documents_truncated += 1
        return ""
    else:
        token_ids = tokenizer.encode(document, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return document
        documents_truncated += 1
        return ""


def truncate_document(document: str, max_tokens: int, tokenizer, quick: bool) -> str:
    global documents_truncated

    if quick:
        # Quick truncation based on character count. ~20x faster than tokenization.
        # Less precise and may cut off in the middle of a word
        char_limit = max_tokens * CHARS_PER_TOKEN
        if len(document) <= char_limit:
            return document
        documents_truncated += 1
        return document[:char_limit] + "\n[TRUNCATED]"

    else:
        token_ids = tokenizer.encode(document, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return document

        # Truncate and decode back to text, ensuring we don't cut off in the middle of a token
        truncated = tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        documents_truncated += 1
        return truncated + "\n[TRUNCATED]"


def build_messages(
    document: str,
    descriptors: list[str],
    tokenizer,
    max_doc_tokens: int,
    quick_truncation: bool,
    drop_long_docs: bool = True,
) -> list[dict[str, str]]:

    if drop_long_docs:
        document = drop_long_document(
            document, max_doc_tokens, tokenizer, quick=quick_truncation
        )
    else:
        document = truncate_document(
            document, max_doc_tokens, tokenizer, quick=quick_truncation
        )

    # If the document is empty after dropping/truncation, return an empty list to skip this example
    if not document.strip():
        return []

    assistant_response = json.dumps(
        {"descriptors": descriptors},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(content=document)},
        {"role": "assistant", "content": assistant_response},
    ]


def expand_input_paths(path_spec: str) -> list[str]:
    if os.path.isdir(path_spec):
        paths = glob.glob(os.path.join(path_spec, "**", "*.jsonl"), recursive=True)
        if not paths:
            paths = glob.glob(os.path.join(path_spec, "*.jsonl"))
    else:
        paths = glob.glob(path_spec)
        if not paths and os.path.isfile(path_spec):
            paths = [path_spec]

    return sorted({path for path in paths if os.path.isfile(path)})


def example_generator(
    paths: list[str],
    tokenizer,
    max_doc_tokens: int,
    interleave_buffer_size: int,
    seed: int,
    shuffle_files: bool,
    quick_truncation: bool,
    drop_long_docs: bool,
) -> Iterator[dict]:
    rng = random.Random(seed)
    ordered_paths = list(paths)
    if shuffle_files:
        rng.shuffle(ordered_paths)

    def file_example_stream(path: str) -> Iterator[dict]:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON at {path}:{line_number}")
                    continue

                if not "text" in obj:
                    text = obj.get("document", "")
                else:
                    text = obj.get("text", "")
                raw_descriptors = obj.get("harmonized_descriptors", [])

                if not isinstance(text, str) or not isinstance(raw_descriptors, list):
                    continue

                descriptors = parse_descriptor_list(raw_descriptors)
                if not text or not descriptors:
                    # Skip examples with empty text or no valid descriptors after parsing
                    continue

                messages = build_messages(
                    document=text,
                    descriptors=descriptors,
                    tokenizer=tokenizer,
                    max_doc_tokens=max_doc_tokens,
                    quick_truncation=quick_truncation,
                    drop_long_docs=drop_long_docs,
                )

                if not messages:
                    continue
                yield {"messages": messages}

    # Initialize a stream for each file and interleave examples in batches
    active_streams: list[tuple[str, Iterator[dict]]] = [
        (path, file_example_stream(path)) for path in ordered_paths
    ]

    while active_streams:
        next_streams: list[tuple[str, Iterator[dict]]] = []
        batch: list[dict] = []

        # Pull examples from each active stream up to the interleave buffer size
        for path, stream in active_streams:
            consumed = 0
            exhausted = False
            while consumed < interleave_buffer_size:
                try:
                    batch.append(next(stream))
                except StopIteration:
                    exhausted = True
                    break
                consumed += 1

            if not exhausted:
                next_streams.append((path, stream))

        if not batch:
            break

        # Shuffle the batch to mix examples from different files
        # Keep examples from the same file together
        rng.shuffle(batch)
        for example in batch:
            yield example

        active_streams = next_streams


def tokenize_batch(batch: dict, tokenizer) -> dict:
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages in batch["messages"]
    ]

    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=False,
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "length": [len(ids) for ids in tokenized["input_ids"]],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL files for chat LLM fine-tuning."
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help='Glob or path to .jsonl files, e.g. "/data/*.jsonl"',
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory or file path containing JSONL inputs.",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=HF_HUB_CACHE)
    parser.add_argument("--max-doc-tokens", type=int, default=60000)
    parser.add_argument(
        "--quick-truncation",
        action="store_true",
        help="Truncate by character count, not tokens. Max characters is counted as max-doc-tokens * CHARS_PER_TOKEN. "
        "Also works with drop-long-docs to quickly drop long documents",
    )
    parser.add_argument(
        "--drop-long-docs",
        action="store_true",
        help="Drop documents longer than max-doc-tokens instead of truncating them.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--writer-batch-size", type=int, default=64)
    shuffle_group = parser.add_mutually_exclusive_group()
    shuffle_group.add_argument(
        "--shuffle-files",
        dest="shuffle_files",
        action="store_true",
        help="Shuffle the order of input files before interleaving.",
    )
    shuffle_group.add_argument(
        "--no-shuffle-files",
        dest="shuffle_files",
        action="store_false",
        help="Keep input files in sorted order.",
    )
    parser.set_defaults(shuffle_files=True)
    parser.add_argument(
        "--interleave-buffer-size",
        type=int,
        default=8,
        help="Number of examples to pull from each file before mixing them into one batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for file shuffling and batch mixing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_path and args.input_dir:
        raise ValueError("Specify only one of --data-path or --input-dir.")

    path_spec = args.input_dir or args.data_path
    if not path_spec:
        raise ValueError("Either --data-path or --input-dir is required.")

    if args.interleave_buffer_size <= 0:
        raise ValueError("--interleave-buffer-size must be greater than zero.")

    output_dir = (
        args.output_dir + "/" + args.run_id
    ) or f"./preprocessed/{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)

    paths = expand_input_paths(path_spec)
    if not paths:
        raise FileNotFoundError(f"No files matched: {path_spec}")

    print(
        f"Run ID: {args.run_id}; "
        f"{len(paths)} input file(s); "
        f"shuffle_files={args.shuffle_files}; "
        f"interleave_buffer_size={args.interleave_buffer_size}; "
        f"seed={args.seed}; "
        f"quick_truncation={args.quick_truncation}",
        f"drop_long_docs={args.drop_long_docs}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    dataset = Dataset.from_generator(
        example_generator,
        gen_kwargs={
            "paths": paths,
            "tokenizer": tokenizer,
            "max_doc_tokens": args.max_doc_tokens,
            "interleave_buffer_size": args.interleave_buffer_size,
            "seed": args.seed,
            "shuffle_files": args.shuffle_files,
            "quick_truncation": args.quick_truncation,
            "drop_long_docs": args.drop_long_docs,
        },
        cache_dir=args.cache_dir,
    )

    # Filter out examples with empty messages
    # This can happen if the original document was empty or if it was dropped/truncated to empty by our preprocessing
    def debug_filter(ex):
        if ex["messages"] is None:
            print("Dropped: messages is None")
            return False
        if not isinstance(ex["messages"], list):
            print(f"Dropped: messages is not a list: {type(ex['messages'])}")
            return False
        if len(ex["messages"]) != 3:
            print(f"Dropped: messages does not have 3 elements: {len(ex['messages'])}")
            return False
        for i, msg in enumerate(ex["messages"]):
            if not isinstance(msg, dict):
                print(f"Dropped: messages[{i}] is not a dict: {type(msg)}")
                return False
            if not msg.get("content"):
                print(f"Dropped: messages[{i}] has no 'content' key or it's empty")
                return False
        return True

    docs_before = len(dataset)
    print(f"Total documents before filtering: {docs_before}", flush=True)
    dataset = dataset.filter(debug_filter, num_proc=os.cpu_count())
    docs_after = len(dataset)
    print(f"Total documents after filtering: {docs_after}", flush=True)
    print(f"Total documents dropped by filter: {docs_before - docs_after}", flush=True)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
        desc="Applying chat template and tokenizing",
    )

    tokenized.save_to_disk(output_dir)
    print(f"Saved tokenized dataset to: {output_dir}", flush=True)
    if args.drop_long_docs:
        print(f"Long documents dropped: {documents_truncated}", flush=True)
    else:
        print(f"Long documents truncated: {documents_truncated}", flush=True)


if __name__ == "__main__":
    main()
