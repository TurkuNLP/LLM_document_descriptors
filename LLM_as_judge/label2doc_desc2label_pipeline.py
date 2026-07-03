from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from LLM_as_judge import LLMJudge  # type: ignore
import prompts

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_rows(path: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def extract_answer_text(response: str) -> str:
    text = response.strip().lower()
    if "answer:" in text:
        _, suffix = text.split("answer:", 1)
        return suffix.strip(" *:!?.-\n\r\t")
    return text


def parse_binary_response(response: str) -> str:
    answer = extract_answer_text(response)
    if re.match(r"^yes\b", answer):
        return "yes"
    if re.match(r"^no\b", answer):
        return "no"
    return "invalid"


def normalize_parsed_response(response: str) -> str:
    normalized = response.strip().lower()
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    return "invalid"


def format_parsed_response(response: str) -> str:
    if response == "yes":
        return "Yes"
    if response == "no":
        return "No"
    return "Invalid"


def collect_label_examples(
    rows: list[dict[str, Any]],
    label_type: str,
) -> list[dict[str, Any]]:
    return [
        {
            "document": row["document"],
            "label": row[label_type],
        }
        for row in rows
    ]


def collect_descriptor_examples(
    rows: list[dict[str, Any]],
    label_type: str,
    descriptor_type: str,
) -> list[dict[str, Any]]:
    def select_descriptors(row: dict[str, Any]) -> list[str]:
        if descriptor_type == "harmonized":
            return row.get("harmonized_descriptors", [])

        similarity_scores = row.get("similarity", [])
        descriptors = row.get("descriptors", [])
        if similarity_scores and descriptors:
            best_idx = max(
                range(len(similarity_scores)), key=similarity_scores.__getitem__
            )
            if best_idx < len(descriptors):
                selected_descriptors = descriptors[best_idx]
                if isinstance(selected_descriptors, list):
                    return selected_descriptors
                if isinstance(selected_descriptors, str):
                    return [selected_descriptors]

        if isinstance(descriptors, list):
            if descriptors and all(
                isinstance(descriptor, str) for descriptor in descriptors
            ):
                return descriptors
            if descriptors and all(
                isinstance(descriptor, list) for descriptor in descriptors
            ):
                first_group = descriptors[0]
                if isinstance(first_group, list):
                    return [str(descriptor) for descriptor in first_group]
        return []

    return [
        {
            "document": row["document"],
            "label": row[label_type],
            "descriptors": select_descriptors(row),
        }
        for row in rows
    ]


def preprocess_examples(
    judge: LLMJudge,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    processed_examples: list[dict[str, Any]] = []

    for example in examples:
        processed = dict(example)
        if "document" in processed:
            processed["document"] = judge.tokenize_and_truncate(processed["document"])
        processed_examples.append(processed)

    return processed_examples


def run_stage(
    judge: LLMJudge,
    sampling_params: Any,
    examples: list[dict[str, Any]],
    prompt_builder,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    processed_examples = preprocess_examples(judge, examples)
    prompts_list = [prompt_builder(example) for example in processed_examples]

    print(f"Got {len(prompts_list)} prompts. Starting evaluation...", flush=True)
    if prompts_list:
        print("Sample prompt:", flush=True)
        print(prompts_list[0], flush=True)
        print("", flush=True)

    responses = judge.generate(sampling_params, prompts_list)
    parsed_responses = [parse_binary_response(response) for response in responses]
    return processed_examples, responses, parsed_responses


def print_binary_results(title: str, parsed_responses: list[str]) -> None:
    counter = Counter(parsed_responses)
    total = sum(counter.values())
    yes_count = counter.get("yes", 0)
    no_count = counter.get("no", 0)
    invalid_count = counter.get("invalid", 0)

    yes_percentage = (yes_count / total * 100) if total > 0 else 0
    no_percentage = (no_count / total * 100) if total > 0 else 0
    invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

    print(f"{title} Results (n={total}):")
    print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
    print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
    print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


def save_results(
    path: str,
    examples: list[dict[str, Any]],
    responses: list[str],
    parsed_responses: list[str],
) -> None:
    if not path.endswith(".jsonl"):
        path += ".jsonl"
    with open(path, "w", encoding="utf-8") as file:
        for example, response, parsed_response in zip(
            examples, responses, parsed_responses
        ):
            result_row = dict(example)
            result_row["model_response"] = response
            result_row["parsed_response"] = format_parsed_response(parsed_response)
            json.dump(result_row, file, ensure_ascii=False)
            file.write("\n")


def load_saved_responses(path: str) -> list[str]:
    responses: list[str] = []
    for row in iter_jsonl(path):
        response = row.get("parsed_response")
        if isinstance(response, str):
            responses.append(normalize_parsed_response(response))
            continue

        response = row.get("response")
        if isinstance(response, str):
            responses.append(normalize_parsed_response(response))
        else:
            responses.append("invalid")
    return responses


def save_detailed_results(
    path: str,
    examples: list[dict[str, Any]],
    responses: list[str],
) -> None:
    if not path.endswith(".jsonl"):
        path += "_detailed.jsonl"
    else:
        path = path[: -len(".jsonl")] + "_detailed.jsonl"

    with open(path, "w", encoding="utf-8") as file:
        for example, response in zip(examples, responses):
            json.dump(
                {"example": example, "response": response}, file, ensure_ascii=False
            )
            file.write("\n")


def append_output_suffix(path: str, suffix: str) -> str:
    if path.endswith(".jsonl"):
        return path[: -len(".jsonl")] + f"_{suffix}.jsonl"
    return f"{path}_{suffix}.jsonl"


def get_label_output_path(output_path: str | None, data_path: str) -> str:
    if output_path:
        return append_output_suffix(output_path, "label2document")
    return append_output_suffix(data_path, "label2document")


def resolve_label_results_path(
    explicit_path: str | None,
    output_path: str | None,
    data_path: str,
) -> str:
    if explicit_path:
        return explicit_path
    return get_label_output_path(output_path, data_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Label2DocumentCorrespondence first and then run Descriptors2LabelCorrespondence only on documents that received a Yes."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128000,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the model.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/weborganizer/topic_format_edu.jsonl",
        help="Path to the JSONL evaluation data. Multiple paths can be comma-separated.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Base path to save the evaluation results. Stage-specific suffixes will be added automatically.",
    )
    parser.add_argument(
        "--label2document-results-path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to Label2DocumentCorrespondence parsed results (.jsonl). "
            "If provided, this path is loaded and used for caching stage 1."
        ),
    )
    parser.add_argument(
        "--detailed-output",
        action="store_true",
        help="Whether to save detailed results including prompts and raw responses.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--label-type",
        choices=["format", "topic"],
        default="format",
        help="How the document label should be interpreted in the first stage.",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=["harmonized", "raw"],
        default="harmonized",
        help="Which set of descriptors to use.",
    )

    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default="local",
        help="Whether to run the judge with a local vLLM model or a remote API.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. If omitted, OPENAI_API_KEY is used.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    data_paths = args.data_path.split(",")
    output_paths = (
        args.output_path.split(",") if args.output_path else [None] * len(data_paths)
    )
    if len(output_paths) < len(data_paths):
        print(
            "Warning: Fewer output paths than data paths. Some results will not be saved.",
            flush=True,
        )

    judge = LLMJudge(args)
    sampling_params = judge.get_sampling_params(args.model)

    for index, data_path in enumerate(data_paths):
        rows = load_rows(data_path)
        output_path = output_paths[index] if index < len(output_paths) else None
        explicit_label_path = None
        if args.label2document_results_path:
            explicit_paths = args.label2document_results_path.split(",")
            explicit_label_path = (
                explicit_paths[index]
                if index < len(explicit_paths)
                else explicit_paths[-1]
            )

        label_output_path = resolve_label_results_path(
            explicit_label_path,
            output_path,
            data_path,
        )

        print(f"Loaded {len(rows)} rows from {data_path}", flush=True)

        label_examples: list[dict[str, Any]] = []
        label_responses: list[str] = []
        label_parsed: list[str] = []

        if os.path.exists(label_output_path):
            loaded_responses = load_saved_responses(label_output_path)
            if len(loaded_responses) == len(rows):
                label_parsed = loaded_responses
                print(
                    f"Using existing Label2DocumentCorrespondence results from {label_output_path}",
                    flush=True,
                )
            else:
                print(
                    (
                        f"Existing Label2DocumentCorrespondence results in {label_output_path} "
                        f"have {len(loaded_responses)} rows, expected {len(rows)}. Recomputing."
                    ),
                    flush=True,
                )

        if not label_parsed:
            label_examples = collect_label_examples(rows, args.label_type)
            label_examples, label_responses, label_parsed = run_stage(
                judge,
                sampling_params,
                label_examples,
                lambda example: prompts.get_label2document_correspondence_prompt(
                    example["document"],
                    example["label"],
                    args.label_type,
                ),
            )
            save_results(label_output_path, rows, label_responses, label_parsed)
            print(
                f"Saved Label2DocumentCorrespondence results to {label_output_path}",
                flush=True,
            )

        print_binary_results("Label2DocumentCorrespondence", label_parsed)

        positive_rows = [
            row
            for row, parsed_response in zip(rows, label_parsed)
            if parsed_response == "yes"
        ]
        print(
            f"Label2DocumentCorrespondence passed for {len(positive_rows)} of {len(rows)} documents.",
            flush=True,
        )

        descriptor_examples: list[dict[str, Any]] = []
        descriptor_responses: list[str] = []
        descriptor_parsed: list[str] = []

        if positive_rows:
            descriptor_examples = collect_descriptor_examples(
                positive_rows,
                args.label_type,
                args.descriptor_type,
            )
            descriptor_examples, descriptor_responses, descriptor_parsed = run_stage(
                judge,
                sampling_params,
                descriptor_examples,
                lambda example: prompts.get_descriptors2label_correspondence_prompt(
                    example["descriptors"],
                    example["label"],
                ),
            )
            print_binary_results("Descriptors2LabelCorrespondence", descriptor_parsed)
        else:
            print(
                "No documents received a Yes response from Label2DocumentCorrespondence; skipping Descriptors2LabelCorrespondence.",
                flush=True,
            )

        if output_path:
            if args.detailed_output and label_examples and label_responses:
                save_detailed_results(
                    label_output_path, label_examples, label_responses
                )

            if positive_rows:
                descriptor_output_path = append_output_suffix(
                    output_path, "descriptors2label"
                )
                save_results(
                    descriptor_output_path,
                    positive_rows,
                    descriptor_responses,
                    descriptor_parsed,
                )
                if args.detailed_output:
                    save_detailed_results(
                        descriptor_output_path,
                        descriptor_examples,
                        descriptor_responses,
                    )


if __name__ == "__main__":
    print("Starting Label2Document -> Descriptors2Label pipeline...", flush=True)
    main()
