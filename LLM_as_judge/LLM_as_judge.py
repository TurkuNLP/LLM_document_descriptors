import argparse
from abc import ABC, abstractmethod
from collections import Counter
import json
import os
import random
from typing import Any, Iterator
import re
import numpy as np  # type: ignore

import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore

import prompts

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class LLMJudge:
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        cache_dir = args.cache_dir or os.getenv("HF_HUB_CACHE")
        self.llm = self._setup_llm(args.model, args.max_model_len, cache_dir)
        self.max_tokens = args.max_tokens

    def generate(
        self,
        sampling_params: SamplingParams,
        inputs: list[str],
    ) -> list[str]:
        outputs = self.llm.chat(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        response_texts: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None) or []
            if candidates:
                response_texts.append(getattr(candidates[0], "text", "") or "")
            else:
                response_texts.append("")

        return response_texts

    def get_sampling_params(self, model_name: str) -> SamplingParams:
        common_params = {
            "repetition_penalty": 1.0,
            "max_tokens": self.max_tokens,
        }

        llama_params = {
            "temperature": 0.2,
            "top_p": 0.5,
        }

        qwen_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
        }

        common_params.update(llama_params if "Llama" in model_name else qwen_params)
        return SamplingParams(**common_params)

    def tokenize_and_truncate(self, text: str) -> str:
        max_input_len = self.llm.llm_engine.model_config.max_model_len - self.max_tokens
        if max_input_len <= 0:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be less than the model's max_model_len ({self.llm.llm_engine.model_config.max_model_len})."
            )
        tokenizer = self.llm.get_tokenizer()
        token_ids = tokenizer.encode(text)

        if len(token_ids) > max_input_len:
            token_ids = token_ids[:max_input_len]

        return tokenizer.decode(token_ids)

    def _setup_llm(
        self,
        model: str,
        max_model_len: int,
        cache_dir: str | None = None,
    ) -> LLM:
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "dtype": "bfloat16",
            "max_model_len": max_model_len,
            "tensor_parallel_size": max(1, torch.cuda.device_count()),
            "enforce_eager": False,
            "gpu_memory_utilization": 0.8,
        }

        if cache_dir:
            llm_kwargs["download_dir"] = cache_dir

        return LLM(**llm_kwargs)


class BaseTask(ABC):
    name: str

    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        return {}

    def include_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> bool:
        return True

    @abstractmethod
    def build_examples(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    def preprocess_example(
        self,
        judge: LLMJudge,
        example: dict[str, Any],
    ) -> dict[str, Any]:
        processed = dict(example)
        if "document" in processed:
            processed["document"] = judge.tokenize_and_truncate(processed["document"])
        return processed

    @abstractmethod
    def parse_response(self, response: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def print_results(
        self,
        parsed_responses: list[str],
        args: argparse.Namespace,
    ) -> None:
        raise NotImplementedError

    def save_detailed_results(self, path, examples, responses) -> None:
        if not path.endswith(".jsonl"):
            path += "_detailed.jsonl"
        with open(path, "w") as f:
            for example, response in zip(examples, responses):
                json.dump(
                    {"example": example, "response": response}, f, ensure_ascii=False
                )
                f.write("\n")

    def save_results(self, path, parsed_responses) -> None:
        if not path.endswith(".jsonl"):
            path += ".jsonl"
        with open(path, "w") as f:
            for response in parsed_responses:
                json.dump({"response": response}, f, ensure_ascii=False)
                f.write("\n")


class QueryDescriptorMatchTask(BaseTask):
    """Evaluates whether a descriptor corresponds to a query."""

    name = "QueryDescriptorMatch"

    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        return {"query": args.query}

    def include_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> bool:
        return True

    def build_examples(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        descriptor = row.get("descriptor", "")

        return [{"descriptor": descriptor, "query": args.query}]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_descriptor_correspondence_prompt(
            example["query"],
            example["descriptor"],
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, {"yes", "no"})

    def print_results(
        self, parsed_responses: list[str], args: argparse.Namespace
    ) -> None:
        counter = Counter(parsed_responses)
        total = sum(counter.values())
        yes_count = counter.get("yes", 0)
        no_count = counter.get("no", 0)
        invalid_count = counter.get("invalid", 0)

        yes_percentage = (yes_count / total * 100) if total > 0 else 0
        no_percentage = (no_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

        print(f"Query Correspondence Evaluation Results (n={total}):")
        print(f"QUERY: {args.query}")
        print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
        print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class QueryDocMatchTask(BaseTask):
    """Evaluates whether a document corresponds to a query."""

    name = "QueryDocMatch"

    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        return {"doc_ids": load_doc_ids(args.doc_ids_path)}

    def include_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> bool:
        return row.get("doc_id") in context["doc_ids"]

    def build_examples(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        return [{"document": row["document"], "query": args.query}]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_query_correspondence_prompt(
            example["document"], example["query"]
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, {"yes", "no"})

    def print_results(
        self, parsed_responses: list[str], args: argparse.Namespace
    ) -> None:
        counter = Counter(parsed_responses)
        total = sum(counter.values())
        yes_count = counter.get("yes", 0)
        no_count = counter.get("no", 0)
        invalid_count = counter.get("invalid", 0)

        yes_percentage = (yes_count / total * 100) if total > 0 else 0
        no_percentage = (no_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

        print(f"Query Correspondence Evaluation Results (n={total}):")
        print(f"QUERY: {args.query}")
        print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
        print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class DescriptorAccuracyTask(BaseTask):
    """Evaluates the accuracy of descriptors for documents."""

    name = "DescriptorAccuracy"
    VALID_CLASSES = [
        "Accurate",
        "Mostly accurate",
        "Partially accurate",
        "Mostly inaccurate",
        "Inaccurate",
    ]

    def include_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> bool:
        return random.random() <= args.sample_percent

    def build_examples(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
        args: argparse.Namespace,
    ) -> list[dict[str, Any]]:
        document = row["document"]
        if args.descriptor_type == "harmonized":
            descriptors = row.get("harmonized_descriptors", [])
        elif args.descriptor_type == "raw":
            similarity_scores = row.get("similarity", [])
            if similarity_scores:
                best_idx = np.argmax(similarity_scores)
                descriptors = row["descriptors"][best_idx]
            else:
                descriptors = row.get("descriptors", [])
        else:
            raise ValueError(f"Invalid descriptor type: {args.descriptor_type}")

        return [
            {
                "document": document,
                "descriptor": descriptor,
            }
            for descriptor in descriptors
        ]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_descriptor_accuracy_prompt(
            example["document"],
            example["descriptor"],
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, self.VALID_CLASSES)

    def print_results(
        self, parsed_responses: list[str], args: argparse.Namespace
    ) -> None:
        counter = Counter(parsed_responses)
        total = sum(counter.values())
        print(f"Descriptor type: {args.descriptor_type}")
        print(f"Descriptor Accuracy Evaluation Results (n={total}):")
        for label in self.VALID_CLASSES:
            count = counter.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{label}: {count} ({percentage:.2f}%)")

        if "invalid" in parsed_responses:
            invalid_count = counter.get("invalid", 0)
            invalid_percentage = (invalid_count / total * 100) if total > 0 else 0
            print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


TASKS: dict[str, BaseTask] = {
    "QueryDocMatch": QueryDocMatchTask(),
    "DescriptorAccuracy": DescriptorAccuracyTask(),
    "QueryDescriptorMatch": QueryDescriptorMatchTask(),
}


def extract_answer_text(response: str) -> str:
    text = response.strip().lower()

    if "answer:" in text:
        _, suffix = text.split("answer:", 1)
        return suffix.strip(" *:!?.-\n\r\t")  # remove common punctuation and whitespace

    # returns the whole response if "answer:" is not found
    return text


def parse_label_response(response: str, valid_labels: list[str]) -> str:
    answer = extract_answer_text(response).strip().lower()

    for label in sorted(valid_labels, key=len, reverse=True):
        pattern = rf"^{re.escape(label.lower())}\b"
        if re.match(pattern, answer):
            return label

    return "invalid"


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_doc_ids(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def load_examples(path: str, task: BaseTask, args) -> list[dict]:
    context = task.setup(args)
    examples = []

    for row in iter_jsonl(path):
        if task.include_row(row, context, args):
            examples.extend(task.build_examples(row, context, args))
            
    if not examples:
        raise ValueError("No examples were included for evaluation. Please check your data and filtering criteria.")

    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM as Judge")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
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
        default="../results/harmonized/fineweb-edu/concatenated/"
        "descriptors_fineweb-edu_harmonized.jsonl",
        help="Path to the JSONL evaluation data.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save the evaluation results (JSONL format). If not specified, results will not be saved, only printed to stdout.",
    )
    parser.add_argument(
        "--detailed-output",
        action="store_true",
        help="Whether to save detailed results including prompts and raw responses. Requires --output-path to be specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by tasks that sample rows.",
    )

    subparsers = parser.add_subparsers(dest="task", required=True)

    # Subparser for query correspondence task
    query_parser = subparsers.add_parser(
        "QueryDocMatch",
        help="Evaluate query correspondence prompts on a selected set of doc IDs.",
    )
    query_parser.add_argument(
        "--query", type=str, help="The query to evaluate correspondence for."
    )
    query_parser.add_argument(
        "--doc-ids-path",
        type=str,
        required=True,
        help="Path to the file containing document IDs to evaluate.",
    )

    # Subparser for descriptor accuracy task
    descriptor_parser = subparsers.add_parser(
        "DescriptorAccuracy",
        help="Evaluate descriptor accuracy on a random sample of documents.",
    )
    descriptor_parser.add_argument(
        "--sample-percent",
        type=float,
        default=0.05,
        help="Fraction of documents to sample for evaluation.",
    )
    descriptor_parser.add_argument(
        "--descriptor-type",
        choices=["harmonized", "raw"],
        default="harmonized",
        help="Type of descriptors to evaluate (harmonized or raw).",
    )

    # Subparser for descriptor correspondence task
    descriptor_query_parser = subparsers.add_parser(
        "QueryDescriptorMatch",
        help="Evaluate descriptor correspondence to query.",
    )
    descriptor_query_parser.add_argument(
        "--query", type=str, help="The query to evaluate correspondence for."
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    
    print("Selected task:", args.task)
    print("Loading model and preparing prompts...")
    task = TASKS[args.task]    
    examples = load_examples(args.data_path, task, args)
    
    judge = LLMJudge(args)
    
    examples = [task.preprocess_example(judge, example) for example in examples]
    input_prompts = [task.build_prompt(example) for example in examples]

    sampling_params = judge.get_sampling_params(args.model)
    responses = judge.generate(sampling_params, input_prompts)
    parsed_responses = [task.parse_response(response) for response in responses]
    task.print_results(parsed_responses, args)
    if args.output_path:
        task.save_results(args.output_path, parsed_responses)
        if args.detailed_output:
            task.save_detailed_results(args.output_path, examples, responses)


if __name__ == "__main__":
    main()
