import argparse
from abc import ABC, abstractmethod
from collections import Counter
import json
import os
import random
from typing import Any, Iterator
import re
import numpy as np  # type: ignore
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI  # type: ignore
import random

import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
import prompts

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

random.seed(42)

class LLMJudge:
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        self.backend = args.backend
        self.model = args.model
        self.max_tokens = args.max_tokens
        self.max_model_len = args.max_model_len

        if self.backend == "local":
            cache_dir = args.cache_dir or os.getenv("HF_HUB_CACHE")
            self.llm = self._setup_llm(args.model, args.max_model_len, cache_dir)
            self.client = None

        elif self.backend == "api":
            self.llm = None
            self.client = OpenAI(
                api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=args.api_base_url,
            )

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def generate(
        self,
        sampling_params: SamplingParams | dict[str, Any],
        inputs: list[str],
    ) -> list[str]:
        if self.backend == "local":
            return self._generate_local(sampling_params, inputs)

        if self.backend == "api":
            return self._generate_api(sampling_params, inputs)

        raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_local(
        self,
        sampling_params: SamplingParams,
        inputs: list[str],
    ) -> list[str]:
        assert self.llm is not None

        outputs = self.llm.chat(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        response_texts: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None) or []
            if candidates:
                response_texts.append(getattr(candidates[0], "text", "") or "")
            else:
                response_texts.append("")

        return response_texts

    def _generate_api(
        self,
        sampling_params: dict[str, Any],
        inputs: list[str],
    ) -> list[str]:
        assert self.client is not None

        def call_api(prompt: str) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_output_tokens=sampling_params["max_tokens"],
                temperature=sampling_params.get("temperature"),
                top_p=sampling_params.get("top_p"),
            )

            return response.choices[0].message.content or ""

        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(call_api, inputs))

    def get_sampling_params(
        self,
        model_name: str,
    ) -> SamplingParams | dict[str, Any]:
        common_params = {
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

        if self.backend == "local":
            return SamplingParams(
                repetition_penalty=1.0,
                **common_params,
            )

        if self.backend == "api":
            # Most APIs do not support vLLM-only params.
            api_params = {
                "max_tokens": common_params["max_tokens"],
                "temperature": common_params.get("temperature"),
                "top_p": common_params.get("top_p"),
            }
            return api_params

        raise ValueError(f"Unknown backend: {self.backend}")

    def tokenize_and_truncate(self, text: str) -> str:
        if self.backend == "local":
            assert self.llm is not None

            max_input_len = (
                self.llm.llm_engine.model_config.max_model_len - self.max_tokens
            )

            if max_input_len <= 0:
                raise ValueError(
                    f"max_tokens ({self.max_tokens}) must be less than "
                    f"the model's max_model_len "
                    f"({self.llm.llm_engine.model_config.max_model_len})."
                )

            tokenizer = self.llm.get_tokenizer()
            token_ids = tokenizer.encode(text)

            if len(token_ids) > max_input_len:
                token_ids = token_ids[:max_input_len]

            return tokenizer.decode(token_ids)

        if self.backend == "api":
            max_input_len = self.max_model_len - self.max_tokens

            if max_input_len <= 0:
                raise ValueError(
                    f"max_tokens ({self.max_tokens}) must be less than "
                    f"max_model_len ({self.max_model_len})."
                )

            # Rough fallback.
            max_chars = max_input_len * 4
            return text[:max_chars]

        raise ValueError(f"Unknown backend: {self.backend}")

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
            "gpu_memory_utilization": 0.9,
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
    valid_labels = {"yes", "no"}

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
        query = row["query"]
        results = row.get("results", [])

        return [
            {
                "query": query,
                "descriptor": result["descriptor"],
            }
            for result in results
            if "descriptor" in result
        ]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_descriptor_correspondence_prompt(
            example["query"],
            example["descriptor"],
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, valid_labels=self.valid_labels)

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
    valid_labels = {"yes", "no", "partial"}

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
        if "document" in row:
            return [{"document": row["document"], "query": args.query}]
        elif "text" in row:
            return [{"document": row["text"], "query": args.query}]
        else:
            raise ValueError("Row must contain either 'document' or 'text' field.")
        
    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_query_correspondence_prompt(
            example["document"], example["query"]
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, valid_labels=self.valid_labels)

    def print_results(
        self, parsed_responses: list[str], args: argparse.Namespace
    ) -> None:
        counter = Counter(parsed_responses)
        total = sum(counter.values())
        yes_count = counter.get("yes", 0)
        partial_count = counter.get("partial", 0)
        no_count = counter.get("no", 0)
        invalid_count = counter.get("invalid", 0)

        yes_percentage = (yes_count / total * 100) if total > 0 else 0
        partial_percentage = (partial_count / total * 100) if total > 0 else 0
        no_percentage = (no_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

        print(f"Query Correspondence Evaluation Results (n={total}):")
        print(f"QUERY: {args.query}")
        print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
        print(f"ANSWER: Partial: {partial_count} ({partial_percentage:.2f}%)")
        print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class DescriptorAccuracyTask(BaseTask):
    """Evaluates the accuracy of descriptors for documents."""

    name = "DescriptorAccuracy"
    valid_labels = [
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
        if "document" in row:
            document = row["document"]
        elif "text" in row:
            document = row["text"]
        else:
            raise ValueError("Row must contain either 'document' or 'text' field.")

        if args.descriptor_type == "harmonized":
            descriptors = row.get("harmonized_descriptors", [])
        elif args.descriptor_type == "raw":
            if "raw_descriptors" in row:
                descriptors = row.get("raw_descriptors", [])
            else:
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
        return parse_label_response(response, self.valid_labels)

    def print_results(
        self, parsed_responses: list[str], args: argparse.Namespace
    ) -> None:
        counter = Counter(parsed_responses)
        total = sum(counter.values())
        print(f"Descriptor type: {args.descriptor_type}")
        print(f"Descriptor Accuracy Evaluation Results (n={total}):")
        for label in self.valid_labels:
            count = counter.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{label}: {count} ({percentage:.2f}%)")

        if "invalid" in parsed_responses:
            invalid_count = counter.get("invalid", 0)
            invalid_percentage = (invalid_count / total * 100) if total > 0 else 0
            print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class Descriptors2LabelCorrespondenceTask(BaseTask):
    """Evaluates whether the descriptors of a document correspond to it WebOrganizer label."""

    name = "Descriptors2LabelCorrespondence"
    valid_labels = {"yes", "no"}

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
        if "document" in row:
            document = row["document"]
        elif "text" in row:
            document = row["text"]
        else:
            raise ValueError("Row must contain either 'document' or 'text' field.")

        label = row.get("label", "")
        descriptors = row.get("descriptors", [])

        return [
            {
                "document": document,
                "label": label,
                "descriptors": descriptors,
            }
        ]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_descriptors2label_correspondence_prompt(
            example["document"],
            example["label"],
            example["descriptors"],
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, self.valid_labels)

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

        print(f"Descriptor to Label Correspondence Evaluation Results (n={total}):")
        print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
        print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class Label2DocumentCorrespondenceTask(BaseTask):
    """Evaluate whether the WebOrganizer label of a document corresponds to the document content."""

    name = "Label2DocumentCorrespondence"
    valid_labels = {"yes", "no"}

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
        if "document" in row:
            document = row["document"]
        elif "text" in row:
            document = row["text"]
        else:
            raise ValueError("Row must contain either 'document' or 'text' field.")
        
        label = row.get("label", "")

        return [
            {
                "document": document,
                "label": label,
            }
        ]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_label2document_correspondence_prompt(
            example["document"],
            example["label"],
        )

    def parse_response(self, response: str) -> str:
        return parse_label_response(response, self.valid_labels)

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

        print(f"Label to Document Correspondence Evaluation Results (n={total}):")
        print(f"ANSWER: Yes: {yes_count} ({yes_percentage:.2f}%)")
        print(f"ANSWER: No: {no_count} ({no_percentage:.2f}%)")
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")


class InferLabelsFromDescriptorsTask(BaseTask):
    """Infers all reasonable labels from a document's descriptors."""

    name = "InferLabelsFromDescriptors"

    format_labels = [
        "About (Org.)",
        "About (Personal)",
        "Academic Writing",
        "Audio Transcript",
        "Comment Section",
        "Content Listing",
        "Creative Writing",
        "Documentation",
        "FAQ",
        "Knowledge Article",
        "Legal Notices",
        "Listicle",
        "News (Org.)",
        "News Article",
        "Nonfiction Writing",
        "Personal Blog",
        "Product Page",
        "Q&A Forum",
        "Spam / Ads",
        "Structured Data",
        "Customer Support",
        "Truncated",
        "Tutorial",
        "User Review",
    ]
    topic_labels = [
        "Adult",
        "Art & Design",
        "Crime & Law",
        "Education & Jobs",
        "Entertainment",
        "Fashion & Beauty",
        "Finance & Business",
        "Food & Dining",
        "Games",
        "Hardware",
        "Health",
        "History",
        "Home & Hobbies",
        "Industrial",
        "Literature",
        "Politics",
        "Religion",
        "Science & Technology",
        "Social Life",
        "Software",
        "Software Development",
        "Sports & Fitness",
        "Transportation",
        "Travel",
    ]

    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        if args.label_type == "format":
            labels = self.format_labels
            random.shuffle(labels)  # Shuffle to avoid bias in label order
        elif args.label_type == "topic":
            labels = self.topic_labels
            random.shuffle(labels)  # Shuffle to avoid bias in label order
        else:
            raise ValueError("--label-type must be either 'format' or 'topic'.")

        self.label_type = args.label_type
        self.label_vocab = labels
        return {"labels": labels}

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
        if args.descriptor_type == "harmonized":
            descriptors = row.get("harmonized_descriptors", [])
        elif args.descriptor_type == "raw":
            similarity_scores = row.get("similarity", [])
            if similarity_scores:
                best_idx = np.argmax(similarity_scores)
                descriptors = row["descriptors"][best_idx]
            else:
                descriptors = row.get("descriptors", [])

        if args.label_type == "format":
            true_label = row.get("format", "")
        elif args.label_type == "topic":
            true_label = row.get("topic", "")

        return [
            {
                "descriptors": descriptors,
                "labels": context["labels"],
                "true_label": true_label,
            }
        ]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return prompts.get_infer_labels_from_descriptors_prompt(
            example["descriptors"],
            example["labels"],
            self.label_type,
        )

    def parse_response(self, response: str) -> list[str] | None:
        return parse_label_list_response(response, self.label_vocab)

    def print_results(
        self, parsed_responses: list[list[str] | None], args: argparse.Namespace
    ) -> None:
        label_counter: Counter[str] = Counter()
        invalid_count = 0
        empty_count = 0

        for response in parsed_responses:
            if response is None:
                invalid_count += 1
                continue

            if not response:
                empty_count += 1

            label_counter.update(response)

        total = len(parsed_responses)
        total_labels = sum(label_counter.values())
        empty_percentage = (empty_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

        print(f"Infer Labels From Descriptors Evaluation Results (n={total}):")
        print(f"Label vocabulary: {', '.join(self.label_vocab)}")
        print(f"Total inferred labels: {total_labels}")
        print(
            f"Documents with no inferred labels: {empty_count} ({empty_percentage:.2f}%)"
        )
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")
        for label in self.label_vocab:
            count = label_counter.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{label}: {count} ({percentage:.2f}%)")
            
            
class DescriptorClassificationTask(BaseTask):
    """Classifies descriptors into predefined categories."""
    
    categories = (
    "Content & Subject Matter: e.g. topic, domain, subtopic, intent, cultural context, temporal relevance",
    "Structure & Format: e.g. genre, document type (blog, news, etc.), media format, length",
    "Style & Tone: e.g. formality, tone, technicality (e.g. jargon), voice (active/passive), syntax",
    "Quality & Trustworthiness: e.g. factuality, authority, originality, spam, bias, depth (superficial/in-depth)",
    "Audience & Engagement: e.g. target audience, engagement level, monetization, call-to-action, shareability",
    "Linguistic Features: e.g. readability, sentiment, figurative language, multilinguality, repetition",
    "Ethical & Legal: e.g. toxicity, misinformation, legal compliance, accessibility, dark patterns",
    "Other: miscellaneous, not fitting into other categories",
    )
    
    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        return {"categories": self.categories}
    
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
        
        if args.descriptor_type == "harmonized":
            descriptors = row.get("harmonized_descriptors", [])
        elif args.descriptor_type == "raw":
            if "raw_descriptors" in row:
                descriptors = row.get("raw_descriptors", [])
            else:
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
                "descriptor": descriptor,
                "categories": context["categories"],
            }
            for descriptor in descriptors
        ]
        
    def build_prompt(self, example: dict[str, Any]) -> str:
        
        formatted_categories = "\n".join(f"{i+1} {category}" for i, category in enumerate(example["categories"]))
        
        return prompts.get_descriptor_classification_prompt(
            example["descriptor"],
            formatted_categories,
        )
        
    def parse_response(self, response: str) -> list[str] | None:
        answer = extract_answer_text(response).strip()
        # answer looks like this "1, 3, 5"
        selected_indices = re.findall(r"\d+", answer)
        
        if not selected_indices:
            return None
        
        selected_categories = []
        for index in selected_indices:
            idx = int(index) - 1  # Convert to 0-based index
            if 0 <= idx < len(self.categories):
                selected_categories.append(self.categories[idx])
        
        return selected_categories if selected_categories else None
    
    def print_results(
        self, parsed_responses: list[list[str] | None], args: argparse.Namespace
    ) -> None:
        category_counter: Counter[str] = Counter()
        invalid_count = 0
        empty_count = 0

        for response in parsed_responses:
            if response is None:
                invalid_count += 1
                continue

            if not response:
                empty_count += 1

            category_counter.update(response)

        total = len(parsed_responses)
        total_categories = sum(category_counter.values())
        empty_percentage = (empty_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0

        print(f"Descriptor Classification Evaluation Results (n={total}):")
        print(f"Total classified categories: {total_categories}")
        print(
            f"Descriptors with no classified categories: {empty_count} ({empty_percentage:.2f}%)"
        )
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")
        for category in self.categories:
            count = category_counter.get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{category}: {count} ({percentage:.2f}%)")
            
            
class AspectCoverageTask(BaseTask):
    """For the descriptors of a document, evuluate whether the given aspect are covered by the descriptors."""
    
    aspects = (
        "Content & Subject Matter: e.g. topic, domain, subtopic, intent, cultural context, temporal relevance",
        "Structure & Format: e.g. genre, document type (blog, news, etc.), media format, length",
        "Style & Tone: e.g. formality, tone, technicality (e.g. jargon), voice (active/passive), syntax",
        "Quality & Trustworthiness: e.g. factuality, authority, originality, spam, bias, depth (superficial/in-depth)",
        "Audience & Engagement: e.g. target audience, engagement level, monetization, call-to-action, shareability",
        "Linguistic Features: e.g. readability, sentiment, figurative language, multilinguality, repetition",
        "Ethical & Legal: e.g. toxicity, misinformation, legal compliance, accessibility, dark patterns",
        "Other: miscellaneous, not fitting into other categories"
    )
    
    def setup(self, args: argparse.Namespace) -> dict[str, Any]:
        return {"aspects": self.aspects}
    
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
        
        if args.descriptor_type == "harmonized":
            descriptors = row.get("harmonized_descriptors", [])
        elif args.descriptor_type == "raw":
            if "raw_descriptors" in row:
                descriptors = row.get("raw_descriptors", [])
            else:
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
                "descriptors": descriptors,
                "aspects": context["aspects"],
            }
        ]
        
    def build_prompt(self, example: dict[str, Any]) -> str:
        formatted_aspects = "\n".join(f"{i+1} {aspect}" for i, aspect in enumerate(example["aspects"]))
        
        return prompts.get_aspect_coverage_prompt(
            example["descriptors"],
            formatted_aspects,
        )
        
    def parse_response(self, response: str) -> list[str] | None:
        answer = extract_answer_text(response).strip()
        # answer looks like this "1, 3, 5"
        selected_indices = re.findall(r"\d+", answer)
        
        if not selected_indices:
            return None
        
        selected_aspects = []
        for index in selected_indices:
            idx = int(index) - 1  # Convert to 0-based index
            if 0 <= idx < len(self.aspects):
                selected_aspects.append(self.aspects[idx])
        
        return selected_aspects if selected_aspects else None
    
    def print_results(
        self, parsed_responses: list[list[str] | None], args: argparse.Namespace
    ) -> None:
        """Print the number of documents that cover each aspect, documents that cover all-1 aspects, all-2 aspects, etc.
        Also print the number of documents that do not cover any aspect, and the number of invalid responses.
        Also, print the percentage of aspects covered by the descriptors across all documents.
        """
        aspect_counter: Counter[str] = Counter()
        invalid_count = 0
        empty_count = 0

        for response in parsed_responses:
            if response is None:
                invalid_count += 1
                continue

            if not response:
                empty_count += 1

            aspect_counter.update(response)

        total = len(parsed_responses)
        total_aspects = sum(aspect_counter.values())
        empty_percentage = (empty_count / total * 100) if total > 0 else 0
        invalid_percentage = (invalid_count / total * 100) if total > 0 else 0
        
        print(f"Aspect Coverage Evaluation Results (n={total}):")
        print(f"Descriptor type: {args.descriptor_type}")
        print(f"Total covered aspects: {total_aspects}")
        print(
            f"Documents with no covered aspects: {empty_count} ({empty_percentage:.2f}%)"
        )
        print(f"Invalid answers: {invalid_count} ({invalid_percentage:.2f}%)")
        for num_aspects in range(len(self.aspects) + 1):
            exact_count = sum(1 for response in parsed_responses if response and len(response) == num_aspects)
            exact_percentage = (exact_count / total * 100) if total > 0 else 0
            print(f"Documents covering exactly {num_aspects} aspects: {exact_count} ({exact_percentage:.2f}%)")
            at_least_count = sum(1 for response in parsed_responses if response and len(response) >= num_aspects)
            at_least_percentage = (at_least_count / total * 100) if total > 0 else 0
            print(f"Documents covering at least {num_aspects} aspects: {at_least_count} ({at_least_percentage:.2f}%)")
        
        for aspect in self.aspects:
            count = aspect_counter.get(aspect, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{aspect}: {count} ({percentage:.2f}%)")    


TASKS: dict[str, BaseTask] = {
    "QueryDocMatch": QueryDocMatchTask(),
    "DescriptorAccuracy": DescriptorAccuracyTask(),
    "QueryDescriptorMatch": QueryDescriptorMatchTask(),
    "Descriptors2LabelCorrespondence": Descriptors2LabelCorrespondenceTask(),
    "Label2DocumentCorrespondence": Label2DocumentCorrespondenceTask(),
    "InferLabelsFromDescriptors": InferLabelsFromDescriptorsTask(),
    "DescriptorClassification": DescriptorClassificationTask(),
    "AspectCoverage": AspectCoverageTask(),
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


def parse_label_list_response(
    response: str,
    valid_labels: list[str],
) -> list[str] | None:
    answer = extract_answer_text(response).strip()
    # answer looks like this "["Health", "Social Life"]"
    answer = re.sub(r"[\[\]\"']", "", answer)  # remove brackets and quotes

    if not answer:
        return None

    label_lookup = {label.casefold(): label for label in valid_labels}

    selected_labels = []
    if "," in answer:
        # Split by commas and strip whitespace
        selected_labels = [label.strip() for label in answer.split(",")]
        for i, label in enumerate(selected_labels):
            if label.casefold() in label_lookup:
                selected_labels[i] = label_lookup[label.casefold()]
            else:
                selected_labels[i] = None  # Mark invalid labels as None
        # Filter out None values (invalid labels)
        selected_labels = [label for label in selected_labels if label is not None]

    return selected_labels


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
        raise ValueError(
            "No examples were included for evaluation. Please check your data and filtering criteria."
        )

    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM as Judge")
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

    # API options
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

    subparsers = parser.add_subparsers(dest="task", required=True)

    # Subparser for query correspondence task
    query_parser = subparsers.add_parser(
        "QueryDocMatch",
        help="Evaluate query correspondence prompts on a selected set of doc IDs.",
    )
    query_parser.add_argument(
        "--query", type=str, help="The query to evaluate correspondence for."
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

    # Subparser for labels inference task
    infer_labels_parser = subparsers.add_parser(
        "InferLabelsFromDescriptors",
        help="Infer all reasonable labels from a document's descriptors.",
    )
    infer_labels_parser.add_argument(
        "--label-type",
        type=str,
        required=True,
        choices=["format", "topic"],
        help="The type of labels to infer from the descriptors.",
    )
    infer_labels_parser.add_argument(
        "--descriptor-type",
        type=str,
        required=True,
        choices=["harmonized", "raw"],
        help="Which descriptor set to use for inference.",
    )
    
    # Subparser for descriptor classification task
    descriptor_classification_parser = subparsers.add_parser(
        "DescriptorClassification",
        help="Classify descriptors into predefined categories.",
    )
    descriptor_classification_parser.add_argument(
        "--sample-percent",
        type=float,
        default=0.05,
        help="Fraction of documents to sample for evaluation.",
    )
    descriptor_classification_parser.add_argument(
        "--descriptor-type",
        type=str,
        required=True,
        choices=["harmonized", "raw"],
        help="Which descriptor set to use for classification.",
    )
    
    # Subparser for aspect coverage task
    aspect_coverage_parser = subparsers.add_parser(
        "AspectCoverage",
        help="Evaluate whether the given aspects are covered by the descriptors.",
    )
    aspect_coverage_parser.add_argument(
        "--sample-percent",
        type=float,
        default=0.05,
        help="Fraction of documents to sample for evaluation.",
    )
    aspect_coverage_parser.add_argument(
        "--descriptor-type",
        type=str,
        required=True,
        choices=["harmonized", "raw"],
        help="Which descriptor set to use for evaluation.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    print("Selected task:", args.task, flush=True)
    print("Loading model and preparing prompts...", flush=True)
    task = TASKS[args.task]

    # If many data_paths are given, run once for each
    data_paths = args.data_path.split(",")
    output_paths = (
        args.output_path.split(",") if args.output_path else [None] * len(data_paths)
    )
    if len(output_paths) < len(data_paths):
        print(
            "Warning: Fewer output paths than data paths. Some results will not be saved.",
            flush=True,
        )

    for i, data_path in enumerate(data_paths):
        examples = load_examples(data_path, task, args)
        output_path = output_paths[i] if i < len(output_paths) else None

        judge = LLMJudge(args)

        examples = [task.preprocess_example(judge, example) for example in examples]
        input_prompts = [task.build_prompt(example) for example in examples]

        print(f"Got {len(input_prompts)} prompts. Starting evaluation...", flush=True)
        print("Sample prompt:", flush=True)
        print(input_prompts[0], flush=True)
        print("", flush=True)

        sampling_params = judge.get_sampling_params(args.model)
        responses = judge.generate(sampling_params, input_prompts)
        parsed_responses = [task.parse_response(response) for response in responses]
        task.print_results(parsed_responses, args)
        if args.output_path:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if args.detailed_output:
                task.save_detailed_results(output_path, examples, responses)
            else:
                task.save_results(output_path, parsed_responses)


if __name__ == "__main__":
    print("Starting LLM as Judge evaluation...", flush=True)
    main()
