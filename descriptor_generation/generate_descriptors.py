# Standard libraries
import argparse
from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
import warnings

# Silence annoying user warning
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
)

# Third party imports
import json_repair  # type: ignore
import pandas as pd  # type: ignore
from pydantic import BaseModel  # type: ignore
import torch  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore

# Local imports
from embed import StellaEmbedder, QwenEmbedder
import descriptor_prompts
from utils import (
    load_documents,
    save_descriptors,
    init_results,
    save_results,
    log_execution_time,
)

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use CK attention instead of Triton attention. For example SWA
# attention used by Mistral models does not properly with Triton
# attention and AMD GPUs.
# os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Suppress sentence_transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Suppress transformers logging (used internally by sentence_transformers)
logging.getLogger("transformers").setLevel(logging.WARNING)
# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def configure_logging(log_file: Path) -> None:
    # reset handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    file_h = logging.FileHandler(log_file)
    file_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_h)
    logging.root.addHandler(stream_h)

    logging.info("=" * 40)
    logging.info("Logging configured to: %s", log_file)
    logging.info("SLURM Job ID: %s", os.environ.get("SLURM_JOB_ID", "N/A"))


class DescriptorGenerator:
    def __init__(self, args):
        self.cache_dir = args.cache_dir
        self.model = args.model
        self.start_index = args.start_index
        self.num_batches = args.num_batches
        self.run_id = args.run_id
        self.num_rewrites = args.num_rewrites
        self.batch_size = args.batch_size
        self.data_source = args.data_source
        self.temperature = args.temperature
        self.max_model_len = args.max_model_len
        self.checkpoint_interval = args.checkpoint_interval
        self.base_dir = Path("..") / "results" / self.run_id
        self.embedder = (
            QwenEmbedder(self.cache_dir)
            if args.embedder_model == "qwen"
            else StellaEmbedder(self.cache_dir)
        )
        self.text_column = args.text_column
        self.prompt_format = args.prompt_format
        self.log_similarity = args.log_similarity

    @log_execution_time
    def LLM_setup(self):
        # Common LLM kwargs
        LLM_kwargs = {
            "model": self.model,
            "dtype": "bfloat16",
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": torch.cuda.device_count(),
            "enforce_eager": False,
            "gpu_memory_utilization": 0.8,
        }

        # Set cache dir if given
        if self.cache_dir:
            LLM_kwargs["download_dir"] = self.cache_dir

        # Model specific kwargs
        if self.model == "deepseek-ai/DeepSeek-V3.2":
            LLM_kwargs["block_size"] = 1  # block_size has to be 1 for deepseek on ROCM

        if self.model == "moonshotai/Kimi-K2.5":
            LLM_kwargs["trust_remote_code"] = True

        return LLM(**LLM_kwargs)

    def remove_thinking_tokens(self, text):
        """Remove thinking tokens from the model output.
        gpt-oss models start their output with 'analysis' and
        end the thinking with 'assistantfinal'."""

        end_marker = "assistantfinal"
        end = text.lower().find(end_marker.lower())
        if end == -1:
            return text

        return text[end + len(end_marker) :].lstrip()

    def generate(self, input, stage):
        response_schema = self.get_response_format(stage)
        max_tokens = {
            "initial": 2000,
            "rewrite": 4000,
            "revise": 5000,
        }

        # Sampling parameters
        common_params = {
            "temperature": self.temperature,
            "top_p": 0.5,
            "repetition_penalty": 1,  # 1 = no penalty, >1 penalty
            "max_tokens": max_tokens.get(stage, 5000),  # max tokens to generate
            "structured_outputs": StructuredOutputsParams(
                json=response_schema
            )
        }

        sampling_params = SamplingParams(**common_params)

        start = time.perf_counter()
        if self.prompt_format == "chat":
            outputs = self.llm.chat(
                input, sampling_params=sampling_params, use_tqdm=False
            )
        else:
            outputs = self.llm.generate(
                input, sampling_params=sampling_params, use_tqdm=False
            )
        elapsed = time.perf_counter() - start if start else 0.0

        response_texts = []
        in_tok = 0
        gen_tok = 0
        for o in outputs:
            if getattr(o, "prompt_token_ids", None):
                in_tok += len(o.prompt_token_ids)

            outs = getattr(o, "outputs", None) or []
            if outs:
                cand = outs[0]
                if getattr(cand, "token_ids", None):
                    gen_tok += len(cand.token_ids)
                txt = getattr(cand, "text", "") or ""
                response_texts.append(self.remove_thinking_tokens(txt))
            else:
                response_texts.append("{}")

        tot_tok = gen_tok + in_tok
        if elapsed > 0 and tot_tok > 0:
            logging.info(
                "LLM throughput: %.1f tok/s (%.1f gen tok/s) — %s tokens in %.2fs",
                tot_tok / elapsed,
                gen_tok / elapsed if gen_tok else 0,
                tot_tok,
                elapsed,
            )

        return response_texts

    def validate_output(self, output, stage):
        "Validate and reformat the model output to ensure it is in the expected format."

        # First, check that output is valid JSON. If not, try to repair it.
        # If it still cannot be repaired, replace with empty dict.
        validated = json_repair.loads(output)
        if not isinstance(validated, dict):
            logging.warning(
                f"Failed to format output: {validated}. Replacing with empty dict."
            )
            validated = {}

        # Next, check if there are empty fields in output
        # If there are, log a warning.
        for key, value in validated.items():
            if not value:
                logging.warning(f"Output has empty field '{key}'.")

        # Check that all required keys are present.
        # If not, add them with empty values and log a warning.
        response_format = self.get_response_format(stage)
        for output_key in response_format["properties"].keys():
            if output_key not in validated:
                logging.warning(
                    f"Output is missing required key '{output_key}'. Adding empty value."
                )
                validated[output_key] = (
                    []
                    if response_format["properties"][output_key]["type"] == "array"
                    else ""
                )

        # Ensure all descriptors and specifics are strings (in case model returned numbers or other types).
        for key in validated:
            if key in ["descriptors", "specifics"]:
                for i, item in enumerate(validated[key]):
                    if not isinstance(item, str):
                        item = str(item)
                        validated[key][i] = item

        return validated

    def determine_start_index(self, start_index: str, data: list) -> int:
        if start_index == "auto":
            descriptor_file = self.base_dir / f"descriptors_{self.run_id}.jsonl"
            if descriptor_file.exists():
                with open(descriptor_file, "r") as f:
                    start_index = sum(1 for _ in f)
            else:
                start_index = 0
            logging.info(f"Start index determined as {start_index}.")
        elif start_index.isdigit():
            start_index = int(start_index)

        # Validate start_index
        if not isinstance(start_index, int):
            raise ValueError("start_index must be 'auto' or an integer.")
        elif start_index < 0:
            raise ValueError("start_index must be a non-negative integer.")
        # Uncommented because this does not work if full dataset is not loaded into memory
        # elif start_index >= len(data):
        #    raise ValueError("start_index is out of bounds for the dataset.")

        return start_index

    def batched(self, data, batch_size=None, start_index=None):
        if not batch_size:
            batch_size = self.batch_size
        if not start_index:
            start_index = self.start_index

        start_index = self.determine_start_index(start_index, data)

        batch = []
        for i, doc in enumerate(data):
            if i < start_index:
                continue
            batch.append(doc)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @log_execution_time
    def initial_stage(self, documents):
        stage = "initial"
        prompts = [
            self.format_prompt(stage=stage, original=document) for document in documents
        ]
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output, stage) for output in batched_output
        ]

        return validated_outputs

    @log_execution_time
    def rewrite_stage(self, general, specific):
        """
        Processes lists of descriptors through the rewrite generation stage.

        Args:
            stage (str): The stage of the pipeline.
            general (list of list): A list of lists of general descriptors.
            specific (list of list): A list of lists of specific descriptors.
            llm (object): The language model used for generating responses.

        Returns:
            list: A list of validated and possibly reformatted outputs in JSON format.
        """
        stage = "rewrite"
        prompts = []
        for g, s in zip(general, specific):
            prompts.append(self.format_prompt(stage=stage, descriptors=g, specifics=s))
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output, stage) for output in batched_output
        ]

        return validated_outputs

    @log_execution_time
    def revise_stage(self, document, rewritten, general, specific):
        """
        Processes lists of descriptors and rewrites through the revise generation stage.

        Args:
            stage (str): The current stage of the pipeline.
            document (list of str): The original documents.
            rewritten (list of str): The rewritten documents.
            general (list of list): General descriptors for the documents.
            specific (list of list): Specific descriptors for the documents.
            llm (object): The language model used for generating outputs.

        Returns:
            list of dict: A list of validated outputs, each containing general and specific descriptors.
        """
        stage = "revise"
        prompts = []
        for d, r, g, s in zip(document, rewritten, general, specific):
            prompts.append(
                self.format_prompt(
                    stage=stage,
                    original=d,
                    rewritten=r,
                    descriptors=g,
                    specifics=s,
                )
            )
        batched_output = self.generate(prompts, stage)
        validated_outputs = [
            self.validate_output(output, stage) for output in batched_output
        ]

        return validated_outputs

    def update_descriptor_vocab(self, descriptor_path):
        desc_counts = Counter()
        with open(self.base_dir / f"descriptors_{self.run_id}.jsonl", "r") as f:
            for line in f.readlines():
                doc = json.loads(line.strip())
                for descriptor_list in doc["descriptors"]:
                    if descriptor_list:
                        descriptors_without_explanations = self.remove_explanations(
                            descriptor_list
                        )
                    desc_counts.update(descriptors_without_explanations)

        save_descriptors(desc_counts, descriptor_path)
        self.log_descriptor_growth(desc_counts)

    def log_descriptor_growth(self, vocab):
        with open(
            self.base_dir / f"descriptor_count_growth_{self.run_id}.txt", "a"
        ) as f:
            f.write(f"{len(vocab)}\n")

    def format_prompt(
        self,
        stage,
        original=None,
        rewritten=None,
        descriptors=None,
        specifics=None,
    ):
        # Truncate original document to max length
        if original:
            original = self.tokenize_and_truncate(original)
        if stage == "initial":
            if self.prompt_format == "chat":
                prompt = descriptor_prompts.initial_chat_prompt(original)
            else:
                prompt = descriptor_prompts.initial_prompt(original)
        elif stage == "rewrite":
            if self.prompt_format == "chat":
                prompt = descriptor_prompts.rewrite_chat_prompt(descriptors, specifics)
            else:
                prompt = descriptor_prompts.rewrite_prompt(descriptors, specifics)
        elif stage == "revise":
            if self.prompt_format == "chat":
                prompt = descriptor_prompts.revise_chat_prompt(
                    original, rewritten, descriptors, specifics
                )
            else:
                prompt = descriptor_prompts.revise_prompt(
                    original, rewritten, descriptors, specifics
                )

        return prompt

    @staticmethod
    def get_response_format(stage):
        if stage == "initial":

            class ResponseFormat(BaseModel):
                descriptors: list[str]
                specifics: list[str]

        elif stage == "rewrite":

            class ResponseFormat(BaseModel):
                text: str

        elif stage == "revise":

            class ResponseFormat(BaseModel):
                differences: str
                descriptors: list[str]
                specifics: list[str]

        return ResponseFormat.model_json_schema()

    def tokenize_and_truncate(self, document):
        max_input_len = (
            self.llm.llm_engine.model_config.max_model_len - 5000
        )  # leave room for generation
        tokenizer = self.llm.get_tokenizer()
        prompt_token_ids = tokenizer.encode(document)
        if len(prompt_token_ids) > max_input_len:  # leave room for generation
            logging.warning(
                f"Document is too long: ({len(prompt_token_ids)} tokens). Truncating..."
            )
            prompt_token_ids = prompt_token_ids[:max_input_len]

        return tokenizer.decode(prompt_token_ids)

    @staticmethod
    def update_results(results, general=None, specific=None, rewrites=None):
        """Updates results dictionary with new descriptors or rewrites."""
        for index in results:
            if general:
                results[index]["descriptors"].append(general[index])
            if specific:
                results[index]["specifics"].append(specific[index])
            if rewrites:
                results[index]["rewrite"].append(rewrites[index])

    def make_checkpoint(self):
        # Calculate how many documents we have processed so far
        with open(
            self.base_dir / f"descriptor_count_growth_{self.run_id}.txt", "r"
        ) as f:
            num_batches = len(f.readlines())
        docs_processed = num_batches * self.batch_size
        checkpoint_dir = self.base_dir / f"checkpoint_{docs_processed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through the files in the directory
        for item_path in self.base_dir.iterdir():
            if item_path.is_file():
                # Destination path in the checkpoint directory
                destination_path = checkpoint_dir / item_path.name
                # Copy the file
                shutil.copy2(item_path, destination_path)

    @staticmethod
    def remove_explanations(list_of_descriptors):
        just_descriptors = []
        for descriptor in list_of_descriptors:
            if isinstance(descriptor, str) and ";" in descriptor:
                just_descriptors.append(descriptor.split(";")[0].strip())
            else:
                just_descriptors.append(
                    descriptor.strip() if isinstance(descriptor, str) else descriptor
                )

        return just_descriptors

    def log_avg_similarity(self):
        similarities = []
        with open(self.base_dir / f"descriptors_{self.run_id}.jsonl", "r") as f:
            for line in f:
                doc = json.loads(line.strip())
                similarities.append(doc["similarity"])
        if not similarities:
            logging.info("No similarities found in the results.")

        for i in range(self.num_rewrites):
            round_similarities = [sim[i] for sim in similarities if len(sim) > i]
            if round_similarities:
                avg_similarity = sum(round_similarities) / len(round_similarities)
                logging.info(
                    f"Average similarity for rewrite round {i+1}: {avg_similarity:.4f}"
                )
            else:
                logging.info(f"No similarities found for rewrite round {i+1}.")

    def pipeline(self):

        self.llm = self.LLM_setup()
        logging.info("Model loaded.")
        data = load_documents(self.data_source, self.cache_dir)
        logging.info("Data loaded.")

        # The text needs to be in a column called "text".
        # If not, give the column name in --text-column arg and it will be renamed to "text"
        if self.text_column != "text":
            data = data.rename_column(self.text_column, "text")

        descriptor_path = self.base_dir / f"descriptor_vocab_{self.run_id}.tsv"

        for batch_num, batch in enumerate(self.batched(data)):
            if self.num_batches == 0:
                break
            logging.info(f"===============New Batch: {batch_num}===============")
            start = time.time()

            # Initialise empty results dictionary
            results = init_results(batch)

            # Generate initial descriptors for document.
            documents = [doc["text"] for doc in batch]
            logging.info("Stage: initial.")
            model_outputs = self.initial_stage(documents)

            # Extract output and append to results.
            general_descriptors = [
                output.get("descriptors", "Generation failed.")
                for output in model_outputs
            ]
            specific_descriptors = [
                output.get("specifics", "Generation failed.")
                for output in model_outputs
            ]
            self.update_results(
                results, general=general_descriptors, specific=specific_descriptors
            )

            # Generate rewrites of the document based on descriptors.
            # After the rewrite, we revise the descriptors to create an even better rewrite.
            for round_num in range(self.num_rewrites):
                # Remove explanations from descriptors for rewriting.
                general_without_explanations = [
                    self.remove_explanations(g) for g in general_descriptors
                ]
                specific_without_explanations = [
                    self.remove_explanations(s) for s in specific_descriptors
                ]
                # Rewrite doc based on the descriptors.
                logging.info(f"Stage: rewrite {round_num+1}.")
                model_outputs = self.rewrite_stage(
                    general_without_explanations, specific_without_explanations
                )

                # Extract output and append to results.
                rewrites = [
                    output.get("text", "Generation failed.") for output in model_outputs
                ]
                self.update_results(results, rewrites=rewrites)

                if not round_num == self.num_rewrites - 1:
                    # Evaluate rewrite and revise descriptors.
                    # This stage is skipped on the last round: since we do not do another rewrite
                    # we do not need another set of descriptors.
                    logging.info(f"Stage: revise {round_num+1}.")
                    model_outputs = self.revise_stage(
                        documents,
                        rewrites,
                        general_descriptors,
                        specific_descriptors,
                    )

                    # Extract output and append to results.
                    general_descriptors = [
                        output.get("descriptors", "Generation failed.")
                        for output in model_outputs
                    ]
                    specific_descriptors = [
                        output.get("specifics", "Generation failed.")
                        for output in model_outputs
                    ]
                    self.update_results(
                        results,
                        general=general_descriptors,
                        specific=specific_descriptors,
                    )

            # Calculate similarity between rewrites and original.
            # Append to results.
            if self.num_rewrites > 0:
                for index in results:
                    similarities = self.embedder.calculate_similarity(
                        results[index]["text"], results[index]["rewrite"]
                    )
                    results[index]["similarity"].extend(similarities)

            # Save all results so far
            save_results(results, self.base_dir, self.run_id, only_best=False)
            logging.info("Results saved.")

            # Update the descriptor vocabulary with new descriptors
            self.update_descriptor_vocab(descriptor_path)

            # Log execution time
            end = time.time()
            execution_time = end - start
            logging.info(
                f"Processing batch took {time.strftime('%H:%M:%S', time.gmtime(execution_time))}."
            )

            if self.checkpoint_interval > 0:
                if batch_num > 0 and (batch_num + 1) % self.checkpoint_interval == 0:
                    self.make_checkpoint()
                    logging.info(f"Checkpoint created after {batch_num + 1} batches.")

            # Stop iterating through new data after num_batches batches have been processed.
            if self.num_batches == -1:
                continue
            elif batch_num + 1 >= self.num_batches:
                if self.log_similarity:
                    self.log_avg_similarity()
                break


def main(args):

    start = time.time()
    dg = DescriptorGenerator(args)
    dg.pipeline()
    end = time.time()
    total_execution_time = end - start
    logging.info(
        f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_execution_time))}."
    )
    logging.info("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A script for getting document descriptors with LLMs."
    )

    # Basic arguments
    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of model to use.",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        choices=["stella", "qwen"],
        default="qwen",
        help="Name of embedding model to use for similarity evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HUB_CACHE", ""),
        help="Path to cache directory, where model is or will be saved.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Model temperature."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=64_000,
        help="Maximum context length for the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of documents given to the model at one time.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=["generate", "chat"],
        default="chat",
        help="Whether to use generate or chat format for prompts.",
    )

    # Data processing arguments
    parser.add_argument(
        "--start-index",
        type=str,
        default="auto",
        help="Index of first document to analyse or 'auto' to find start index based on already processed documents.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=-1,
        help="Number of batches of size batch-size to process. Set to -1 to process all data.",
    )
    parser.add_argument(
        "--num-rewrites",
        type=int,
        default=3,
        help="How many rewriting cycles the script should go through.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Number of batches after which all results so far will be saved into a checkpoint. "
        "If 0, no checkpoints are created. Default: 0 (no checkpoints).",
    )
    parser.add_argument(
        "--log-similarity",
        action="store_true",
        help="Whether to log average similarity between rewrites and original documents at the end of run.",
    )

    # Data arguments
    parser.add_argument(
        "--data-source", type=str, default="fineweb", help="Which data set to process."
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )

    args = parser.parse_args()

    # Create required directories
    os.makedirs("../results", exist_ok=True)
    os.makedirs(f"../results/{args.run_id}", exist_ok=True)

    # Configure logging
    log_file = Path(f"../results/{args.run_id}/{args.run_id}.log")
    configure_logging(log_file)

    # Log the run settings
    with open(f"../results/{args.run_id}/{args.run_id}_settings.txt", "a") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f.write(f"{arg}: {value}\n")
        f.write("===========================\n")

    main(args)
