from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore
import os
import torch  # type: ignore
import torch.distributed as dist  # type: ignore
import time
import prompts
import json
from datasets import load_dataset  # type: ignore
import random
from random import shuffle
from collections import defaultdict
from typing import Optional, Dict, Union
import argparse
import re
import logging
from pydantic import BaseModel  # type: ignore
import re


# Configure logging
slurm_job_id = os.environ.get("SLURM_JOB_ID", "default_id")
logging.basicConfig(
    filename=f"../logs/{slurm_job_id}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Suppress sentence_transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Suppress transformers logging (used internally by sentence_transformers)
logging.getLogger("transformers").setLevel(logging.WARNING)


def LLM_setup(model, cache_dir):
    """
    Set up and initialize a Large Language Model (LLM) with specified configurations.

    Args:
        model (str): The name or path of the model to be loaded.
        cache_dir (str): The directory where the model and related files will be cached.

    Returns:
        LLM: An instance of the initialized Large Language Model with the specified configurations.

    Configuration:
        - dtype: Data type used for model parameters, set to "bfloat16".
        - max_model_len: Maximum length of the model, set to 128,000 tokens.
        - tensor_parallel_size: Number of GPUs to use for tensor parallelism, determined by the number of available CUDA devices.
        - enforce_eager: Whether to enforce eager execution, set to False.
        - gpu_memory_utilization: Fraction of GPU memory to utilize, set to 0.9.
        - pipeline_parallel_size: Optional, can be set to 2 if multiple nodes are needed (currently commented out).

    Note:
        Ensure that the required dependencies, such as PyTorch, are installed and CUDA devices are available for optimal performance.
    """
    return LLM(
        model=model,
        download_dir=cache_dir,
        dtype="bfloat16",
        max_model_len=128_000,
        tensor_parallel_size=torch.cuda.device_count(),
        # pipeline_parallel_size=2, # use if multiple nodes are needed
        enforce_eager=False,
        gpu_memory_utilization=0.8,
    )


def format_prompt(descriptors):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You will be given a list of descriptive words and phrases that characterize the style, tone, or contents of documents. These descriptors represent potential candidates for fine-tuning a Large Language Model (LLM).  

Your task is to **evaluate each descriptor**. For each descriptor, think about these questions:
1. Does the descriptor suggest the document contains **step-by-step instructions**? (Yes/No)
2. Does the descriptor suggest the document includes **problem-solving examples**? (Yes/No)
3. Does the descriptor suggest the document contains **multi-turn dialogue**? (Yes/No)
4. Does the descriptor suggest the document demonstrates **chain-of-thought reasoning**? (Yes/No)

If the answer to any of the above is "Yes," label the descriptor as a **‘good candidate’**. Otherwise, label it as a **‘bad candidate’**.

**Do not modify the descriptors in any way.**
All given descriptors **must** be placed in one of the categories. Do not skip any.
If unsure, give the label **‘bad candidate’**
If all descriptors belong to only one category, return an empty list for the other category.

Return the results following this JSON format:
{{"good_candidates": ["item", "item", "item", ...],
  "bad_candidates": ["item", "item", "item", ...]
}}<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the list:
{descriptors}<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return prompt


def chat(llm, message):
    """
    Generates a response from a language model (LLM) based on the provided message.

    Args:
        llm: The language model instance to use for generating the response.
        message: The input message to which the LLM should respond.

    Returns:
        str: The generated response text from the LLM, stripped of leading and trailing
             spaces, backticks, and newline characters.

    Notes:
        - This is currently only used for reformatting JSON strings.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.5,
        max_tokens=3_000,  # max tokens to generate
    )

    output = (
        llm.chat(
            messages=message,
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0]
        .outputs[0]
        .text.strip(" `\njson")
    )  # The model tends to generate these ticks
    # around JSON strings, which cause issues.

    return output


def generate(llm, batched_input, response_schema):
    """
    Generates text using a language model with specified sampling parameters.

    Args:
        llm: The language model to use for text generation.
        batched_input: A batch of input data for the language model.
        response_schema: JSON schema to guide output.

    Returns:
        A list of generated text outputs, stripped of leading and trailing spaces, backticks, and newlines.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.5,
        repetition_penalty=1,  # 1 = no penalty, >1 penalty
        max_tokens=3000,  # max tokens to generate
        guided_decoding=response_schema,
    )

    batched_outputs = llm.generate(
        batched_input, sampling_params=sampling_params, use_tqdm=True
    )
    
    # Remove reasoning part of the output if it exists.
    batched_outputs = [re.sub(r"<think>.*?</think>", "", out.outputs[0].text, flags=re.DOTALL) for out in batched_outputs]
    return [out.strip(" `\njson") for out in batched_outputs]


def get_response_format():
    class ResponseFormat(BaseModel):
        good_candidates: list[str]
        bad_candidates: list[str]

    json_schema = ResponseFormat.model_json_schema()
    return GuidedDecodingParams(json=json_schema)


def load_data(name):
    if name == "initial":
        with open("../data/all_unique_descriptors.txt", "r") as f:
            file = f.readlines()[:10_000]
    else:
        with open("../results/good_candidate_descriptors.txt", "r") as f:
            file = f.readlines()
    return file


def reformat_output(llm, output):
    """
    Reformats a given output string to ensure it is valid JSON.

    This function attempts to fix common JSON formatting issues such as:
    - Removing any text outside curly brackets.
    - Replacing single quotes with double quotes.
    - Removing trailing commas.
    - Adding double quotes around keys if missing.
    - Adding double quotes around string values if missing.

    If the initial regex-based fixes do not result in valid JSON, the function
    will attempt to use a language model to correct the JSON format.

    Args:
        llm: The language model to use for fixing JSON if regex-based fixes fail.
        output (str): The output string to be reformatted.

    Returns:
        dict or str: The reformatted JSON as a dictionary if successful, or "FAIL" if all attempts fail.
    """
    logging.warning("Fixing JSON formatting.")
    for i in range(3):
        # Remove newlines and spaces from start and end
        output = output.strip(" \n")
        # Remove any text outside curly brackets
        json_start = output.find("{")
        json_end = output.find("}")
        if json_start != -1 and json_end != -1:
            output = output[json_start : json_end + 1]  # Include the '}'
        # Replace single quotes with double quotes.
        output = output.replace("'", '"')
        # Remove trailing commas
        output = re.sub(r",\s*([\]}])", r"\1", output)
        # Add double quotes around keys (if missing)
        output = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', output)
        # Add double quotes around string values (if missing)
        output = re.sub(r':\s*([^"\s\d\[{].*?)(?=[,}\]])', r':"\1"', output)
        valid_json = validate_output(output)
        if valid_json:
            logging.warning(f"Fixed JSON formatting with {i} LLM call(s).")
            return json.loads(output, strict=False)

        # If fixing JSON with regex does not work,
        # we try giving it to the model to fix.
        # This takes quite a lot of time (~1-2 min/attempt), so consider lowering
        # number of attempts to speed up the process.
        prompt = prompts.reformat_output_prompt(output)
        output = chat(llm, prompt)
        valid_json = validate_output(output)
        if valid_json:
            logging.warning(f"Fixed JSON formatting with {i+1} LLM call(s).")
            return json.loads(output, strict=False)

    # If fixing does not work, save the malformed JSON to disk for later inspection.
    # Return "FAIL"
    logging.warning("Failed to fix JSON formatting.")
    with open("../results/malformed_JSON_output.txt", "a") as f:
        f.write(f"{output}\n======================\n")
    return "FAIL"


def validate_output(output):
    """
    Validates if the given output is a valid JSON string.

    This function attempts to parse the provided output as JSON. If the parsing
    is successful, it returns True. If the parsing fails due to a JSON decoding
    error, it logs the error and the invalid output, then returns False.

    Args:
        output (str): The output string to be validated as JSON.

    Returns:
        bool: True if the output is a valid JSON string, False otherwise.
    """
    try:
        json.loads(output, strict=False)
        return True
    except json.JSONDecodeError as e:
        logging.debug(e)
        logging.debug("Invalid JSON output:")
        logging.debug(repr(output))
        return False


def batched(descriptors, batch_size, batch_count):
    """
    Splits a list of strings into a list of sublists,
    each containing max inner_list_size strings. If the input list does not
    neatly split into the smaller lists, the last lists can be shorter.

    :param descriptors: List of strings
    :return: List of lists
    """
    # Create the list of lists
    result = []
    for i in range(0, len(descriptors), batch_size * batch_count):
        chunk = descriptors[i:i + batch_size * batch_count]
        sublists = [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
        result.append(sublists)

    return result
   
        
def check_output_format(output, llm):
    validated_output = []
    for out in output:
        valid_json = validate_output(out)
        if valid_json:
            validated_output.append(json.loads(out, strict=False))
        else:
            reformatted = reformat_output(llm, out)
            if not reformatted == "FAIL":
                validated_output.append(reformatted)
            else:
                validated_output.append(json.loads('{"good_candidates": [], "bad_candidates": []}'))
                
    return validated_output


def save_results(results):
    with open(f"../results/good_candidate_descriptors.txt", "w") as f:
        for line in results:
            for desc in line.get("good_candidates"):
                desc = desc.strip(" \n")
                f.write(desc+"\n")
    with open(f"../results/bad_candidate_descriptors.txt", "a") as f:
        for line in results:
            for desc in line.get("bad_candidates"):
                desc = desc.strip(" \n")
                f.write(desc+"\n")


def main(args):
    run_id = args.run_id
    cache_dir = args.cache_dir
    model = args.model
    num_batches = args.num_batches
    batch_size = args.batch_size
    global temperature
    temperature = args.temperature

    logging.info("Loading model...")
    llm = LLM_setup(model, cache_dir)
    

    logging.info("Starting document processing pipeline...")
    
    
    for i in range(3):
        results = []
        if i == 0:
            data = load_data("initial")
        else:
            data = load_data("next")
        for batch_num, batch in enumerate(batched(data, batch_size, num_batches)):
            prompts = [
                format_prompt(descs) for descs in batch
            ]
            json_schema = get_response_format()
            batched_outputs = generate(llm, prompts, json_schema)
            output = check_output_format(batched_outputs, llm)
            results.extend(output)
        save_results(results)
        logging.info("Results saved.")

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A script for getting document descriptors with LLMs."
    )
    
    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../.cache",
        help="Path to cache directory, where model is or will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        help="Name of model to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of descriptors to give to the model at once.",
    )
    parser.add_argument(
        "--num-batches", type=int, default=100,
        help="Number of LLM calls to do at once."
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Model temperature."
    )
    args = parser.parse_args()

    # Log the run settings
    with open(f"../results/{args.run_id}_settings.txt", "w") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f.write(f"{arg}: {value}\n")

    # Create required directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    main(args)
    logging.info("Done.")