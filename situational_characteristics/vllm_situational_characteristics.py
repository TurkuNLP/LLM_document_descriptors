from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore
import os
import torch  # type: ignore
import torch.distributed as dist  # type: ignore
import time
import prompts
import json
from datasets import load_dataset  # type: ignore
from random import shuffle
from collections import defaultdict
from typing import Optional, Dict, Union
import argparse
import re
import logging
from pydantic import BaseModel  # type: ignore


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

  
def format_prompt(document):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
Based on this framework for analysis of situational characteristics, analyse the given document.   

Framework:
- Participants:
    - Addressor(s) (i.e., speaker or author)
        - single / plural / institutional / unidentified
        - social characteristics: e.g., age, education, profession

    - Addressee(s)
        - single / plural / unenumerated
        - self / other

    - Are there onlookers?

- Relations among participants
    - Interactiveness
    - Social roles: relative status or power
    - Personal relationship: e.g., friends, colleagues, strangers
    - Shared knowledge: personal, specialist

- Channel
    - Mode: speech / writing / signing
    - Specific medium:
        - permanent: e.g., taped, transcribed, printed, handwritten, email
        - transient: e.g., face-to-face, telephone, radio, TV

- Processing circumstances
    - Production: real time / planned / scripted / revised and edited
    - Comprehension: real time / skimming / careful reading

- Setting
    - Are the time and place of communication shared by participants?
    - Place of communication
        - private / public
        - specific setting
    - Time: contemporary / historical time period

- Communicative purposes
    - General purposes: e.g., narrate/report, describe, inform/explain/interpret, persuade, how-to/procedural, entertain, edify, reveal self
    - Specific purposes: e.g., summarize information from numerous sources, describe methods, present new research findings, teach moral through personal story
    - Purported factuality: factual, opinion, speculative, imaginative
    - Expression of stance: epistemic, attitudinal, no overt stance

- Topic
    - General topical domain: e.g., domestic, daily activities, business/workplace, science, education/academic, government/legal/politics, religion, sports, art/entertainment
    - Specific topic
    - Social status of person being referred to<|eot_id|><|start_header_id|>user<|end_header_id|>

{document}<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
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
        batched_input, sampling_params=sampling_params, use_tqdm=False
    )

    return [out.outputs[0].text.strip(" `\njson") for out in batched_outputs]


def get_response_format():

    class PlaceOfCommunication(BaseModel):
        private_public: str
        specific_setting: str

    class Addressee(BaseModel):
        type: str
        self_other: str

    class Addressor(BaseModel):
        type: str
        social_characteristics: str

    class Participants(BaseModel):
        addressor: Addressor
        addressee: Addressee
        are_there_onlookers: str

    class RelationsAmongParticipants(BaseModel):
        interactiveness: str
        social_roles: str
        personal_relationship: str
        shared_knowledge: str

    class Channel(BaseModel):
        mode: str
        specific_medium: str

    class ProcessingCircumstances(BaseModel):
        production: str
        comprehension: str

    class Setting(BaseModel):
        are_time_and_place_shared: str
        place_of_communication: PlaceOfCommunication
        time: str

    class CommunicativePurposes(BaseModel):
        general_purposes: str
        specific_purposes: str
        purported_factuality: str
        expression_of_stance: str

    class Topic(BaseModel):
        general_topical_domain: str
        specific_topic: str
        social_status_of_person_referred_to: str

    class ResponseFormat(BaseModel):
        participants: Participants
        relations_among_participants: RelationsAmongParticipants
        channel: Channel
        processing_circumstances: ProcessingCircumstances
        setting: Setting
        communicative_purposes: CommunicativePurposes
        topic: Topic


    json_schema = ResponseFormat.model_json_schema()
    return GuidedDecodingParams(json=json_schema)


def load_documents():
    """
    Load documents from a specified data source.
    This function provides two options for loading documents:
    1. From the HuggingFace FineWeb dataset (commented out by default).
    2. From a local JSONL file containing a 40k sample.

    Returns:
        list: A list of documents loaded from the selected data source.
    """
    # Comment/uncomment to choose data source

    # Original fineweb sample
    # return load_dataset("HuggingFaceFW/fineweb",
    #                    name="sample-10BT",
    #                    split="train",
    #                    streaming=True)

    # Our 40k sample
    with open("../data/fineweb_40k.jsonl", "r") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]


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


def batched(data, batch_size, start_index):
    """
    Generator function that yields batches of data from a given start index.

    Args:
        data (list): The list of documents to be batched.
        batch_size (int): The size of each batch.
        start_index (int): The index from which to start batching.

    Yields:
        list: A batch of documents of size `batch_size`. The last batch may be smaller if there are not enough documents left.
    """
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
                
    return validated_output


def save_results(results, run_id):
    with open(f"../results/sit_chars/results_{run_id}.jsonl", "a") as f:
        for line in results:
            f.write(f"{json.dumps(line)}\n")


def main(args):
    """
    Main function to generate descriptors for documents.
    Documents are processed in batches, with each batch going through the pipeline.
    The pipeline consists of three stages:
    - Initial: Generate general and specific descriptors for each document.
    - Rewrite: Rewrite the document based on the descriptors.
    - Revise: Evaluate the rewrite and revise the descriptors.
    Finally, similarity score between rewrites and the original documents are calculated.
    The best rewrite and descriptors are saved to a JSONL file.

    Args:
        args (Namespace): Command line arguments containing the following attributes:
            cache_dir (str): Directory to cache the model.
            model (str): Model name or path.
            start_index (int): Starting index for processing documents.
            num_batches (int): Number of batches to process.
            use_previous_descriptors (bool): Flag to use previous descriptors.
            descriptor_path (str): Path to save descriptors.
            run_id (str): Unique identifier for the run.
            num_rewrites (int): Number of rewrites to perform.
            batch_size (int): Number of documents per batch.
            max_vocab (int): Maximum number of descriptors in the vocabulary.
            temperature (float): Temperature parameter for the model.

    Returns:
        None
    """
    cache_dir = args.cache_dir
    model = args.model
    start_index = args.start_index
    num_batches = args.num_batches
    run_id = args.run_id
    batch_size = args.batch_size
    global temperature
    temperature = args.temperature

    logging.info("Loading model...")
    llm = LLM_setup(model, cache_dir)
    logging.info("Loading data...")
    data = load_documents()

    logging.info("Starting document processing pipeline...")
    for batch_num, batch in enumerate(batched(data, batch_size, start_index)):

        start_time = time.time()
        
        documents = [doc["text"] for doc in batch]
        
        prompts = [
            format_prompt(document)
            for document in documents
        ]
        json_schema = get_response_format()
        batched_outputs = generate(llm, prompts, json_schema)
        output = check_output_format(batched_outputs, llm)
        
        save_results(output, run_id)
        logging.info("Results saved.")

        end_time = time.time()

        logging.info(
            f"Processed {len(output)} documents in {time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))}."
        )
        logging.info(f"Processed a total of {(batch_num+1)*len(output)} documents.")

        # Stop run after num_batches batches have been processed.
        # If -1, we continue until we run out of data or time.
        if num_batches == -1:
            continue
        elif batch_num + 1 >= num_batches:
            break
        

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
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of model to use.",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Index of first document to analyse."
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=-1,
        help="Number of batches of size batch-size to process. Set to -1 to process all data.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Model temperature."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents given to the model at one time.",
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
