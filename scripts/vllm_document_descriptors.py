from vllm import LLM, SamplingParams # type: ignore
from vllm.sampling_params import GuidedDecodingParams #type: ignore
import os
import torch # type: ignore
import torch.distributed as dist # type: ignore
import time
import prompts
import json
from datasets import load_dataset # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from random import shuffle
from collections import defaultdict
import argparse
import re
import logging
from pydantic import BaseModel #type: ignore


# Configure logging
slurm_job_id = os.environ.get("SLURM_JOB_ID", "default_id")
logging.basicConfig(
    filename=f"../logs/{slurm_job_id}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
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
        #pipeline_parallel_size=2, # use if multiple nodes are needed
        enforce_eager=False,
        gpu_memory_utilization=0.9,
    )


def calculate_doc_similarity(original, rewrite, cache):
    """Calculate the similarity between the original document and its rewrites.

    Args:
        original (str): The original document as a string.
        rewrite (list of str): A list of rewritten documents.
        cache (str): The path to the cache folder for storing model data.

    Returns:
        list of float: A list of similarity scores between the original document and each rewritten document, rounded to four decimal places.
    """
    model = SentenceTransformer("jinaai/jina-embeddings-v3",
                                trust_remote_code=True,
                                cache_folder=cache)
    original_embedding = model.encode([original])
    rewrite_embeddings = model.encode(rewrite)

    # Compute cosine similarities
    similarity = model.similarity(original_embedding, rewrite_embeddings)

    # Return similarity between documents
    return [round(float(sim), 4) for sim in similarity[0]]


def format_prompt(stage, original=None, rewritten=None, general=None, specific=None, vocab=None):
    """
    Formats a prompt message based on the given stage and parameters.

    Args:
        stage (str): The stage of code execution. Can be "initial", "rewrite" or "revise".
        original (str, optional): The original text. Default is None.
        rewritten (str, optional): The rewritten text. Default is None.
        general (str, optional): General descriptors. Default is None.
        specific (str, optional): Specific descriptors. Default is None.
        vocab (str, optional): Current descriptor vocabulary. Default is None.

    Returns:
        str: The formatted prompt message.
    """
    if stage == "initial":
        message = prompts.initial_prompt(original, vocab)
    elif stage == "rewrite":
        message = prompts.rewrite_prompt(general, specific)
    else:
        message = prompts.revise_keyphrases_prompt(original, rewritten, general, specific, vocab)
    return message


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
           max_tokens=3_000, # max tokens to generate
           )

    output = llm.chat(
        messages=message,
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0].outputs[0].text.strip(" `\n") # The model tends to generate these ticks
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
        repetition_penalty=1, # 1 = no penalty, >1 penalty
        max_tokens=3000, #max tokens to generate
        guided_decoding=response_schema,
    )

    batched_outputs = llm.generate(
        batched_input,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    return [out.outputs[0].text.strip(" `\n") for out in batched_outputs]


def get_response_format(stage):
    """
    Generates a JSON schema for the response format based on the given stage.
    
    Args:
        stage (str): The stage of code execution. Can be 'initial', 'rewrite', or 'revise'.
    
    Returns:
        GuidedDecodingParams: An object containing the JSON schema for the response format.
    """
    if stage == 'initial':
        class ResponseFormat(BaseModel):
            general: list[str]
            specific: list[str]
    elif stage == 'rewrite':
        class ResponseFormat(BaseModel):
            document: str
    else:
        class ResponseFormat(BaseModel):
            differences: str
            general: list[str]
            specific: list[str]
            
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
    #return load_dataset("HuggingFaceFW/fineweb",
    #                    name="sample-10BT",
    #                    split="train",
    #                    streaming=True)
    
    
    #Our 40k sample
    with open("../data/fineweb_40k.jsonl", "r") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]


def initial_stage(stage, documents, vocab, llm):
    """
    Processes a list of documents through the initial generation stage.

    Args:
        documents (list of str): A list of documents to be processed.
        vocab (list of str): The current general descriptor vocabulary.
        stage (str): The current stage of the pipeline.
        llm (object): The language model to be used for generating outputs.

    Returns:
        list of dict: A list of validated and possibly reformatted outputs in JSON format.
    """
    if len(vocab) == 0:
        vocab = "The list of general descriptors is currently empty."
    else:
        vocab = '\n'.join(vocab)

    prompts = [format_prompt(stage=stage, original=document, vocab=vocab) for document in documents]
    json_schema = get_response_format(stage)
    batched_outputs = generate(llm, prompts, json_schema)
    validated_outputs = []
    for output in batched_outputs:
        valid_json = validate_output(output)
        if valid_json:
            validated_outputs.append(json.loads(output, strict=False))
        else:
            reformatted = reformat_output(llm, output)
            if reformatted == "FAIL":
                validated_outputs.append(json.loads('{"general": ["Generation failed"], "specific": ["Generation failed"]}'))
            else:
                validated_outputs.append(reformatted)

    return validated_outputs


def rewrite_stage(stage, general, specific, llm):
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
    prompts = []
    for g, s in zip(general, specific):
        prompts.append(format_prompt(stage=stage, general=g, specific=s))
    json_schema = get_response_format(stage)
    batched_output = generate(llm, prompts, json_schema)
    validated_outputs = []
    for output in batched_output:
        valid_json = validate_output(output)
        if valid_json:
            validated_outputs.append(json.loads(output, strict=False))
        else:
            reformatted = reformat_output(llm, output)
            if reformatted == "FAIL":
                validated_outputs.append(json.loads('{"document": "Generation failed."}'))
            else:
                validated_outputs.append(reformatted)

    return validated_outputs


def revise_stage(stage, document, rewritten, general, specific, vocab, llm):
    """
    Processes lists of descriptors and rewrites through the revise generation stage.

    Args:
        stage (str): The current stage of the pipeline.
        document (list of str): The original documents.
        rewritten (list of str): The rewritten documents.
        general (list of list): General descriptors for the documents.
        specific (list of list): Specific descriptors for the documents.
        vocab (list of str): The current general descriptor vocabulary.
        llm (object): The language model used for generating outputs.

    Returns:
        list of dict: A list of validated outputs, each containing general and specific descriptors.
    """
    vocab = '\n'.join(vocab)
    prompts = []
    for d,r,g,s in zip(document,rewritten, general, specific):
        prompts.append(format_prompt(stage=stage, original=d,
                                     rewritten=r, general=g,
                                     specific=s, vocab=vocab))
    json_schema = get_response_format(stage)
    batched_output = generate(llm, prompts, json_schema)
    validated_outputs = []
    for output in batched_output:
        valid_json = validate_output(output)
        if valid_json:
            validated_outputs.append(json.loads(output, strict=False))
        else:
            reformatted = reformat_output(llm, output)
            if reformatted == "FAIL":
                validated_outputs.append(json.loads('{"general": ["Generation failed"], "specific": ["Generation failed"]}'))
            else:
                validated_outputs.append(reformatted)

    return validated_outputs


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
    # Remove any text outside curly brackets
    json_start = output.find('{')
    json_end = output.find('}')
    if json_start != -1 and json_end != -1:
        output = output[json_start:json_end + 1]  # Include the '}'
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
        return json.loads(output, strict=False)

    # If fixing JSON with regex does not work,
    # we try giving it to the model to fix.
    for _ in range(3):
        prompt = prompts.reformat_output_prompt(output)
        output = chat(llm, prompt)
        valid_json = validate_output(output)
        if valid_json:
            return json.loads(output, strict=False)

    logging.warning("Failed to fix JSON formatting.")
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


def save_best_results(results, run_id):
    """
    Save the best results from the given results dictionary to a JSONL file.

    This function iterates through the results dictionary, finds the best result
    based on the highest similarity score, and writes the best result to a JSONL
    file. The file is named using the provided run_id.

    Args:
        results (dict): A dictionary where keys are document identifiers and values
                        are dictionaries containing the following keys:
                        - "general": List of general descriptors.
                        - "specific": List of specific descriptors.
                        - "rewrite": List of document rewrites.
                        - "similarity": List of similarity scores.
        run_id (str): Identifier for the current run, used to name the output file.

    Returns:
        list: A list of the best general descriptors for each document in the results.

    Raises:
        Exception: If an error occurs during file writing, a warning is logged and
                   the function returns the list of best general descriptors.
    """
    try:
        with open(f"../results/descriptors_{run_id}.jsonl", "a", encoding="utf8") as f:
            for doc in results.values():
                best_index = doc["similarity"].index(max(doc["similarity"]))
                doc["general"] = doc["general"][best_index]
                doc["specific"] = doc["specific"][best_index]
                doc["rewrite"] =  doc["rewrite"][best_index].encode("utf-8", errors="ignore").decode("utf-8"), # Remove possible code breaking chars.
                doc["similarity"] = doc["similarity"][best_index]
                json_line = json.dumps(doc, ensure_ascii=False)
                f.write(json_line + "\n")

        return [doc["general"] for doc in results.values()]

    except Exception as e:
        logging.warning("Saving  results failed. Results will be discarded.")
        logging.warning(e)
        return [doc["general"] for doc in results.values()]


def initialise_descriptor_vocab(use_previous_descriptors, path):
    """
    Initializes a vocabulary of descriptors from a file or creates an empty dictionary.

    Args:
        use_previous_descriptors (bool): If True, attempts to load descriptors from the specified file.
                                         If False, initializes an empty dictionary.
        path (str): The file path to load the descriptors from if use_previous_descriptors is True.

    Returns:
        defaultdict: A dictionary with descriptors as keys and their frequencies as values.
    """
    descriptors = defaultdict(int)

    if use_previous_descriptors:
        logging.info("use_previous_descriptors=True")
        logging.info("Set this to False if you want to start with an empty dictionary.")
        try:
            with open(path, "r") as f:
                file = f.readlines()
                for line in file:
                    line = line.strip().split("\t")
                    desc, freq = line
                    descriptors[desc] = int(freq)
            return descriptors
        except FileNotFoundError:
            logging.info("No previous descriptors found. Defaulting to empty dictionary.")
            return descriptors
    else:
        return descriptors


def save_descriptors(vocab, path):
    """
    Save vocabulary descriptors and their frequencies to a file.

    Args:
        vocab (list of tuples): A list of tuples where each tuple contains a descriptor (str) and its frequency (int).
        path (str): The file path where the descriptors will be saved.

    Writes:
        A tab-separated file with each line containing a descriptor and its frequency.
    """
    with open(path, "w", encoding="utf8") as f:
        for desc, freq in vocab:
            f.write(f"{desc}\t{freq}\n")
            
            
def count_unique_descriptors(vocab, run_id):
    """
    Counts the number of unique descriptors in the given vocabulary and appends the count to a results file.

    Args:
        vocab (set): A set containing unique descriptors.
        run_id (str): A unique identifier for the current run, used to name the results file.

    Writes:
        Appends the count of unique descriptors to a file named "descriptor_count_growth_<run_id>.txt" located in the "../results/" directory.
    """
    with open(f"../results/descriptor_count_growth_{run_id}.txt", "a") as f:
        f.write(f"{len(vocab)}\n")


def return_top_descriptors(descriptor_counts_sorted, max_vocab):
    """
    Returns the top descriptors from a sorted list of descriptor counts.

    Args:
        descriptor_counts_sorted (list of tuples): A list of tuples where each tuple contains a descriptor and its count, sorted by count in descending order.
        max_vocab (int): The maximum number of descriptors to return. If -1, all descriptors are returned.

    Returns:
        list: A list of the top descriptors, limited by max_vocab if specified.
    """
    if max_vocab == -1:
        return [desc[0] for desc in descriptor_counts_sorted]
    else:
        return [desc[0] for desc in descriptor_counts_sorted][:max_vocab]


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
            batch= []
    if batch:
        yield batch


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
    use_previous_descriptors = args.use_previous_descriptors
    descriptor_path = args.descriptor_path
    run_id = args.run_id
    num_rewrites = args.num_rewrites
    batch_size = args.batch_size
    max_vocab=args.max_vocab
    global temperature
    temperature = args.temperature

    logging.info("Loading model...")
    llm = LLM_setup(model, cache_dir)
    logging.info("Loading data...")
    data = load_documents()

    if not descriptor_path:
        descriptor_path = f"../results/descriptor_vocab_{run_id}.tsv"
    descriptor_counts = initialise_descriptor_vocab(use_previous_descriptors, descriptor_path)
    # Keep the top 100 general descriptors. These will be given to the model as possible options.
    descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
    descriptor_vocab = return_top_descriptors(descriptor_counts_sorted, max_vocab)
    # Shuffle the list of descriptors to avoid ordering bias
    shuffle(descriptor_vocab)

    logging.info("Starting document processing...")
    for batch_num, batch in enumerate(batched(data, batch_size, start_index)):

        start_time = time.time()

        results = {
            index: {
                "document": doc["text"],
                "doc_id": doc["id"],
                "general": [],
                "specific": [],
                "rewrite": [],
                "similarity": []
            }
            for index, doc in enumerate(batch)
        }

        # Generate initial descriptors for document.
        documents = [doc["text"] for doc in batch]
        stage = "initial"
        logging.info(f"Stage: {stage}")
        model_outputs = initial_stage(stage, documents, descriptor_vocab, llm)
        general_descriptors = [output.get("general", "Generation failed.") for output in model_outputs]
        specific_descriptors = [output.get("specific", "Generation failed.") for output in model_outputs]

        for index in results:
            results[index]["general"].append(general_descriptors[index])
            results[index]["specific"].append(specific_descriptors[index])

        # Generate num_rewrites rewrites of the document based on descriptors.
        # After the rewrite, we revise the descriptors to create an even better rewrite.

        for round_num in range(num_rewrites):
            # Rewrite doc based on the descriptors.
            stage = "rewrite"
            logging.info(f"Stage: {stage}")
            model_outputs = rewrite_stage(stage,
                                          general_descriptors,
                                          specific_descriptors,
                                          llm)

            rewrites = [output.get("document", "Generation failed.") for output in model_outputs]
            for index in results:
                results[index]["rewrite"].append(rewrites[index])

            if not round_num == num_rewrites-1:
                # Evaluate rewrite and revise descriptors.
                # This stage is skipped on the last round: since we do not do another rewrite
                # we do not need another set of descriptors.
                # This saves us one LLM call.
                stage = "revise"
                logging.info(f"Stage: {stage}")
                model_outputs = revise_stage(stage,
                                             documents,
                                             rewrites,
                                             general_descriptors,
                                             specific_descriptors,
                                             descriptor_vocab,
                                             llm)

                general_descriptors = [output.get("general", "Generation failed.") for output in model_outputs]
                specific_descriptors = [output.get("specific", "Generation failed.") for output in model_outputs]

                for index in results:
                    results[index]["general"].append(general_descriptors[index])
                    results[index]["specific"].append(specific_descriptors[index])

        for index in results:
            similarities = calculate_doc_similarity(results[index]["document"],
                                                    results[index]["rewrite"],
                                                    cache_dir)
            results[index]["similarity"].extend(similarities)

        # Save best result based on similarity score between original and rewrite.
        # Return the best general descriptors.
        best_descriptors = save_best_results(results, run_id)
        logging.info("Results saved.")

        # Update descriptor counts.
        for desc_list in best_descriptors:
            for desc in desc_list:
                descriptor_counts[desc] += 1

        # Sort descriptors by their frequency.
        # This creates a sorted list of tuples (descriptor, count)
        descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
        save_descriptors(descriptor_counts_sorted, descriptor_path)
        count_unique_descriptors(descriptor_counts_sorted, run_id)

        # Keep max_vocab most common general descriptors. These will be given to the model as possible options.
        descriptor_vocab = return_top_descriptors(descriptor_counts_sorted, max_vocab)
        # Shuffle the list of descriptors to avoid ordering bias
        shuffle(descriptor_vocab)

        end_time = time.time()

        logging.info(f"Processed {len(results)} documents in {round(end_time-start_time, 2)} seconds.")

        # Stop run after num_batches batches have been processed.
        # If -1, we continue until we run out of data or time.
        if num_batches == -1:
            continue
        elif batch_num+1 >= num_batches:
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for getting document descriptors with LLMs.")

    parser.add_argument("--run-id", type=str, required=True,
                        help="ID for this run, e.g. run1")
    parser.add_argument("--cache-dir", type=str, default="../.cache",
                        help="Path to cache directory, where model is or will be saved.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Name of model to use.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Index of first document to analyse.")
    parser.add_argument("--num-batches", type=int, default=-1,
                        help="Number of batches of size batch-size to process. Set to -1 to process all data.")
    parser.add_argument("--use-previous-descriptors", action="store_true",
                        help="Use descriptors used in a previous run as a starting point.")
    parser.add_argument("--descriptor-path", type=str, default="",
                        help="Path to descriptors, if using previous descriptors")
    parser.add_argument("--num-rewrites", type=int, default=0,
                        help="How many rewriting cycles the script should go through.")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Model temperature.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of documents given to the model at one time.")
    parser.add_argument("--max-vocab", type=int, default=-1,
                        help="Max number of descriptors given in the prompt. Give -1 to use all descriptors.")

    args = parser.parse_args()

    # Log the run settings
    with open(f"../results/{args.run_id}_settings.txt", "w") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Create required directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    main(args)

