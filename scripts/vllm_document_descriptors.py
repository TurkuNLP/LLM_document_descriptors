from vllm import LLM, SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import huggingface_hub
import os
import torch
import torch.distributed as dist
import time
import prompts
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import sys
from pydantic import BaseModel
from random import shuffle
from collections import defaultdict
import argparse
import logging


# Configure logging
slurm_job_id = os.environ.get('SLURM_JOB_ID')
# Configure logging
logging.basicConfig(
    filename=f'../logs/{slurm_job_id}.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def LLM_setup(model, cache_dir):
    """
    Sets up the Language Model (LLM) with specified parameters.

    Args:
        cache_dir (str): Directory to cache the downloaded model.

    Returns:
        LLM: An instance of the LLM class initialized with the specified settings.
    """
    return LLM(
        #model="meta-llama/Llama-3.1-70B-Instruct",
        model=model,
        download_dir=cache_dir,
        dtype='bfloat16',
        max_model_len=128_000,
        tensor_parallel_size=torch.cuda.device_count(),
        #pipeline_parallel_size=2, # use if us need run on multiple nodes
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        #quantization="bitsandbytes",
        #load_format="bitsandbytes",
    )


def get_sampling_params(stage, llm):
    """
    Creates and returns sampling parameters for generating responses.

    Returns:
        SamplingParams: An instance of SamplingParams with set parameters for text generation.
    """

    response_format = get_response_format(stage)
    logits_processor = JSONLogitsProcessor(schema=response_format, llm=llm.llm_engine)

    return SamplingParams(
        temperature=0.0,
        top_p=0.5,
        max_tokens=8_000, # max tokens to generate
        logits_processors=[logits_processor] # ensure correct JSON formatting
    )


def calculate_doc_similarity(original, rewrite):
    """
    Calculates cosine similarities between the original document and the rewrites.

    Args:
        original (str): The original document.
        rewrites (list): List of the rewritten documents.

    Returns:
        dict: Dictionary of rewritten documents and their similarity scores.
    """
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    original_embedding = model.encode([original])
    rewrite_embedding = model.encode([rewrite])

    # Compute cosine similarities
    similarity = model.similarity(original_embedding, rewrite_embedding)

    # Return similarity between documents
    return round(float(similarity), 4)


def get_response_format(stage):
    """
    Returns the appropriate response format based on the stage of execution.

    Args:
        stage (str): The stage of execution the program is in.

    Returns:
        ResponseFormat: A Pydantic model for formatting responses.
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

    return ResponseFormat


def format_prompt(stage, original=None, rewritten=None, general=None, specific=None, vocab=None):
    """
    Formats the prompt based on the current processing stage.

    Args:
        stage (str): The processing stage ('initial', 'rewrite', or 'revise').
        original (str): The original document text.
        rewritten (str, optional): The rewritten text if available. Defaults to None.
        general (list, optional): List of general descriptors if available. Defaults to None.
        specific (list, optional): List of specific descriptors if available. Defaults to None.
        vocab (list, optional): Vocabulary to include in the prompt. Defaults to None.

    Returns:
        str: The formatted prompt based on the stage and inputs.
    """
    if stage == 'initial':
        message = prompts.initial_prompt(original, vocab)
    elif stage == 'rewrite':
        message = prompts.rewrite_prompt(general, specific)
    else:
        message = prompts.revise_keyphrases_prompt(original, rewritten, general, specific, vocab)
    return message


def generate(llm, message, stage):
    """
    Generates a response from the LLM based on the input message and sampling parameters.

    Args:
        llm (LLM): The language model instance.
        message (str): The input message for the LLM to process.
        sampling_params (SamplingParams): The parameters used for generating the response.

    Returns:
        str: The generated text output from the LLM.
    """
    # Disabled JSON formatted output to test effect on thoughput
    #sampling_params = get_sampling_params(stage, llm)

    sampling_params = SamplingParams(
           temperature=temperature,
           top_p=0.5,
           max_tokens=3_000, # max tokens to generate
           )

    output = llm.chat(
        messages=message,
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0].outputs[0].text.strip(" `") # The model tends to generate these ticks
                                     # around JSON strings, which cause issues.

    return output


def load_documents():
    """
    Loads and streams documents from a specified dataset.

    Returns:
        Dataset: A streaming dataset split to be processed in training mode.
    """
    return load_dataset('HuggingFaceFW/fineweb',
                        name='sample-10BT',
                        split='train',
                        streaming=True)


def initial_stage(document, vocab, stage, llm):
    """
    Generates initial descriptors for a given document.

    Args:
        document (str): Document text.
        vocab (list): Vocabulary for descriptor generation.
        stage (str): Current processing stage.
        client (OpenAI): OpenAI client instance.

    Returns:
        tuple: General and specific descriptors.
    """
    if len(vocab) == 0:
        vocab = "The list of general descriptors is currently empty."
    else:
        vocab = '\n'.join(vocab)

    prompt = format_prompt(stage=stage, original=document, vocab=vocab)
    output = generate(llm, prompt, stage)
    valid_json = validate_output(output)
    if valid_json:
        output = json.loads(output, strict=False)
    else:
        output = reformat_output(llm, output)
        # If reformatting fails, return skip and move on to next document.
        if output == "skip":
            return "skip", "skip"

    return output['general'], output['specific']


def rewrite_stage(stage, general, specific, llm):
    """
    Rewrites a document based on provided descriptors.

    Args:
        stage (str): Current processing stage.
        general (list): General descriptors.
        specific (list): Specific descriptors.
        client (OpenAI): OpenAI client instance.

    Returns:
        str: Rewritten document.
    """
    prompt = format_prompt(stage=stage,
                           general=general,
                           specific=specific)
    output = generate(llm, prompt, stage)
    valid_json = validate_output(output)
    if valid_json:
        output = json.loads(output, strict=False)
    else:
        output = reformat_output(llm, output)
        # If reformatting fails, return skip and move on to next document.
        if output == "skip":
            return "skip"

    return output['document']


def revise_stage(stage, document, rewritten, general, specific, vocab, llm):
    """
    Revises descriptors based on a rewritten document.

    Args:
        stage (str): Current processing stage.
        document (str): Original document text.
        rewritten (str): Rewritten document.
        general (list): General descriptors.
        specific (list): Specific descriptors.
        vocab (list): Vocabulary for descriptor generation.
        client (OpenAI): OpenAI client instance.
    Returns:
        tuple: Revised general and specific descriptors.
    """
    vocab = '\n'.join(vocab)
    prompt = format_prompt(stage=stage,
                           original=document,
                           rewritten=rewritten,
                           general=general,
                           specific=specific,
                           vocab=vocab)
    output = generate(llm, prompt, stage)
    valid_json = validate_output(output)
    if valid_json:
        output = json.loads(output, strict=False)
    else:
        output = reformat_output(llm, output)
        # If reformatting fails, return skip and move on to next document.
        if output == "skip":
            return "skip", "skip"

    return output['general'], output['specific']


def reformat_output(llm, output):
    """
    If model output is not valid JSON, try reformatting it by calling LLM.
    If after 3 tries the output is still not valid, skip the document.
    """

    for _ in range(3):
        prompt = prompts.reformat_output_prompt(output)
        output = generate(llm, prompt, None)
        valid_json = validate_output(output)
        if valid_json:
            return json.loads(output, strict=False)

    return "skip"


def validate_output(output):
    try:
        parsed_json = json.loads(output, strict=False)
        return True
    except json.JSONDecodeError as e:
        #print(e)
        #print('Invalid JSON output:')
        #print(repr(output))
        #print()
        return False


def save_best_results(document, doc_id, rewrites, general, specific, similarity_scores, run_id, print_results=False):
    """
    Saves the best results (highest similarity) among multiple rewrites.

    Args:
        document (str): Original document.
        rewrites (list): List of rewritten documents.
        general (list): General descriptors for each rewrite.
        specific (list): Specific descriptors for each rewrite.
        similarity_scores (list): Similarity scores for each rewrite.
        run_id (str): Run identifier.

    Returns:
        list: Best general descriptors.
    """
    if len(rewrites) == 0:
        best_index = 0
        results = {
                'document': document,
                'id': doc_id,
                'general_descriptors': general[0],
                'specific_descriptors': specific[0],
                }
    else:
        best_index = similarity_scores.index(max(similarity_scores))
        results = {
            'document': document,
            'id': doc_id,
            'rewrite': rewrites[best_index].encode('utf-8', errors='ignore').decode('utf-8'), # Remove possible code breaking chars.
            'similarity': similarity_scores[best_index],
            'general_descriptors': general[best_index],
            'specific_descriptors': specific[best_index],
        }

    if print_results:
        print('======================')
        print('BEST RESULTS:')
        for key, value in results.items():
            print(key)
            print(value)
            print()
        print('======================')

    with open(f'../results/descriptors_{run_id}.jsonl', 'a', encoding='utf8') as f:
        f.write(json.dumps(results, ensure_ascii=False))
        f.write('\n')

    return general[best_index]


def initialise_descriptor_vocab(use_previous_descriptors, path):
    """
    Initializes the descriptor vocabulary.

    Args:
        use_previous_descriptors (bool): Whether to load previous descriptors.
        path (str): Path to the previous descriptors file.

    Returns:
        defaultdict: Initialized descriptor vocabulary.
    """

    descriptors = defaultdict(int)

    if use_previous_descriptors:
        print('use_previous_descriptors=True')
        print('Set this to False if you want to start with an empty dictionary.')
        try:
            with open(path, 'r') as f:
                file = f.readlines()
                for line in file:
                    line = line.strip().split('\t')
                    desc, freq = line
                    descriptors[desc] = int(freq)
            return descriptors
        except FileNotFoundError:
            print('No previous descriptors found. Defaulting to empty dictionary.')
            return descriptors
    else:
        return descriptors


def save_descriptors(vocab, path):
    """
    Saves the current descriptor vocabulary to a file.

    Args:
        vocab (defauldict): Dict of descriptors and their frequency.
        path (str): Path to save the vocabulary.
    """
    with open(path, 'w', encoding='utf8') as f:
        for desc, freq in vocab:
            f.write(f"{desc}\t{freq}\n")


def return_top_descriptors(descriptor_counts_sorted):
    return [desc[0] for desc in descriptor_counts_sorted][:100]


def main(args):
    """
    Main function to set up the model, generate responses, and save the results.

    - Initializes cache directory and sets up the LLM.
    - Iterates through each document in the dataset, generating responses for each stage.
    - Collects and saves results.
    """

    cache_dir = args.cache_dir
    model = args.model
    start_index = args.start_index
    end_index = args.end_index
    use_previous_descriptors = args.use_previous_descriptors
    run_id = args.run_id
    num_rewrites = args.num_rewrites
    global temperature
    temperature = args.temperature

    logging.info('Loading model...')
    llm = LLM_setup(model, cache_dir)
    logging.info('Loading data...')
    data = load_documents()


    descriptor_path = "../results/descriptor_vocab_{run_id}"
    descriptor_counts = initialise_descriptor_vocab(use_previous_descriptors, descriptor_path)
    # Keep the top 100 general descriptors. These will be given to the model as possible options.
    descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
    descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)

    for i, doc in enumerate(data):
        if i < start_index:
            continue

        start_time = time.time()

        # This is where we collect all descriptors, rewrites and document similarities.
        # At the end, we compare similarity scores and only keep the best ones.
        general_descriptor_lists = []
        specific_descriptor_lists = []
        rewrites = []
        doc_similarities = []

        document = doc['text']
        doc_id = doc['id']

        logging.info(f'Working on document {i}, id: {doc_id}')

        # Generate initial descriptors for document.
        stage = 'initial'
        general_descriptors, specific_descriptors = initial_stage(document, descriptor_vocab, stage, llm)

        # If generation fails we skip and move on.
        if general_descriptors == "skip":
            logging.info(f"Document {i} skipped.")
            continue

        general_descriptor_lists.append(general_descriptors)
        specific_descriptor_lists.append(specific_descriptors)

        # Generate num_rewrites rewrites of the document based on descriptors.
        # After the rewrite, we revise the descriptors to create an even better rewrite.


        # I'VE SET THE NUMBER OF REWRITES TO ZERO TO SKIP THIS WHOLE PROCESS!
        # It takes too long if we want to run this on 100,000 documents!
        # We can always return to it later!
        for round_num in range(num_rewrites):
            # Rewrite doc based on the descriptors.
            stage = 'rewrite'
            rewritten = rewrite_stage(stage,
                                      general_descriptors,
                                      specific_descriptors,
                                      llm)

            # If generation fails, we skip and move on.
            if rewritten == "skip":
                logging.info(f"Document {i} skipped.")
                break

            rewrites.append(rewritten)

            if not round_num == num_rewrites-1:
                # Evaluate rewrite and revise descriptors.
                # This stage is skipped on the last round because since we do not do another rewrite
                # we do not need another set of descriptors.
                # This saves us one LLM call.
                stage = 'revise'
                general_descriptors, specific_descriptors = revise_stage(stage,
                                                                        document,
                                                                        rewritten,
                                                                        general_descriptors,
                                                                        specific_descriptors,
                                                                        descriptor_vocab,
                                                                        llm)

                # If generation fails, we skip and move on
                if general_descriptors == "skip":
                    logging.info(f"Document {i} skipped.")
                    break

                general_descriptor_lists.append(general_descriptors)
                specific_descriptor_lists.append(specific_descriptors)


            doc_similarities.append(calculate_doc_similarity(document, rewritten))

        # Save best result based on similarity score between original and rewrite.
        # Return the best general descriptors.
        best_descriptors = save_best_results(document,
                                             doc_id,
                                             rewrites,
                                             general_descriptor_lists,
                                             specific_descriptor_lists,
                                             doc_similarities,
                                             run_id)

        # Update descriptor counts.
        for desc in best_descriptors:
            descriptor_counts[desc] += 1

        # Sort descriptors by their frequency and save.
        descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
        save_descriptors(descriptor_counts_sorted, descriptor_path)

        # Keep the 100 most common general descriptors. These will be given to the model as possible options.
        descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)

        end_time = time.time()

        logging.info(f"Time taken to generate descriptors for document {i}:{round(end_time-start_time, 2)} seconds.")

        # Stop run at given index.
        # If -1, we continue until we run out of data or time.
        if end_index == -1:
            continue
        elif i >= end_index:
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script for getting document descriptors with LLMs.')

    parser.add_argument('--run-id', type=str, required=True,
                        help='ID for this run, e.g. run1')
    parser.add_argument('--cache-dir', type=str, default='../hf_cache',
                        help='Path to cache directory, where model is or will be saved.')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-70B-Instruct',
                        help='Name of model to use.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Index of first document to analyse.')
    parser.add_argument('--end-index', type=int, default=-1,
                        help='Index of last document to analyse. Give -1 to set no stopping index.')
    parser.add_argument('--use-previous-descriptors', action='store_true',
                        help='Use descriptors used in a previous run as a starting point.')
    parser.add_argument('--num-rewrites', type=int, default=0,
                        help='How many rewriting cycles the script should go through.')
    parser.add_argument('--temperature', type=float, default=0,
                       help='Model temperature.')

    args = parser.parse_args()

    # Log the run settings
    with open(f'../results/{args.run_id}_settings.txt', 'w') as f:
        f.write(f'run id: {args.run_id}\n')
        f.write(f'cache dir: {args.cache_dir}\n')
        f.write(f'model: {args.model}\n')
        f.write(f'start index: {args.start_index}\n')
        f.write(f'end index: {args.end_index}\n')
        f.write(f'use previous descriptors: {args.use_previous_descriptors}\n')
        f.write(f'num rewrites: {args.num_rewrites}\n')
        f.write(f'temperature: {args.temperature}\n')

    main(args)

