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


def LLM_setup(cache_dir):
    """
    Sets up the Language Model (LLM) with specified parameters.

    Args:
        cache_dir (str): Directory to cache the downloaded model.

    Returns:
        LLM: An instance of the LLM class initialized with the specified settings.
    """
    return LLM(
        #model="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        model="meta-llama/Llama-3.1-70B-Instruct",
        download_dir=cache_dir,
        dtype='bfloat16',
        tensor_parallel_size=4, #or use torch.cuda.device_count(),
        #pipeline_parallel_size=2,
        enforce_eager=False,
        gpu_memory_utilization=1,
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
        top_p=0.95,
        max_tokens=8_000,
        logits_processors=[logits_processor]
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
    model = SentenceTransformer("all-MiniLM-L6-v2")
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


def print_execution_time(start_time, end_time):
    """
    Prints the total time taken to process data.

    Args:
        start_time (float): Start time of the process.
        end_time (float): End time of the process.
    """
    print('='*20)
    print(f'Total time taken to generate answers: {end_time - start_time}')
    print('='*20)


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
    sampling_params = get_sampling_params(stage, llm)
    
    return llm.chat(
        messages=message,
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0].outputs[0].text


def load_documents():
    """
    Loads and streams documents from a specified dataset.

    Returns:
        Dataset: A streaming dataset split to be processed in training mode.
    """
    #with open('../data/test_data.txt', 'r', encoding='utf8') as f:
    #    file = f.readlines()
    #    file = [doc.strip() for doc in file]
    #return file
    
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
    output = json.loads(generate(llm, prompt, stage))
    general = output['general']
    specific = output['specific']
    
    return general, specific


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
    output = json.loads(generate(llm, prompt, stage))
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
    output = json.loads(generate(llm, prompt, stage))
    general = output['general']
    specific = output['specific']

    return general, specific


def save_best_results(document, rewrites, general, specific, similarity_scores, run_id, print_results=False):
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
    best_index = similarity_scores.index(max(similarity_scores))
    results = {
        'document': document,
        'rewrite': rewrites[best_index],
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


def main(start_at_index=0, stop_at_index=100, use_previous_descriptors=False, descriptor_path=None, run_id='run1'):
    """
    Main function to set up the model, generate responses, and save the results.

    - Initializes cache directory and sets up the LLM.
    - Iterates through each document in the dataset, generating responses for each stage.
    - Collects and saves results.
    """
    
    cache_dir = "../hf_cache"
    
    print('Loading model...')
    llm = LLM_setup(cache_dir)
    print('Loading data...')
    data = load_documents()

    descriptor_counts = initialise_descriptor_vocab(use_previous_descriptors, descriptor_path)
    # Keep the top 100 general descriptors. These will be given to the model as possible options.
    descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
    descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)
    
    for i, line in enumerate(data):
        if i < start_at_index:
            continue
        
        print(f'Working on document {i}')

        file_id = f'{run_id}_doc{i}'
        general_descriptor_lists = []
        specific_descriptor_lists = []
        rewrites = []
        doc_similarities = []
        
        # CHANGE THIS WHEN USING THE REAL DATASET!
        document = line['text']
        #document = line
        
        # Generate initial descriptors for document
        stage = 'initial'
        general_descriptors, specific_descriptors = initial_stage(document, descriptor_vocab, stage, llm)
        general_descriptor_lists.append(general_descriptors)
        specific_descriptor_lists.append(specific_descriptors)


        for _ in range(5):
            # Rewrite doc based on the descriptors
            stage = 'rewrite'
            rewritten = rewrite_stage(stage,
                                      general_descriptors,
                                     specific_descriptors,
                                      llm)
            rewrites.append(rewritten)


            # Evaluate rewrite and revise descriptors
            stage = 'revise'
            general_descriptors, specific_descriptors = revise_stage(stage,
                                                                     document,
                                                                     rewritten,
                                                                     general_descriptors,
                                                                     specific_descriptors,
                                                                     descriptor_vocab,
                                                                     llm)
            general_descriptor_lists.append(general_descriptors)
            specific_descriptor_lists.append(specific_descriptors)
  

            doc_similarities.append(calculate_doc_similarity(document, rewritten))

        # Save best result based on similarity score between original and rewrite
        # Return the best general descriptors
        best_descriptors = save_best_results(document,
                                             rewrites,
                                             general_descriptor_lists,
                                             specific_descriptor_lists,
                                             doc_similarities,
                                             run_id)

        # Update descriptor counts and save
        for desc in best_descriptors:
            descriptor_counts[desc] += 1
            
        # Sort descriptors by their frequency and save
        descriptor_counts_sorted = sorted(descriptor_counts.items(), key=lambda item: item[1], reverse=True)
        save_descriptors(descriptor_counts_sorted, descriptor_path)
        
        # Keep the 100 most common general descriptors. These will be given to the model as possible options.
        descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)

        # Stop at given index
        if stop_at_index == -1:
            continue
        elif i >= stop_at_index:
            break

if __name__ == '__main__':
    main(start_at_index=0,
         stop_at_index=-1,
         use_previous_descriptors=False,
         descriptor_path='../results/descriptors_vllm_70B_1.tsv',
         run_id='vllm_70B_1')
