from vllm import LLM, SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import huggingface_hub
import os
import torch
import torch.distributed as dist
import time
import prompts
import json
import ast
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import evaluate
import sys


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

    

def LLM_setup(cache_dir):
    """
    Sets up the Language Model (LLM) with specified parameters.

    Args:
        cache_dir (str): Directory to cache the downloaded model.

    Returns:
        LLM: An instance of the LLM class initialized with the specified settings.
    """
    return LLM(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        download_dir=cache_dir,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=False,
        gpu_memory_utilization=0.95,
        max_model_len=3_000
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
        temperature=0.5,
        top_p=0.95,
        max_tokens=3000,
        logits_processors=[logits_processor]
    )


def print_gpu_memory_summary():
    num_gpus = torch.cuda.device_count()

    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        print(f'Memory summary for GPU {gpu_id}')
        print(torch.cuda.memory_summary(device=gpu_id))
        print('-' * 80)


def get_response_format(stage):
    if stage == 'initial':
        class ResponseFormat(BaseModel):
            keyphrases: list
    elif stage == 'rewrite':
        class ResponseFormat(BaseModel):
            document: str
    else:
        class ResponseFormat(BaseModel):
            differences: str
            keyphrases: list

    return ResponseFormat


def print_execution_time(start_time, end_time):
    """
    Prints the total time taken to process data.

    Args:
        end_time (float): End time of the process.
        start_time (float): Start time of the process.
    """
    print('========================')
    print(f'Total time taken to generate answers: {end_time - start_time}')
    print('========================')


def format_prompt(stage, original, rewritten='', list=''):
    """
    Formats the prompt based on the current processing stage.

    Args:
        stage (str): The processing stage ('initial', 'rewrite', or 'revise').
        original (str): The original document text.
        rewritten (str, optional): The rewritten text if available. Defaults to ''.
        list (list, optional): List of keyphrases if available. Defaults to ''.

    Returns:
        str: The formatted prompt based on the stage and inputs.
    """
    if stage == 'initial':
        message = prompts.initial_prompt(original)
    elif stage == 'rewrite':
        message = prompts.rewrite_prompt(list)
    else:
        message = prompts.revise_keyphrases_prompt(original, rewritten, list)
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
        #extra_body={"guided_json": ResponseFormat.model_json_schema()}
    )[0].outputs[0].text


def extract_output(stage, output):
    """
    Extracts keyphrases from the model output based on the processing stage.

    Args:
        stage (str): The current processing stage ('initial' or 'revise').
        output (str): The output text from the LLM.

    Returns:
        list or None: Extracted keyphrases as a list, or None if extraction fails.
    """
    if stage == 'initial':
        return ast.literal_eval(output).get('keyphrases')
    elif stage == 'revise':
        return ast.literal_eval(output).get('keyphrases')


def load_documents():
    """
    Loads and streams documents from a specified dataset.

    Returns:
        Dataset: A streaming dataset split to be processed in training mode.
    """
    return load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)


def save_results(name, document, rewrites, keyphrase_lists):
    """
    Saves generated results to a text file in JSON format.

    Args:
        results (list): List of dictionaries containing generated results.
    """
    with open(f'../results/{name}_vllm_document_original.txt', 'w', encoding='utf-8') as f:
        f.write(document)

    with open(f'../results/{name}_vllm_document_rewrites.txt', 'w', encoding='utf-8') as f:
        for doc in rewrites:
            f.write(doc)
    
    with open(f'../results/{name}_vllm_document_keyphrases.txt', 'w', encoding='utf-8') as f:
        for i in keyphrase_lists:
            f.write('=======================\n')
            for phrase in i:
                f.write(f'{phrase}\n')
            f.write('=======================\n')


def report_memory_usage(out=sys.stdout):
    print(f'Max memory allocation:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)


def check_devices(model):
    print(f'Devices:', file=sys.stdout)
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(f'  {name}.{param_name}:{param.device}', file=sys.stderr)
            if param.device.type != 'cuda':
                warning(f'{name}.{param_name} on device {param.device}')


def main(start_at_index=0, stop_at_index=100, file_id='run1'):
    """
    Main function to set up the model, generate responses, and save the results.

    - Initializes cache directory and sets up the LLM.
    - Iterates through each document in the dataset, generating responses for each stage.
    - Collects and saves results, and prints execution time.
    """
    
    cache_dir = "/scratch/project_2011109/otto/LLM_data_labelling/hf_cache"

    llm = LLM_setup(cache_dir)
    data = load_documents()
    keyphrase_lists = []
    rewrites = []

    print_gpu_memory_summary()
    report_memory_usage()

    for i, line in enumerate(data):
        if i < start_at_index:
            continue
        
        document = line['text']
        print('Original document:')
        print(document)
        print('================================')
        start_time = time.time()
        stage = 'initial'
        prompt = format_prompt(stage=stage, original=document)
        output = generate(llm, prompt, stage)
        keyphrases = extract_output(stage=stage, output=output)
        keyphrase_lists.append(keyphrases)
        print('Keyphrases:')
        print(keyphrases)
        print('================================')
        
        doc_similarity = 0
        while doc_similarity <= 0.9:
            stage = 'rewrite'
            prompt = format_prompt(stage=stage, original=document, list=keyphrases)
            rewritten = generate(llm, prompt, stage)
            rewrites.append(rewritten)
    
            stage = 'revise'
            prompt = format_prompt(stage=stage, original=document, rewritten=rewritten, list=keyphrases)
            output = generate(llm, prompt, stage)
            keyphrases = extract_output(stage=stage, output=output)
            keyphrase_lists.append(keyphrases)

            #doc_similarity = calculate_doc_similarity(document, rewritten)
            #print(f'Similarity score: {doc_similarity}')
            end_time = time.time()
            print_execution_time(start_time, end_time)
            save_results(file_id, document, rewrites, keyphrase_lists)

            report_memory_usage()
            #check_devices(llm)

        if i >= stop_at_index:
            break


if __name__ == '__main__':
    main(stop_at_index = 2,
        start_at_index=2,
        file_id='doc2')
