from vllm import LLM, SamplingParams
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
from random import shuffle
from collections import defaultdict
import argparse
import re
import logging


# Configure logging
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_id')
logging.basicConfig(
    filename=f'../logs/{slurm_job_id}.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Suppress sentence_transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# Suppress transformers logging (used internally by sentence_transformers)
logging.getLogger("transformers").setLevel(logging.WARNING)


def LLM_setup(model, cache_dir):
    return LLM(
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


def calculate_doc_similarity(original, rewrite, cache):
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
    if stage == 'initial':
        message = prompts.initial_prompt(original, vocab)
    elif stage == 'rewrite':
        message = prompts.rewrite_prompt(general, specific)
    else:
        message = prompts.revise_keyphrases_prompt(original, rewritten, general, specific, vocab)
    return message


def chat(llm, message):
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


def generate(llm, batched_input):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.5,
        repetition_penalty=1.2, # small repetition penalty
        max_tokens=3000, #max tokens to generate
    )

    batched_outputs = llm.generate(
        batched_input,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    return [out.outputs[0].text.strip(" `\n") for out in batched_outputs]



def load_documents():
    return load_dataset('HuggingFaceFW/fineweb',
                        name='sample-10BT',
                        split='train',
                        streaming=True)


def initial_stage(documents, vocab, stage, llm):
    if len(vocab) == 0:
        vocab = "The list of general descriptors is currently empty."
    else:
        vocab = '\n'.join(vocab)

    prompts = [format_prompt(stage=stage, original=document, vocab=vocab) for document in documents]
    batched_outputs = generate(llm, prompts)
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
    prompts = []
    for g, s in zip(general, specific):
        prompts.append(format_prompt(stage=stage, general=g, specific=s))
    batched_output = generate(llm, prompts)
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
    vocab = '\n'.join(vocab)
    prompts = []
    for d,r,g,s in zip(document,rewritten, general, specific):
        prompts.append(format_prompt(stage=stage, original=d,
                                     rewritten=r, general=g,
                                     specific=s, vocab=vocab))
    batched_output = generate(llm, prompts)
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

    for _ in range(3):
        prompt = prompts.reformat_output_prompt(output)
        output = chat(llm, prompt)
        valid_json = validate_output(output)
        if valid_json:
            return json.loads(output, strict=False)

    # If unable to fix output, return generic dictionary.
    # This should allow the code to keep running.
    logging.warning("Failed to fix JSON formatting.")
    return json.loads('FAIL')


def validate_output(output):
    try:
        parsed_json = json.loads(output, strict=False)
        return True
    except json.JSONDecodeError as e:
        logging.debug(e)
        logging.debug('Invalid JSON output:')
        logging.debug(repr(output))
        return False


def save_best_results(results, run_id):
    try:
        with open(f'../results/descriptors_{run_id}.jsonl', 'a', encoding='utf8') as f:
            for doc in results.values():
                best_index = doc['similarity'].index(max(doc['similarity']))
                doc['general'] = doc['general'][best_index]
                doc['specific'] = doc['specific'][best_index]
                doc['rewrite'] =  doc['rewrite'][best_index].encode('utf-8', errors='ignore').decode('utf-8'), # Remove possible code breaking chars.
                doc['similarity'] = doc['similarity'][best_index]
                json_line = json.dumps(doc, ensure_ascii=False)
                f.write(json_line + '\n')

        return [doc["general"] for doc in results.values()]

    except Exception as e:
        logging.warning("Saving  results failed.")
        logging.warning(e)
        return [doc["general"] for doc in results.values()]


def initialise_descriptor_vocab(use_previous_descriptors, path):
    descriptors = defaultdict(int)

    if use_previous_descriptors:
        logging.info('use_previous_descriptors=True')
        logging.info('Set this to False if you want to start with an empty dictionary.')
        try:
            with open(path, 'r') as f:
                file = f.readlines()
                for line in file:
                    line = line.strip().split('\t')
                    desc, freq = line
                    descriptors[desc] = int(freq)
            return descriptors
        except FileNotFoundError:
            logging.info("No previous descriptors found. Defaulting to empty dictionary.")
            return descriptors
    else:
        return descriptors


def save_descriptors(vocab, path):
    with open(path, "w", encoding="utf8") as f:
        for desc, freq in vocab:
            f.write(f"{desc}\t{freq}\n")


def return_top_descriptors(descriptor_counts_sorted):
    return [desc[0] for desc in descriptor_counts_sorted][:100]


def batched(data, batch_size, start_index):
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
    cache_dir = args.cache_dir
    model = args.model
    start_index = args.start_index
    end_index = args.end_index
    use_previous_descriptors = args.use_previous_descriptors
    descriptor_path = args.descriptor_path
    run_id = args.run_id
    num_rewrites = args.num_rewrites
    batch_size = args.batch_size
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
    descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)

    logging.info("Starting document processing...")
    for batch in batched(data, batch_size, start_index):

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
        model_outputs = initial_stage(documents, descriptor_vocab, stage, llm)
        general_descriptors = [output.get("general") for output in model_outputs]
        specific_descriptors = [output.get("specific") for output in model_outputs]

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

            rewrites = [output.get("document") for output in model_outputs]
            for index in results:
                results[index]["rewrite"].append(rewrites[index])

            if not round_num == num_rewrites-1:
                # Evaluate rewrite and revise descriptors.
                # This stage is skipped on the last round because since we do not do another rewrite
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

                general_descriptors = [output.get("general") for output in model_outputs]
                specific_descriptors = [output.get("specific") for output in model_outputs]

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

        # Keep the 100 most common general descriptors. These will be given to the model as possible options.
        descriptor_vocab = return_top_descriptors(descriptor_counts_sorted)

        end_time = time.time()

        logging.info(f"Processed {len(results)} documents in {round(end_time-start_time, 2)} seconds.")

        # Stop run at given index.
        # If -1, we continue until we run out of data or time.
        if end_index == -1:
            continue
        elif i >= end_index:
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for getting document descriptors with LLMs.")

    parser.add_argument("--run-id", type=str, required=True,
                        help="ID for this run, e.g. run1")
    parser.add_argument("--cache-dir", type=str, default="../hf_cache",
                        help="Path to cache directory, where model is or will be saved.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Name of model to use.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Index of first document to analyse.")
    parser.add_argument("--end-index", type=int, default=-1,
                        help="Index of last document to analyse. Give -1 to set no stopping index.")
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

    args = parser.parse_args()

    # Log the run settings
    with open(f'../results/{args.run_id}_settings.txt', 'w') as f:
        f.write(f'slurm id: {os.environ.get("SLURM_JOB_ID")}\n')
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

    # Create required directories
    os.makedirs('../logs', exist_ok=True)
    os.makedirs('../results', exist_ok=True)

    main(args)

