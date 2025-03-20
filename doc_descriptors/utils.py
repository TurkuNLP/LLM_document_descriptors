from collections import Counter
import copy
from datasets import load_dataset  # type: ignore
import functools
import json
import logging
import time

# Get the root logger (inherits settings from main function)
logger = logging.getLogger(__name__)

def save_results(results, path, run_id, only_best=True):
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
        only_best (bool): Whether to save all results or only those with best rewrite score.
    """
    if only_best:
        with open(path / f"descriptors_{run_id}.jsonl", "a", encoding="utf8") as f:
            for doc in results.values():
                doc_copy = copy.deepcopy(doc)  # Create a deep copy of the one set of results
                best_index = doc_copy["similarity"].index(max(doc_copy["similarity"]))
                doc_copy["general"] = doc_copy["general"][best_index]
                doc_copy["specific"] = doc_copy["specific"][best_index]
                doc_copy["rewrite"] = (
                    doc_copy["rewrite"][best_index]
                    .encode("utf-8", errors="ignore")
                    .decode("utf-8"),
                )  # Remove possible code breaking chars.
                doc_copy["similarity"] = doc_copy["similarity"][best_index]
                json_line = json.dumps(doc_copy, ensure_ascii=False)
                f.write(json_line + "\n")
                
    else:
        with open(path / f"descriptors_{run_id}.jsonl", "a", encoding="utf8") as f:
            for doc in results.values():
                json_line = json.dumps(doc, ensure_ascii=False)
                f.write(json_line + "\n")
                

def save_descriptors(desc_counts, path):
    """
    Save vocabulary descriptors and their frequencies to a file.

    Args:
        vocab (list of tuples): A list of tuples where each tuple contains a descriptor (str) and its frequency (int).
        path (str): The file path where the descriptors will be saved.

    Writes:
        A tab-separated file with each line containing a descriptor and its frequency.
    """
    with open(path, "w", encoding="utf8") as f:
        for item in desc_counts.most_common():
            f.write(f"{item[0]}\t{item[1]}\n")
            

def load_documents(source, cache):
    """
    Load documents from a specified data source.
    This function provides two options for loading documents:
    1. From the HuggingFace FineWeb dataset (commented out by default).
    2. From a local JSONL file containing a 40k sample.

    Returns:
        list: A list of documents loaded from the selected data source.
    """

    # Original fineweb sample
    if source == "original":
        return load_dataset("HuggingFaceFW/fineweb",
                            name="sample-10BT",
                            split="train",
                            streaming=True,
                            cache_dir=cache)

    # Our 40k sample
    elif source == "40k":
        with open("../data/fineweb_40k.jsonl", "r") as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines]
        
        
    elif source =="oscar":
        data = []
        with open("../data/en.tsv", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t", 1)
                data.append({"id": line[0], "text": line[1].strip()})
        return data
    
    else:
        raise ValueError(f"Invalid data source '{source}'. Should be 'original', '40k' or 'oscar'.")
    
    
def init_results(batch):
    return {
        index: {
            "document": doc["text"],
            "doc_id": doc["id"],
            "general": [],
            "specific": [],
            "rewrite": [],
            "similarity": [],
        }
        for index, doc in enumerate(batch)
    }
    
    
def initialise_descriptor_vocab(path):
    """
    Initializes a vocabulary of descriptors from a file or creates an empty dictionary.

    Args:
        use_previous_descriptors (bool): If True, attempts to load descriptors from the specified file.
                                         If False, initializes an empty dictionary.
        path (str): The file path to load the descriptors from if use_previous_descriptors is True.

    Returns:
        defaultdict: A dictionary with descriptors as keys and their frequencies as values.
    """
    descriptors = Counter()

    try:
        with open(path, "r") as f:
            file = f.readlines()
            for line in file:
                line = line.strip().split("\t")
                desc, freq = line[0], int(line[1])
                descriptors[desc] += freq
        return descriptors
    except FileNotFoundError:
        return descriptors
    
    
def save_synonym_dict(groups, path, run_id):
    # Save groups for later inspection
    with open(path / f"synonyms_{run_id}.jsonl", "a") as f:
        json_line = json.dumps(groups, ensure_ascii=False)
        f.write(json_line + "\n")
        

def log_execution_time(func):
    """Decorator that logs the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution of {func.__name__} took {time.strftime('%H:%M:%S', time.gmtime(execution_time))}.")
        return result
    return wrapper
        
        
