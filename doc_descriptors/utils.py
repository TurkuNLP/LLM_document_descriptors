from collections import Counter
import copy
from datasets import load_dataset  # type: ignore
import functools
import json
import logging
import time
import numpy as np
from pathlib import Path

# Get the root logger (inherits settings from main function)
logger = logging.getLogger(__name__)

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


def sanitize_unicode(obj):
    if isinstance(obj, str):
        return obj.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')
    elif isinstance(obj, dict):
        return {sanitize_unicode(k): sanitize_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_unicode(i) for i in obj]
    return obj  


def save_results(results, path, run_id, only_best=True):
    output_file = path / f"descriptors_{run_id}.jsonl"

    with open(output_file, "a", encoding="utf-8") as f:
        if only_best:
            best_results = get_best_results(results)
            for doc in best_results.values():
                # Sanitize the document to avoid encoding issues
                doc = sanitize_unicode(doc)
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")
        else:
            for doc in results.values():
                # Sanitize the document to avoid encoding issues
                doc = sanitize_unicode(doc)
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")


def get_best_results(results):
    best_results = {}
    for idx, doc in results.items():
        # Find index of results with best similarity score
        best_index = np.argmax(doc["similarity"])

        best_results[idx] = {
            "document": doc["document"],
            "doc_id": doc["doc_id"],
            "descriptors": doc["general"][best_index],
            "specifics": doc["specific"][best_index],
            "rewrite": doc["rewrite"][best_index],
            "similarity": doc["similarity"][best_index],
        }

    return best_results


def get_best_descriptors(results):
    best_descriptors = []
    
    for doc in results.values():
        # Find index of descriptors with best similarity score
        best_index = np.argmax(doc["similarity"])
        best_descriptors.extend(doc["descriptors"][best_index])

    return best_descriptors


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
    Load documents from a specified data source. Could be a local file or a dataset from Hugging Face.

    Returns:
        list: A list of documents loaded from the selected data source.
    """
    
    # Try, if source is a JSONL
    try:
        path = Path(source)
        if path.is_file() and path.suffix == ".jsonl":
            with open(path, "r") as f:
                lines = f.readlines()
                return [json.loads(line) for line in lines]
    except Exception:
        pass
    
    # Try, if source if parquet file
    try:
        path = Path(source)
        if path.is_file() and path.suffix == ".parquet":
            import pandas as pd #type:ignore
            df = pd.read_parquet(source)
            data = df.to_dict(orient="records")
            return data
    except Exception:
        pass

    # Original fineweb sample
    if source.lower() == "fineweb":
        return load_dataset("HuggingFaceFW/fineweb",
                            name="sample-10BT",
                            split="train",
                            streaming=True,
                            cache_dir=cache)
        
    elif source.lower() =="core":
        data = []
        with open("../data/core/train.tsv", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t")
                data.append({"id": line[1], "text": line[2].strip()})
        return data
    
    elif source.lower() == "tweet_sentiment_extraction":
        return load_dataset("mteb/tweet_sentiment_extraction",
                    split="train",
                    streaming=True,
                    cache_dir=cache)
                
    elif source.lower() == "emotion":
        return load_dataset("mteb/emotion",
                            split="train",
                            streaming=True,
                            cache_dir=cache)
        
    elif source.lower() == "arxiv":
        return load_dataset("mteb/ArxivClassification",
                            split="train",
                            streaming=True,
                            cache_dir=cache)
        
    elif source.lower() == "imdb":
        return load_dataset("stanfordnlp/imdb",
                            split="train",
                            streaming=True,
                            cache_dir=cache)
        
    else:
        try:
            return load_dataset(source,
                            split="train",
                            streaming=True,
                            cache_dir=cache)
        
        except:
            raise ValueError(f"Invalid data source '{source}'.")
    
def init_results(batch):
    return {
        index: {
            "document": doc["text"],
            "doc_id": doc.get("id", 0),
            "descriptors": [],
            "specifics": [],
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
                try:
                    line = line.strip().split("\t")
                    desc, freq = line[0], int(line[1])
                    descriptors[desc] += freq
                except:
                    logging.warning(f"Skipping malformed line in descriptor file: {line}")
                    continue
        return descriptors
    except FileNotFoundError:
        logging.info(f"No previous descriptors found at {path}")
        logging.info(f"Initialising with empty descriptor vocabulary.")
        return descriptors
    
    
def save_synonym_dict(groups, path, run_id):
    # Save groups for later inspection
    with open(path / f"synonyms_{run_id}.jsonl", "a") as f:
        json_line = json.dumps(groups, ensure_ascii=False)
        f.write(json_line + "\n")
