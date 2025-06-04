# Standard library imports
import argparse
import functools
import json
import logging
import numpy as np
import os
from pathlib import Path
import shutil
import time
from tqdm import tqdm

# Third-party imports
import faiss
from sqlitedict import SqliteDict #type: ignore


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

class NNSearcher:
    def __init__(self, args):
        self.run_id = args.run_id
        self.data_path = Path(args.data_path)
        self.base_dir = Path(f"../data/faiss/{self.run_id}")
        self.sqlite_path = Path(args.sqlite_path) if args.sqlite_path else self.base_dir / "descriptors.sqlite"
        self.faiss_index_path = Path(args.faiss_index_path) if args.faiss_index_path else self.base_dir / "index.index"
        self.checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else self.base_dir / "checkpoints"
        self.merge_log_path = self.base_dir/ "merge_log.jsonl"
        self.k = args.k
        self.stop_index = args.stop_index
        self.nlist = args.nlist
        self.nprobe = args.nprobe

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        log_file = self.base_dir / "out.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        faiss.omp_set_num_threads(os.cpu_count())
        self.logger.info(f"FAISS using {faiss.omp_get_max_threads()} threads")

    @log_execution_time
    def load_data(self, path, stop_index=-1):
        self.logger.info("Loading JSONL data...")
        with open(path, "r") as f:
            if stop_index > 0:
                lines = [next(f) for _ in range(stop_index)]
            else:
                lines = f.readlines()

        records = [json.loads(line) for line in lines]
        embeddings = np.array([rec["embedding"] for rec in records], dtype=np.float32)
        descriptors = [rec["descriptor"] for rec in records]
        self.logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings, descriptors

    @log_execution_time
    def build_index(self, embeddings):        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index_flat = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(index_flat)

        ids = np.arange(len(embeddings), dtype=np.int64)
        index.add_with_ids(embeddings, ids)

        return index, ids

    @log_execution_time
    def initialize_sqlite(self, ids, descriptors):
        self.logger.info("Initializing SQLite database...")
        descriptor_db = SqliteDict(self.sqlite_path, autocommit=False)
        for idx, descriptor in zip(ids, descriptors):
            descriptor_db[int(idx)] = descriptor
        descriptor_db.commit()
        return descriptor_db

    def save_checkpoint(self, index, descriptor_db, iteration):
        # Save current state to checkpoint
        faiss.write_index(index, str(self.checkpoint_dir / f"faiss_index_iter{iteration}.index"))
        descriptor_checkpoint = self.checkpoint_dir / f"descriptors_iter{iteration}.sqlite"
        shutil.copy(self.sqlite_path, descriptor_checkpoint)
        descriptor_db.commit()
        self.logger.info(f"Checkpoint saved: Iteration {iteration}")
        
    @log_execution_time
    def merge_neighbors(self, index, descriptor_db, mutual_pairs, iteration):
        new_embeddings = []
        new_descriptors = []
        remove_ids = []

        for i, j in mutual_pairs:
            emb_i = index.reconstruct(int(i))
            emb_j = index.reconstruct(int(j))
            avg_emb = (emb_i + emb_j) / 2.0
            new_embeddings.append(avg_emb)

            desc_i = descriptor_db[int(i)]
            desc_j = descriptor_db[int(j)]
            merged_desc = (desc_i, desc_j)
            new_descriptors.append(merged_desc)

            remove_ids.extend([int(i), int(j)])

            # Log merge operation
            self.log_merge(iteration, [int(i), int(j)], merged_desc)
            
        return new_embeddings, new_descriptors, remove_ids
    
    def log_merge(self, iteration, merged_ids, merged_descriptors):
        # Log merge operation to JSONL file
        with open(self.merge_log_path, "a") as f:
            f.write(json.dumps({
                "iteration": iteration,
                "merged_ids": merged_ids,
                "merged_descriptors": merged_descriptors
            }) + "\n")
    
    @log_execution_time
    def remove_merged(self, remove_ids, index, descriptor_db):
        # Remove merged vectors
        remove_ids_np = np.array(remove_ids, dtype=np.int64)
        selector = faiss.IDSelectorBatch(remove_ids_np.astype(np.int64))
        index.remove_ids(selector)
        for rid in remove_ids:
            del descriptor_db[rid]
        descriptor_db.commit()

    def find_nn(self):
        # Load data
        embeddings, descriptors = self.load_data(args.data_path, stop_index=args.stop_index)

        # Build FAISS index
        index, ids = self.build_index(embeddings)

        # Initialize SQLite database
        descriptor_db = self.initialize_sqlite(ids, descriptors)

        # Prepare for merging
        next_id = ids.max() + 1
        iteration = 0
        
        # Save initial checkpoint
        self.save_checkpoint(index, descriptor_db, iteration)
        
        self.logger.info("Starting merge loop...")
        
        while True:
            iteration += 1
            self.logger.info(f"[Iteration {iteration}] Searching for mutual nearest neighbors...")
            ntotal = index.ntotal
            if ntotal < 2:
                self.logger.info("Not enough vectors left to merge.")
                break

            # Query nearest neighbors
            _, neighbors = index.search(embeddings, self.k + 1)
            all_neighbors = neighbors[:, 1:]
            if self.k == 1:
                all_neighbors = all_neighbors.reshape(-1, 1)

            # Find mutual pairs
            mutual_pairs = []
            visited = set()

            for i in range(len(all_neighbors)):
                for j in all_neighbors[i]:
                    if i in visited or j in visited:
                        continue
                    if j < len(all_neighbors) and i in all_neighbors[j]:
                        mutual_pairs.append((i, j))
                        visited.update([i, j])
                        break

            if not mutual_pairs:
                self.logger.info("No more mutual nearest neighbors found. Halting.")
                break

            self.logger.info(f"Found {len(mutual_pairs)} mutual pairs. Merging...")
            new_embeddings, new_descriptors, remove_ids = self.merge_neighbors(index,
                                                                               descriptor_db,
                                                                               mutual_pairs,
                                                                               iteration
                                                                               )

            self.remove_merged(remove_ids, index, descriptor_db)
            
            # Add new averaged vectors
            new_embeddings_np = np.stack(new_embeddings).astype(np.float32)
            new_ids = np.arange(next_id, next_id + len(new_embeddings), dtype=np.int64)
            index.add_with_ids(new_embeddings_np, new_ids)
            for nid, desc in zip(new_ids, new_descriptors):
                descriptor_db[int(nid)] = desc

            next_id += len(new_embeddings)
            embeddings = new_embeddings_np

            # Save checkpoint
            self.save_checkpoint(index, descriptor_db, iteration)

        self.logger.info(f"Done. Final number of vectors in index: {index.ntotal}")
        
def main(args):
    searcher = NNSearcher(args)
    searcher.find_nn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FAISS neighbors and descriptors.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique identifier for this run")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the JSONL data file")
    parser.add_argument("--sqlite-path", type=str, help="Path to the SQLite database file")
    parser.add_argument("--faiss-index-path", type=str, help="Path to the FAISS index file")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors to consider")
    parser.add_argument("--stop-index", type=int, default=-1, help="Stop loading data after this many records (-1 for all)")
    parser.add_argument("--nlist", type=int, default=1000, help="Number of clusters (nlist) for IVFFlat")
    parser.add_argument("--nprobe", type=int, default=50, help="Number of clusters to search (nprobe)")
    args = parser.parse_args()
    
    main(args)