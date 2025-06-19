# Standard library imports
import argparse
import functools
import json
import logging
import numpy as np
import os
from pathlib import Path
import time

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
        self.base_dir = Path(args.save_dir) if args.save_dir else Path(f"../data/faiss/{self.run_id}")
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.merge_log_path = self.base_dir/ "merge_log.jsonl"
        self.last_checkpoint_path = self.base_dir / "last_checkpoint.txt"
        self.resume = args.resume
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
    def build_index(self, embeddings, iteration):        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)

        # Fit the embeddings to self.nlist voronoi cells
        index.train(embeddings)
        ids = np.arange(len(embeddings), dtype=np.int64)
        index.add_with_ids(embeddings, ids)
        index.nprobe = self.nprobe
        index.make_direct_map()
        # Save index to file        
        faiss.write_index(index, str(self.checkpoint_dir / f"faiss_index_iter{iteration}.index"))
        
        return index, ids

    @log_execution_time
    def build_sqlite(self, ids, descriptors, iteration):
        assert len(ids) == len(descriptors), "IDs and descriptors should have the same length"
        self.logger.info("Initializing SQLite database...")
        descriptor_db = SqliteDict(str(self.checkpoint_dir / f"descriptors_iter{iteration}.sqlite"))
        for idx, descriptor in zip(ids, descriptors):
            descriptor_db[int(idx)] = descriptor
        descriptor_db.commit()
        return descriptor_db

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
            
        new_embeddings = np.stack(new_embeddings).astype(np.float32)
            
        return new_embeddings, new_descriptors, remove_ids
    
    def log_merge(self, iteration, merged_ids, merged_descriptors):
        # Log merge operation to JSONL file
        with open(self.merge_log_path, "a") as f:
            f.write(json.dumps({
                "iteration": iteration,
                "merged_ids": merged_ids,
                "merged_descriptors": merged_descriptors
            }) + "\n")
    
    def remove_merged_entries(self, remove_ids, embeddings, descriptors):
        # Remove merged emebeddings
        embeddings = np.delete(embeddings, remove_ids, 0)
        
        # Remove merged descriptors
        mask = np.ones(len(descriptors), dtype=bool)
        mask[remove_ids] = False
        descriptors = [desc for i, desc in enumerate(descriptors) if mask[i]]

        return embeddings, descriptors
    
    @log_execution_time
    def search_index(self, index, embeddings):
        _, neighbors = index.search(embeddings, self.k + 1)
        all_neighbors = neighbors[:, 1:]
        if self.k == 1:
            all_neighbors = all_neighbors.reshape(-1, 1)
            
        return all_neighbors
    
    def find_mutual_pairs(self, neighbors):
        mutual_pairs = []
        visited = set()
        for i in range(len(neighbors)):
            for j in neighbors[i]:
                if i in visited or j in visited:
                    continue
                if j < len(neighbors) and i in neighbors[j]:
                    mutual_pairs.append((i, j))
                    visited.update([i, j])
                    break
        
        return mutual_pairs
    
    def reconstruct_embeddings_and_descriptors(self, index, descriptor_db):
        embeddings = np.stack([index.reconstruct(i) for i in range(index.ntotal)]).astype(np.float32)
        descriptors = [descriptor_db[i] for i in range(index.ntotal)]
        return embeddings, descriptors
    
    def get_last_checkpoint(self):
        if self.last_checkpoint_path.exists():
            return int(self.last_checkpoint_path.read_text().strip())
        return 0
    
    def save_final_results(self, index, descriptor_db):
        results = []
        for i in range(index.ntotal):
            results.append({
                "embedding": index.reconstruct(i).tolist(),
                "descriptor": descriptor_db[i]
            })
        with open(self.base_dir / "merge_results.jsonl", "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
                
    def find_nn(self):
        self.logger.info("Starting run...")
        if self.resume:
            # If continuing from a checkpoint, load the last saved index and descriptors
            iteration = self.get_last_checkpoint()
            self.logger.info(f"Resuming from iteration {iteration}")
            index = faiss.read_index(str(self.checkpoint_dir / f"faiss_index_iter{iteration}.index"))
            descriptor_db = SqliteDict(str(self.checkpoint_dir / f"descriptors_iter{iteration}.sqlite"))
            embeddings, descriptors = self.reconstruct_embeddings_and_descriptors(index, descriptor_db)
        else:
            # If starting fresh, load data and build index
            embeddings, descriptors = self.load_data(self.data_path, stop_index=self.stop_index)
            iteration = 0
            index, ids = self.build_index(embeddings, iteration)
            descriptor_db = self.build_sqlite(ids, descriptors, iteration)
        
        ntotal = index.ntotal
        self.logger.info("Starting merge loop...")
        while True:
            iteration += 1
            self.logger.info(f"[Iteration {iteration}] Searching for mutual nearest neighbors...")
            self.logger.info(f"Total vectors in index: {ntotal}")
            # Check if there are enough vectors to merge
            if ntotal < 2:
                self.logger.info("Not enough vectors left to merge.")
                break
            
            # Query nearest neighbors
            nearest_neighbors = self.search_index(index, embeddings)
            # Find mutual nearest pairs
            mutual_pairs = self.find_mutual_pairs(nearest_neighbors)

            if not mutual_pairs:
                self.logger.info("No more mutual nearest neighbors found. Halting.")
                break

            self.logger.info(f"Found {len(mutual_pairs)} mutual pairs. Merging...")
            new_embeddings, new_descriptors, remove_ids = self.merge_neighbors(index,
                                                                               descriptor_db,
                                                                               mutual_pairs,
                                                                               iteration
                                                                               )
            embeddings, descriptors = self.remove_merged_entries(remove_ids, embeddings, descriptors)
            
            # Combine old and new embeddings and descriptors
            embeddings = np.concatenate((embeddings, new_embeddings))
            descriptors = descriptors + new_descriptors
            
            assert len(embeddings) == len(descriptors), "Embeddings and descriptors should have the same length after merging"

            # Save checkpoint and build new index
            index, ids= self.build_index(embeddings, iteration)
            ntotal = index.ntotal
            # Close old SQLite database and build a new one
            descriptor_db.close()
            descriptor_db = self.build_sqlite(ids, descriptors, iteration)
            
            self.last_checkpoint_path.write_text(str(iteration))

        self.logger.info(f"Merging done. Final number of vectors in index: {index.ntotal}")
        self.logger.info(f"Saving final results to {self.base_dir / 'merge_results.jsonl'}")
        
        # Save final results
        self.save_final_results(index, descriptor_db)
        descriptor_db.close()
        self.logger.info("Results saved. Exiting.")
        
def main(args):
    searcher = NNSearcher(args)
    searcher.find_nn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FAISS neighbors and descriptors.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique identifier for this run")
    parser.add_argument("--data-path", type=str, help="Path to the JSONL data file")
    parser.add_argument("--save-dir", type=str, help="Directory to save results and checkpoints")
    parser.add_argument("--resume", action='store_true', help="Resume from last checkpoint")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors to consider")
    parser.add_argument("--stop-index", type=int, default=-1, help="Stop loading data after this many records (-1 for all)")
    parser.add_argument("--nlist", type=int, default=100, help="Number of clusters (nlist) for IVFFlat")
    parser.add_argument("--nprobe", type=int, default=10, help="Number of clusters to search (nprobe)")
    args = parser.parse_args()
    
    main(args)