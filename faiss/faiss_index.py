import faiss #type: ignore
import numpy as np #type: ignore
import json
import os

class FaissIndex:
    def __init__(self, dimension, index_type="FlatL2", nlist=100):
        self.dimension = dimension
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dimension) 
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            raise ValueError("Unsupported index type")
        self.id_to_data = {}
        
    def train(self, training_vectors):
        training_vectors = np.asarray(training_vectors, dtype=np.float32)
        if isinstance(self.index, faiss.IndexIVFFlat):
            assert not self.index.is_trained
            self.index.train(training_vectors)
            assert self.index.is_trained

    def add(self, vector, data=None):
        vector = np.asarray(vector, dtype=np.float32)
        idx = len(self.id_to_data)
        self.index.add(vector.reshape(1, -1))
        if data is not None:
            self.id_to_data[idx] = data

    def search(self, query_vector, top_k=5, nprobe=10):
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = nprobe
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return distances, indices
    
    def save_index(self, path):
        faiss.write_index(self.index, path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.id_to_data.items()}, f, ensure_ascii=False, indent=2)
        
    def load_index(self, path):
        self.index = faiss.read_index(path)
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.id_to_data = {int(k): v for k, v in raw.items()}
        else:
            self.id_to_data = {}