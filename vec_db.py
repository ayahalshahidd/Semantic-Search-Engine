from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import pickle
from typing import Dict, List, Annotated
import heapq

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:

    def __init__(self, database_file_path = "saved_db.csv", index_file_path = "index.csv", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.nlist = 60  # Number of coarse clusters
        self.nsubquantizers = 4  # Number of subquantizers for PQ
        self.centroids_level1 = []
        self.inverted_lists_level1 = {}
        self.nested_centroids = {}
        self.nested_inverted_lists = {}

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            pass

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                "centroids_level1": self.centroids_level1,
                "inverted_lists_level1": self.inverted_lists_level1,
                "nested_centroids": self.nested_centroids,
                "nested_inverted_lists": self.nested_inverted_lists
            }, f)

    def _build_index(self):
        num_records = self._get_num_records()
        if num_records == 0:
            raise ValueError("The database is empty. Add records before building the index.")
        
        data = self.get_all_rows()

        # Level 1 Clustering (IVF)
        minibatch_kmeans_level1 = MiniBatchKMeans(n_clusters=self.nlist, random_state=42, batch_size=1000)
        minibatch_kmeans_level1.fit(data)
        self.centroids_level1 = minibatch_kmeans_level1.cluster_centers_
        
        cluster_assignments_level1 = minibatch_kmeans_level1.labels_
        self.inverted_lists_level1 = {i: [] for i in range(self.nlist)}
        for idx, cluster_idx in enumerate(cluster_assignments_level1):
            self.inverted_lists_level1[cluster_idx].append(idx)

        # Level 2 PQ for each Level 1 cluster
        self.nested_centroids = {}
        self.nested_inverted_lists = {}
        
        for level1_idx, level1_vectors in self.inverted_lists_level1.items():
            level1_cluster_vectors = data[level1_vectors]
            
            # Perform Product Quantization on each cluster's vectors
            minibatch_kmeans_level2 = MiniBatchKMeans(n_clusters=self.nsubquantizers, random_state=42, batch_size=1000)
            minibatch_kmeans_level2.fit(level1_cluster_vectors)
            
            level2_centroids = minibatch_kmeans_level2.cluster_centers_
            self.nested_centroids[level1_idx] = level2_centroids
            
            # Create inverted lists for Level 2 PQ
            cluster_assignments_level2 = minibatch_kmeans_level2.labels_
            inverted_lists_level2 = {i: [] for i in range(self.nsubquantizers)}
            for idx, cluster_idx in enumerate(cluster_assignments_level2):
                inverted_lists_level2[cluster_idx].append(level1_vectors[idx])
            
            self.nested_inverted_lists[level1_idx] = inverted_lists_level2
        
        self._save_index()

    def retrieve(self, query: np.ndarray, top_k=5, top_level1_k=3, top_level2_k=5):
        self._load_index()
        heap = []  # Min-heap for top-k results

        # Step 1: Retrieve top-level centroids (level 1)
        centroid_scores_level1 = []
        for i, centroid in enumerate(self.centroids_level1):
            score = self._cal_score(query, centroid)
            centroid_scores_level1.append((score, i))

        closest_level1_centroids = [x[1] for x in sorted(centroid_scores_level1, reverse=True)[:top_level1_k]]

        # Step 2: Retrieve nested centroids (level 2)
        candidates = []
        for level1_idx in closest_level1_centroids:
            for nested_centroid_idx, nested_centroid in enumerate(self.nested_centroids.get(level1_idx, [])):
                score = self._cal_score(query, nested_centroid)
                heapq.heappush(candidates, (score, level1_idx, nested_centroid_idx))
                if len(candidates) > top_level2_k:
                    heapq.heappop(candidates)  # Keep only top_k candidates

        # Step 3: Retrieve final candidates
        for score, level1_idx, nested_centroid_idx in candidates:
            vector_ids = self.nested_inverted_lists.get(level1_idx, {}).get(nested_centroid_idx, [])
            for vector_id in vector_ids:
                heapq.heappush(heap, (self._cal_score(query, self.get_one_row(vector_id)), vector_id))
                if len(heap) > top_k:
                    heapq.heappop(heap)  # Keep only top_k results

        # Return top-k results
        final_candidates = [x[1] for x in heap]
        return final_candidates

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    def _load_index(self):
        """
        Load centroids and inverted index from a file.
        """
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
                self.centroids_level1 = index_data["centroids_level1"]
                self.inverted_lists_level1 = index_data["inverted_lists_level1"]
                self.nested_centroids = index_data["nested_centroids"]
                self.nested_inverted_lists = index_data["nested_inverted_lists"]
                print("Index successfully loaded.")
        except Exception as e:
            raise ValueError(f"Error loading the index: {e}")
