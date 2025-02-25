import numpy as np
import pandas as pd
import random

class KMeans:
    def __init__(self, data, k=3, initial_centers=None):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        
        self.data = data
        self.initial_centers = initial_centers
        self.k = len(initial_centers) if initial_centers is not None else k
        self.labels_ = None
        self.best_error_ = None
        self.centroids_ = None

        if self.k <= 0 or self.k > len(self.data):
            raise ValueError(f"Invalid k={self.k} for {len(self.data)} data points")

    def _initialize_centers(self):
        if self.initial_centers is not None:
            centers = np.asarray(self.initial_centers)
            if centers.shape[1] != self.data.shape[1]:
                raise ValueError(f"Centers have {centers.shape[1]} features but data has {self.data.shape[1]} features")
            return centers
            
        random_indices = np.random.choice(len(self.data), size=self.k, replace=False)
        return self.data[random_indices].copy()

    def _run_kmeans(self, max_iter=100, n_init=50):
        n_samples, n_features = self.data.shape
        best_error = np.inf
        best_assignments = None
        n_runs = 1 if self.initial_centers is not None else n_init

        for _ in range(n_runs):
            current_centroids = self._initialize_centers()
            
            for _ in range(max_iter):
                expanded_data = self.data[:, np.newaxis, :]
                expanded_centroids = current_centroids[np.newaxis, :, :]
                
                distances = np.sum((expanded_data - expanded_centroids) ** 2, axis=2)
                current_assignments = np.argmin(distances, axis=1)
                
                total_error = np.sum(np.min(distances, axis=1))
                
                assignment_matrix = np.zeros((n_samples, self.k))
                assignment_matrix[np.arange(n_samples), current_assignments] = 1
                
                cluster_sizes = assignment_matrix.sum(axis=0)
                new_centroids = (assignment_matrix.T @ self.data) / np.maximum(cluster_sizes[:, np.newaxis], 1)
                
                empty_clusters = cluster_sizes == 0
                new_centroids[empty_clusters] = current_centroids[empty_clusters]
                
                if total_error < best_error:
                    best_error = total_error
                    best_assignments = current_assignments.copy()
                    self.centroids_ = current_centroids.copy()
                
                if np.allclose(current_centroids, new_centroids):
                    break
                    
                current_centroids = new_centroids.copy()

        return best_error, best_assignments

    def fit(self):
        self.best_error_, self.labels_ = self._run_kmeans()
        return self.labels_
