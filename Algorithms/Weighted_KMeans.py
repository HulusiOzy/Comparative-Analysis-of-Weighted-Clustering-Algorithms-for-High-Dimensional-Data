import pandas as pd 
import numpy as np 
import random

class WKMeans:
    def __init__(self, data, k=3, beta=2, initial_centers=None, initial_weights=None):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        self.data = data
        self.k = len(initial_centers) if initial_centers is not None else k
        self.beta = beta
        self.initial_centers = initial_centers
        self.initial_weights = initial_weights
        self.n_init = 1 if (initial_centers is not None and initial_weights is not None) else 50
        
        self.labels_ = None
        self.weights_ = None
        self.centroids_ = None
        self.partition_matrix_ = None
        self.best_score_ = None

    def _validate_centers(self, centers):
        if centers is None:
            return None
            
        centers = np.array(centers)
        if centers.shape[1] != self.data.shape[1]:
            raise ValueError(f"Centers have {centers.shape[1]} features but data has {self.data.shape[1]} features")
            
        if not np.issubdtype(centers.dtype, np.number):
            raise ValueError("Centers must contain numeric values only")
            
        return centers
    
    def _validate_weights(self, weights, N):
        if weights is None:
            return None
            
        weights = np.array(weights)
        if len(weights) != N:
            raise ValueError(f"Weights dimension {len(weights)} does not match feature dimension {N}")
            
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
            
        return weights

    def _P0(self, data_points, U, Z, W):
        M, N = data_points.shape
        k = Z.shape[0]
        total_error = 0
        W_beta = W**self.beta
        
        for l in range(k):
            cluster_mask = U[:, l] == 1
            if np.any(cluster_mask):
                cluster_points = data_points[cluster_mask]
                diff = cluster_points[:, np.newaxis, :] - Z[l]
                total_error += np.sum(np.sqrt((diff * diff).sum(axis=2)) * W_beta)
        
        return total_error
        
    def _P1(self, data_points, Z, W):
        M = len(data_points)
        W_beta = W**self.beta
        
        expanded_data = data_points[:, np.newaxis, :]
        expanded_centers = Z[np.newaxis, :, :]
        diff = expanded_data - expanded_centers
        weighted_dists = np.sqrt(np.sum((diff * diff) * W_beta, axis=2))
        
        U = np.zeros((M, self.k))
        U[np.arange(M), np.argmin(weighted_dists, axis=1)] = 1
        return U

    def _P2(self, data_points, U, Z, W):
        M = len(data_points)
        cluster_sizes = U.sum(axis=0)
        cluster_sizes = np.maximum(cluster_sizes, np.finfo(float).eps)
        
        Z_new = (U.T @ data_points) / cluster_sizes[:, np.newaxis]
        
        empty_clusters = cluster_sizes == 0
        if np.any(empty_clusters):
            random_indices = np.random.choice(M, np.sum(empty_clusters), replace=False)
            Z_new[empty_clusters] = data_points[random_indices]
        
        return Z_new

    def _P3(self, U, Z, data_points):
        N = data_points.shape[1]
        Dj = np.zeros(N)
        
        expanded_points = data_points[:, np.newaxis, :]
        expanded_centers = Z[np.newaxis, :, :]
        
        for j in range(N):
            diffs = expanded_points[:, :, j] - expanded_centers[:, :, j]
            cluster_assignments = U.astype(bool)
            masked_diffs = diffs[cluster_assignments]
            Dj[j] = np.sum(np.sqrt(masked_diffs * masked_diffs))
        
        W = np.zeros(N)
        nonzero_mask = Dj != 0
        
        if np.any(nonzero_mask):
            for j in range(N):
                if Dj[j] != 0:
                    ratios = np.array([
                        (Dj[j]/Dj[t])**(1/(self.beta-1))
                        for t in range(N)
                        if Dj[t] != 0
                    ])
                    W[j] = 1.0 / np.sum(ratios)
        
        W = W / np.maximum(np.sum(W), np.finfo(float).eps)
        return W

    def _single_run(self):
        M, N = self.data.shape
        
        if self.initial_centers is not None:
            Z = self._validate_centers(self.initial_centers)
        else:
            indices = np.random.choice(M, self.k, replace=False)
            Z = self.data[indices]
        
        if self.initial_weights is not None:
            W = self._validate_weights(self.initial_weights, N)
        else:
            W = np.random.rand(N)
            W = W/np.sum(W)
        
        U = np.zeros((M, self.k))
        prev_P0 = float('inf')
        
        U = self._P1(self.data, Z, W)
        
        for _ in range(100):
            U = self._P1(self.data, Z, W)
            Z = self._P2(self.data, U, Z, W)
            W = self._P3(U, Z, self.data)
            
            curr_P0 = self._P0(self.data, U, Z, W)
            
            if abs(curr_P0 - prev_P0) < 1e-6:
                break
                
            prev_P0 = curr_P0
            
        return curr_P0, U, Z, W

    def fit(self):
        best_P0 = float('inf')
        best_U = None
        best_Z = None
        best_W = None
        
        for _ in range(self.n_init):
            curr_P0, U, Z, W = self._single_run()
            
            if curr_P0 < best_P0:
                best_P0 = curr_P0
                best_U = U.copy()
                best_Z = Z.copy()
                best_W = W.copy()
        
        self.labels_ = np.argmax(best_U, axis=1)
        self.weights_ = best_W
        self.centroids_ = best_Z
        self.partition_matrix_ = best_U
        self.best_score_ = best_P0
        
        return self.labels_