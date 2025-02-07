import pandas as pd
import numpy as np
from typing import Tuple, Optional

class SWKMeans:
    def __init__(self, data, k=3, beta=2, initial_centers=None, initial_weights=None):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        if beta <= 1.0:
            raise ValueError("Beta must be greater than 1 for SWKMeans algorithm")

        self.data = data
        self.initial_centers = initial_centers
        self.initial_weights = initial_weights
        self.k = len(initial_centers) if initial_centers is not None else k
        self.beta = beta
        
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

    def _validate_weights(self, weights):
        weights = np.array(weights)
        N = self.data.shape[1]
        
        if weights.shape != (self.k, N):
            raise ValueError(f"Weights matrix must have shape ({self.k}, {N})")
            
        if not np.issubdtype(weights.dtype, np.number):
            raise ValueError("Weights must contain numeric values only")
                
        return weights

    def _calculate_objective(self, U, Z, W):
        M, N = self.data.shape
        expanded_data = self.data[:, np.newaxis, :]
        expanded_Z = Z[np.newaxis, :, :]
        
        diffs = expanded_data - expanded_Z
        distances = np.sqrt(np.square(diffs))
        
        W_beta = np.power(W, self.beta)[np.newaxis, :, :]
        weighted_distances = distances * W_beta
        
        expanded_U = U[:, :, np.newaxis]
        return np.sum(expanded_U * weighted_distances)

    '''
    def _calculate_objective(self, U, Z, W):
        M, N = self.data.shape
        expanded_data = self.data[:, np.newaxis, :]
        expanded_Z = Z[np.newaxis, :, :]
        diffs = expanded_data - expanded_Z
        distances = np.sqrt(np.square(diffs))
        cluster_distances = np.sum(distances, axis=2)
        W_beta = np.power(W, self.beta)
        weighted_distances = cluster_distances * np.sum(W_beta, axis=1)[np.newaxis, :]
        expanded_U = U
        return np.sum(expanded_U * weighted_distances)
    '''

    def _solve_P1(self, Z, W):
        M, N = self.data.shape
        expanded_data = self.data[:, np.newaxis, :]
        expanded_Z = Z[np.newaxis, :, :]
        W_beta = np.power(W, self.beta)[np.newaxis, :, :]
        
        diffs = expanded_data - expanded_Z
        distances = np.sqrt(np.square(diffs))
        weighted_distances = distances * W_beta
        
        cluster_distances = np.sum(weighted_distances, axis=2)
        
        U = np.zeros((M, self.k))
        U[np.arange(M), np.argmin(cluster_distances, axis=1)] = 1
        
        return U

    '''
    def _solve_P1(self, Z, W):
        M, N = self.data.shape
        expanded_data = self.data[:, np.newaxis, :]
        expanded_Z = Z[np.newaxis, :, :]
        diffs = expanded_data - expanded_Z
        distances = np.sqrt(np.square(diffs))
        cluster_distances = np.sum(distances, axis=2)
        W_beta = np.power(W, self.beta)
        weighted_distances = cluster_distances * np.sum(W_beta, axis=1)[np.newaxis, :]
        U = np.zeros((M, self.k))
        U[np.arange(M), np.argmin(weighted_distances, axis=1)] = 1
        
        return U
    '''

    def _solve_P2(self, U):
        M, N = self.data.shape
        cluster_sizes = U.sum(axis=0)
        Z = (U.T @ self.data) / np.maximum(cluster_sizes[:, np.newaxis], 1)
        
        empty_clusters = cluster_sizes == 0
        if np.any(empty_clusters):
            random_points = self.data[np.random.choice(M, np.sum(empty_clusters), replace=False)]
            Z[empty_clusters] = random_points
            
        return Z

    def _solve_P3(self, U, Z):
        M, N = self.data.shape
        expanded_data = self.data[:, np.newaxis, :]
        expanded_Z = Z[np.newaxis, :, :]
        diffs = np.abs(expanded_data - expanded_Z)
        
        expanded_U = U[:, :, np.newaxis]
        masked_diffs = diffs * expanded_U
        total_dispersion = np.sum(masked_diffs)
        count = np.sum(U) * N
        sigma = total_dispersion / max(count, 1e-10)
        
        squared_diffs = np.square(diffs + sigma)
        D_lj = np.sum(squared_diffs * expanded_U, axis=0)
        
        W = np.zeros((self.k, N))
        for l in range(self.k):
            valid_features = D_lj[l] > 0
            if np.any(valid_features):
                D_ratios = np.zeros((N, N))
                D_ratios[valid_features, :] = D_lj[l, valid_features][:, np.newaxis]
                D_ratios[:, valid_features] /= D_lj[l, valid_features]
                
                power = 1/(self.beta-1)
                denominators = np.sum(np.power(D_ratios, power), axis=1)
                W[l, valid_features] = 1 / np.maximum(denominators[valid_features], 1e-10)
            
            if np.sum(W[l]) > 0:
                W[l] /= np.sum(W[l])
                
        return W

    def _subspace_kmeans(self, max_iter=100, tol=1e-6):
        M, N = self.data.shape
        
        if self.initial_centers is not None:
            Z = self._validate_centers(self.initial_centers)
        else:
            idx = np.random.choice(M, self.k, replace=False)
            Z = self.data[idx].copy()
        
        if self.initial_weights is not None:
            W = self._validate_weights(self.initial_weights)
        else:
            W = np.ones((self.k, N)) / N 

        W = np.ones((self.k, N)) / N
        
        best_obj = float('inf')
        best_solution = None
        prev_obj = float('inf')
        
        for _ in range(max_iter):
            U = self._solve_P1(Z, W)
            Z = self._solve_P2(U)
            W = self._solve_P3(U, Z)
            
            obj = self._calculate_objective(U, Z, W)
            
            if abs(obj - prev_obj) < tol:
                break
                
            if obj < best_obj:
                best_obj = obj
                best_solution = (U.copy(), Z.copy(), W.copy())
                
            prev_obj = obj
        
        return best_solution or (U, Z, W)

    def fit(self, n_init=50):
        actual_n_init = 1 if self.initial_centers is not None else n_init
        
        best_objective = float('inf')
        best_solution = None
        
        for iteration in range(actual_n_init):
            U, Z, W = self._subspace_kmeans()
            current_objective = self._calculate_objective(U, Z, W)
            
            if current_objective < best_objective:
                best_objective = current_objective
                best_solution = (U, Z, W)
        
        U, Z, W = best_solution
        self.partition_matrix_ = U
        self.labels_ = np.argmax(U, axis=1)
        self.centroids_ = Z
        self.weights_ = W
        self.best_score_ = best_objective
        
        return self.labels_