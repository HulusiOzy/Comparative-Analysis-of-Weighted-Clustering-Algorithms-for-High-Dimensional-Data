import pandas as pd
import numpy as np
import heapq

class Wards:
    def __init__(self, data, k=3):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        self.data = data
        self.n_clusters = k
        self.labels_ = None
        self.centroids_ = None
        self.cluster_sizes = None
        self.distance_heap = None
        self.cluster_versions = None
        self.current_centroids = None

    def _initialize_clusters(self):
        n_samples = len(self.data)
        self.cluster_sizes = {i: 1 for i in range(n_samples)}
        self.current_centroids = {i: self.data[i].copy() for i in range(n_samples)}
        self.distance_heap = []
        self.cluster_versions = {i: 0 for i in range(n_samples)}
        self._initialize_distances()
    
    def _initialize_distances(self):
        n_samples = len(self.data)
        indices = np.triu_indices(n_samples, k=1)
        
        for i, j in zip(*indices):
            distance = self._calculate_ward_distance(i, j)
            heapq.heappush(self.distance_heap, 
                          (distance, i, j, self.cluster_versions[i], self.cluster_versions[j]))
    
    def _calculate_ward_distance(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        centroid_diff = self.current_centroids[label1] - self.current_centroids[label2]
        squared_dist = np.sum(centroid_diff * centroid_diff)      
        return (n1 * n2) / (n1 + n2) * squared_dist
    
    def _find_closest_clusters(self):
        while self.distance_heap:
            distance, label1, label2, ver1, ver2 = heapq.heappop(self.distance_heap)
            if (label1 in self.cluster_versions and 
                label2 in self.cluster_versions and
                self.cluster_versions[label1] == ver1 and 
                self.cluster_versions[label2] == ver2):
                return (label1, label2), distance
        return (None, None), float('inf')
    
    def _merge_clusters(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        new_size = n1 + n2
        
        new_centroid = (n1 * self.current_centroids[label1] + n2 * self.current_centroids[label2]) / new_size
        
        self.cluster_versions[label1] += 1
        self.cluster_sizes[label1] = new_size
        self.current_centroids[label1] = new_centroid

        self.cluster_sizes.pop(label2)
        self.current_centroids.pop(label2)
        self.cluster_versions.pop(label2)
        
        remaining_clusters = list(self.cluster_sizes.keys())
        for cluster_id in remaining_clusters:
            if cluster_id != label1:
                smaller_cluster = min(label1, cluster_id)
                larger_cluster = max(label1, cluster_id)
                distance = self._calculate_ward_distance(smaller_cluster, larger_cluster)
                heapq.heappush(self.distance_heap,
                             (distance, smaller_cluster, larger_cluster,
                              self.cluster_versions[smaller_cluster],
                              self.cluster_versions[larger_cluster]))

    def _compute_final_centroids(self, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        self.centroids_ = np.zeros((len(unique_labels), self.data.shape[1]))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            self.centroids_[i] = np.mean(self.data[mask], axis=0)

    def _hierarchical_cluster(self):
        cluster_labels = np.arange(len(self.data))
        
        while len(np.unique(cluster_labels)) > self.n_clusters:
            (label1, label2), min_distance = self._find_closest_clusters()
            
            if label1 is None:
                break
            
            cluster_labels[cluster_labels == label2] = label1
            self._merge_clusters(label1, label2)

        return cluster_labels

    def fit(self):
        self._initialize_clusters()
        self.labels_ = self._hierarchical_cluster()
        self._compute_final_centroids(self.labels_)
        return self.labels_, self.centroids_