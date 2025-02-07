import pandas as pd 
import numpy as np 
from collections import defaultdict

class UPGMA:
    def __init__(self, data, n_clusters=3):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        self.data = data
        self.n_clusters = n_clusters
        self.labels_ = None
        self.centroids_ = None
        self.cluster_sizes = None
        self.cluster_points = None
        self.distances = None

    def _euclidean_distance(self, centroid1, centroid2):
        return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

    def _initialize_clusters(self):
        self.cluster_sizes = {i: 1 for i in range(len(self.data))}
        self.cluster_points = {i: [i] for i in range(len(self.data))}
        self.distances = defaultdict(dict)
        self._initialize_distances()

    def _initialize_distances(self):
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                distance = self._calculate_upgma_distance(i, j)
                self.distances[i][j] = distance

    def _calculate_upgma_distance(self, label1, label2):
        points1 = [self.data[idx] for idx in self.cluster_points[label1]]
        points2 = [self.data[idx] for idx in self.cluster_points[label2]]
        
        total_distance = 0
        for point1 in points1:
            for point2 in points2:
                total_distance += self._euclidean_distance(point1, point2)
        
        return total_distance / (len(points1) * len(points2))
    
    def _find_closest_clusters(self):
        min_distance = float('inf')
        min_pair = None

        for label1 in self.distances:
            for label2, distance in self.distances[label1].items():
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (label1, label2)

        if min_pair is None:
            return (None, float('inf'))
        return min_pair, min_distance
    
    def _calculate_centroid(self, cluster_label):
        cluster_indices = self.cluster_points[cluster_label]
        cluster_data = self.data[cluster_indices]
        return np.mean(cluster_data, axis=0)

    def _update_cluster(self, label1, label2):
        self.cluster_points[label1].extend(self.cluster_points[label2])
        self.cluster_sizes[label1] = len(self.cluster_points[label1])
        
        self.cluster_points.pop(label2)
        self.cluster_sizes.pop(label2)
        
        if label2 in self.distances:
            del self.distances[label2]
        for label in self.distances:
            if label2 in self.distances[label]:
                del self.distances[label][label2]
        
        remaining_clusters = list(self.cluster_sizes.keys())
        for cluster_id in remaining_clusters:
            if cluster_id != label1:
                smaller_cluster = min(label1, cluster_id)
                larger_cluster = max(label1, cluster_id)
                distance = self._calculate_upgma_distance(smaller_cluster, larger_cluster)
                self.distances[smaller_cluster][larger_cluster] = distance

    def _compute_final_centroids(self):
        unique_labels = np.unique(self.labels_)
        self.centroids_ = np.zeros((len(unique_labels), self.data.shape[1]))
        
        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            self.centroids_[i] = np.mean(self.data[mask], axis=0)
    
    def _hierarchical_cluster(self):
        cluster_labels = np.arange(len(self.data))
        
        while len(np.unique(cluster_labels)) > self.n_clusters:
            (label1, label2), min_distance = self._find_closest_clusters()
            
            if label1 is None:
                break
            
            cluster_labels[cluster_labels == label2] = label1
            self._update_cluster(label1, label2)

        return cluster_labels

    def fit(self):
        self._initialize_clusters()
        self.labels_ = self._hierarchical_cluster()
        self._compute_final_centroids()
        return self.labels_