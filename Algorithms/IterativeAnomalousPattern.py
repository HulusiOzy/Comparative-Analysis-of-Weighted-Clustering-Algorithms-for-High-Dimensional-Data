import numpy as np
import pandas as pd

class IterativeAnomalousPattern:
    def __init__(self, data, k=3):
        self.data = data
        self.k = k
        self.centroids_ = None
        self.remaining_points = None
        self.labels_ = None
        
    def _single_ap_run(self, data_points, assigned_points_mask):
        grand_mean = np.mean(data_points, axis=0)
        distances_to_mean = np.sum((data_points[:, np.newaxis, :] - grand_mean) ** 2, axis=2)
        centroid = data_points[np.argmax(distances_to_mean)]
        
        old_assignments = None
        while True:
            dist_to_centroid = np.sum((data_points - centroid) ** 2, axis=1)
            dist_to_mean = np.sum((data_points - grand_mean) ** 2, axis=1)
            assignments = (dist_to_centroid < dist_to_mean).astype(int)
            
            if np.array_equal(assignments, old_assignments):
                break
                
            old_assignments = assignments
            cluster_points = data_points[assignments == 1]
            
            if len(cluster_points) == 0:
                break
                
            centroid = np.mean(cluster_points, axis=0)
            
        return centroid, np.where(assignments == 1)[0]

    def find_centroids(self):
        n_samples = len(self.data)
        self.remaining_points = self.data.copy()
        self.labels_ = np.full(n_samples, -1)
        assigned_points_mask = np.zeros(n_samples, dtype=bool)
        centroids = []

        for cluster_id in range(self.k):
            if len(self.remaining_points) < 2:
                break

            centroid, cluster_indices = self._single_ap_run(self.remaining_points, assigned_points_mask)
            
            original_indices = np.where(~assigned_points_mask)[0][cluster_indices]
            self.labels_[original_indices] = cluster_id
            
            assigned_points_mask[original_indices] = True
            centroids.append(centroid)
            
            self.remaining_points = self.data[~assigned_points_mask]
            
        unassigned_mask = self.labels_ == -1
        if np.any(unassigned_mask):
            distances = np.array([np.sum((self.data[unassigned_mask] - c) ** 2, axis=1) 
                                for c in centroids])
            self.labels_[unassigned_mask] = np.argmin(distances, axis=0)

        self.centroids_ = np.array(centroids)
        return self.labels_, self.centroids_