import pandas as pd
import numpy as np
import random

class KMeans:
    def __init__(self, filename='iris.data', k=3):

        self.filename = filename
        self.k = k
        self.labels_ = None
        self.best_error_ = None
        self.best_indices_ = None
        
    def _distance(self, point1, point2):
        return np.sum((point1 - point2)**2)
    
    def _k_means_iteration(self, data_points, centroids):
        cluster_assignments = {}
        S_k = [[] for _ in range(self.k)]
        
        for i, point in enumerate(data_points):
            distances = []
            for k in range(self.k):
                dist = self._distance(point, centroids[k])
                distances.append((dist, k))
            
            min_dist = float('inf')
            nearest_k = None
            for dist, k in distances:
                if dist < min_dist:
                    min_dist = dist
                    nearest_k = k
            
            cluster_assignments[i] = nearest_k
            S_k[nearest_k].append(point)
        
        return cluster_assignments, S_k
    
    def _cluster_recenter(self, S_k, dimension):
        new_centroids = []
        for k in range(self.k):
            if len(S_k[k]) > 0:
                centroid = []
                for v in range(dimension):
                    feature_values = [point[v] for point in S_k[k]]
                    feature_mean = sum(feature_values) / len(feature_values)
                    centroid.append(feature_mean)
                new_centroids.append(centroid)
            else:
                new_centroids.append([0] * dimension)
        return new_centroids
    
    def _square_error(self, data_points, cluster_assignments, centroids):
        total_error = 0
        cluster_errors = []
        
        for k in range(self.k):
            cluster_error = 0
            cluster_points = []
            for i, assigned_cluster in cluster_assignments.items():
                if assigned_cluster == k:
                    cluster_points.append(data_points[i])
            
            for point in cluster_points:
                squared_dist = self._distance(point, centroids[k])
                cluster_error += squared_dist
            cluster_errors.append(cluster_error)
            total_error += cluster_error
        
        return total_error, cluster_errors
    
    def _run_kmeans(self, data_points, max_iter=1000, n_init=10):
        best_error = float('inf')
        best_initial_indices = None
        best_cluster_assignments = None
        
        for init in range(n_init):
            random_indices = random.sample(range(len(data_points)), self.k)
            centroids = [data_points[i] for i in random_indices]
            last_iteration = None
            iteration = 0
            
            while iteration < max_iter:
                cluster_assignments, S_k = self._k_means_iteration(data_points, centroids)
                if last_iteration and cluster_assignments == last_iteration:
                    break
                centroids = self._cluster_recenter(S_k, len(data_points[0]))
                last_iteration = cluster_assignments
                iteration += 1
            
            total_error, cluster_errors = self._square_error(data_points, cluster_assignments, centroids)
            if total_error < best_error:
                best_error = total_error
                best_initial_indices = random_indices
                best_cluster_assignments = cluster_assignments
        
        return best_error, best_initial_indices, best_cluster_assignments
    
    def fit(self):
        df = pd.read_csv(self.filename, header = None)
        data = df.to_numpy()
        
        self.best_error_, self.best_indices_, cluster_assignments = self._run_kmeans(data)
        
        n_samples = len(data)
        self.labels_ = [cluster_assignments[i] for i in range(n_samples)]
        
        return self.labels_

if __name__ == "__main__":
    input_filename = 'Depression Student Dataset.csv.data'
    kmeans = KMeans(filename=input_filename, k=2)
    
    final_labels = kmeans.fit()
    
    base_filename = input_filename.split('.')[0]
    output_filename = f"{base_filename}.predicted"
    
    with open(output_filename, 'w') as f:
        for label in final_labels:
            f.write(f"{label}\n")