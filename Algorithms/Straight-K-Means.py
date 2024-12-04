import pandas as pd
import numpy as np
import random

def distance(point1, point2):
    return np.sum((point1 - point2)**2)

def k_means_iteration(data_points, centroids):
    K = len(centroids)
    cluster_assignments = {}
    S_k = [[] for _ in range(K)]
    
    for i, point in enumerate(data_points):
        distances = []
        for k in range(K):
            dist = distance(point, centroids[k])
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

def cluster_recenter(S_k, dimension):
    K = len(S_k)
    new_centroids = []
    for k in range(K):
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

def square_error(data_points, cluster_assignments, centroids):
    K = len(centroids)
    total_error = 0
    cluster_errors = []
    
    for k in range(K):
        cluster_error = 0
        cluster_points = []
        for i, assigned_cluster in cluster_assignments.items():
            if assigned_cluster == k:
                cluster_points.append(data_points[i])
        
        for point in cluster_points:
            squared_dist = distance(point, centroids[k])
            cluster_error += squared_dist
        cluster_errors.append(cluster_error)
        total_error += cluster_error
    
    return total_error, cluster_errors

def run_kmeans(data_points, K, max_iter=1000, n_init=10):
    best_error = float('inf')
    best_initial_indices = None
    best_cluster_assignments = None
    
    for init in range(n_init):
        random_indices = random.sample(range(len(data_points)), K)
        centroids = [data_points[i] for i in random_indices]
        last_iteration = None
        iteration = 0
        
        while iteration < max_iter:
            cluster_assignments, S_k = k_means_iteration(data_points, centroids)
            if last_iteration and cluster_assignments == last_iteration:
                break
            centroids = cluster_recenter(S_k, len(data_points[0]))
            last_iteration = cluster_assignments
            iteration += 1
        
        total_error, cluster_errors = square_error(data_points, cluster_assignments, centroids)
        if total_error < best_error:
            best_error = total_error
            best_initial_indices = random_indices
            best_cluster_assignments = cluster_assignments
    
    return best_error, best_initial_indices, best_cluster_assignments

input_filename = 'iris.data.data'
df = pd.read_csv(input_filename)
data = df.to_numpy()
K = int(input("Number of initial clusters: "))

best_error, best_indices, cluster_assignments = run_kmeans(data, K)
print("Best error:", best_error)
print("Best initial centroid indices:", best_indices)

cluster_sizes = [0] * K
for cluster in cluster_assignments.values():
    cluster_sizes[cluster] += 1
print("Cluster sizes:", cluster_sizes)

base_filename = input_filename.split('.')[0]
output_filename = f"{base_filename}.predicted"
n_samples = len(data)
final_labels = [cluster_assignments[i] for i in range(n_samples)]

with open(output_filename, 'w') as f:
    for label in final_labels:
        f.write(f"{label}\n")