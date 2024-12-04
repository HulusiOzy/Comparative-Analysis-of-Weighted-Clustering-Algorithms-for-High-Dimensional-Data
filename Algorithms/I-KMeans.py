import pandas as pd
import numpy as np
import random

def calculating_gravity_center(data_points):
    N = data_points.shape[0] #maybe use .shape[0] if some datasets give issues
    gravity_center = (np.sum(data_points, axis=0) / N) # (1/N) * (Σ y_{iv})
    return gravity_center

def getting_the_furthest_point_from(point, data_points):
    distances = np.sum((data_points - point) ** 2, axis=1) # Σ(y_iv - point)^2
    furthest = np.argmax(distances)
    return data_points[furthest]

def min_distance_rule(centroid1, centroid2, data_points):
    cluster_assignments = {} #Maybe dont store in a dictionary
    for i, point in enumerate(data_points):
        dist_to_c = np.sum((point - centroid1)**2) # Σ(y_iv - point)^2
        dist_to_mean = np.sum((point - centroid2)**2) # Σ(y_iv - point)^2
        if dist_to_c < dist_to_mean:
            cluster_assignments[i] = 1 # 1 = centroid
        else:
            cluster_assignments[i] = 0 # 0 = grand mean
    return cluster_assignments

def cluster_recenter(S, data_points):
    temp_data_points = np.array([
        data_points[i] 
        for i in range(len(data_points)) 
        if S[i] == 1
        ]) #If assigned to 1 put it its value into the list
    if len(temp_data_points) == 0: #Just incase
        return None
    return calculating_gravity_center(temp_data_points)

def run_AP(data_points):
    counter = 1
    grand_mean = calculating_gravity_center(data_points)
    c = getting_the_furthest_point_from(grand_mean, data_points)
    old_S = None
    while True:
        S = min_distance_rule(c, grand_mean, data_points)
        if old_S == S: #Transfer S from a dictionary to a np array then use np.array_equal to check
            break
        old_S = S
        c = cluster_recenter(S, data_points)
        counter += 1
    return S, c

#No function calls because I dont trust my own code
def k_means_iteration(data_points, centroids):
    K = len(centroids)
    cluster_assignments = {}
    S_k = [[] for _ in range(K)] #Not numpy array, maybe bad
    for i, point in enumerate(data_points):
        distances = []
        for k in range(K):
            dist = np.sum((point - centroids[k])**2)
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

def iterated_anomalous_patterns(input_filename, max_clusters):
    df = pd.read_csv(input_filename, header=None) #Was going to do this bit out of the func but who cares
    data_points = df.to_numpy()
    all_clusters = [] #Maybe use np array but not needed for now
    remaining_points = np.arange(len(data_points)) #Track remaining points, maybe change this to use data_points instead but this looks cleaner to me
    while len(remaining_points) > 0 and len(all_clusters) < max_clusters:
        current_data = data_points[remaining_points]
        S, cluster_center = run_AP(current_data)
        all_clusters.append(cluster_center)
        cluster_points = []
        for point_idx, assignment in S.items(): #Get points assigned to current cluster center
            if assignment == 1:
                cluster_points.append(point_idx)
        #Remove points in current cluster from remaining points
        remaining_mask = np.ones(len(remaining_points), dtype=bool) #Boolean masking black magic
        remaining_mask[cluster_points] = False #Sets the points in the current cluster to false
        remaining_points = remaining_points[remaining_mask] #Removes the remaining points in the list
        #remaining_points = np.delete(remaining_points, cluster_points) #Optional, but I prefer magic
    return np.array(all_clusters)

def iterated_k_means(data_points, initial_centroids):
    current_centroids = np.array(initial_centroids) #Time: 3AM, too lazy to check if this is returned as an np array so will turn it into one anyways
    #Maybe add these to function parameters
    max_iterations = 100
    tolerance=1e-4
    
    for iteration in range(max_iterations):
        assignments, S_k = k_means_iteration(data_points, current_centroids)
        
        new_centroids = [] #Probably redundant but who cares
        for cluster_points in S_k: #Doing this so I can calculate "gravity centers" of clusters not whole datasets
            if len(cluster_points) > 0: #Normally I would say this is useless but remember that the centroids arent points anymore and I dont wanna think too much
                cluster_array = np.array(cluster_points) #Function only accepts numpy arrays
                new_centroid = calculating_gravity_center(cluster_array)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(current_centroids[len(new_centroids)]) #Using len(new_centroids) to get current centroid index, smartass.
        
        new_centroids = np.array(new_centroids)
        
        centroid_movement = np.sum((new_centroids - current_centroids) ** 2) #What a genius I am
        if centroid_movement < tolerance:
            break
            
        current_centroids = new_centroids
    
    return assignments#, current_centroids #If needed

input_filename = 'iris.data.data'
cluster_centers = iterated_anomalous_patterns(input_filename, 3)

df = pd.read_csv(input_filename, header=None)
data = df.to_numpy()
cluster_assignments = iterated_k_means(data, cluster_centers)

#For output
base_filename = input_filename.split('.')[0]
output_filename = f"{base_filename}.predicted"
n_samples = len(data)
final_labels = [cluster_assignments[i] for i in range(n_samples)]
with open(output_filename, 'w') as f:
    for label in final_labels:
        f.write(f"{label}\n")