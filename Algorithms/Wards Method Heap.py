import pandas as pd
import numpy as np
import heapq

def euclidean_distance(centroid1, centroid2):
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

class ClusterInfo:
    def __init__(self, data):
        self.cluster_sizes = {i: 1 for i in range(len(data))}
        self.centroids = {i: data[i] for i in range(len(data))}
        self.distance_heap = [] #Priority queue because why not
        self.cluster_versions = {i: 0 for i in range(len(data))}
        self._initialize_distances(data)
    
    def _initialize_distances(self, data):
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                distance = self._calculate_ward_distance(i, j)
                heapq.heappush(self.distance_heap, (distance, i, j, self.cluster_versions[i], self.cluster_versions[j]))#Store as (distance, smaller_label, larger_label)
    
    def _calculate_ward_distance(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        centroid1 = self.centroids[label1]
        centroid2 = self.centroids[label2]
        squared_dist = euclidean_distance(centroid1, centroid2) ** 2
        return (n1 * n2) / (n1 + n2) * squared_dist
    
    def find_closest_clusters(self):
        while self.distance_heap:
            distance, label1, label2, ver1, ver2 = heapq.heappop(self.distance_heap)
            if (label1 in self.cluster_versions and 
                label2 in self.cluster_versions and
                self.cluster_versions[label1] == ver1 and 
                self.cluster_versions[label2] == ver2):
                return (label1, label2), distance
        return (None, None), float('inf')
    
    def update_cluster(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        c1 = self.centroids[label1]
        c2 = self.centroids[label2]
        new_size = n1 + n2
        new_centroid = (n1 * c1 + n2 * c2) / new_size
        
        self.cluster_versions[label1] += 1 #Increment version of surviving cluster BEFORE updates
        
        self.cluster_sizes[label1] = new_size
        self.centroids[label1] = new_centroid
        self.cluster_sizes.pop(label2)
        self.centroids.pop(label2)
        self.cluster_versions.pop(label2)
        
        for cluster_id in self.cluster_sizes.keys(): #Calculate new distances with updated version numbers
            if cluster_id != label1:
                smaller_cluster = min(label1, cluster_id)
                larger_cluster = max(label1, cluster_id)
                distance = self._calculate_ward_distance(smaller_cluster, larger_cluster)
                heapq.heappush(self.distance_heap, 
                             (distance, smaller_cluster, larger_cluster,
                              self.cluster_versions[smaller_cluster],
                              self.cluster_versions[larger_cluster]))

def hierarchical_cluster(data, n_clusters=2):
    cluster_labels = np.arange(len(data))
    cluster_info = ClusterInfo(data)
    
    while len(np.unique(cluster_labels)) > n_clusters:
        (label1, label2), min_distance = cluster_info.find_closest_clusters()
        
        if label1 is None:
            break
            
        cluster_labels[cluster_labels == label2] = label1
        cluster_info.update_cluster(label1, label2)
        
        print(f"Merged {label1}, {label2}")
        print(np.unique(cluster_labels))

    print("\nFinal cluster assignments:")
    assignments = [f"{i} = {cluster_labels[i]}" for i in range(len(cluster_labels))]
    print("[" + ", ".join(assignments) + "]")
    
    return cluster_labels

df = pd.read_csv('heart_failure_clinical_records_dataset.csv.data', header = None)
Y = df.to_numpy()
final_labels = hierarchical_cluster(Y, n_clusters=1)

input_filename = 'heart_failure_clinical_records_dataset.csv.data'
base_filename = input_filename.split('.')[0]  #Split for . bc im lazy :)
output_filename = f"{base_filename}.predicted"

with open(output_filename, 'w') as f: #Each on a new line to mimic rows, hopefully this works
    for label in final_labels:
        f.write(f"{label}\n")