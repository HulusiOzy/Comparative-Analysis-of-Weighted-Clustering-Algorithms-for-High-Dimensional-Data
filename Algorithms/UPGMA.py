import pandas as pd 
import numpy as np 
from collections import defaultdict #For missing keys

# Length of each value between clusters is
# (|A| . |B|)^-1 . sum{x∈A} sum{x∈B} d(x,y)

def euclidean_distance(centroid1, centroid2):
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

class ClusterInfo:
    def __init__(self, data):
        self.data = data  #Slight differences from Ward's :D
        self.cluster_sizes = {i: 1 for i in range(len(data))}
        self.cluster_points = {i: [i] for i in range(len(data))} #For UPGMA instead of storing centroids we store points
        
        self.distances = defaultdict(dict)
        self._initialize_distances(data)
    
    def _initialize_distances(self, data):
        for i in range(len(data)):
            for j in range(i + 1, len(data)): #Only upper traingle for speed purposes
                distance = self._calculate_upgma_distance(i, j)
                self.distances[i][j] = distance

    def _calculate_upgma_distance(self, label1, label2):
        #Only need points for this one, no need for cluster centers :D
        points1 = [self.data[idx] for idx in self.cluster_points[label1]]
        points2 = [self.data[idx] for idx in self.cluster_points[label2]]
        
        total_distance = 0
        for point1 in points1:
            for point2 in points2:
                total_distance += euclidean_distance(point1, point2)
        
        return total_distance / (len(points1) * len(points2)) # (|A| . |B|)^-1 . sum{x∈A} sum{x∈B} d(x,y)
    
    def find_closest_clusters(self):
        min_distance = float('inf')
        min_pair = None

        for label1 in self.distances:
            for label2, distance in self.distances[label1].items():
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (label1, label2)

        if min_pair is None:
            return (None, float('inf'))
        else:
            return min_pair, min_distance
    
    def update_cluster(self, label1, label2):
        ##Merge label 2 INTO label 1

        #Merge/Update 1
        self.cluster_points[label1].extend(self.cluster_points[label2])
        self.cluster_sizes[label1] = len(self.cluster_points[label1])
        
        #Remove 2
        self.cluster_points.pop(label2)
        self.cluster_sizes.pop(label2)
        
        #For distances
        self._update_distances(label1, label2)

    def _update_distances(self, label1, label2):
        #Same as Wards but I didnt comment there so I will comment here :D
        #Remove old distances with label 2
        if label2 in self.distances:
            del self.distances[label2]
        for label in self.distances:
            if label2 in self.distances[label]:
                del self.distances[label][label2]
        
        existing_labels = list(self.cluster_sizes.keys())
        for cluster_id in existing_labels: #Basically only calculates the row
            if cluster_id != label1:
                smaller_cluster = min(label1, cluster_id)
                larger_cluster = max(label1, cluster_id)
                self.distances[smaller_cluster][larger_cluster] = self._calculate_upgma_distance(smaller_cluster, larger_cluster)

def hierarchical_cluster(data, n_clusters):
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

    return cluster_labels

input_filename = 'iris.data.data'

df = pd.read_csv(input_filename, header = None)
Y = df.to_numpy()
cluster_labels = np.arange(len(Y))
final_labels = hierarchical_cluster(Y, 3)

base_filename = input_filename.split('.')[0]
output_filename = f"{base_filename}.predicted"

with open(output_filename, 'w') as f:
    for label in final_labels:
        f.write(f"{label}\n")