import pandas as pd
import numpy as np
import heapq

class WardsHeap:
    def __init__(self, filename='iris.data', n_clusters=3):
        self.filename = filename
        self.n_clusters = n_clusters
        self.labels_ = None
        self.data = None
        self.cluster_sizes = None
        self.centroids = None
        self.distance_heap = None
        self.cluster_versions = None

    def _euclidean_distance(self, centroid1, centroid2):
        return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

    def _initialize_clusters(self):
        self.cluster_sizes = {i: 1 for i in range(len(self.data))}
        self.centroids = {i: self.data[i] for i in range(len(self.data))}
        self.distance_heap = [] #Priority queue because why not
        self.cluster_versions = {i: 0 for i in range(len(self.data))}
        self._initialize_distances()
    
    def _initialize_distances(self):
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                distance = self._calculate_ward_distance(i, j)
                heapq.heappush(self.distance_heap, (distance, i, j, self.cluster_versions[i], self.cluster_versions[j]))#Store as (distance, smaller_label, larger_label)
    
    def _calculate_ward_distance(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        centroid1 = self.centroids[label1]
        centroid2 = self.centroids[label2]
        squared_dist = self._euclidean_distance(centroid1, centroid2) ** 2
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
    
    def _update_cluster(self, label1, label2):
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

    def _hierarchical_cluster(self):
        cluster_labels = np.arange(len(self.data))
        
        while len(np.unique(cluster_labels)) > self.n_clusters:
            (label1, label2), min_distance = self._find_closest_clusters()
            
            if label1 is None:
                break
                
            cluster_labels[cluster_labels == label2] = label1
            self._update_cluster(label1, label2)
            
            print(f"Merged {label1}, {label2}")
            print(np.unique(cluster_labels))

        print("\nFinal cluster assignments:")
        assignments = [f"{i} = {cluster_labels[i]}" for i in range(len(cluster_labels))]
        print("[" + ", ".join(assignments) + "]")
        
        return cluster_labels

    def fit(self):
        df = pd.read_csv(self.filename, header=None)
        self.data = df.to_numpy()
        self._initialize_clusters()
        
        self.labels_ = self._hierarchical_cluster()
        return self.labels_

    def save_predictions(self, output_filename=None):
        if self.labels_ is None: #MVP level check right here
            raise ValueError("Must call fit() before saving predictions")
            
        if output_filename is None:
            base_filename = self.filename.split('.')[0]  #Split for . bc im lazy :)
            output_filename = f"{base_filename}.predicted"
            
        with open(output_filename, 'w') as f: #Each on a new line to mimic rows, hopefully this works
            for label in self.labels_:
                f.write(f"{label}\n")

if __name__ == "__main__":
    input_filename = 'heart_failure_clinical_records_dataset.csv.data'
    wards_heap = WardsHeap(filename=input_filename, n_clusters=1)
    wards_heap.fit()
    wards_heap.save_predictions()