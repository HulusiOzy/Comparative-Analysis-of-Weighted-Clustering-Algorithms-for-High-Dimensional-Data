import pandas as pd 
import numpy as np 
from collections import defaultdict
import os

#   Very simple
#   ESS = Σ Σ (xij - x̄i)²
#   xiv is the j'th observation of i'th cluster, x̄i is the mean of i'th cluster

#   Wards calculates the increase in ESS that would result in merging two clusters
#   ΔE(p,q) = [ (npnq)/(np+nq) ] ||x̄p - x̄q||²
#   np and nq are the number of observations in cluster p and q (The sizes)
#   x̄p and x̄q are the centroids of p and q
#   ||x̄p - x̄q|| is the Euclidean distance between p and q

#   (|p| . |q| / |p u q|) . ||x̄p - x̄q||²

class Wards:
    def __init__(self, filename='iris.data', n_clusters=3):
        self.filename = filename
        self.n_clusters = n_clusters
        self.labels_ = None
        self.data = None
        self.cluster_sizes = None
        self.centroids = None
        self.distances = None

    def _euclidean_distance(self, centroid1, centroid2):
        return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

    def _initialize_clusters(self):
        self.cluster_sizes = {i: 1 for i in range(len(self.data))}  #Keep track of size of each cluster, instead of recounting
        self.centroids = {i: self.data[i] for i in range(len(self.data))}  #Keeps track of cluster center for each data, instead of recalculating
        self.distances = defaultdict(dict) #ABSOLUTE FUCKING OVERKILL, we should already know whether points exist or not given the fact that we create them in order but I want to use something new anyway, also less checks :D
        self._initialize_distances()
    
    def _initialize_distances(self):
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)): #Only upper traingle for speed purposes
                distance = self._calculate_ward_distance(i, j)
                self.distances[i][j] = distance
    
    def _calculate_ward_distance(self, label1, label2):
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        centroid1 = self.centroids[label1]
        centroid2 = self.centroids[label2]

        squared_dist = self._euclidean_distance(centroid1, centroid2) ** 2
        return (n1 * n2) / (n1 + n2) * squared_dist
    
    def _find_closest_clusters(self):
        min_distance = float('inf') #Maybe better way exists
        min_pair = None

        for label1 in self.distances:
            for label2, distance in self.distances[label1].items():
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (label1, label2)

        if min_pair == None:
            return (None, float('inf'))
        else:
            return min_pair, min_distance        

    def _update_cluster(self, label1, label2):
        ##Merge label 2 INTO label 1

        #NOTE: To future me

        #FIRST update cluster properties
        n1 = self.cluster_sizes[label1]
        n2 = self.cluster_sizes[label2]
        c1 = self.centroids[label1]
        c2 = self.centroids[label2]
        new_size = n1 + n2
        new_centroid = (n1 * c1 + n2 * c2) / new_size

        #SECOND update tracking dictionaries
        self.cluster_sizes[label1] = new_size
        self.centroids[label1] = new_centroid
        self.cluster_sizes.pop(label2)
        self.centroids.pop(label2)

        #THIRD handle the distances
        self._update_distances(label1, label2)
    
    def _update_distances(self, label1, label2):
        if label2 in self.distances:
            del self.distances[label2]
        for label in self.distances:
            if label2 in self.distances[label]:
                del self.distances[label][label2]
        
        existing_labels = list(self.cluster_sizes.keys()) #List, set, anything between doesnt matter.
        
        for cluster_id in existing_labels:
            if cluster_id != label1:
                ##According to wizards

                #In a nested dictionary
                #We want larger labels as data and smaller labels as keys
                #Apparently it avoids duplications

                #If you ask me it looks beautiful

                # distances{
                #   0: {1: n, 2: n}:
                #   1: {2: n}
                #   2: {}
                #}
                
                smaller_cluster = min(label1, cluster_id) #Outer key
                larger_cluster = max(label1, cluster_id) #Inner key
                self.distances[smaller_cluster][larger_cluster] = self._calculate_ward_distance(smaller_cluster, larger_cluster)

    def _hierarchical_cluster(self):
        cluster_labels = np.arange(len(self.data))
        
        while len(np.unique(cluster_labels)) > self.n_clusters:
            (label1, label2), min_distance = self._find_closest_clusters()
            
            if label1 is None:
                break
            
            cluster_labels[cluster_labels == label2] = label1 #Array masking black magic I found on the deep web.
            self._update_cluster(label1, label2) #Also check notebook on how array masking works
            
            print(f"Merged {label1}, {label2}")
            print(np.unique(cluster_labels))

        return cluster_labels

    def fit(self):
        df = pd.read_csv(self.filename, header=None)
        self.data = df.to_numpy()
        self._initialize_clusters()
        
        self.labels_ = self._hierarchical_cluster()
        return self.labels_

    def save_predictions(self, output_filename=None):
        if self.labels_ is None:
            raise ValueError("Must call fit() before saving predictions")
            
        if output_filename is None:
            base_filename = self.filename.split('.')[0]  #Split for . bc im lazy :)
            output_filename = f"{base_filename}.predicted"
            
        with open(output_filename, 'w') as f: #Each on a new line to mimic rows, hopefully this works
            for label in self.labels_:
                f.write(f"{label}\n")

if __name__ == "__main__":
    input_filename = 'iris.data.data'
    wards = Wards(filename=input_filename, n_clusters=3)
    wards.fit()
    wards.save_predictions()