import pandas as pd 
import numpy as np 
from collections import defaultdict #For missing keys

# Length of each value between clusters is
# (|A| . |B|)^-1 . sum{x∈A} sum{x∈B} d(x,y)

#Will need some vectorization later down the line but this should be fine
class UPGMA:
    def __init__(self, filename='iris.data', n_clusters=3):
        self.filename = filename
        self.n_clusters = n_clusters
        self.labels_ = None
        self.data = None
        self.cluster_sizes = None
        self.cluster_points = None
        self.distances = None
    
    def _euclidean_distance(self, centroid1, centroid2):
        return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

    #Moved actual inits here because im expanding this class, if I see fit will move it back to __init__
    def _initialize_clusters(self):
        self.cluster_sizes = {i: 1 for i in range(len(self.data))} #For dict of cluster sizes/ how many points in each cluser
        self.cluster_points = {i: [i] for i in range(len(self.data))} #For dict of cluster points/ check notebook for simple example if you end up frogetting
        self.distances = defaultdict(dict)
        self._initialize_distances()

    def _initialize_distances(self):
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)): #Only upper traingle for speed purposes
                distance = self._calculate_upgma_distance(i, j)
                self.distances[i][j] = distance

    def _calculate_upgma_distance(self, label1, label2):
        #Only need points for this one, no need for cluster centers :D
        points1 = [self.data[idx] for idx in self.cluster_points[label1]]
        points2 = [self.data[idx] for idx in self.cluster_points[label2]]
        
        total_distance = 0
        for point1 in points1:
            for point2 in points2:
                total_distance += self._euclidean_distance(point1, point2)
        
        return total_distance / (len(points1) * len(points2)) # (|A| . |B|)^-1 . sum{x∈A} sum{x∈B} d(x,y)
    
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
        else:
            return min_pair, min_distance
    
    def _update_cluster(self, label1, label2):
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
    
    #Moved this into the class, for future self: This just runs the functions above, shouldnt really do anything more
    #If more is needed consider making a new class :D
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

        return cluster_labels

    def fit(self):
        df = pd.read_csv(self.filename, header=None)
        self.data = df.to_numpy()
        self._initialize_clusters()
        
        self.labels_ = self._hierarchical_cluster()
        return self.labels_

    def save_predictions(self, output_filename=None):
        if self.labels_ is None: #BIGTIME ERROR CHECKING
            raise ValueError("Must call fit() before saving predictions")
            
        if output_filename is None:
            base_filename = self.filename.split('.')[0]
            output_filename = f"{base_filename}.predicted"
            
        with open(output_filename, 'w') as f:
            for label in self.labels_:
                f.write(f"{label}\n")

if __name__ == "__main__":
    input_filename = 'Depression Student Dataset.csv.data'
    upgma = UPGMA(filename=input_filename, n_clusters=2)
    upgma.fit()
    upgma.save_predictions()