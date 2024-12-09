import pandas as pd
import numpy as np
import random

class IKMeans:
    #Might get confusing to read later down the line but trust me on this, also notebook should have psuedo code
    def __init__(self, filename='iris.data', max_clusters=3, max_iterations=100, tolerance=1e-4):
        self.filename = filename
        self.max_clusters = max_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.labels_ = None
        self.cluster_centers_ = None
        
    #I think this is what they meant by overengineering
    def _distance(self, point1, point2, distance_type='single'):
        if distance_type == 'single':
            return np.sum((point1 - point2)**2)
        elif distance_type == 'one_to_many':
            return np.sum((point1 - point2) ** 2, axis=1)
        elif distance_type == 'many_to_many':
            return np.sum((point1 - point2) ** 2)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")

    def _calculating_gravity_center(self, data_points):
        N = data_points.shape[0] #Consider Use of .shape[0]
        gravity_center = (np.sum(data_points, axis=0) / N) # (1/N) * (Σ y_{iv})
        return gravity_center
    
    def _getting_the_furthest_point_from(self, point, data_points):
        #distances = np.sum((data_points - point) ** 2, axis=1) 
        distances = self._distance(data_points, point, distance_type='one_to_many')# Σ(y_iv - point)^2
        furthest = np.argmax(distances)
        return data_points[furthest]
    
    def _min_distance_rule(self, centroid1, centroid2, data_points):
        cluster_assignments = {} #Consider use of a np array
        for i, point in enumerate(data_points):
            #dist_to_c = np.sum((point - centroid1)**2) 
            dist_to_c = self._distance(point, centroid1) # Σ(y_iv - point)^2
            #dist_to_mean = np.sum((point - centroid2)**2) 
            dist_to_mean = self._distance(point, centroid2) # Σ(y_iv - point)^2
            if dist_to_c < dist_to_mean:
                cluster_assignments[i] = 1 # 1 = centroid
            else:
                cluster_assignments[i] = 0 # 0 = grand mean
        return cluster_assignments
    
    def _cluster_recenter_ap(self, S, data_points):
        temp_data_points = np.array([
            data_points[i] 
            for i in range(len(data_points)) 
            if S[i] == 1
        ]) #If assigned to 1 put its value into the list
        if len(temp_data_points) == 0: #You never know
            return None
        return self._calculating_gravity_center(temp_data_points)
    
    def _run_ap(self, data_points):
        counter = 1
        grand_mean = self._calculating_gravity_center(data_points)
        c = self._getting_the_furthest_point_from(grand_mean, data_points)
        old_S = None
        
        while True:
            S = self._min_distance_rule(c, grand_mean, data_points)
            if old_S == S: #Transfer S from a dictionary to a np array then use np.array_equal to check
                break
            old_S = S
            c = self._cluster_recenter_ap(S, data_points)
            counter += 1
        return S, c
    
    #K means without using other files
    def _k_means_iteration(self, data_points, centroids):
        K = len(centroids)
        cluster_assignments = {}
        S_k = [[] for _ in range(K)] #Consider use of numpy array
        
        for i, point in enumerate(data_points):
            distances = []
            for k in range(K):
                # dist = np.sum((point - centroids[k])**2)
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
    
    def _iterated_anomalous_patterns(self, data_points):
        all_clusters = [] #Consider use of np array
        remaining_points = np.arange(len(data_points)) #Track remaining points, maybe change this to use data_points instead but this looks cleaner to me
        
        while len(remaining_points) > 0 and len(all_clusters) < self.max_clusters:
            current_data = data_points[remaining_points]
            S, cluster_center = self._run_ap(current_data)
            all_clusters.append(cluster_center)
            
            cluster_points = []
            for point_idx, assignment in S.items(): #Points assigned to current cluster
                if assignment == 1:
                    cluster_points.append(point_idx)
                    
            #Remove points in current cluster from remaining points
            remaining_mask = np.ones(len(remaining_points), dtype=bool) #Check notebook for detailed cluster masking info if you froget how to do this
            remaining_mask[cluster_points] = False
            remaining_points = remaining_points[remaining_mask]
            #remaining_points = np.delete(remaining_points, cluster_points) #Optional, but I prefer magic

        return np.array(all_clusters)
    
    def _iterated_k_means(self, data_points, initial_centroids):
        current_centroids = np.array(initial_centroids)
        
        for iteration in range(self.max_iterations):
            assignments, S_k = self._k_means_iteration(data_points, current_centroids)
            
            new_centroids = [] #Probably redundant, can be refractored but who cares
            for cluster_points in S_k: #To only calculate gravity centers of cluster, efficency!
                if len(cluster_points) > 0:
                    cluster_array = np.array(cluster_points) #Func only accepts numpy arrays
                    new_centroid = self._calculating_gravity_center(cluster_array)
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(current_centroids[len(new_centroids)]) #Using len(new_centroids) to get current centroid index
            
            new_centroids = np.array(new_centroids)
            #centroid_movement = np.sum((new_centroids - current_centroids) ** 2) #What a genius I am
            centroid_movement = self._distance(new_centroids, current_centroids, distance_type='many_to_many') #Vectorization done
            
            if centroid_movement < self.tolerance:
                break
                
            current_centroids = new_centroids
        
        return assignments#, current centroids #If needed
    
    def fit(self):
        df = pd.read_csv(self.filename, header=None)
        data = df.to_numpy()
        
        #Anomalous Pattern
        self.cluster_centers_ = self._iterated_anomalous_patterns(data)
        
        #Straight K Means
        cluster_assignments = self._iterated_k_means(data, self.cluster_centers_)

        n_samples = len(data)
        self.labels_ = [cluster_assignments[i] for i in range(n_samples)]
        
        return self.labels_
    
    #Using a function for this unlike before, hopefully makes my code cleaner, Also dont call if not needed
    def save_predictions(self, output_filename=None):
        if self.labels_ is None:
            raise ValueError("Must call fit() before saving predictions")
            
        if output_filename is None:
            base_filename = self.filename.split('.')[0]
            output_filename = f"{base_filename}.predicted"
            
        with open(output_filename, 'w') as f:
            for label in self.labels_:
                f.write(f"{label}\n")

if __name__ == "__main__":
    input_filename = 'iris.data.data'
    ikm = IKMeans(filename=input_filename, max_clusters=3)
    ikm.fit()
    ikm.save_predictions()