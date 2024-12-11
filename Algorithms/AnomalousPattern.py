import pandas as pd
import numpy as np

class AnomalousPattern:
    def __init__(self, filename='iris.data'):
        self.filename = filename
        self.labels_ = None
        self.grand_mean_ = None
        self.centroid_ = None
 
    #Cooked this up last night while turning AP into a class
    #Youre prolly gonna look at this and go "whats the point of this? they are both np.sum anyways"
    #And I will say ur wrong bc this is vectorization so I dont have one liner loops that make the code look disgusting
    #Trust hulusi of the past and dont remove this
    def _distance(self, point1, point2, distance_type='single'):
        if distance_type == 'single':
            return np.sum((point1 - point2)**2)
        elif distance_type == 'one_to_many':
            return np.sum((point1 - point2) ** 2, axis=1) #For laterall sum, sum across features
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")

    def _calculating_gravity_center(self, data_points):
        N = len(data_points) #Couldve used .shape[0] dont know whats faster
        gravity_center = (np.sum(data_points, axis=0) / N) # (1/N) * (Σ y_{iv})
        return gravity_center
    
    def _getting_the_furthest_point_from(self, point, data_points):
        distances = self._distance(data_points, point, distance_type='one_to_many') # Σ(y_iv - point)^2
        furthest = np.argmax(distances)
        return data_points[furthest]
    
    def _min_distance_rule(self, centroid, grand_mean, data_points):
        cluster_assignments = {} #Really dumb to store in a dictionary BUT it doesnt need to be changed YET
        for i, point in enumerate(data_points):
            dist_to_c = self._distance(point, centroid) # Σ(y_iv - point)^2
            dist_to_mean = self._distance(point, grand_mean) # Σ(y_iv - point)^2
            if dist_to_c < dist_to_mean:
                cluster_assignments[i] = 1  # 1 = centroid
            else:
                cluster_assignments[i] = 0  # 0 = grand mean
        return cluster_assignments
    
    def _cluster_recenter(self, S, data_points):
        temp_data_points = []
        for i in range(len(data_points)):
            if S[i] == 1:
                temp_data_points.append(data_points[i])
        if len(temp_data_points) == 0: #You never know
            return None
        return self._calculating_gravity_center(temp_data_points)
    
    def fit(self):
        df = pd.read_csv(self.filename, header=None)
        data_points = df.to_numpy() #I thought about this before but initing it here is more memory efficent, why idk just trust me
        
        counter = 1 #Just to keep track, book says it so I do it
        self.grand_mean_ = self._calculating_gravity_center(data_points)
        self.centroid_ = self._getting_the_furthest_point_from(self.grand_mean_, data_points)
        old_S = None
        
        while True:
            S = self._min_distance_rule(self.centroid_, self.grand_mean_, data_points)
            print(f"It {counter}: {S} \n")
            
            if old_S == S:
                break
                
            old_S = S
            self.centroid_ = self._cluster_recenter(S, data_points)
            counter += 1
        
        self.labels_ = [S[i] for i in range(len(data_points))]
        return self.labels_

if __name__ == "__main__":
    input_filename = 'iris.data.data'
    ap = AnomalousPattern(filename=input_filename)
    
    final_labels = ap.fit()
    
    base_filename = input_filename.split('.')[0]
    output_filename = f"{base_filename}.predicted"
    
    with open(output_filename, 'w') as f:
        for label in final_labels:
            f.write(f"{label}\n")