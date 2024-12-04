import pandas as pd
import numpy as np

def calculating_gravity_center(data_points):
    N = len(data_points) #maybe use .shape[0] if some datasets give issues
    gravity_center = (np.sum(data_points, axis=0) / N) # (1/N) * (Σ y_{iv})
    return gravity_center

def getting_the_furthest_point_from(point, data_points):
    distances = np.sum((data_points - point) ** 2, axis=1) # Σ(y_iv - point)^2
    furthest = np.argmax(distances)
    return data_points[furthest]

def min_distance_rule(centroid, grand_mean, data_points):
    cluster_assignments = {} #Maybe dont store in a dictionary
    for i, point in enumerate(data_points):
        dist_to_c = np.sum((point - centroid)**2) # Σ(y_iv - point)^2
        dist_to_mean = np.sum((point - grand_mean)**2) # Σ(y_iv - point)^2
        if dist_to_c < dist_to_mean:
            cluster_assignments[i] = 1 # 1 = centroid
        else:
            cluster_assignments[i] = 0 # 0 = grand mean
    return cluster_assignments

def cluster_recenter(S, data_points):
    temp_data_points = []
    for i in range(len(data_points)):
        if S[i] == 1:
            temp_data_points.append(data_points[i])
    if len(temp_data_points) == 0: #Just incase
        return None
    return calculating_gravity_center(temp_data_points)

input_filename = 'iris.data.data'
df = pd.read_csv(input_filename, header = None)
data_points = df.to_numpy()


counter = 1
grand_mean = calculating_gravity_center(data_points)
c = getting_the_furthest_point_from(grand_mean, data_points)
old_S = None

while True:
    S = min_distance_rule(c, grand_mean, data_points)
    print(f"It {counter}: {S} \n")
    
    if old_S == S: #Transfer S from a dictionary to a np array then use np.array_equal to check
        break
        
    old_S = S
    c = cluster_recenter(S, data_points)
    counter += 1

base_filename = input_filename.split('.')[0]  #Split for . bc im lazy :)
output_filename = f"{base_filename}.predicted"

final_labels = [S[i] for i in range(len(data_points))] #Convert dictionary to ordered list

with open(output_filename, 'w') as f:  #Each on a new line to mimic rows
    for label in final_labels:
        f.write(f"{label}\n")