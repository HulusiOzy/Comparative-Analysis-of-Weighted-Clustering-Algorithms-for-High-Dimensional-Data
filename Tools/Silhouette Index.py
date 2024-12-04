import pandas as pd
import numpy as np


#Overall Silhouette Index = (∑((b(i) - a(i)) / max(a(i), b(i)))) / N
def silhouette_index(ai_values, bi_values):
    si_values = np.zeros(len(ai_values))
    
    #The loop is to calculate s(i) for each point
    for i in range(len(ai_values)):
        max_value = max(ai_values[i], bi_values[i]) #Apparently the max in the formula is to avoid division by 0
        if max_value != 0:  #But we still check for 0 here because we like unnecessary code
            si_values[i] = (bi_values[i] - ai_values[i]) / max_value
    
    overall_index = np.mean(si_values) #.mean because why divide by length when you can use numpy functions
    
    return overall_index, si_values

#Shit way to calculate a(i) values, improve later
#a(i) = (1/(|Ci|-1)) ∑ d(i,j) for all j in Ci, it says i!=j but if i=j the d is 0 no?
def internal_distance(data):
    labels = data.iloc[:, -1] #Get labels from last column
    points = data.iloc[:, :-1].values #Data without labels
    #This method makes it so that im going have to match them in the future, keep in mind    

    ai_values = np.zeros(len(data))
    
    for i in range(len(data)):
        current_point = points[i]
        current_label = labels.iloc[i]
        
        same_cluster_mask = (labels == current_label) & (np.arange(len(data)) != i) #Masking for label and NOT itself
        cluster_points = points[same_cluster_mask] #Put the mask into a list
        
        if len(cluster_points) > 0: #Super unnecessary
            distances = np.sum(np.abs(cluster_points - current_point), axis=1) #Sum of distances to other points in the same cluster
            ai_values[i] = np.sum(distances) / len(cluster_points)
    
    return ai_values

#Shit way to calculate b(i) values, improve later
#b(i) = min (1/|Ck|) ∑ d(i,j) for all j in Ck where Ck represents all clusters other than Ci
def nearest_neighbor_distance(data):
    labels = data.iloc[:, -1] #Get labels from last column
    unique_labels = np.unique(labels)
    points = data.iloc[:, :-1].values#Data without labels
    #This method makes it so that im going have to match them in the future, keep in mind    
    
    bi_values = np.zeros(len(data))
    
    for i in range(len(data)):
        current_point = points[i]
        current_label = labels.iloc[i]
        cluster_averages = [] #Going to send this into bi_values[] which is already a numpy array so no need to init
        
        for label in unique_labels:
            if label == current_label: #BIG CHECK
                continue
                
            other_cluster_mask = (labels == label) #Masking for label
            other_cluster_points = points[other_cluster_mask] #Put the mask into a list
            
            if len(other_cluster_points) > 0: #Super unnecessary
                distances = np.sum(np.abs(other_cluster_points - current_point), axis=1) #Sum of distances to other points in the different cluster
                cluster_average = np.mean(distances)
                cluster_averages.append(cluster_average)
        
        if cluster_averages: #Super unnecessary
            bi_values[i] = np.min(cluster_averages)
            
    return bi_values

data = pd.read_csv('iris.data.data', header=None)
predicted = pd.read_csv('iris.predicted', header=None)

data[len(data.columns)] = predicted

ai = internal_distance(data)
bi = nearest_neighbor_distance(data)
overall_silhouette_index, si = silhouette_index(ai, bi)

print(overall_silhouette_index)
