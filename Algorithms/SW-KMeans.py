import pandas as pd 
import numpy as np 
import random

##We have lots of formulas to go through, but the basic one is

#P(U, Z, W) = Σ^{k}_{l = 1} Σ^{M}_{i = 1} Σ^{N}_{j = 1} u_{i, l} . w^{β}_{j} . d(x_{i, j}, z_{l, j})

#Where
#X = Dataset
#M = Rows/Records
#N = Columns/Features
#U is a M.k partition matrix where u_{i,l} is a binary variable that indicates whether i belongs to cluster l
#Z is a set of vectors representing cluster centers
#W is a set of weights for each feature
#β is a input parameter

#Optimization Solutions
#P1: Fix Z = Z^ and W = W^; solve the reduced problem P(U, Z^, W^)
#P2: Fix U = U^ and W = W^; solve the reduced problem P(U^, Z, W^)
#P3: Fix U = U^ and Z = Z^; solve the reduced problem P(U^, Z^, W)

#Formal P1:
#u_{i, l} = 1 IF Σ^{N}_{j = 1}w^{β}_{j}.d{x_{i,j},z_{l, j}} ≤ Σ^{N}_{j = 1}w^{β}_{j}.d{x_{i,j},z_{t, j}} FOR 1 ≤ t ≤ K
#u_{i, l} = 0 for t != l
#NOTE: We check each distance by seeing the difference between each feature, we can add weights to this by multiplying each feature by w

#Formula P2:
#z_{l, j} = Σ^{M}_{i = 1}u_{i,l}.x_{i, j}/Σ^{M}_{i = 1}.u_{i, l} FOR 1 ≤ l ≤ k AND 1 ≤ j ≤ N

#For Numerical: The center of cluster l for feature j is just the average of that feature's values for all points in cluster l

#z_{l, j} = a^{r}_{j}

#For Categorical: Its the mode of j/ most common value of j.

#Formula P3:
#D_{j} = Σ^{k}_{l = 1}Σ^{M}_{i = 1} = u^_{i, l} . d(x_{i,j). z_{l, j})

#For feature j, calculate difference between j and the mean of j for that cluster.

#w^_{j} = (Σ^{h}_{t = 1} [D_{j}/D_{t}]^((β - 1)^-1))^-1 IF D_{j} != 0
#w^_{j} = 0 IF D_{j} = 0

#If dispersion is 0 the weight is 0
#If dispersion is anything but 0, sum up the ratio of the dispersion of j's(current feature) to every features dispersion (t), and then raised to the power of 1/(β - 1). THEN GETTING EVERYTHING TO THE POWER OF -1

##STEPS
#Step 1: Initialize Z, W randomly and determine U so that P(U^0, Z^0, W^0) is minimized.
#Step 2: Fix Z and W, and solve the problem P(U, Z^, W^). If P(U^t+1, Z^, W^) = P(U^t, Z^, W^), then output P(U^t, Z^, W^) and stop. If not go to step 3.
#Step 3: Fix U and W, and solve the problem P(U^, Z, W^). If P(U^t, Z^t+1, W^) = P(U^, Z^t, W^), then output P(U^, Z^t, W^) and stop. If not go to step 4.
#Step 4: Fix U and Z, and solve the problem P(U^, Z^, W). If P(U^t, Z^, W^t+1) = P(U^, Z^, W^t), then output P(U^, Z^, W^t) and stop. If not go to step 2.

def main(filename, k, beta):
    data_points = pd.read_csv(filename, header=None).values #.values puts everything into numpy array, we efficent now
    M, N = data_points.shape
    
    indices = np.random.choice(M, k, replace=False) #Pick random points by row
    Z = data_points[indices]
    
    W = np.random.rand(k, N)
    W = W/np.sum(W, axis = 1)[:, np.newaxis] #Normalize to 1
    
    U = np.zeros((M, k))
    prev_P0 = float('inf')
    
    U = P1(data_points, Z, W, beta) #For Step1

    for iteration in range(1000):
        #Bit spagetti will hopefully fix later, not spagetti anymore
        Z_old = Z.copy()
        W_old = W.copy()
        U_old = U.copy()
        
        U = P1(data_points, Z, W, beta)
        Z = P2(data_points, U, Z, W, beta)
        W = P3(U, Z, beta, data_points)

        curr_P0 = P0(data_points, U, Z, W, beta)

        if abs(curr_P0 - prev_P0) < 1e-6:
            break

        prev_P0 = curr_P0
    
    return np.argmax(U, axis=1), W, Z, U, data_points

#No squareroot euclidean distance
def distance(x, z, categorical = False):
    if categorical: #Categorical doesnt exist for now but who cares
        return 0 if x == z else 1
    diff = x - z
    return diff * diff #The brains on me is actually insane

#P(U, Z, W) = Σ^{k}_{l = 1} Σ^{M}_{i = 1} Σ^{N}_{j = 1} u_{i, l} . w^{β}_{j} . d(x_{i, j}, z_{l, j})
#Where this is used to check for convergence
#From what I understand its summed square error criterion
def P0(data_points, U, Z, W, beta):
    M, N = data_points.shape
    k = Z.shape[0]

    total_error = 0
    W_beta = np.power(W, beta) #So I dont repeat this every single iteration
    
    
    for l in range(k):
        cluster_mask = U[:, l] == 1 #Uil
        if np.any(cluster_mask): #The if checks never end, probably redundant but who cares
            cluster_points = data_points[cluster_mask] #Get points in cluster thru mask
            diff = cluster_points[:, np.newaxis, :] - Z[l] #Calculate distances from the center
            total_error += np.sum((diff * diff) * W_beta[l]) #Sum of sum bc formula :D
        
    return total_error

#P1: Fix Z = Z^ and W = W^; solve the reduced problem P(U, Z^, W^)
#u_{i, l} = 1 IF Σ^{N}_{j = 1}w^{β}_{j}.d{x_{i,j},z_{l, j}} ≤ Σ^{N}_{j = 1}w^{β}_{j}.d{x_{i,j},z_{t, j}} FOR 1 ≤ t ≤ K
#u_{i, l} = 0 for t != l
#Basically min distance
def P1(data_points, Z, W, beta):
    M, N = data_points.shape
    k = Z.shape[0]

    distances = np.zeros((M, k)) #Same shape as U but for distances rather than if it belongs or not
    W_beta = np.power(W, beta)

    for l in range(k): #Still need minimum one loop :(
        diff = data_points - Z[l] #Take out cluster l first
        weighted_diffs = np.sum((diff * diff) * W_beta[l], axis=1) #Weighted sum over features
        distances[:, l] = weighted_diffs #Store results for cluster l

    U = np.zeros((M, k))

    #For future self
    #np.argmin(distances, axis=1) finds the cluster which gives the minimum distance for each point
    #np.arange(M) sets the column to 1 for that row if its the closes one
    U[np.arange(M), np.argmin(distances, axis=1)] = 1 #Im really pushing my luck with numpy, ask supervisor if I can use it to this extent
    
    return U

#P2: Fix U = U^ and W = W^; solve the reduced problem P(U^, Z, W^)
#z_{l, j} = Σ^{M}_{i = 1}u_{i,l}.x_{i, j}/Σ^{M}_{i = 1}.u_{i, l} FOR 1 ≤ l ≤ k AND 1 ≤ j ≤ N
#For Numerical: The center of cluster l for feature j is just the average of that feature's values for all points in cluster l
#For Categorical: Its the mode of j/ most common value of j.
def P2(data_points, U, Z, W, beta, categorical_features=None):
    M, N = data_points.shape
    k = Z.shape[0]

    Z_new = np.zeros_like(Z) # _like for same shape
    for l in range(k):
        cluster_mask = U[:, l] == 1
        if np.any(cluster_mask):
            #Use simple mean as books's formula
            #z_{l,j} = Σ(u_{i,l} * x_{i,j}) / Σ(u_{i,l})
            Z_new[l] = np.mean(data_points[cluster_mask], axis=0)
        else:
            #If cluster is empty, initialize with random point
            Z_new[l] = data_points[np.random.randint(M)]

    return Z_new
    #NOTE: maybe change how empty points are dealt with

#P3: Fix U = U^ and Z = Z^; solve the reduced problem P(U^, Z^, W)
#D_{j} = Σ^{k}_{l = 1}Σ^{M}_{i = 1} = u^_{i, l} . d(x_{i,j). z_{l, j})
#For feature j, calculate difference between j and the mean of j for that cluster.
#w^_{j} = (Σ^{h}_{t = 1} [D_{j}/D_{t}]^((β - 1)^-1))^-1 IF D_{j} != 0
#If dispersion is anything but 0, sum up the ratio of the dispersion of j's(current feature) to every features dispersion (t), and then raised to the power of 1/(β - 1). THEN GETTING EVERYTHING TO THE POWER OF -1
#w^_{j} = 0 IF D_{j} = 0
#If dispersion is 0 the weight is 0
def P3(U, Z, beta, data_points):
    M, N = data_points.shape
    k = Z.shape[0]

    Dlj = np.zeros((k, N))
    
    #1. Avarage dispersion for all features so we can use it in Dlj
    total_dispersion = 0
    count = 0
    
    #Basically do Dj Twice
    for l in range(k):
        cluster_mask = U[:, l] == 1 #Masking to iterate through only the cluster l
        if np.any(cluster_mask):
            diff = data_points[cluster_mask] - Z[l] #Calculates the difference
            squared_diff = diff * diff #Yes my code is readable, yes I am a sane person
            total_dispersion += np.sum(squared_diff)
            count += np.sum(cluster_mask) * N
            
    sigma = total_dispersion / count if count > 0 else 1.0 #https://i.imgflip.com/8cgkuf.png
    
    #Calculate Dj with sigma
    #Yes I know its Dlj but Dj sounds better
    for l in range(k):
        cluster_mask = U[:, l] == 1
        if np.any(cluster_mask):
            diff = data_points[cluster_mask] - Z[l]
            squared_diff = diff * diff + sigma  #Sigma + distance 
            Dlj[l] = np.sum(squared_diff, axis=0)

    W = np.zeros((k, N))

    #Weights for each feature using within cluster ratio
    for l in range(k):
        for j in range(N):
            ratios = Dlj[l, j] / Dlj[l, :]  #Calculate ratios within cluster l, might vectorise these loops soon
            denominator = np.sum(ratios ** (1/(beta-1)))
            W[l, j] = 1 / denominator #Logical, logical
        
        W[l] = W[l] / np.sum(W[l]) #Normalize to 1
    
    return W

input_filename = 'iris.data.data'
base_filename = input_filename.split('.')[0]
output_filename = f"{base_filename}.predicted"

best_p0 = float('inf')
best_result = None

for i in range(1):
    clusters, W, Z, U, data = main(input_filename, k=3, beta=2) #U is really redundant
    curr_p0 = P0(data, U, Z, W, beta=-1)
    print(f"Run {i+1} P0: {curr_p0}")
    
    if curr_p0 < best_p0: #Scenario where P0 value of 26.98 is bigger than P0 value of 26.63 but still gives higher accuracy value
        best_p0 = curr_p0
        best_result = (clusters, W, Z, U) #U is really redundant
        print(f"New best P0: {best_p0}")

with open(output_filename, 'w') as f:
    for label in best_result[0]:
        f.write(f"{label}\n")