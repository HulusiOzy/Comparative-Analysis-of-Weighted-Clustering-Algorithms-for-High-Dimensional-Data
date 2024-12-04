import numpy as np
from itertools import permutations

def accuracy(matrix):
    #matrix = np.array(matrix)
    diagonal_sum = np.trace(matrix)
    total_sum = np.sum(matrix) #I could have used just the size of the original list but whatever
    return diagonal_sum / total_sum

def confusion_matrix(actual, predicted):
    labels = list(set(actual)) #Order dont matter because we look at permutations
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)} #Creates a label to idx dictionary
    matrix = np.array([[0] * n_labels] * n_labels) #Use np.zeros nexttime
    for true, pred in zip(actual, predicted):
        pred_idx = label_to_idx[pred] #Column 
        acc_idx = label_to_idx[true] #Row
        matrix[acc_idx, pred_idx] += 1
    return matrix

def match_values(actual_values, predicted_values):
    actual_unique = sorted(list(set(actual_values)))
    pred_unique = sorted(list(set(predicted_values)))
    
    if len(actual_unique) != len(pred_unique): #Dumb optimizations
        raise ValueError("Number of unique labels in actual and predicted must match")
    
    label_map = dict(zip(pred_unique, actual_unique)) #Mapping
    
    matched_values = [label_map[pred_label] for pred_label in predicted_values] #New list
    
    return matched_values

def load_labels(filename):
    label_dict = {}
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            label_dict[idx] = int(line.strip())
    return label_dict

#Bismillah, this is peak spaghetti code
def try_all_permutations(matrix):
    n = matrix.shape[0] #Get dimensions, yes I could of just passed in labels but whatever
    best_accuracy = 0
    best_matrix = None
    
    for perm in permutations(range(n)): #Maybe implement my own permutations but this is ok for now
        permuted_matrix = matrix[:, perm] #Create new matrix with columns in current permutation order
        acc = accuracy(permuted_matrix)
        print(permuted_matrix, acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_matrix = permuted_matrix
    
    return best_matrix, best_accuracy

actual = load_labels('iris.actual')
predicted = load_labels('iris.predicted')

actual_values = list(actual.values())
predicted_values = list(predicted.values())

predicted_values = match_values(actual_values, predicted_values)

matrix = confusion_matrix(actual_values, predicted_values)
best_matrix, best_accuracy = try_all_permutations(matrix)
print("Confusion Matrix:")
print(best_matrix)
print(f"Accuracy: {best_accuracy}")

##Check notebook, i got notes for TP/FN/FP/TN calculations for bigger matrices
##Then do TPR/FNR/FPR/TNR based off of those :D.