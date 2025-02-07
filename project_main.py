import pandas as pd
import numpy as np

from functions import (
    list_datasets, list_algorithms, list_tools,
    save, load, drop, init_weights,
    confusion_matrix, ari, pca, silhouette_index,
    data, predicted, centers, weights
)
from algorithms import (
    kmeans, wkmeans, swkmeans,
    mwkmeans, upgma, wards, iap
)

__all__ = [
    'list_datasets', 'list_algorithms', 'list_tools',
    'save', 'load', 'save_results', 'drop'

    'confusion_matrix', 'ari', 'pca', 'silhouette_index',

    'kmeans', 'wkmeans', 'swkmeans',
    'mwkmeans', 'upgma', 'wards',

    'data', 'predicted', 'centers', 'weights'

    'actual', 'predicted', 'accuracy', 'ari'
]

import functions
import algorithms
import numpy as np

def feature_removal():
    impact_dict = {}
    
    functions.load('Depression Student Dataset.csv', 1)
    n_cols = functions.data.shape[1]
    
    algorithms.iap(k=2)
    initial_centers = functions.centers.copy()
    algorithms.kmeans(k=2, initial_centers=initial_centers)
    baseline = functions.silhouette_index(display=0)
    
    for i in range(n_cols):
        print(f"Processing column {i}")
        functions.load('Depression Student Dataset.csv', 1)
        functions.drop(i)
        algorithms.iap(k=2)
        initial_centers = functions.centers.copy()
        algorithms.kmeans(k=2, initial_centers=initial_centers)
        new_si = functions.silhouette_index(display=0)
        impact_dict[i] = new_si - baseline
        
    sorted_impact_dict = sorted(impact_dict.items(), key=lambda x: x[1])
    new_order = [x[0] for x in sorted_impact_dict]
    functions.load('Depression Student Dataset.csv', 1)
    functions.data = functions.data[functions.data.columns[new_order]]
    
    results = []
    for i in range(len(functions.data.columns)):
        algorithms.iap(k=2)
        initial_centers = functions.centers.copy()
        algorithms.kmeans(k=2, initial_centers=initial_centers)
        acc = functions.confusion_matrix(0)
        results.append((i, acc))
        print(f"Dropping col:{new_order[i]}, Accuracy:{acc}")
        functions.drop(0)
        
    return results

def feature_removal_two():
    functions.load('Depression Student Dataset.csv', 1)
    results = []
    
    while functions.data.shape[1] > 1:
        baseline = functions.silhouette_index(display=0)
        best_impact = -float('inf')
        best_col = None
        
        for col in range(functions.data.shape[1]):
            data_copy = functions.data.copy()
            functions.drop(col)
            algorithms.wkmeans(k=2, beta=2)
            impact = functions.silhouette_index(display=0) - baseline
            
            if impact > best_impact:
                best_impact = impact
                best_col = col
            
            functions.data = data_copy
        
        functions.drop(best_col)
        algorithms.wkmeans(k=2, beta=2)
        acc = functions.confusion_matrix(display=0)
        results.append((best_col, acc))
        print(f"Removed column {best_col}, Accuracy: {acc}")
    
    return results

def data_order():
    impact_dict = {}
    
    functions.load('Depression Student Dataset.csv', 1)
    n_cols = functions.data.shape[1]
    
    algorithms.iap(k=2)
    initial_centers = functions.centers.copy()
    algorithms.kmeans(k=2, initial_centers=initial_centers)
    baseline = functions.silhouette_index(display=0)
    
    for i in range(n_cols):
        print(f"Processing column {i}")
        functions.load('Depression Student Dataset.csv', 1)
        functions.drop(i)
        algorithms.iap(k=2)
        initial_centers = functions.centers.copy()
        algorithms.kmeans(k=2, initial_centers=initial_centers)
        new_si = functions.silhouette_index(display=0)
        impact_dict[i] = new_si - baseline
        
    sorted_impact_dict = sorted(impact_dict.items(), key=lambda x: x[1], reverse=True)
    new_order = [x[0] for x in sorted_impact_dict]
    functions.load('Depression Student Dataset.csv', 1)
    functions.data = functions.data[functions.data.columns[new_order]]  

    return functions.data