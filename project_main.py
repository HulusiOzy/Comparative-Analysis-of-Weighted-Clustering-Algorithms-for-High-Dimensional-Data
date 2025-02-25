import pandas as pd
import numpy as np

from functions import (
    list_datasets, list_algorithms, list_tools,
    save, load, drop, init_weights, init_centers, init_data,
    confusion_matrix, ari, pca, silhouette_index, ch_index,
    data, predicted, centers, weights
)
from algorithms import (
    kmeans, wkmeans, swkmeans,
    mwkmeans, upgma, wards, iap,
    global_kmeans
)

__all__ = [
    'list_datasets', 'list_algorithms', 'list_tools',
    'save', 'load', 'save_results', 'drop'

    'confusion_matrix', 'ari', 'pca', 'silhouette_index', 'ch_index'

    'kmeans', 'wkmeans', 'swkmeans',
    'mwkmeans', 'upgma', 'wards',
    'iap', 'global_kmeans'

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
    
    init_centers(2)
    init_weights()
    initial_centers = functions.centers
    initial_weights = functions.weights
    algorithms.wkmeans(k=2, beta=2, initial_centers=initial_centers)
    baseline = functions.silhouette_index(display=0)
    
    for i in range(n_cols):
        print(f"Processing column {i}")
        functions.load('Depression Student Dataset.csv', 1)
        functions.drop(i)
        algorithms.wkmeans(k=2, beta=2, initial_centers=initial_centers)
        new_si = functions.silhouette_index(display=0)
        impact_dict[i] = new_si - baseline
        
    sorted_impact_dict = sorted(impact_dict.items(), key=lambda x: x[1])
    new_order = [x[0] for x in sorted_impact_dict]
    functions.load('Depression Student Dataset.csv', 1)
    functions.data = functions.data[functions.data.columns[new_order]]
    
    results = []
    for i in range(len(functions.data.columns)):
        algorithms.wkmeans(k=2, beta=2, initial_centers=initial_centers)
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

def help():
    print('\nload(filename, preprocess=0)')
    print('Load dataset from the Datasets folder. Set preprocess=1 for automatic data preprocessing.')
    print('--------------------')

    print('\nsave(filename)')
    print('Save current data, actual labels, and predicted labels to output files.')
    print('--------------------')

    print('\ndrop(n)')
    print('Remove the nth column from the loaded dataset.')
    print('--------------------')

    print('\ninit_data(n_samples=100, n_features=2, n_clusters=3, cluster_std=1.0)')
    print('Generate synthetic clustering data for testing.')
    print('--------------------')

    print('\ninit_weights()')
    print('Initialize feature weights using various methods (random, uniform, variance-based, etc.).')
    print('--------------------')

    print('\ninit_centers(k)')
    print('Initialize cluster centers using either random points or sampled data points.')
    print('--------------------')

    print('\nkmeans(k, initial_centers=None)')
    print('Basic K-Means clustering algorithm.')
    print('--------------------')

    print('\nwkmeans(k, beta, initial_centers=None, initial_weights=None)')
    print('Weighted K-Means with feature weights.')
    print('--------------------')

    print('\nswkmeans(k, beta, initial_centers=None, initial_weights=None)')
    print('Subspace Weighted K-Means for high-dimensional data.')
    print('--------------------')

    print('\nmwkmeans(k, p, initial_centers=None, initial_weights=None)')
    print('Minkowski Weighted K-Means.')
    print('--------------------')

    print('\nupgma(k)')
    print('Unweighted Pair Group Method with Arithmetic Mean.')
    print('--------------------')

    print('\nwards(k)')
    print("Ward's Hierarchical Clustering.")
    print('--------------------')

    print('\niap(k)')
    print('Iterative Anomalous Pattern detection.')
    print('--------------------')

    print('\nconfusion_matrix(display=0)')
    print('Calculate and optionally display confusion matrix between actual and predicted labels.')
    print('--------------------')

    print('\nari(display=0)')
    print('Calculate Adjusted Rand Index between actual and predicted labels.')
    print('--------------------')

    print('\nsilhouette_index(display=1)')
    print('Calculate Silhouette Index for clustering quality evaluation.')
    print('--------------------')

    print('\nch_index(display=1)')
    print('Calculate Calinski-Harabasz Index for clustering quality evaluation.')
    print('--------------------')

    print('\npca(dimensions=2, display=1, data=None, highlight_actual=0, highlight_centers=False)')
    print('Perform PCA dimensionality reduction and optionally plot results.')
    print('Set highlight_centers=True to display initial centers in the visualization.')
    print('--------------------')

    print('\nlist_datasets()')
    print('Display available datasets in the Datasets folder.')
    print('--------------------')

    print('\nlist_algorithms()')
    print('Display available clustering algorithms and their parameters.')
    print('--------------------')

    print('\nlist_tools()')
    print('Display available analysis and preprocessing tools.')
    print('--------------------')
