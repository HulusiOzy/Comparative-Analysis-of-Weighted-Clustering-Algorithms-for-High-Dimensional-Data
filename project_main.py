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

import functions
import algorithms

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

ALGORITHMS = {
    'kmeans': {'function': kmeans, 'params': ['k']},
    'global_kmeans': {'function': global_kmeans, 'params': ['k']},
    'wkmeans': {'function': wkmeans, 'params': ['k', 'beta']},
    'swkmeans': {'function': swkmeans, 'params': ['k', 'beta']},
    'mwkmeans': {'function': mwkmeans, 'params': ['k', 'p']},
    'upgma': {'function': upgma, 'params': ['k']},
    'wards': {'function': wards, 'params': ['k']},
    'iap': {'function': iap, 'params': ['k']}
}

def _select_algorithm(prompt="Select algorithm"):
    print("\nAvailable algorithms:")
    for i, key in enumerate(ALGORITHMS.keys()):
        print(f"{i+1}. {key}")

    algo_idx = int(input(f"\n{prompt} (number): ")) - 1
    algo_key = list(ALGORITHMS.keys())[algo_idx]
    return algo_key, ALGORITHMS[algo_key]

def _get_params(algo_key, algo_info, k=None):
    params = {'k': k if k is not None else int(input(f"K for {algo_key}: "))}

    if 'beta' in algo_info['params']:
        params['beta'] = float(input(f"Beta for {algo_key}: "))

    if 'p' in algo_info['params']:
        p_input = input(f"P for {algo_key} (inf for infinity): ")
        params['p'] = float('inf') if p_input.lower() == 'inf' else float(p_input)

    return params

def _calculate_metrics(predicted):
    metrics = {
        'silhouette': silhouette_index(display=0, predicted=predicted),
        'ch_index': ch_index(display=0, predicted=predicted),
        'accuracy': None,
        'ari': None
    }

    if functions.actual is not None:
        metrics['accuracy'] = confusion_matrix(display=0, predicted=predicted)
        metrics['ari'] = ari(display=0, predicted=predicted)

    return metrics

def feature_removal():
    filename = input("\nEnter dataset filename: ")
    algo_key, algo_info = _select_algorithm()
    params = _get_params(algo_key, algo_info, k=2)
    
    functions.load(filename, 1)
    original_data = functions.data.copy()
    n_cols = original_data.shape[1]
    
    min_cols = max(2, n_cols // 3)
    
    algo_info['function'](**params)
    baseline = silhouette_index(display=0)
    
    impact_dict = {}
    for i in range(n_cols):
        print(f"Testing column {i}")
        functions.data = original_data.drop(original_data.columns[i], axis=1)
        algo_info['function'](**params)
        impact_dict[i] = silhouette_index(display=0) - baseline
    
    sorted_impact = sorted(impact_dict.items(), key=lambda x: x[1])
    column_order = [x[0] for x in sorted_impact]
    
    functions.load(filename, 1)
    results = []
    
    max_to_remove = n_cols - min_cols
    
    for i in range(min(max_to_remove, len(column_order))):
        col_to_remove = column_order[i]
        col_name = original_data.columns[col_to_remove]
        
        current_index = list(functions.data.columns).index(col_name)
        
        algo_info['function'](**params)
        acc = confusion_matrix(0)
        results.append((col_to_remove, acc))
        print(f"Dropping col:{col_to_remove}, Accuracy:{acc}")
        
        functions.drop(current_index)
    
    functions.load(filename, 1)
    
    return results

def compare():
    if functions.data is None or len(functions.data) == 0:
        print("No data loaded. Use load() function first.")
        return

    algorithms_dict = {
        'kmeans': {'function': kmeans, 'params': ['k']},
        'global_kmeans': {'function': global_kmeans, 'params': ['k']},
        'wkmeans': {'function': wkmeans, 'params': ['k', 'beta']},
        'swkmeans': {'function': swkmeans, 'params': ['k', 'beta']},
        'mwkmeans': {'function': mwkmeans, 'params': ['k', 'p']},
        'upgma': {'function': upgma, 'params': ['k']},
        'wards': {'function': wards, 'params': ['k']},
        'iap': {'function': iap, 'params': ['k']}
    }
    
    print("\nAvailable algorithms:")
    for i, (key, _) in enumerate(algorithms_dict.items()):
        print(f"{i+1}. {key}")

    algo1_idx = int(input("\nSelect first algorithm: ")) - 1
    algo1_key = list(algorithms_dict.keys())[algo1_idx]
    algo1 = algorithms_dict[algo1_key]

    algo2_idx = int(input("\nSelect second algorithm: ")) - 1
    algo2_key = list(algorithms_dict.keys())[algo2_idx]
    algo2 = algorithms_dict[algo2_key]

    algo1_params = {'k': int(input(f"K for {algo1_key}: "))}
    for param in algo1['params']:
        if param == 'beta' and param not in algo1_params:
            algo1_params['beta'] = float(input(f"Beta for {algo1_key}: "))
        elif param == 'p' and param not in algo1_params:
            p_input = input(f"P for {algo1_key} (inf for infinity): ")
            algo1_params['p'] = float('inf') if p_input.lower() == 'inf' else float(p_input)

    algo2_params = {'k': int(input(f"K for {algo2_key}: "))}
    for param in algo2['params']:
        if param == 'beta' and param not in algo2_params:
            algo2_params['beta'] = float(input(f"Beta for {algo2_key}: "))
        elif param == 'p' and param not in algo2_params:
            p_input = input(f"P for {algo2_key} (inf for infinity): ")
            algo2_params['p'] = float('inf') if p_input.lower() == 'inf' else float(p_input)

    algo1['function'](**algo1_params)
    predicted1 = functions.predicted.copy()

    algo2['function'](**algo2_params)
    predicted2 = functions.predicted.copy()

    si1 = silhouette_index(display=0, predicted=predicted1)
    ch1 = ch_index(display=0, predicted=predicted1)
    si2 = silhouette_index(display=0, predicted=predicted2)
    ch2 = ch_index(display=0, predicted=predicted2)

    cm1, ari1, cm2, ari2 = None, None, None, None
    if functions.actual is not None:
        cm1 = confusion_matrix(display=0, predicted=predicted1)
        ari1 = ari(display=0, predicted=predicted1)
        cm2 = confusion_matrix(display=0, predicted=predicted2)
        ari2 = ari(display=0, predicted=predicted2)

    print(f"\nResults: {algo1_key} vs {algo2_key}")
    print(f"Silhouette: {si1:.4f} vs {si2:.4f}")
    print(f"CH Index: {ch1:.4f} vs {ch2:.4f}")
    if cm1 is not None:
        print(f"Accuracy: {cm1:.4f} vs {cm2:.4f}")
        print(f"ARI: {ari1:.4f} vs {ari2:.4f}")

    return {
        'algorithm1': {
            'name': algo1_key,
            'silhouette': si1,
            'ch_index': ch1,
            'accuracy': cm1,
            'ari': ari1
        },
        'algorithm2': {
            'name': algo2_key,
            'silhouette': si2,
            'ch_index': ch2,
            'accuracy': cm2,
            'ari': ari2
        }
    }

def compare_all():
    if functions.data is None or len(functions.data) == 0:
        print("No data loaded. Use load() function first.")
        return

    algorithms_dict = {
        'kmeans': {'function': kmeans, 'params': ['k']},
        'global_kmeans': {'function': global_kmeans, 'params': ['k']},
        'wkmeans': {'function': wkmeans, 'params': ['k', 'beta']},
        'swkmeans': {'function': swkmeans, 'params': ['k', 'beta']},
        'mwkmeans': {'function': mwkmeans, 'params': ['k', 'p']},
        'upgma': {'function': upgma, 'params': ['k']},
        'wards': {'function': wards, 'params': ['k']},
        'iap': {'function': iap, 'params': ['k']}
    }

    k = int(input("Number of clusters (k): "))
    beta = float(input("Beta parameter for weighted algorithms: "))
    p_input = input("Minkowski parameter (p), 'inf' for infinity: ")
    p = float('inf') if p_input.lower() == 'inf' else float(p_input)

    results = {}

    for algo_key, algo in algorithms_dict.items():
        print(f"Running {algo_key}...")

        params = {'k': k}
        if 'beta' in algo['params']:
            params['beta'] = beta
        if 'p' in algo['params']:
            params['p'] = p

        try:
            algo['function'](**params)
            predicted = functions.predicted.copy()

            si = silhouette_index(display=0, predicted=predicted)
            ch = ch_index(display=0, predicted=predicted)

            cm, ari_score = None, None
            if functions.actual is not None:
                cm = confusion_matrix(display=0, predicted=predicted)
                ari_score = ari(display=0, predicted=predicted)

            results[algo_key] = {
                'silhouette': si,
                'ch_index': ch,
                'accuracy': cm,
                'ari': ari_score,
                'params': params
            }

        except Exception as e:
            print(f"Error with {algo_key}: {str(e)}")
            results[algo_key] = {'error': str(e)}

    print("\nResults:")
    for algo in results:
        if 'error' in results[algo]:
            print(f"{algo}: ERROR")
        else:
            si = results[algo]['silhouette']
            ch = results[algo]['ch_index']
            acc = results[algo]['accuracy']
            ari_val = results[algo]['ari']
            print(f"{algo}: SI={si:.4f}, CH={ch:.4f}", end="")
            if acc is not None:
                print(f", ACC={acc:.4f}, ARI={ari_val:.4f}", end="")
            print()

    return results

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

    print('\nglobal_kmeans(k)')
    print('Global K-Means clustering algorithm with deterministic initialization.')
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

    print('\ncompare()')
    print('Compare two clustering algorithms and evaluate their performance using various validity indices.')
    print('--------------------')
