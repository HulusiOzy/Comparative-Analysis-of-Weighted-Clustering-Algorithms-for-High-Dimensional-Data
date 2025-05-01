import pandas as pd
import numpy as np
import functions

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
    'save', 'load', 'drop',
    'confusion_matrix', 'ari', 'pca', 'silhouette_index', 'ch_index',
    'kmeans', 'wkmeans', 'swkmeans',
    'mwkmeans', 'upgma', 'wards',
    'iap', 'global_kmeans',
    'data', 'predicted', 'centers', 'weights',
    'actual', 'accuracy'
]

def feature_removal():
    import algorithms
    
    algorithms_dict = {
        'kmeans': algorithms.kmeans,
        'global_kmeans': algorithms.global_kmeans,
        'wkmeans': algorithms.wkmeans,
        'swkmeans': algorithms.swkmeans,
        'mwkmeans': algorithms.mwkmeans,
        'upgma': algorithms.upgma,
        'wards': algorithms.wards,
        'iap': algorithms.iap
    }
    
    filename = input("\nEnter dataset filename: ")
    
    print("\nAvailable algorithms:")
    for i, key in enumerate(algorithms_dict.keys()):
        print(f"{i+1}. {key}")
    
    algo_idx = int(input("\nSelect algorithm (number): ")) - 1
    algorithm = list(algorithms_dict.keys())[algo_idx]
    algo_func = algorithms_dict[algorithm]
    
    params = {'k': 2}
    if algorithm in ['wkmeans', 'swkmeans']:
        params['beta'] = float(input("Beta parameter: "))
    elif algorithm == 'mwkmeans':
        p_input = input("Minkowski parameter (p), 'inf' for infinity: ")
        params['p'] = float('inf') if p_input.lower() == 'inf' else float(p_input)
    
    impact_dict = {}
    functions.load(filename, 1)
    n_cols = functions.data.shape[1]
    
    algo_func(**params)
    baseline = functions.silhouette_index(display=0)
    
    for i in range(n_cols):
        print(f"Testing column {i}")
        functions.load(filename, 1)
        functions.drop(i)
        algo_func(**params)
        impact_dict[i] = functions.silhouette_index(display=0) - baseline
    
    sorted_impact = sorted(impact_dict.items(), key=lambda x: x[1])
    new_order = [x[0] for x in sorted_impact]
    
    functions.load(filename, 1)
    functions.data = functions.data[functions.data.columns[new_order]]
    
    results = []
    for i in range(len(functions.data.columns)):
        algo_func(**params)
        acc = functions.confusion_matrix(0)
        results.append((new_order[i], acc))
        print(f"Dropping col:{new_order[i]}, Accuracy:{acc}")
        functions.drop(0)
    
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

    print('\ncompare_all()')
    print('Compare all the cluster algorithms on a dataseti.')
    print('-------------------')
