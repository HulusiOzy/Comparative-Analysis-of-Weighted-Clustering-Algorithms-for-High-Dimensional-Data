import pandas as pd
import numpy as np
import os

from Tools.PreProcessing import Preprocessing
from Tools.Table_of_Confusion import TableOfConfusion
from Tools.Silhouette_Index import SilhouetteIndex
from Tools.CH_Index import CHIndex
from Tools.ARI import ARI
from Tools.PCA import PCA

np.set_printoptions(threshold=np.inf) 

data = pd.DataFrame()
actual = None
predicted = None
centers = None
weights = None

accuracy = None
silhouetteIndex = None
adjusted = None

DATASETS = {
    'iris.data': {
        'description': 'Iris flower dataset',
        'format': 'CSV with no headers'
    },
    'search_engine_data.csv': {
        'description': 'Search engine performance data',
        'format': 'CSV with headers'
    },
    'StudentPerformanceFactors.csv': {
        'description': 'Student academic performance factors',
        'format': 'CSV with headers'
    },
    'Depression Student Dataset.csv': {
        'description': 'Student depression indicators',
        'format': 'CSV with headers'
    }
}

TOOLS = {
    'PCA': {
        'description': 'Dimensionality reduction tool',
        'purpose': 'Reduces data dimensions while preserving variance'
    },
    'Preprocessing': {
        'description': 'Data preparation tool',
        'purpose': 'Normalizes and standardizes input data'
    },
    'SilhouetteIndex': {
        'description': 'Clustering evaluation tool',
        'purpose': 'Measures clustering quality and cohesion'
    },
    'TableOfConfusion': {
        'description': 'Results analysis tool',
        'purpose': 'Evaluates clustering accuracy and visualizes results'
    }
}

ALGORITHMS = {
    'kmeans': {
        'description': 'Basic K-Means clustering',
        'parameters': {
            'required': ['k'],
            'optional': ['initial_centers']
        }
    },   
    'global_kmeans': {
        'description': 'Global K-means clustering - deterministic incremental approach',
        'parameters': {
            'required': ['k'],
            'optional': []
        }
    },
    'iap': {
        'description': 'Iterative Anomalous Pattern',
        'parameters': {
            'required': ['k'],
            'optional': []
        }
    },
    'wkmeans': {
        'description': 'Weighted K-Means',
        'parameters': {
            'required': ['k', 'beta'],
            'optional': ['initial_centers', 'initial_weights']
        }
    },
    'swkmeans': {
        'description': 'Subspace Weighted K-Means',
        'parameters': {
            'required': ['k', 'beta'],
            'optional': ['initial_centers', 'initial_weights']
        }
    },
    'mwkmeans': {
        'description': 'Minkowski Weighted K-Means',
        'parameters': {
            'required': ['k', 'p'],
            'optional': ['initial_centers', 'initial_weights']
        }
    },
    'upgma': {
        'description': 'Unweighted Pair Group Method with Arithmetic Mean',
        'parameters': {
            'required': ['k'],
            'optional': []
        }
    },
    'wards': {
        'description': "Ward's Hierarchical Clustering",
        'parameters': {
            'required': ['k'],
            'optional': []
        }
    }
}

def list_datasets():
    print("\nAvailable Datasets:")
    for name, info in DATASETS.items():
        print(f"\n{name}")
        print(f"Description: {info['description']}")
        print(f"Format: {info['format']}")

def list_algorithms():
    print("\nAvailable Clustering Algorithms:")
    for name, info in ALGORITHMS.items():
        print(f"\n{name.upper()}")
        print(f"Description: {info['description']}")
        print(f"Required parameters: {', '.join(info['parameters']['required'])}")
        if info['parameters']['optional']:
            print(f"Optional parameters: {', '.join(info['parameters']['optional'])}")

def list_tools():
    print("\nAvailable Tools:")
    for name, info in TOOLS.items():
        print(f"\n{name}")
        print(f"Description: {info['description']}")
        print(f"Purpose: {info['purpose']}")

def load(filename, preprocess=0, target=1, bin=0, target_col=-1):
    global data, actual

    dataset_path = os.path.join('Datasets', filename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File '{filename}' not found in Datasets folder")
    
    if preprocess:
        preprocessor_target_col = None
        if target_col >= 0:
            preprocessor_target_col = str(target_col)
        elif target_col < -1:
            preprocessor_target_col = str(target_col)
        
        preprocessor = Preprocessing(input_file=dataset_path, 
                                     target_column=preprocessor_target_col, 
                                     column_style='N', 
                                     bin_option=bin, 
                                     has_target=(target==1))
        processed_path, actual_path = preprocessor.fit()
        data = pd.read_csv(processed_path, header=None)
        
        if target and actual_path:
            actual = pd.read_csv(actual_path, header=None).iloc[:, 0].values
            os.remove(actual_path)
        else:
            actual = None
            
        os.remove(processed_path)
    else:
        data = pd.read_csv(dataset_path)
        if target:
            if target_col < 0:
                target_col = data.shape[1] + target_col
            actual = data.iloc[:, target_col].values
            data = data.drop(data.columns[target_col], axis=1)
        else:
            actual = None
    
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['data'] = data
            ipython.user_ns['actual'] = actual
    except NameError:
        pass
    
    return data

def save(filename):
    if actual is not None:
        np.savetxt('output.actual', actual, delimiter=',')
    if predicted is not None:
        np.savetxt('output.predicted', predicted, delimiter=',')
    if data is not None:
        np.savetxt('output.data', data, delimiter=',')

def drop(n):
    global data
    
    if data is None or len(data) == 0:
        raise ValueError("No data loaded. Use load() function first.")
        
    if not 0 <= n < data.shape[1]:
        raise ValueError(f"Column index {n} out of bounds for data with {data.shape[1]} columns")
    
    data = data.drop(data.columns[n], axis=1)
    
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['data'] = data
    except NameError:
        pass
    
    return data

def confusion_matrix(display=0, actual=None, predicted=None):
    actual_data = actual if actual is not None else globals()['actual']
    predicted_data = predicted if predicted is not None else globals()['predicted']
    
    tc = TableOfConfusion(actual_data, predicted_data)
    accuracy = float(tc.fit())
    
    if display:
        tc.plot_confusion_matrix()
        
    return accuracy

def ari(display = 0, actual = None, predicted = None):
    actual_data = actual if actual is not None else globals()['actual']
    predicted_data = predicted if predicted is not None else globals()['predicted']

    ari_calculator = ARI()
    score = float(ari_calculator.fit(actual_data, predicted_data))

    if display:
        ari_calculator.plot()
        
    return score

def pca(dimensions=2, display=1, data=None, highlight_actual=False):
    input_data = data if data is not None else globals()['data']

    pca_transformer = PCA()
    transformed_data = pca_transformer.fit_transform(
        input_data,
        dimensions=dimensions,
        display=display,
        highlight_actual=highlight_actual,
    )

    return transformed_data

def silhouette_index(display=1, data=None, predicted=None):
    input_data = data if data is not None else globals()['data']
    predicted_labels = predicted if predicted is not None else globals()['predicted']
    
    si_calculator = SilhouetteIndex(input_data, predicted_labels)
    silhouetteIndex = float(si_calculator.fit())
    
    if display:
        print(f"\nSilhouette Index: {silhouetteIndex:.4f}")
    
    return silhouetteIndex  

def ch_index(display=0, data=None, predicted=None):
    input_data = data if data is not None else globals()['data']
    predicted_labels = predicted if predicted is not None else globals()['predicted']
    
    ch_calculator = CHIndex(input_data, predicted_labels)
    ch_score = float(ch_calculator.fit())
    
    if display:
        ch_calculator.plot()
    
    return ch_score

def init_weights():
    global data

    if data is None or len(data) == 0:
        raise ValueError("No data loaded. Use load() function first.")

    working_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    for col in working_data.columns:
        if not pd.api.types.is_numeric_dtype(working_data[col]):
            working_data[col], _ = pd.factorize(working_data[col])

    X = working_data.values
    n_features = X.shape[1]

    print("\nInitialization methods:")
    print("1. Random")
    print("2. Uniform")
    print("3. Variance-based")
    print("4. Entropy-based")
    print("5. Dispersion-based")

    while True:
        try:
            choice = int(input("\nSelect method (1-5): "))
            if choice not in [1, 2, 3, 4, 5]:
                print("Please enter 1, 2, 3, 4, or 5")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    if choice == 1:
        weights = np.random.rand(n_features)
    elif choice == 2:
        weights = np.ones(n_features)
    elif choice == 3:
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-10
        weights = 1 / (stds * stds)
    elif choice == 4:
        weights = np.zeros(n_features)
        n_samples = len(X)

        for i in range(n_features):
            xi_iqr = np.subtract(*np.percentile(X[:, i], [75, 25]))
            xi_width = 2 * xi_iqr / np.cbrt(n_samples) if xi_iqr else None

            if xi_width:
                xi_bins = int(np.ceil((np.max(X[:, i]) - np.min(X[:, i])) / xi_width))
            else:
                xi_bins = int(np.ceil(1 + np.log2(n_samples)))

            xi_hist = np.histogram(X[:, i], bins=xi_bins)[0]
            xi_probs = xi_hist[xi_hist > 0] / n_samples
            H_Xi = -np.sum(xi_probs * np.log2(xi_probs))

            H_Xi_given_j = np.zeros(n_features)
            for j in range(n_features):
                if i != j:
                    xj_iqr = np.subtract(*np.percentile(X[:, j], [75, 25]))
                    xj_width = 2 * xj_iqr / np.cbrt(n_samples) if xj_iqr else None

                    if xj_width:
                        xj_bins = int(np.ceil((np.max(X[:, j]) - np.min(X[:, j])) / xj_width))
                    else:
                        xj_bins = int(np.ceil(1 + np.log2(n_samples)))

                    joint_hist = np.histogram2d(X[:, i], X[:, j], bins=[xi_bins, xj_bins])[0]
                    joint_probs = joint_hist / n_samples

                    xj_probs = np.sum(joint_probs, axis=1)
                    mask = xj_probs > 0
                    cond_probs = (joint_probs[mask] / xj_probs[mask, np.newaxis])
                    cond_probs = cond_probs[cond_probs > 0]

                    H_Xi_given_j[j] = -np.sum(cond_probs * np.log2(cond_probs))

            gain = np.sum(H_Xi_given_j[H_Xi_given_j > 0]) - ((len(H_Xi_given_j) - 1) * H_Xi)
            weights[i] = abs(gain)
    else:  
        print("\nDispersion calculation methods:")
        print("1. Range Based")
        print("2. Standard Deviation")
        print("3. Interquartile Range (IQR)")
        print("4. Median Absolute Deviation (MAD)")

        while True:
            try:
                disp_choice = int(input("\nSelect dispersion method (1-4): "))
                if disp_choice not in [1, 2, 3, 4]:
                    print("Please enter 1, 2, 3, or 4")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")

        dispersions = np.zeros(n_features)

        for i in range(n_features):
            if disp_choice == 1:  
                dispersions[i] = np.max(X[:, i]) - np.min(X[:, i])
            elif disp_choice == 2:  
                dispersions[i] = np.std(X[:, i])
            elif disp_choice == 3:  
                dispersions[i] = np.subtract(*np.percentile(X[:, i], [75, 25]))
            else:  
                median = np.median(X[:, i])
                dispersions[i] = np.median(np.abs(X[:, i] - median))

        dispersions[dispersions == 0] = 1e-10
        weights = 1 / (dispersions * dispersions)

    weights = weights / np.sum(weights)

    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['weights'] = weights
    except NameError:
        pass

    return weights

def init_centers(k):
    global data

    if data is None or len(data) == 0:
        raise ValueError("No data loaded. Use load() function first.")

    if k <= 0 or k > len(data):
        raise ValueError(f"Invalid k={k} for {len(data)} data points")

    print("\nInitialization methods:")
    print("1. Standard (Random points within data bounds)")
    print("2. Sampling (Random data points)")

    while True:
        try:
            choice = int(input("\nSelect method (1 or 2): "))
            if choice not in [1, 2]:
                print("Please enter 1 or 2")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    if choice == 1:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        centers = np.random.uniform(
            low=min_vals,
            high=max_vals,
            size=(k, data.shape[1])
        )
    else:
        indices = np.random.choice(len(data), size=k, replace=False)
        centers = data.iloc[indices].to_numpy() if isinstance(data, pd.DataFrame) else data[indices]

    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['centers'] = centers
    except NameError:
        pass

    return centers

def init_data(n_samples=100, n_features=2, n_clusters=3, cluster_std=1.0, random_state=None):
    """
    https://github.com/scikit-learn/scikit-learn/blob/6a0838c41/sklearn/datasets/_samples_generator.py#L917
    """
    global data, actual

    if random_state is not None:
        np.random.seed(random_state)

    centers = np.random.uniform(-10, 10, size=(n_clusters, n_features))

    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    current_idx = 0
    for i in range(n_clusters):
        n = samples_per_cluster + (1 if i < remainder else 0)
        end_idx = current_idx + n

        X[current_idx:end_idx] = np.random.normal(
            loc=centers[i],
            scale=cluster_std,
            size=(n, n_features)
        )
        y[current_idx:end_idx] = i
        current_idx = end_idx

    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    data = pd.DataFrame(X)
    actual = y

    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['data'] = data
            ipython.user_ns['actual'] = actual
    except NameError:
        pass

    return data
