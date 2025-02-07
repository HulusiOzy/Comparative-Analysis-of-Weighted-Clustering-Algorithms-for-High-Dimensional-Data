import pandas as pd
import numpy as np
import os

from Tools.PreProcessing import Preprocessing
from Tools.Table_of_Confusion import TableOfConfusion
from Tools.Silhouette_Index import SilhouetteIndex
from Tools.ARI import ARI
from Tools.PCA import PCA

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
    'iap': {
        'description': 'Iterative Anomalous Pattern',
        'parameters': {
            'required': ['k'],
        }
    },
    'wkmeans': {
        'description': 'Weighted K-Means',
        'parameters': {
            'required': ['k', 'beta'],
            'optional': ['initial_centers']
        }
    },
    'swkmeans': {
        'description': 'Subspace Weighted K-Means',
        'parameters': {
            'required': ['k', 'beta'],
            'optional': ['initial_centers']
        }
    },
    'mwkmeans': {
        'description': 'Minkowski Weighted K-Means',
        'parameters': {
            'required': ['k', 'p'],
            'optional': ['initial_centers']
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

def load(filename, preprocess=0):
    global data, actual

    dataset_path = os.path.join('Datasets', filename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File '{filename}' not found in Datasets folder")
    
    if preprocess:
        preprocessor = Preprocessing(input_file=dataset_path, target_column=None, column_style='N')
        processed_path, actual_path = preprocessor.fit()
        data = pd.read_csv(processed_path, header=None)
        actual = pd.read_csv(actual_path, header=None).iloc[:, 0].values
        os.remove(processed_path)
        os.remove(actual_path)
    else:
        data = pd.read_csv(dataset_path)
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

def pca(dimensions=2, display=1, data=None):
    input_data = data if data is not None else globals()['data']
    
    pca_transformer = PCA()
    transformed_data = pca_transformer.fit_transform(input_data, dimensions=dimensions, display=display)
    
    return transformed_data

def silhouette_index(display=1, data=None, predicted=None):
    input_data = data if data is not None else globals()['data']
    predicted_labels = predicted if predicted is not None else globals()['predicted']
    
    si_calculator = SilhouetteIndex(input_data, predicted_labels)
    silhouetteIndex = float(si_calculator.fit())
    
    if display:
        print(f"\nSilhouette Index: {silhouetteIndex:.4f}")
    
    return silhouetteIndex  

def init_weights(): #Only for 1/(qi^2)
    global data
    
    if data is None or len(data) == 0:
        raise ValueError("No data loaded. Use load() function first.")
        
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1e-10 #So no division by zero
    weights = 1 / (stds * stds)
    weights = weights / np.sum(weights) #Normaize
    
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns['weights'] = weights
    except NameError:
        pass
        
    return weights