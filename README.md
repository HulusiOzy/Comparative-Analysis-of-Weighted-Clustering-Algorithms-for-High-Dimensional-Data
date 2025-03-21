# Clustering Framework

A comprehensive Python framework for clustering analysis with various algorithms, metrics, and evaluation tools.

## Overview

This framework provides a collection of clustering algorithms, preprocessing tools, and evaluation metrics to support data analysis tasks. It's designed to be used interactively in IPython, making it ideal for exploratory data analysis and educational purposes.

## System Requirements

- Python 3.13.2
- IPython 8.32.0
- NumPy 2.2.3
- Pandas 2.2.3
- Matplotlib 3.10.0
- Seaborn 0.13.2

## Project Structure

```
.
├── Algorithms/
│   ├── IterativeAnomalousPattern.py
│   ├── KMeans.py
│   ├── Minkowski_Weighted_KMeans.py
│   ├── SubSpace_Weighted_KMeans.py
│   ├── UPGMA.py
│   ├── Wards_Method.py
│   └── Weighted_KMeans.py
├── Tools/
│   ├── PCA.py
│   ├── PreProcessing.py
│   ├── Silhouette_Index.py
│   └── Table_of_Confusion.py
├── Metrics/
│   └── MinkowskiMetric.py
├── Datasets/
│   ├── Depression Student Dataset.csv
│   ├── Student Depression Dataset.csv
│   ├── StudentPerformanceFactors.csv
│   ├── iris.data
│   ├── search_engine_data.csv
│   ├── search_engine_data_2.csv
│   └── Iris/
│       ├── bezdekIris.data
│       ├── iris.data
│       ├── iris.names
│       ├── iris.zip
│       ├── Index
│       └── PreProcessing.py
├── functions.py
├── algorithms.py
└── project_main.py
```

## Getting Started

1. Start IPython in the project directory:
   ```
   $ ipython
   ```

2. Import the project modules:
   ```python
   from project_main import *
   ```

3. List available datasets, algorithms, and tools:
   ```python
   list_datasets()
   list_algorithms()
   list_tools()
   ```

4. Load a dataset:
   ```python
   load('iris.data', preprocess=1)  # Set preprocess=1 for automatic data preprocessing
   ```

5. Run a clustering algorithm:
   ```python
   kmeans(k=3)  # Basic K-means with 3 clusters
   ```

6. Evaluate the results:
   ```python
   silhouette_index()  # Calculate and display Silhouette Index
   confusion_matrix(display=1)  # Display confusion matrix if actual labels exist
   ```

## Core Functions

### Data Management

- `load(filename, preprocess=0)` - Load a dataset from the Datasets folder
- `save(filename)` - Save current data and results
- `drop(n)` - Remove a column from the loaded dataset
- `init_data(n_samples, n_features, n_clusters, cluster_std)` - Generate synthetic clustering data

### Clustering Algorithms

- `kmeans(k, initial_centers=None)` - Standard K-means algorithm
- `wkmeans(k, beta, initial_centers=None, initial_weights=None)` - Weighted K-means
- `swkmeans(k, beta, initial_centers=None, initial_weights=None)` - Subspace Weighted K-means
- `mwkmeans(k, p, initial_centers=None, initial_weights=None)` - Minkowski Weighted K-means
- `upgma(k)` - Unweighted Pair Group Method with Arithmetic Mean
- `wards(k)` - Ward's Hierarchical Clustering
- `iap(k)` - Iterative Anomalous Pattern detection

### Analysis Tools

- `confusion_matrix(display=0)` - Calculate confusion matrix between actual and predicted labels
- `silhouette_index(display=1)` - Calculate Silhouette Index for clustering quality
- `ch_index(display=1)` - Calculate Calinski-Harabasz Index
- `ari(display=0)` - Calculate Adjusted Rand Index
- `pca(dimensions=2, display=1)` - Perform PCA dimensionality reduction

### Initialization

- `init_weights()` - Initialize feature weights using various methods
- `init_centers(k)` - Initialize cluster centers

## Global Variables

The framework maintains several global variables for easy access:

- `data` - Current dataset in use
- `actual` - Actual labels (if available)
- `predicted` - Predicted cluster labels from the most recent run
- `centers` - Cluster centroids from the most recent run
- `weights` - Feature weights from the most recent run

## Example Workflow

```python
# Load and preprocess a dataset
load('Depression Student Dataset.csv', preprocess=1)

# Visualize the data with PCA
pca(dimensions=2, display=1, highlight_actual=True)

# Initialize cluster centers
centers = init_centers(k=3)

# Run K-means clustering
kmeans(k=3, initial_centers=centers)

# Evaluate results
silhouette_index()
ch_index(display=1)
confusion_matrix(display=1)
```

## Feature Engineering Examples

```python
# Automatic feature selection by impact on clustering quality
load('Depression Student Dataset.csv', preprocess=1)
feature_removal()

# Weighted clustering with custom feature importance
load('iris.data', preprocess=1)
weights = init_weights()  # Choose method interactively
wkmeans(k=3, beta=2, initial_weights=weights)
```

## Advanced Features

### Feature Selection

The framework includes methods for automatic feature selection:

```python
# Remove features one by one and observe impact on cluster quality
feature_removal()

# Alternative approach that optimizes for Silhouette Index improvement
feature_removal_two()
```

### Dimensionality Reduction and Visualization

```python
# Apply PCA and visualize results
pca(dimensions=2, display=1)

# Visualize with class labels (if available)
pca(dimensions=2, display=1, highlight_actual=True)
```

### Working with Different Datasets

```python
# Using iris dataset
load('iris.data', preprocess=1)

# Or using the Bezdek version from the Iris subfolder
load('Iris/bezdekIris.data', preprocess=1)

# Working with the depression datasets
load('Depression Student Dataset.csv', preprocess=1)
load('Student Depression Dataset.csv', preprocess=1)
```

## Help

For a complete list of available functions and their parameters:

```python
help()
```

## Notes

- When using `preprocess=1` with the `load()` function, the framework will automatically normalize and standardize the data.
- The `display` parameter in evaluation functions toggles visualization output.
- For reproducible results, you can initialize centers and weights before running algorithms.
- The framework automatically updates global variables after each algorithm run, making it easy to chain operations.
