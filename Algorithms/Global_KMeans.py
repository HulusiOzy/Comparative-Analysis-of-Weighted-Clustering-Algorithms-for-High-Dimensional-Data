import numpy as np

class GlobalKMeans:
    def __init__(self, data, k=3):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        self.data = data
        self.k = k
        self.labels_ = None
        self.centroids_ = None
        self.best_error_ = None
        
    def _calculate_error(self, data, centroids):
        expanded_data = data[:, np.newaxis, :]
        expanded_centroids = centroids[np.newaxis, :, :]
        distances = np.sum((expanded_data - expanded_centroids) ** 2, axis=2)
        return np.sum(np.min(distances, axis=1))
        
    def _run_kmeans(self, data, initial_centroids, max_iter=100):
        centroids = initial_centroids.reshape(-1, data.shape[1])
        expanded_data = data[:, np.newaxis, :]
        
        for _ in range(max_iter):
            expanded_centroids = centroids[np.newaxis, :, :]
            distances = np.sum((expanded_data - expanded_centroids) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for i in range(len(centroids)):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return labels, centroids, self._calculate_error(data, centroids)
        
    def fit(self):
        n_samples, n_features = self.data.shape
        current_centroids = np.mean(self.data, axis=0, keepdims=True)
        best_labels, _ = self._run_kmeans(self.data, current_centroids)[:2]
        
        for current_k in range(2, self.k + 1):
            best_error = np.inf
            best_new_config = None
            
            for i in range(n_samples):
                candidate_point = self.data[i].reshape(1, -1)
                candidate_centroids = np.vstack([current_centroids, candidate_point])
                labels, centroids, error = self._run_kmeans(self.data, candidate_centroids)
                
                if error < best_error:
                    best_error = error
                    best_new_config = (labels, centroids)
            
            best_labels, current_centroids = best_new_config
        
        self.labels_ = best_labels
        self.centroids_ = current_centroids
        self.best_error_ = self._calculate_error(self.data, current_centroids)
        return self.labels_
