import numpy as np

class WeightedMinkowskiMetric:
    def __init__(self, p=2, weights=None):
        self.p = p
        self._p_recip = 1/p if p != float('inf') else 0
        self.weights = weights

    def calculate_distance(self, point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Points must have same dimension")
            
        if self.weights is None:
            if self.p == float('inf'):
                return max(abs(a-b) for a, b in zip(point1, point2))
            elif self.p == 1:
                return sum(abs(a - b) for a, b in zip(point1, point2))
            elif self.p == 2:
                return sum((a - b) * (a - b) for a, b in zip(point1, point2)) ** self._p_recip
            
            return sum(abs(a - b) ** self.p for a, b in zip(point1, point2)) ** self._p_recip
            
        if self.p == float('inf'):
            return max(w * abs(a-b) for w, a, b in zip(self.weights, point1, point2))
        elif self.p == 1:
            return sum(w * abs(a - b) for w, a, b in zip(self.weights, point1, point2))
        elif self.p == 2:
            return sum(w * (a - b) * (a - b) for w, a, b in zip(self.weights, point1, point2)) ** self._p_recip
            
        return sum((w**self.p) * abs(a - b)**self.p for w, a, b in zip(self.weights, point1, point2)) ** self._p_recip

    def calculate_dispersion(self, cluster_points, centroid, feature_idx):
        if len(cluster_points) == 0:
            return 0.0
        return sum(abs(point[feature_idx] - centroid[feature_idx])**self.p for point in cluster_points)

    def calculate_weights(self, cluster_points, centroid):
        if len(cluster_points) == 0:
            return np.ones(len(centroid)) / len(centroid)
            
        n_features = len(centroid)
        dispersions = [self.calculate_dispersion(cluster_points, centroid, v) for v in range(n_features)]
        
        weights = []
        for v in range(n_features):
            if dispersions[v] == 0:
                weights.append(1.0)
                continue
                
            dispersion_ratios_sum = sum(
                (dispersions[v]/dispersions[u])**(1/(self.p - 1))
                for u in range(n_features)
                if dispersions[u] != 0
            )
            
            weights.append(1/dispersion_ratios_sum if dispersion_ratios_sum != 0 else 1.0)
            
        total = sum(weights)
        return [w/total for w in weights] if total > 0 else [1.0/n_features] * n_features

    def calculate_center(self, points, weights=None):
        if len(points) == 0:
            return None
            
        if weights is None:
            return np.mean(points, axis=0)
            
        weighted_points = [[point[j] * weights[j] for j in range(len(point))] 
                         for point in points]
        
        weighted_center = np.mean(weighted_points, axis=0)
        
        return [c / w if w != 0 else c 
                for c, w in zip(weighted_center, weights)]


class MinkowskiWeightedKMeans:
    def __init__(self, data, n_clusters, p=2, max_iter=100, tol=1e-4, initial_centers=None, initial_weights=None):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        self.data = data
        self.K = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        
        self.metric = WeightedMinkowskiMetric(p=self.p)

        self.initial_centers = self._validate_centers(initial_centers) if initial_centers is not None else None
        self.initial_weights = self._validate_weights(initial_weights) if initial_weights is not None else None

        self.labels_ = None
        self.centroids_ = None
        self.weights_ = None
        self.n_iter_ = 0
        self.best_score_ = None

    def _P0(self, labels, centers, weights):
        total_error = 0
        for k in range(self.K):
            cluster_mask = (labels == k)
            cluster_points = self.data[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
                
            for point in cluster_points:
                for v in range(len(point)):
                    diff = abs(point[v] - centers[k][v]) ** self.p
                    total_error += (weights[k][v] ** self.p) * diff
                    
        return total_error

    def _initialize_centers(self):
        if self.initial_centers is not None:
            return self.initial_centers
            
        n_samples = len(self.data)
        indices = np.random.choice(n_samples, self.K, replace=False)
        return self.data[indices].copy()
    
    def _initialize_weights(self):
        if self.initial_weights is not None:
            return self.initial_weights

        n_features = self.data.shape[1]
        return np.ones((self.K, n_features)) / n_features

    def _assign_clusters(self, centers, weights):
        n_samples = len(self.data)
        distances = np.zeros((n_samples, self.K))
        
        for k in range(self.K):
            self.metric.weights = weights[k]
            for i in range(n_samples):
                distances[i, k] = self.metric.calculate_distance(self.data[i], centers[k])
                
        return np.argmin(distances, axis=1)

    def _update_centers(self, labels, weights):
        centers = np.zeros((self.K, self.data.shape[1]))
        
        for k in range(self.K):
            mask = labels == k
            if np.any(mask):
                cluster_points = self.data[mask]
                self.metric.weights = weights[k]
                centers[k] = self.metric.calculate_center(cluster_points)
                
        return centers
    
    def _update_weights(self, labels, centers):
        n_features = self.data.shape[1]
        new_weights = np.zeros((self.K, n_features))
        
        for k in range(self.K):
            mask = labels == k
            if np.any(mask):
                cluster_points = self.data[mask]
                self.metric.weights = None
                new_weights[k] = self.metric.calculate_weights(cluster_points, centers[k])
                
        return new_weights

    def fit(self, n_init=1000):
        best_error = float('inf')
        best_labels = None
        best_centers = None
        best_weights = None
        best_n_iter = 0

        actual_n_init = 1 if (self.initial_centers is not None or self.initial_weights is not None) else n_init

        for init in range(actual_n_init):
            n_samples, n_features = self.data.shape
            centers = self._initialize_centers()
            weights = self._initialize_weights()
            
            prev_labels = None
            
            for iteration in range(self.max_iter):
                labels = self._assign_clusters(centers, weights)
                
                if prev_labels is not None and np.all(labels == prev_labels):
                    break
                    
                centers = self._update_centers(labels, weights)
                weights = self._update_weights(labels, centers)
                
                prev_labels = labels.copy()
            
            current_error = self._P0(labels, centers, weights)
            
            if current_error < best_error:
                best_error = current_error
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_weights = weights.copy()
                best_n_iter = iteration + 1
        
        self.labels_ = best_labels
        self.centroids_ = best_centers
        self.weights_ = best_weights
        self.n_iter_ = best_n_iter
        self.best_score_ = best_error
        
        return self