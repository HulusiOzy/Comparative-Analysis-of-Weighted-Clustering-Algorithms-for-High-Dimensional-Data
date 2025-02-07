import numpy as np

class SilhouetteIndex:
    def __init__(self, data=None, predicted=None):
        self.data = data if isinstance(data, np.ndarray) else np.asarray(data)
        self.predicted = predicted if isinstance(predicted, np.ndarray) else np.asarray(predicted)
        self.overall_index = None
        self.si_values = None

    def _silhouette_index(self, ai_values, bi_values):
        max_values = np.maximum(ai_values, bi_values)
        zero_mask = max_values == 0
        si_values = np.zeros_like(ai_values)
        non_zero_mask = ~zero_mask
        si_values[non_zero_mask] = (bi_values[non_zero_mask] - ai_values[non_zero_mask]) / max_values[non_zero_mask]
        si_values[zero_mask] = 0
        
        return np.mean(si_values), si_values

    def _internal_distance(self, data, labels):
        n_samples = len(data)
        expanded_data = np.expand_dims(data, axis=1)
        broadcasted_data = np.broadcast_to(expanded_data, (n_samples, n_samples, data.shape[1]))
        distances = np.sqrt(np.sum((broadcasted_data - data)**2, axis=2))
        label_matches = (labels.reshape(-1, 1) == labels)
        np.fill_diagonal(label_matches, False)
        counts = np.sum(label_matches, axis=1)
        masked_distances = np.where(label_matches, distances, 0)
        ai_values = np.divide(np.sum(masked_distances, axis=1), counts, 
                            where=counts > 0, out=np.zeros_like(counts, dtype=float))
        
        return ai_values

    def _nearest_neighbor_distance(self, data, labels):
        n_samples = len(data)
        unique_labels = np.unique(labels)
        expanded_data = np.expand_dims(data, axis=1)
        broadcasted_data = np.broadcast_to(expanded_data, (n_samples, n_samples, data.shape[1]))
        
        distances = np.sqrt(np.sum((broadcasted_data - data)**2, axis=2))
        bi_values = np.inf * np.ones(n_samples)
        
        for label in unique_labels:
            other_cluster_mask = (labels == label)
            current_mask = (labels != label)
            if np.any(other_cluster_mask):
                cluster_distances = distances[:, other_cluster_mask]
                mean_distances = np.mean(cluster_distances, axis=1)
                bi_values[current_mask] = np.minimum(bi_values[current_mask], mean_distances[current_mask])
        
        return bi_values

    def fit(self):
        ai = self._internal_distance(self.data, self.predicted)
        bi = self._nearest_neighbor_distance(self.data, self.predicted)
        self.overall_index, self.si_values = self._silhouette_index(ai, bi)
        return self.overall_index