import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations

class TableOfConfusion:
    def __init__(self, actual=None, predicted=None):
        self.actual = actual if isinstance(actual, np.ndarray) else np.asarray(actual)
        self.predicted = predicted if isinstance(predicted, np.ndarray) else np.asarray(predicted)
        self.best_matrix = None
        self.best_accuracy = None
        self.matrix = None

    def _accuracy(self, matrix):
        return np.trace(matrix) / np.sum(matrix)

    def _confusion_matrix(self, actual, predicted):
        labels = sorted(list(set(actual)))
        n_labels = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
        
        for true, pred in zip(actual, predicted):
            matrix[label_to_idx[true], label_to_idx[pred]] += 1
            
        return matrix

    def _match_values(self, actual_values, predicted_values):
        actual_unique = sorted(list(set(actual_values)))
        pred_unique = sorted(list(set(predicted_values)))
        label_map = dict(zip(pred_unique, actual_unique))
        return [label_map[pred_label] for pred_label in predicted_values]

    def _try_all_permutations(self, matrix):
        n = matrix.shape[0]
        best_accuracy = 0
        best_matrix = None
        
        for perm in permutations(range(n)):
            permuted_matrix = matrix[:, perm].astype(np.int64)
            acc = self._accuracy(permuted_matrix)
            if acc > best_accuracy:
                best_accuracy = acc
                best_matrix = permuted_matrix
        
        return best_matrix, best_accuracy

    def plot_confusion_matrix(self, title='Confusion Matrix'):
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.best_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{title}\nAccuracy: {self.best_accuracy:.2%}')
        plt.tight_layout()
        plt.show()

    def fit(self):
        predicted_values = self._match_values(self.actual, self.predicted)
        self.matrix = self._confusion_matrix(self.actual, predicted_values)
        self.best_matrix, self.best_accuracy = self._try_all_permutations(self.matrix)
        return self.best_accuracy