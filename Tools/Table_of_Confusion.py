import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TableOfConfusion:
    def __init__(self, actual=None, predicted=None):
        self.actual = np.asarray(actual)
        self.predicted = np.asarray(predicted)
        self.best_matrix = None
        self.best_accuracy = None

    def _confusion_matrix(self):
        actual_unique = sorted(list(set(self.actual)))
        pred_unique = sorted(list(set(self.predicted)))
        label_map = dict(zip(pred_unique, actual_unique))
        
        mapped_pred = [label_map[p] for p in self.predicted]
        n_labels = len(actual_unique)
        label_to_idx = {label: idx for idx, label in enumerate(actual_unique)}
        
        matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
        for true, pred in zip(self.actual, mapped_pred):
            matrix[label_to_idx[true], label_to_idx[pred]] += 1
            
        return matrix

    def _mark_matrix(self, matrix):
        mask_matrix = matrix.copy()
        row_mask = np.zeros(matrix.shape[0], dtype=bool)
        col_mask = np.zeros(matrix.shape[0], dtype=bool)
        
        while True:
            zero_pos = np.where(mask_matrix == 0)
            if len(zero_pos[0]) == 0:
                break
                
            row, col = zero_pos[0][0], zero_pos[1][0]
            mask_matrix[row, :] = 1
            mask_matrix[:, col] = 1
            mask_matrix[row, col] = 2
            row_mask[row] = True
            col_mask[col] = True
            
        return np.where(mask_matrix == 2), row_mask, col_mask

    def _hungarian_algorithm(self, matrix):
        cost_matrix = -matrix.copy()
        n = cost_matrix.shape[0]
        
        cost_matrix -= cost_matrix.min(axis=1)[:, np.newaxis]
        
        cost_matrix -= cost_matrix.min(axis=0)
        
        while True:
            marked_zeros, row_mask, col_mask = self._mark_matrix(cost_matrix)
            if len(marked_zeros[0]) >= n:
                assignment = np.zeros(n, dtype=int)
                assignment[marked_zeros[0]] = marked_zeros[1]
                return matrix[:, assignment]
            
            min_val = cost_matrix[~row_mask][:, ~col_mask].min()
            
            cost_matrix[~row_mask] -= min_val
            
            cost_matrix[:, col_mask] += min_val

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
        matrix = self._confusion_matrix()
        self.best_matrix = self._hungarian_algorithm(matrix)
        self.best_accuracy = np.trace(self.best_matrix) / np.sum(self.best_matrix)
        return self.best_accuracy
