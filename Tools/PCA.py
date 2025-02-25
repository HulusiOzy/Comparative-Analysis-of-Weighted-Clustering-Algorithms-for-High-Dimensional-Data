import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functions

class PCA:
    def __init__(self, standardize=True):
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def _validate_data(self, data):
        if data is None:
            raise ValueError("Input data cant be None")
            
        return data if isinstance(data, np.ndarray) else np.asarray(data)
    
    def _generate_color_palette(self, n_colors):
        return sns.husl_palette(n_colors)
        
    def cov_Matrix(self, data):
        X = self._validate_data(data)
        n = X.shape[0]
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n - 1)
        return covariance_matrix
        
    def eigen_Vectors(self, covMatrix):
        eigenvalues, eigenvectors = np.linalg.eigh(covMatrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return eigenvalues, eigenvectors
        
    def proj_Matrix(self, data, dimensions, eigenValues, eigenVectors):
        if dimensions > len(eigenValues):
            raise ValueError(f"Max dimensions is {eigenValues}")
        
        X = self._validate_data(data)
        X_centered = X - self.mean
        selected_vectors = eigenVectors[:, :dimensions]
        projected_data = np.dot(X_centered, selected_vectors)
        projected_data[:, 0] *= -1
        
        return projected_data
    
    def _create_plot(self, projMatrix, highlight_actual=False):
        plt.figure(figsize=(12, 8))
        
        if highlight_actual and functions.actual is not None:
            unique_labels = np.unique(functions.actual)
            colors = self._generate_color_palette(len(unique_labels))
            
            for idx, label in enumerate(unique_labels):
                mask = functions.actual == label
                plt.scatter(projMatrix[mask, 0], 
                          projMatrix[mask, 1],
                          c=[colors[idx]],
                          label=f'Class {label}',
                          alpha=0.7)
            plt.legend()
        else:
            plt.scatter(projMatrix[:, 0], projMatrix[:, 1], alpha=0.7)
            
        var_explained = self.explained_variance_ratio[:2] * 100
        
        plt.xlabel(f'First Principal Component ({var_explained[0]:.1f}% variance)')
        plt.ylabel(f'Second Principal Component ({var_explained[1]:.1f}% variance)')
        plt.title('PCA Projection with Class Labels' if highlight_actual else 'PCA Projection')
        plt.grid(True)
        
    def fit_transform(self, data, dimensions=2, display=True, highlight_actual=False):
        X = self._validate_data(data)
        
        if dimensions < 1:
            raise ValueError(f"Min dimensions is 1")
        
        covMatrix = self.cov_Matrix(X)
        eigenValues, eigenVectors = self.eigen_Vectors(covMatrix)
        projMatrix = self.proj_Matrix(X, dimensions, eigenValues, eigenVectors)
        
        if dimensions == 2 and display:
            self._create_plot(projMatrix, highlight_actual)
            plt.show()
            
        return projMatrix

