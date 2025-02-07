import numpy as np
import matplotlib.pyplot as plt

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
        
    def fit_transform(self, data, dimensions=2, display=True):
        X = self._validate_data(data)
        
        if dimensions < 1:
            raise ValueError(f"Min dimensions is 1")
        
        covMatrix = self.cov_Matrix(X)
        eigenValues, eigenVectors = self.eigen_Vectors(covMatrix)
        projMatrix = self.proj_Matrix(X, dimensions, eigenValues, eigenVectors)
        
        if dimensions == 2 and display:
            plt.figure(figsize=(12, 8))
            plt.scatter(projMatrix[:, 0], projMatrix[:, 1], alpha=1.0)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA Projection (2D)')
            plt.grid(True)
            plt.show()
            
        return projMatrix