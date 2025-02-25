import numpy as np
import matplotlib.pyplot as plt

class CHIndex:
    def __init__(self, data=None, predicted=None):
        self.data = data if isinstance(data, np.ndarray) else np.asarray(data)
        self.predicted = predicted if isinstance(predicted, np.ndarray) else np.asarray(predicted)
        self.score = None
        self.bcss = None
        self.wcss = None
        
    def _calculate_bcss(self):
        n_clusters = len(np.unique(self.predicted))
        overall_mean = np.mean(self.data, axis=0)
        bcss = 0
        
        for k in range(n_clusters):
            cluster_mask = (self.predicted == k)
            n_points = np.sum(cluster_mask)
            
            if n_points > 0:
                cluster_mean = np.mean(self.data[cluster_mask], axis=0)
                squared_dist = np.sum((cluster_mean - overall_mean) ** 2)
                bcss += n_points * squared_dist
                
        return bcss
        
    def _calculate_wcss(self):
        n_clusters = len(np.unique(self.predicted))
        wcss = 0
        
        for k in range(n_clusters):
            cluster_mask = (self.predicted == k)
            cluster_points = self.data[cluster_mask]
            
            if len(cluster_points) > 0:
                cluster_mean = np.mean(cluster_points, axis=0)
                squared_dists = np.sum((cluster_points - cluster_mean) ** 2, axis=1)
                wcss += np.sum(squared_dists)
                
        return wcss
        
    def plot(self, title=None):
        if self.score is None:
            raise ValueError("Must call fit() before plotting")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = ['Between-cluster\nvariance', 'Within-cluster\nvariance']
        values = [self.bcss, self.wcss]
        
        bars = ax.bar(components, values)
        ax.set_ylabel('Sum of Squares')
        
        if title:
            ax.set_title(f'{title}\nCH Index: {self.score:.2f}')
        else:
            ax.set_title(f'Cluster Quality Components\nCH Index: {self.score:.2f}')
            
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()
        
    def fit(self):
        n_samples = len(self.data)
        n_clusters = len(np.unique(self.predicted))
        
        if n_clusters == 1 or n_clusters == n_samples:
            return 0.0
            
        self.bcss = self._calculate_bcss()
        self.wcss = self._calculate_wcss()
        
        df_between = n_clusters - 1
        df_within = n_samples - n_clusters
        
        self.score = (self.bcss / df_between) / (self.wcss / df_within)
        return self.score
