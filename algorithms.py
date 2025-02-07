import numpy as np
import functions

class BaseClusteringAlgorithm:
    def __init__(self, k=None, **kwargs):
        if k is None:
            raise ValueError("Number of clusters (k) must be provided")
        self.k = k
        self.labels_ = None
        self.centroids_ = None
        self.weights_ = None
        self.extra_params = kwargs

    def _update_global_state(self):
        functions.predicted = self.labels_
        functions.centers = self.centroids_
        functions.weights = self.weights_

    def _update_ipython_namespace(self):
        try:
            ipython = get_ipython()
            if ipython is not None:
                ipython.user_ns['predicted'] = functions.predicted
                ipython.user_ns['centers'] = functions.centers
                ipython.user_ns['weights'] = functions.weights
        except NameError:
            pass

    def fit(self):
        if functions.data is None or len(functions.data) == 0:
            raise ValueError("No data loaded. Use load() function first.")
            
        data_array = functions.data.to_numpy() if not isinstance(functions.data, np.ndarray) else functions.data
        
        self._fit_algorithm(data_array)
        self._update_global_state()
        self._update_ipython_namespace()
            
        return self.labels_

    def _fit_algorithm(self, data_array):
        raise NotImplementedError("Subclasses must implement _fit_algorithm method")

class KMeans(BaseClusteringAlgorithm):
    def _fit_algorithm(self, data_array):
        from Algorithms.KMeans import KMeans as KMeansImpl
        model = KMeansImpl(data=data_array, k=self.k)
        self.labels_ = model.fit()
        self.centroids_ = model.centroids_

class WeightedKMeans(BaseClusteringAlgorithm):
    def __init__(self, k=None, beta=None, **kwargs):
        super().__init__(k, **kwargs)
        if beta is None:
            raise ValueError("beta parameter is required for Weighted K-Means")
        self.beta = beta

    def _fit_algorithm(self, data_array):
        from Algorithms.Weighted_KMeans import WKMeans
        model = WKMeans(data=data_array, k=self.k, beta=self.beta)
        self.labels_ = model.fit()
        self.weights_ = model.weights_
        self.centroids_ = model.centroids_

class SubspaceWeightedKMeans(BaseClusteringAlgorithm):
    def __init__(self, k=None, beta=None, **kwargs):
        super().__init__(k, **kwargs)
        if beta is None:
            raise ValueError("beta parameter is required for Subspace Weighted K-Means")
        self.beta = beta

    def _fit_algorithm(self, data_array):
        from Algorithms.SubSpace_Weighted_KMeans import SWKMeans
        model = SWKMeans(data=data_array, k=self.k, beta=self.beta)
        self.labels_ = model.fit()
        self.weights_ = model.weights_
        self.centroids_ = model.centroids_

class MinkowskiWeightedKMeans(BaseClusteringAlgorithm):
    def __init__(self, k=None, p=None, **kwargs):
        super().__init__(k, **kwargs)
        if p is None:
            raise ValueError("p parameter is required for Minkowski Weighted K-Means")
        self.p = p

    def _fit_algorithm(self, data_array):
        from Algorithms.Minkowski_Weighted_KMeans import MinkowskiWeightedKMeans as MWKMeans
        model = MWKMeans(data=data_array, n_clusters=self.k, p=self.p)
        model.fit()
        self.labels_ = model.labels_
        self.weights_ = model.weights_
        self.centroids_ = model.centroids_

class UPGMA(BaseClusteringAlgorithm):
    def _fit_algorithm(self, data_array):
        from Algorithms.UPGMA import UPGMA as UPGMAImpl
        model = UPGMAImpl(data=data_array, n_clusters=self.k)
        self.labels_ = model.fit()
        self.centroids_ = model.centroids_

class Wards(BaseClusteringAlgorithm):
    def _fit_algorithm(self, data_array):
        from Algorithms.Wards_Method import Wards as WardsImpl
        model = WardsImpl(data=data_array, k=self.k)
        self.labels_, self.centroids_ = model.fit()

class IterativeAnomalousPattern(BaseClusteringAlgorithm):
    def _fit_algorithm(self, data_array):
        from Algorithms.IterativeAnomalousPattern import IterativeAnomalousPattern as IAP
        model = IAP(data=data_array, k=self.k)
        self.labels_, self.centroids_ = model.find_centroids()
        
def kmeans(**kwargs): return KMeans(**kwargs).fit()
def wkmeans(**kwargs): return WeightedKMeans(**kwargs).fit()
def swkmeans(**kwargs): return SubspaceWeightedKMeans(**kwargs).fit()
def mwkmeans(**kwargs): return MinkowskiWeightedKMeans(**kwargs).fit()
def upgma(**kwargs): return UPGMA(**kwargs).fit()
def wards(**kwargs): return Wards(**kwargs).fit()
def iap(**kwargs): return IterativeAnomalousPattern(**kwargs).fit()