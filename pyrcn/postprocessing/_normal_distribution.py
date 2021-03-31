import scipy
import scipy.stats
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, PatchExtractor


class NormalDistribution(BaseEstimator, TransformerMixin):
    def __init__(self, size=1):
        self._transformer = scipy.stats.norm
        self._mean = 0
        self._std = 0
        self._size = size

    def fit(self, X, y=None):
        self._mean, self._std = self._transformer.fit(X)
        return self

    def transform(self, X, y=None):
        return self._transformer.rvs(loc=self._mean, scale=self._std, size=self._size)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X)
        return self.transform(X=X)
