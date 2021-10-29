from typing import Union, Dict
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
    """
    Transform an input distribution to a normal distribution.

    Parameters
    ----------
    size : Union[int, np.integer], default=1
        Defining number of random variates
    """
    def __init__(self, size: Union[int, np.integer] = 1):
        self._transformer = scipy.stats.norm
        self._mean = 0
        self._std = 0
        self._size = size

    def fit(self, X: np.ndarray, y: None=None) -> TransformerMixin:
        """
        Fit the NormalDistribution

        Parameters
        ----------
        X : np.ndarray of shape(n_samples, n_features)
            The input features
        y : None
            ignored

        Returns
        -------
        self : returns a trained NormalDistribution.
        """
        self._mean, self._std = self._transformer.fit(X)
        return self

    def transform(self, X: np.ndarray, y: None=None) -> np.ndarray:
        """
        Transforms the input matrix X.

        Parameters
        ----------
        X : ndarray of size (n_samples, n_features)

        Returns
        -------
        y: ndarray of size (n_samples, )
        """
        return self._transformer.rvs(loc=self._mean, scale=self._std, size=self._size)

    def fit_transform(self, X: np.ndarray, y: None=None, 
                      **fit_params: Union[Dict, None]) -> np.ndarray:
        """
        Fit the Estimator and transforms the input matrix X.

        Parameters
        ----------
        X : ndarray of size (n_samples, n_features)
        y : None
            ignored
        **fit_params : Union[Dict, None]
            ignored            

        Returns
        -------
        y: ndarray of size (n_samples, )
        """
        self.fit(X=X)
        return self.transform(X=X)
