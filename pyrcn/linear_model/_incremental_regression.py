"""
Framework for incremental regression.
"""

# Author: Michael Schindler <michael.schindler@maschindler.de>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state, check_X_y, column_or_1d, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class IncrementalRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scaler = StandardScaler(copy=False)

        self._K = None
        self._P = None
        self._output_weights = None

    def fit(self, X, y, n_jobs=1, reset=False):
        if reset:
            self._P = None

        self.partial_fit(self._preprocessing(X), y, n_jobs)

    def predict(self, X):
        if self._output_weights is None:
            raise NotFittedError(self)

        return safe_sparse_dot(self._preprocessing(X), self._output_weights)

    def partial_fit(self, X, y, n_jobs=1, reset=False):
        X_preprocessed = self._preprocessing(X)

        if reset:
            self._K = None
            self._P = None

        if self._K is None:
            self._K = safe_sparse_dot(X_preprocessed.T, X_preprocessed)
        else:
            self._K += safe_sparse_dot(X_preprocessed.T, X_preprocessed)

        self._P = np.linalg.inv(self._K + self.alpha**2 * np.identity(X_preprocessed.shape[1]))

        if self._output_weights is None:
            self._output_weights = np.matmul(self._P, safe_sparse_dot(X_preprocessed.T, y))
        else:
            self._output_weights += np.matmul(self._P, safe_sparse_dot(X_preprocessed.T, (y - safe_sparse_dot(X_preprocessed, self._output_weights))))

    def _preprocessing(self, X):
        X_preprocessed = X

        if self.fit_intercept:
            X_preprocessed = np.hstack((X_preprocessed, np.ones(shape=(X.shape[0], 1))))

        if self.normalize:
            self.scaler.fit_transform(X_preprocessed)

        return X_preprocessed
