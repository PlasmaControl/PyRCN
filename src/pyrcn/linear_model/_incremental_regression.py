"""The :mod:`incremental_regression` contains IncrementalRegression."""

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import validate_data


class IncrementalRegression(RegressorMixin, BaseEstimator):
    """
    Linear regression.

    This linear regression algorithm is able to perform a linear regression
    with the L2 regularization and iterative fit. [1]
    .. [1] https://ieeexplore.ieee.org/document/4012031

    References
    ----------
    N. Liang, G. Huang, P. Saratchandran and N. Sundararajan,
    "A Fast and Accurate Online Sequential Learning Algorithm for Feedforward
    Networks," in IEEE Transactions on Neural Networks, vol. 17, no. 6,
    pp. 1411-1423, Nov. 2006, doi: 10.1109/TNN.2006.880583.

    Parameters
    ----------
    alpha : float, default=1e-5
        L2 regularization parameter
    fit_intercept : bool, default=True
        Fits a constant offset if True. Use this if input values are not
        average free.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    """

    def __init__(self, *, alpha: float = 1e-5,
                 fit_intercept: bool = True):
        """Construct the IncrementalRegression."""
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self._K: np.ndarray = np.ndarray([])
        self._xTy: np.ndarray = np.ndarray([])
        self._output_weights: np.ndarray = np.ndarray([])

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    reset: bool = False, validate: bool = True,
                    postpone_inverse: bool = False) -> IncrementalRegression:
        """
        Fit the regressor partially.

        Parameters
        ----------
        X : ndarray of shape (samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        reset : bool, default=False
            Begin a new fit, drop prior fits.
        validate: bool, default=True
            Validate input data if True.
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation.

        Returns
        -------
        self : returns a partially fitted IncrementalRegression model
        """
        if validate:
            validate_data(self, X, y, multi_output=True)

        X_preprocessed = self._preprocessing(X)

        if reset:
            self._K = np.ndarray([])
            self._xTy = np.ndarray([])
            self._output_weights = np.ndarray([])

        if self._K.shape == ():
            self._K = safe_sparse_dot(X_preprocessed.T, X_preprocessed)
        else:
            self._K += safe_sparse_dot(X_preprocessed.T, X_preprocessed)

        if self._xTy.shape == ():
            self._xTy = safe_sparse_dot(X_preprocessed.T, y)
        else:
            self._xTy += safe_sparse_dot(X_preprocessed.T, y)

        # can only be postponed if output weights have not been initialized yet
        if postpone_inverse and self._output_weights.shape == ():
            return self

        A = self._K + self.alpha * np.identity(X_preprocessed.shape[1])

        if self._output_weights.shape == ():
            self._output_weights = np.linalg.solve(
                A, cast(np.ndarray, self._xTy))
        else:
            self._output_weights += np.linalg.solve(
                A, safe_sparse_dot(X_preprocessed.T, (
                        y - safe_sparse_dot(X_preprocessed,
                                            self._output_weights))))
            # self._output_weights += np.linalg.solve(
            #     A, self._xTy - np.matmul(self._K, self._output_weights))
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> IncrementalRegression:
        """
        Fit the regressor.

        Parameters
        ----------
        X : ndarray of shape (samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        reset : bool, default=False
            Begin a new fit, drop prior fits.
        validate: bool, default=True
            Validate input data if True.
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation.

        Returns
        -------
        self : returns a fitted IncrementalRegression model
        """
        self.partial_fit(X, y, reset=True, validate=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output y according to input X.

        Parameters
        ----------
        X : ndarray of shape (samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        """
        if self._output_weights.shape == ():
            raise NotFittedError(self)

        return safe_sparse_dot(self._preprocessing(X),
                               self._output_weights)

    def _preprocessing(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing on the input data X.

        Parameters
        ----------
        X : ndarray of shape (samples, n_features)

        Returns
        -------
        X_preprocessed : ndarray of shape (n_samples, n_features)
        or (n_samples, n_features+1)
        """
        X_preprocessed = X

        if self.fit_intercept:
            X_preprocessed = np.hstack(
                (X_preprocessed, np.ones(shape=(X.shape[0], 1))))

        return X_preprocessed

    def __sizeof__(self) -> int:
        """
        Return the size of the object in bytes.

        Returns
        -------
        size : int
            Object memory in bytes.
        """
        return object.__sizeof__(self) + self._K.nbytes + self._xTy.nbytes + \
            self._output_weights.nbytes

    @property
    def coef_(self) -> np.ndarray | None:
        """
        Return the output weights without intercept.

        Compatibility to ```sklearn.linear_model.Ridge```.

        Returns
        -------
        coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
            Weight vector(s).
        """
        if self._output_weights.shape != ():
            if self.fit_intercept:
                if self._output_weights.ndim == 2:
                    return np.transpose(self._output_weights)[:, :-1]
                else:
                    return self._output_weights[:-1]
            else:
                return np.transpose(self._output_weights)
        return self._output_weights

    @property
    def intercept_(self) -> np.ndarray:
        """
        Return the intercept of output output weights.

        Compatibility to ```sklearn.linear_model.Ridge```.

        Returns
        -------
        intercept_ : Union[float, np.ndarray]
            Independent term in decision function.
        """
        if self._output_weights.shape != () and self.fit_intercept:
            return np.transpose(self._output_weights)[:, -1]
        return np.array([])
