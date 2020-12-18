from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.fixes import sparse_lsqr
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model._base import _rescale_data, _preprocess_data

import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
from scipy.special import expit
from joblib import Parallel, delayed


class LinearRegression(RegressorMixin, MultiOutputMixin, LinearModel):
    """

    """
    def __init__(self, normalize=False, copy_X=True, n_jobs=1):
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.xTx_ = None
        self.xTy_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=True)
        n_targets = y.shape[1]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        _xTx = np.dot(X.T, X)
        inv_xTx = np.linalg.inv(_xTx)
        _xTy = np.dot(X.T, y)

        if y.ndim < 2:
            self.coef_ = np.dot(inv_xTx, self._xTy)
        else:
            if self.n_jobs is not None:
                self.coef_ = Parallel(n_jobs=self.n_jobs)(delayed(np.dot)(inv_xTx, _xTy[:, n]) for n in range(n_targets))
            else:
                self.coef_ = np.dot(inv_xTx, _xTy)
        return self

    def partial_fit(self, X, y, sample_weight=None, update_output_weights=True):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        update_output_weights : bool
            Whether to compute output weights after a run on partial_fit

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if update_output_weights:
            _xTx = np.dot(X.T, X)
            inv_xTx = np.linalg.inv(_xTx)
            _xTy = np.dot(X.T, y)
            if y.ndim < 2:
                self.coef_ = np.dot(inv_xTx, self._xTy)
            else:
                if self.n_jobs is not None:
                    self.coef_ = Parallel(n_jobs=self.n_jobs)(
                        delayed(np.dot)(inv_xTx, self._xTy[:, n]) for n in range(self.n_outputs_))
                else:
                    self.coef_ = np.dot(inv_xTx, self._xTy)
        if self.xTx_ is not None:
            self.xTx_ = self.xTx_ + np.dot(X.T, X)
            self.xTy_ = self.xTy_ + np.dot(X.T, y)
        else:
            self.xTx_ = np.dot(X.T, X)
            self.xTy_ = np.dot(X.T, y)

    def finalize(self):
        """
        Finalize the training by solving the linear regression problem and set xTx and xTy to None.

        Returns
        -------

        """
        self._finalize()

    def _finalize(self):
        """
        This finalizes the training of a model. No more required attributes, such as activations, xTx, xTy will be
        removed.

        Warnings : The model cannot be improved with additional data afterwards!!!

        Parameters
        ----------
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------

        """
        if self.output_weights_ is None:
            self._compute_output_weights(n_jobs=n_jobs)

        self._xTx = None
        self._xTy = None
        self._activations_mean = None
        self._activations_var = None
        self.is_fitted_ = True


class RidgeRegression(LinearRegression):
    """

    """
