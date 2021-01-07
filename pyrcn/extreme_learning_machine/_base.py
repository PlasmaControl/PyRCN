"""Utilities for the extreme learning machine modules
"""
import scipy
import numpy as np

# noinspection PyProtectedMember
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import NotFittedError


def inplace_bounded_relu(X):
    """Compute the bounded rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.minimum(np.maximum(X, 0), 1, out=X)


def inplace_tanh_inverse(X):
    """Compute the tanh inverse function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.arctanh(X, out=X)


def inplace_identity_inverse(X):
    """Compute the identity inverse function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['identity'](X)


def inplace_logistic_inverse(X):
    """Compute the logistic inverse function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.negative(np.log(1 - X), out=X)


ACTIVATIONS.update({'bounded_relu': inplace_bounded_relu})

ACTIVATIONS_INVERSE = {
    'tanh': inplace_tanh_inverse,
    'identity': inplace_identity_inverse,
    'logistic': inplace_logistic_inverse
}


class InputToNode(BaseEstimator, TransformerMixin):
    """InputToNode class for ELM

    .. versionadded:: 0.00
    """

    def __init__(self,
                 hidden_layer_size=500,
                 sparsity=1.,
                 activation='tanh',
                 input_scaling=1.,
                 bias_scaling=1.,
                 random_state=None):
        self.hidden_layer_size = hidden_layer_size  # read only
        self.sparsity = sparsity  # read only
        self.activation = activation  # read/write
        self.input_scaling = input_scaling  # read/write
        self.bias_scaling = bias_scaling  # read/write
        self.random_state = check_random_state(random_state)  # read only

        self._input_weights = None
        self._bias = None
        self._hidden_layer_state = None

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, n_jobs=None):
        """
        Fit the input_weights_matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        y : Ignored
        n_jobs: Ignored

        Returns
        -------
        self : returns a trained ELM model.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)
        self._set_uniform_random_input_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size * self.sparsity).astype(int),
            random_state=self.random_state)
        self._set_uniform_random_bias(
            hidden_layer_size=self.hidden_layer_size,
            random_state=self.random_state)
        return self

    def _set_uniform_random_input_weights(self, n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
        nr_entries = np.int32(n_features_in * fan_in)
        weights_array = random_state.uniform(low=-1., high=1., size=nr_entries)

        if fan_in < hidden_layer_size:
            indices = np.zeros(shape=nr_entries, dtype=int)
            indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in, step=fan_in)

            for en in range(0, n_features_in * fan_in, fan_in):
                indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
            self._input_weights = scipy.sparse.csr_matrix(
                (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
        else:
            self._input_weights = weights_array.reshape((n_features_in, hidden_layer_size))

    def _set_uniform_random_bias(self, hidden_layer_size: int, random_state):
        self._bias = random_state.uniform(low=-1., high=1., size=hidden_layer_size)

    def transform(self, X):
        if self._input_weights is None or self._bias is None:
            raise NotFittedError(self)

        self._hidden_layer_state =\
            safe_sparse_dot(X, self._input_weights) * self.input_scaling\
            + np.ones(shape=(X.shape[0], 1)) * self._bias * self.bias_scaling
        ACTIVATIONS[self.activation](self._hidden_layer_state)
        return self._hidden_layer_state

    def _validate_hyperparameters(self):
        """
        Validate the hyperparameter. Ensure that the parameter ranges and dimensions are valid.
        Returns
        -------

        """
        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got %s." % self.hidden_layer_size)
        if self.input_scaling <= 0.:
            raise ValueError("input_scaling must be > 0, got %s." % self.input_scaling)
        if self.sparsity <= 0. or self.sparsity > 1.:
            raise ValueError("sparsity must be between 0. and 1., got %s." % self.sparsity)
        if self.bias_scaling < 0:
            raise ValueError("bias must be > 0, got %s." % self.bias_scaling)
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.activation, ACTIVATIONS))
