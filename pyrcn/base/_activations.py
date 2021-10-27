import numpy as np
from sklearn.neural_network._base import ACTIVATIONS


def inplace_bounded_relu(X):
    """
    Compute the bounded rectified linear unit function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.minimum(np.maximum(X, 0, out=X), 1, out=X)


def inplace_tanh_inverse(X):
    """
    Compute the tanh inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.arctanh(X, out=X)


def inplace_identity_inverse(X):
    """
    Compute the identity inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['identity'](X)


def inplace_logistic_inverse(X):
    """
    Compute the logistic inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.negative(np.log(1 - X, out=X), out=X)


def inplace_relu_inverse(X):
    """
    Compute the relu inverse function inplace.

    The relu function is not invertible!
    This is an approximation assuming $x = f^{-1}(y=0) = 0$. It is valid in $x \in [0, \infty]$.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['relu'](X)


def inplace_bounded_relu_inverse(X):
    """
    Compute the bounded relu inverse function inplace.

    The bounded relu function is not invertible!
    This is an approximation assuming $x = f^{-1}(y=0) = 0$ and $x = f^{-1}(y=1) = 1$. It is valid in $x \in [0, 1]$.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['bounded_relu'](X)


ACTIVATIONS.update({'bounded_relu': inplace_bounded_relu})

ACTIVATIONS_INVERSE = {
    'tanh': inplace_tanh_inverse,
    'identity': inplace_identity_inverse,
    'logistic': inplace_logistic_inverse,
    'relu': inplace_relu_inverse,
    'bounded_relu': inplace_bounded_relu_inverse
}

ACTIVATIONS_INVERSE_BOUNDS = {
    'tanh': [-.99, .99],
    'identity': [-np.inf, np.inf],
    'logistic': [0.01, .99],
    'relu': [0, np.inf],
    'bounded_relu': [0, 1]
}