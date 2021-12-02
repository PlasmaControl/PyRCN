"""The :mod:`activations` contains various activation functions for PyRCN."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import numpy as np
from sklearn.neural_network._base import ACTIVATIONS
from typing import Dict, Callable


def inplace_softplus(X: np.ndarray) -> None:
    """
    Compute the softplux activation function inplace:
    .. math::
        f(x) = \\mathrm{ln}(1 + e^{x})

    Parameters
    ----------
    X : numpy.ndarray

    beta : float, default=1
        Scaling factor
    """
    np.log(1 + np.exp(X, out=X), out=X)


def inplace_softmax(X: np.ndarray, beta: float = 1) -> None:
    """
    Compute the softmax activation function inplace:
    .. math::
        y_k = \\frac{e^{x_k}}{\\sum_{i=1}^{n} e^{x_i}}

    Parameters
    ----------
    X : numpy.ndarray

    beta : float, default=1
        Scaling factor
    """
    denominator = np.sum(np.exp(beta*X))
    np.divide(np.exp(beta*X, out=X), denominator, out=X)


def inplace_bounded_relu(X: np.ndarray) -> None:
    """
    Compute the bounded rectified linear unit function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    np.minimum(np.maximum(X, 0, out=X), 1, out=X)


def inplace_tanh_inverse(X: np.ndarray) -> None:
    """
    Compute the tanh inverse function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    np.arctanh(X, out=X)


def inplace_identity_inverse(X: np.ndarray) -> None:
    """
    Compute the identity inverse function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    ACTIVATIONS['identity'](X)


def inplace_logistic_inverse(X: np.ndarray) -> None:
    """
    Compute the logistic inverse function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    np.negative(np.log(1 - X, out=X), out=X)


def inplace_relu_inverse(X: np.ndarray) -> None:
    r"""
    Compute the relu inverse function inplace.

    The relu function is not invertible!
    This is an approximation assuming $x = f^{-1}(y=0) = 0$.
    It is valid in $x \in [0, \infty]$.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    ACTIVATIONS['relu'](X)


def inplace_bounded_relu_inverse(X: np.ndarray) -> None:
    r"""
    Compute the bounded relu inverse function inplace.

    The bounded relu function is not invertible!
    This is an approximation assuming
    $x = f^{-1}(y=0) = 0$ and $x = f^{-1}(y=1) = 1$.
    It is valid in $x \in [0, 1]$.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    ACTIVATIONS['bounded_relu'](X)


ACTIVATIONS.update({'bounded_relu': inplace_bounded_relu,
                    'softmax': inplace_softmax,
                    'softplus': inplace_softplus})

ACTIVATIONS_INVERSE: Dict[str, Callable] = {
    'tanh': inplace_tanh_inverse,
    'identity': inplace_identity_inverse,
    'logistic': inplace_logistic_inverse,
    'relu': inplace_relu_inverse,
    'bounded_relu': inplace_bounded_relu_inverse
}

ACTIVATIONS_INVERSE_BOUNDS: Dict[str, tuple] = {
    'tanh': (-.99, .99),
    'identity': (-np.inf, np.inf),
    'logistic': (0.01, .99),
    'relu': (0, np.inf),
    'bounded_relu': (0, 1)
}
