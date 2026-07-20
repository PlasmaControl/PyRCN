"""The :mod:`activations` contains various activation functions for PyRCN."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import expit as logistic_sigmoid


def inplace_identity(X: np.ndarray) -> None:
    """
    Compute the identity function inplace.

    This is a no-op, kept for API consistency with the other activations.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    # Nothing to do


def inplace_tanh(X: np.ndarray) -> None:
    """
    Compute the hyperbolic tangent function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    np.tanh(X, out=X)


def inplace_logistic(X: np.ndarray) -> None:
    """
    Compute the logistic sigmoid function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    logistic_sigmoid(X, out=X)


def inplace_relu(X: np.ndarray) -> None:
    """
    Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : ndarray
        The input data.
    """
    np.maximum(X, 0, out=X)


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
    inplace_identity(X)


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
    inplace_relu(X)


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
    inplace_bounded_relu(X)


ACTIVATIONS: dict[str, Callable] = {
    'identity': inplace_identity,
    'tanh': inplace_tanh,
    'logistic': inplace_logistic,
    'relu': inplace_relu,
    'bounded_relu': inplace_bounded_relu,
}

ACTIVATIONS_INVERSE: dict[str, Callable] = {
    'tanh': inplace_tanh_inverse,
    'identity': inplace_identity_inverse,
    'logistic': inplace_logistic_inverse,
    'relu': inplace_relu_inverse,
    'bounded_relu': inplace_bounded_relu_inverse
}

ACTIVATIONS_INVERSE_BOUNDS: dict[str, tuple] = {
    'tanh': (-.99, .99),
    'identity': (-np.inf, np.inf),
    'logistic': (0.01, .99),
    'relu': (0, np.inf),
    'bounded_relu': (0, 1)
}
