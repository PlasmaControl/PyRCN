"""Utilities for the extreme learning machine modules
"""

import numpy as np
from sklearn.neural_network._base import ACTIVATIONS


def inplace_bounded_relu(X):
    """Compute the rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.minimum(np.maximum(X, 0, out=X), 1, out=X)


ACTIVATIONS.update({'bounded_relu': inplace_bounded_relu})
