from sklearn.neural_network._base import ACTIVATIONS
import numpy as np


def bounded_relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    np.clip(X, 0, 1, out=X)
    return X


ACTIVATIONS["bounded_relu"] = bounded_relu
