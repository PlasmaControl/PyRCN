"""The :mod:`autoencoder` contains base functionality for PyRCN."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import sys

import numpy as np
import scipy
from scipy.sparse.linalg import eigs as eigens
from scipy.sparse.linalg import ArpackNoConvergence

if sys.version_info >= (3, 8):
    from typing import Union
else:
    from typing import Union


def _antisymmetric_weights(
        weights: Union[np.ndarray, scipy.sparse.csr_matrix]) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Transform a given weight matrix to get antisymmetric, e.g., compute
    weights - weights.T

    Parameters
    ----------
    weights : Union[np.ndarray, scipy.sparse.csr_matrix],
    shape=(hidden_layer_size, hidden_layer_size)
        The given square matrix with weight values.

    Returns
    -------
    antisymmetric_weights : Union[np.ndarray,scipy.sparse.csr_matrix],
    shape=(hidden_layer_size, hidden_layer_size)
        The antisymmetric weight matrix.
    """
    return weights - weights.transpose()


def _unitary_spectral_radius(
        weights: Union[np.ndarray, scipy.sparse.csr_matrix],
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Normalize a given weight matrix to the unitary spectral radius, e.g.,
    the maximum absolute eigenvalue.

    Parameters
    ----------
    weights : Union[np.ndarray, scipy.sparse.csr_matrix],
    shape=(hidden_layer_size, hidden_layer_size)
        The given square matrix with weight values.
    random_state : numpy.random.RandomState

    Returns
    -------
    weights / np.amax(np.abs(eigenvalues)) :
    Union[np.ndarray,scipy.sparse.csr_matrix],
    shape=(hidden_layer_size, hidden_layer_size)
        The weight matrix, divided by its maximum eigenvalue.
    """
    try:
        we = eigens(
            weights, k=np.minimum(10, weights.shape[0] - 2), which='LM',
            v0=random_state.normal(loc=0., scale=1., size=weights.shape[0]),
            return_eigenvectors=False)
    except ArpackNoConvergence as e:
        print("WARNING: No convergence! Returning possibly invalid values!!!")
        we = e.eigenvalues
    return weights / np.amax(np.abs(we))


def _make_sparse(k_in: int, dense_weights: np.ndarray,
                 random_state: np.random.RandomState) \
        -> scipy.sparse.csr_matrix:
    """
    Make a dense weight matrix sparse.

    Parameters
    ----------
    k_in : int
        Determines how many inputs are mapped to one neuron.
    dense_weights : np.ndarray, shape=(n_inputs, n_outputs)
        The randomly initialized layer weights.
    random_state : numpy.random.RandomState

    Returns
    -------
    sparse_weights : scipy.sparse.csr_matrix
        The sparse layer weights
    """
    n_inputs, n_outputs = dense_weights.shape
    nr_entries = int(n_inputs * k_in)

    for neuron in range(n_outputs):
        all_indices = np.arange(n_inputs)
        keep_indices = np.random.choice(n_inputs, k_in, replace=False)
        zero_indices = np.setdiff1d(all_indices, keep_indices)
        dense_weights[zero_indices, neuron] = 0

    return scipy.sparse.csr_matrix(dense_weights, dtype='float64')


def _normal_random_weights(
        n_inputs: int, n_outputs: int, k_in: int,
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Sparse or dense normal random weights.

    Parameters
    ----------
    n_inputs : int
        Number of inputs to the layer (e.g., n_features).
    n_outputs : int
        Number of outputs of the layer (e.g., hidden_layer_size)
    k_in : int
        Determines how many inputs are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    normal_random_weights : Union[np.ndarray,scipy.sparse.csr_matrix],
        shape = (n_inputs, n_outputs)
        The randomly initialized layer weights.
    """
    dense_weights = random_state.normal(
        loc=0., scale=1., size=(n_inputs, n_outputs))

    if k_in < n_outputs:
        return _make_sparse(k_in, dense_weights, random_state)
    else:
        return dense_weights


def _uniform_random_weights(
        n_inputs: int, n_outputs: int, k_in: int,
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Sparse or dense uniform random weights in range [-1, 1].

    Parameters
    ----------
    n_inputs : int
        Number of inputs to the layer (e.g., n_features).
    n_outputs : int
        Number of outputs of the layer (e.g., hidden_layer_size)
    k_in : int
        Determines how many inputs are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_weights : Union[np.ndarray,scipy.sparse.csr_matrix],
        shape = (n_inputs, n_outputs)
        The randomly initialized layer weights.
    """
    dense_weights = random_state.uniform(
        low=-1., high=1., size=(n_inputs, n_outputs))

    if k_in < n_outputs:
        return _make_sparse(k_in, dense_weights, random_state)
    else:
        return dense_weights


def _uniform_random_input_weights(
        n_features_in: int, hidden_layer_size: int, fan_in: int,
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Return uniform random input weights in range [-1, 1].

    Parameters
    ----------
    n_features_in : int
    hidden_layer_size : int
    fan_in : int
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_input_weights : Union[np.ndarray,
    scipy.sparse.csr_matrix], shape = (n_features, hidden_layer_size)
        The randomly initialized input weights.
    """
    input_weights = _uniform_random_weights(
        n_inputs=n_features_in, n_outputs=hidden_layer_size, k_in=fan_in,
        random_state=random_state)
    return input_weights


def _uniform_random_bias(
        hidden_layer_size: int, random_state: np.random.RandomState) \
        -> np.ndarray:
    """
    Return uniform random bias in range [-1, 1].

    Parameters
    ----------
    hidden_layer_size : int
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_bias : ndarray of shape (hidden_layer_size, )
    """
    bias_weights = _uniform_random_weights(
        n_inputs=hidden_layer_size, n_outputs=1, k_in=hidden_layer_size,
        random_state=random_state)
    return bias_weights


def _uniform_random_recurrent_weights(
        hidden_layer_size: int, fan_in: int,
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Return uniformly distributed random reservoir weights.

    Parameters
    ----------
    hidden_layer_size : Union[int, np.integer]
    fan_in : Union[int, np.integer]
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_recurrent_weights : Union[np.ndarray,
    scipy.sparse.csr_matrix], shape=(hidden_layer_size, hidden_layer_size)
    """
    recurrent_weights = _uniform_random_weights(
        n_inputs=hidden_layer_size, n_outputs=hidden_layer_size, k_in=fan_in,
        random_state=random_state)
    return _antisymmetric_weights(recurrent_weights)


def _normal_random_recurrent_weights(
        hidden_layer_size: int, fan_in: int,
        random_state: np.random.RandomState) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Return normally distributed random reservoir weights.

    Parameters
    ----------
    hidden_layer_size : Union[int, np.integer]
    fan_in : Union[int, np.integer]
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    normal_random_recurrent_weights : Union[np.ndarray,
    scipy.sparse.csr_matrix], shape=(hidden_layer_size, hidden_layer_size)
    """
    recurrent_weights = _normal_random_weights(
        n_inputs=hidden_layer_size, n_outputs=hidden_layer_size, k_in=fan_in,
        random_state=random_state)
    return _unitary_spectral_radius(
        weights=recurrent_weights, random_state=random_state)


def _normal_recurrent_attention_weights(
        hidden_layer_size: int, fan_in: int,
        random_state: np.random.RandomState, attention_weights: np.ndarray) \
        -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Return normally distributed random reservoir weights.

    Parameters
    ----------
    hidden_layer_size : int
    fan_in : int
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    normal_random_recurrent_weights : Union[np.ndarray,
    scipy.sparse.csr_matrix] of size (hidden_layer_size, hidden_layer_size)
    """
    recurrent_weights = _normal_random_weights(
        n_inputs=hidden_layer_size, n_outputs=hidden_layer_size, k_in=fan_in,
        random_state=random_state)
    recurrent_weights = np.multiply(
        np.asarray(recurrent_weights), attention_weights)
    return _unitary_spectral_radius(
        recurrent_weights, random_state=random_state)
