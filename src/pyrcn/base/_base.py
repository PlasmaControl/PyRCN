"""The :mod:`autoencoder` contains base functionality for PyRCN."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import sys

import numpy as np

if sys.version_info >= (3, 8):
    from typing import Union
else:
    from typing import Union


def _uniform_random_input_weights(n_features_in: int,
                                  hidden_layer_size: Union[int, np.integer],
                                  fan_in: int,
                                  random_state: np.random.RandomState) \
                                      -> np.ndarray:
    """
    Return uniform random input weights in range [-1, 1].

    Parameters
    ----------
    n_features_in : int
    hidden_layer_size : Union[int, np.integer]
    fan_in : int
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_input_weights : np.ndarray of size
    (n_features, hidden_layer_size)
    """
    nr_entries = int(n_features_in * fan_in)

    if fan_in < hidden_layer_size:
        weights_values = random_state.uniform(
            low=-1., high=1., size=nr_entries)
        weights_array = np.zeros(shape=(n_features_in, hidden_layer_size),
                                 dtype=float)
        indices = np.zeros(shape=nr_entries, dtype=int)
        indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in,
                           step=fan_in)

        for k, en in enumerate(range(0, n_features_in * fan_in, fan_in)):
            indices[en: en + fan_in] = random_state.permutation(
                hidden_layer_size)[:fan_in].astype(int)
            weights_array[k, indices[indptr[k]:indptr[k+1]]] += \
                weights_values[indptr[k]:indptr[k+1]]
    else:
        weights_array = random_state.uniform(
            low=-1., high=1., size=nr_entries).reshape(
            (n_features_in, hidden_layer_size))
    return weights_array


def _uniform_random_bias(hidden_layer_size: Union[int, np.integer],
                         random_state: np.random.RandomState) -> np.ndarray:
    """
    Return uniform random bias in range [-1, 1].

    Parameters
    ----------
    hidden_layer_size : Union[int, np.integer]
    fan_in : Union[int, np.integer]
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_bias : ndarray of size (hidden_layer_size)
    """
    return random_state.uniform(low=-1., high=1., size=hidden_layer_size)


def _normal_random_recurrent_weights(hidden_layer_size: int, fan_in: int,
                                     random_state: np.random.RandomState) \
                                         -> np.ndarray:
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
    normal_random_recurrent_weights : np.ndarray
    of size (hidden_layer_size, hidden_layer_size)
    """
    nr_entries = int(hidden_layer_size * fan_in)

    if fan_in < hidden_layer_size:
        weights_values = random_state.normal(loc=0., scale=1., size=nr_entries)
        recurrent_weights_init = np.zeros(
            shape=(hidden_layer_size, hidden_layer_size), dtype=float)
        indices = np.zeros(shape=nr_entries, dtype=int)
        indptr = np.arange(start=0, stop=(hidden_layer_size + 1) * fan_in,
                           step=fan_in)

        for k, en in enumerate(range(0, hidden_layer_size * fan_in, fan_in)):
            indices[en: en + fan_in] = random_state.permutation(
                hidden_layer_size)[:fan_in].astype(int)
            recurrent_weights_init[k, indices[indptr[k]:indptr[k + 1]]] += \
                weights_values[indptr[k]:indptr[k + 1]]
    else:
        weights_array = random_state.normal(loc=0., scale=1., size=nr_entries)
        recurrent_weights_init = weights_array.reshape(
            (hidden_layer_size, hidden_layer_size))

    we = np.linalg.eigvals(recurrent_weights_init)
    return recurrent_weights_init / np.amax(np.absolute(we))


def _normal_recurrent_attention_weights(hidden_layer_size: int, fan_in: int,
                                        random_state: np.random.RandomState,
                                        attention_weights: np.ndarray) \
        -> np.ndarray:
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
    scipy.sparse.csr.csr_matrix] of size (hidden_layer_size, hidden_layer_size)
    """
    recurrent_weights_init = _normal_random_recurrent_weights(
        hidden_layer_size, fan_in, random_state)

    recurrent_weights_init = np.multiply(
        np.asarray(recurrent_weights_init), attention_weights)
    we = np.linalg.eigvals(recurrent_weights_init)
    return recurrent_weights_init / np.amax(np.absolute(we))
