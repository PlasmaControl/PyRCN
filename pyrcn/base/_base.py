try:
    from typing import Union, Literal
except ImportError:
    from typing import Union
    from typing_extensions import Literal
import numpy as np
import scipy

if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
    from scipy.sparse.linalg import ArpackNoConvergence
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence


def _uniform_random_input_weights(n_features_in: Union[int, np.integer], 
                                  hidden_layer_size: Union[int, np.integer], 
                                  fan_in: Union[int, np.integer], 
                                  random_state: np.random.RandomState) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:
    """
    Returns uniform random input weights in range [-1, 1]

    Parameters
    ----------
    n_features_in : Union[int, np.integer]
    hidden_layer_size : Union[int, np.integer]
    fan_in : Union[int, np.integer]
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    uniform_random_input_weights : Union[np.ndarray, scipy.sparse.csr.csr_matrix]
       of size (n_features, hidden_layer_size)
    """
    nr_entries = np.int32(n_features_in * fan_in)
    weights_array = random_state.uniform(low=-1., high=1., size=nr_entries)

    if fan_in < hidden_layer_size:
        indices = np.zeros(shape=nr_entries, dtype=int)
        indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in, step=fan_in)

        for en in range(0, n_features_in * fan_in, fan_in):
            indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
        return scipy.sparse.csr_matrix(
            (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
    else:
        return weights_array.reshape((n_features_in, hidden_layer_size))


def _uniform_random_bias(hidden_layer_size: Union[int, np.integer], 
                         random_state: np.random.RandomState) -> np.ndarray:
    """
    Returns uniform random bias in range [-1, 1].

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


def _normal_random_recurrent_weights(hidden_layer_size: Union[int, np.integer], 
                                     fan_in: Union[int, np.integer], 
                                     random_state: np.random.RandomState) -> Union[np.ndarray, scipy.sparse.csr.csr_matrix]:
    """
    Returns normally distributed random reservoir weights

    Parameters
    ----------
    hidden_layer_size : Union[int, np.integer]
    fan_in : Union[int, np.integer]
        Determines how many features are mapped to one neuron.
    random_state : numpy.random.RandomState

    Returns
    -------
    normal_random_recurrent_weights : Union[np.ndarray, scipy.sparse.csr.csr_matrix] 
        of size (hidden_layer_size, hidden_layer_size)
    """
    nr_entries = np.int32(hidden_layer_size * fan_in)
    weights_array = random_state.normal(loc=0., scale=1., size=nr_entries)

    if fan_in < hidden_layer_size:
        indices = np.zeros(shape=nr_entries, dtype=int)
        indptr = np.arange(start=0, stop=(hidden_layer_size + 1) * fan_in, step=fan_in)

        for en in range(0, hidden_layer_size * fan_in, fan_in):
            indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
        recurrent_weights_init = scipy.sparse.csr_matrix(
            (weights_array, indices, indptr), shape=(hidden_layer_size, hidden_layer_size), dtype='float64')
    else:
        recurrent_weights_init = weights_array.reshape((hidden_layer_size, hidden_layer_size))

    try:
        we = eigens(recurrent_weights_init, 
                    k=np.minimum(10, hidden_layer_size - 2), 
                    which='LM', 
                    return_eigenvectors=False, 
                    v0=random_state.normal(loc=0., scale=1., size=hidden_layer_size)
                    )
    except ArpackNoConvergence:
        print("WARNING: No convergence! Returning possibly invalid values!!!")
        we = ArpackNoConvergence.eigenvalues
    return recurrent_weights_init / np.amax(np.absolute(we))