import os
import glob
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

sns.set_theme(style="whitegrid")
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 8,
          'font.family': 'lmodern',
         }
plt.rcParams.update(params)
plt.rc('image', cmap='RdBu')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rc('font', **{'family': 'serif'})

import scipy
if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
    from scipy.sparse.linalg import ArpackNoConvergence
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence


def normal_random_recurrent_weights(n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
    """
    Returns normally distributed random reservoir weights

    Parameters
    ----------
    n_features_in : int
    hidden_layer_size : int
    fan_in : int
        Determines how many features are mapped to one neuron.
    random_state : numpy.RandomState

    Returns
    -------
    normal_random_input_weights : ndarray of size (hidden_layer_size, hidden_layer_size)
    """
    if n_features_in != hidden_layer_size:
        raise ValueError("Dimensional mismatch: n_features must match hidden_layer_size, got %s !=%s." %
                            (n_features_in, hidden_layer_size))
    nr_entries = np.int32(n_features_in * fan_in)
    weights_array = random_state.normal(loc=0., scale=1., size=nr_entries)

    if fan_in < hidden_layer_size:
        indices = np.zeros(shape=nr_entries, dtype=int)
        indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in, step=fan_in)

        for en in range(0, n_features_in * fan_in, fan_in):
            indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
        recurrent_weights_init = scipy.sparse.csr_matrix(
            (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
    else:
        recurrent_weights_init = weights_array.reshape((n_features_in, hidden_layer_size))

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


hidden_layer_size = 50
r=1
b=0.5

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(7, 4)

predefined_recurrent_weights = np.zeros((hidden_layer_size, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][0])
axs[0][0].set_title("(DLR)")
we = np.linalg.eig(predefined_recurrent_weights)

predefined_recurrent_weights = np.zeros((hidden_layer_size, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
    predefined_recurrent_weights[k, k+1] = b
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][1])
axs[0][1].set_title("(DLRB)")
we = np.linalg.eig(predefined_recurrent_weights)


predefined_recurrent_weights = np.zeros((hidden_layer_size, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
predefined_recurrent_weights[0, hidden_layer_size - 1] = r
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][2])
axs[0][2].set_title("(SCR)")
we = np.linalg.eig(predefined_recurrent_weights)


predefined_recurrent_weights = normal_random_recurrent_weights(50, 50, 10, check_random_state(42))
sns.heatmap(data=predefined_recurrent_weights.todense(), ax=axs[1][1])
axs[1][0].set_title("Sparse")

predefined_recurrent_weights = normal_random_recurrent_weights(50, 50, 50, check_random_state(42))
sns.heatmap(data=predefined_recurrent_weights, ax=axs[1][2])
axs[1][2].set_title("Dense")

plt.tight_layout()
plt.show()


###########################

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(7, 4)

predefined_input_weights = np.zeros((13, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][0])
axs[0][0].set_title("(DLR)")
we = np.linalg.eig(predefined_recurrent_weights)

predefined_recurrent_weights = np.zeros((hidden_layer_size, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
    predefined_recurrent_weights[k, k+1] = b
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][1])
axs[0][1].set_title("(DLRB)")
we = np.linalg.eig(predefined_recurrent_weights)


predefined_recurrent_weights = np.zeros((hidden_layer_size, hidden_layer_size))
for k in range(hidden_layer_size - 1):
    predefined_recurrent_weights[k+1, k] = r
predefined_recurrent_weights[0, hidden_layer_size - 1] = r
sns.heatmap(data=predefined_recurrent_weights, ax=axs[0][2])
axs[0][2].set_title("(SCR)")
we = np.linalg.eig(predefined_recurrent_weights)


predefined_recurrent_weights = normal_random_recurrent_weights(50, 50, 10, check_random_state(42))
sns.heatmap(data=predefined_recurrent_weights.todense(), ax=axs[1][1])
axs[1][0].set_title("Sparse")

predefined_recurrent_weights = normal_random_recurrent_weights(50, 50, 50, check_random_state(42))
sns.heatmap(data=predefined_recurrent_weights, ax=axs[1][2])
axs[1][2].set_title("Dense")

plt.tight_layout()
plt.show()
