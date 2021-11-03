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

hidden_layer_size = 50
r=0.5
rho=0.9
b=0.05



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
