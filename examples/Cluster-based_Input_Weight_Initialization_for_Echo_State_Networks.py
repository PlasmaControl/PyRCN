import time
import glob
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid, cross_val_score
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from sklearn.cluster import MiniBatchKMeans
from joblib import dump, load
from pyrcn.echo_state_network import SeqToLabelESNClassifier
from pyrcn.base.blocks import PredefinedWeightsInputToNode, NodeToNode
from pyrcn.metrics import accuracy_score
from pyrcn.model_selection import SequentialSearchCV
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#Options
plt.rc('image', cmap='RdBu')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker


arab = np.load(r"E:\multivariate_time_series_dataset\numpy\ARAB.npz")
X_train = np.empty(shape=(6600, ), dtype=object)
y_train = np.empty(shape=(6600, ), dtype=object)
X_test = np.empty(shape=(2200, ), dtype=object)
y_test = np.empty(shape=(2200, ), dtype=object)

for k, (X, y) in enumerate(zip(arab['X'], arab['Y'])):
    X_train[k] = X[X.sum(axis=1)!=0, :]  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X
    y_train[k] = np.tile(y, (X_train[k].shape[0], 1))
scaler = StandardScaler().fit(np.concatenate(X_train))
for k, X in enumerate(X_train):
    X_train[k] = scaler.transform(X=X)  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X

X_train, y_train = shuffle(X_train, y_train, random_state=0)

for k, (X, y) in enumerate(zip(arab['Xte'], arab['Yte'])):
    X_test[k] = scaler.transform(X=X[X.sum(axis=1)!=0, :])  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X
    y_test[k] = np.tile(y, (X_test[k].shape[0], 1))

initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bidirectional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': 1, 'scoring': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True)}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True)}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True)}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = SeqToSeqESNClassifier(**initially_fixed_params)
base_esn.fit(X_train, y_train)
try:
    sequential_search = load("../sequential_search_arab.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search, "../sequential_search_arab.joblib")
