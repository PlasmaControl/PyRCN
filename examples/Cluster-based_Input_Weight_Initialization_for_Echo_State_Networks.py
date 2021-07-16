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
from pyrcn.base import PredefinedWeightsInputToNode, NodeToNode
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


chlo = np.load(r"E:\multivariate_time_series_dataset\numpy\CHLO.npz")
X_train = np.empty(shape=(467, ), dtype=object)
y_train = np.empty(shape=(467, ), dtype=object)
X_test = np.empty(shape=(3840, ), dtype=object)
y_test = np.empty(shape=(3840, ), dtype=object)

for k, (X, y) in enumerate(zip(chlo['X'], chlo['Y'])):
    X_train[k] = X[X.sum(axis=1)!=0, :]  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X
    y_train[k] = np.argwhere(y).ravel()
scaler = StandardScaler().fit(np.concatenate(X_train))
for k, X in enumerate(X_train):
    X_train[k] = scaler.transform(X=X)  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X

X_train, y_train = shuffle(X_train, y_train, random_state=0)

for k, (X, y) in enumerate(zip(chlo['Xte'], chlo['Yte'])):
    X_test[k] = scaler.transform(X=X[X.sum(axis=1)!=0, :])  # Sequences are zeropadded -> should we remove zeros? if not, X_train[k] = X
    y_test[k] = np.argwhere(y).ravel()

kmeans = MiniBatchKMeans(n_clusters=50, n_init=200, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=2, random_state=0)
kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])

initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 3,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bi_directional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_km_esn = SeqToLabelESNClassifier(input_to_node=PredefinedWeightsInputToNode(predefined_input_weights=w_in.T),
                                      **initially_fixed_params)

try:
    sequential_search = load("../sequential_search_chlo_km.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_km_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search, "../sequential_search_chlo_km.joblib")

print(sequential_search.all_best_params_)
print(sequential_search.all_best_score_)
"""
param_grid = {'hidden_layer_size': [50, 100, 200, 400, 800, 1600]}

for params in ParameterGrid(param_grid):
    kmeans = MiniBatchKMeans(n_clusters=params['hidden_layer_size'], n_init=200, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=0, random_state=0)
    kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
    w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
    base_km_esn = clone(sequential_search.best_estimator_)
    base_km_esn.input_to_node.predefined_input_weights=w_in.T
    base_km_esn.set_params(**params)
    gs_cv = GridSearchCV(base_km_esn,
                         param_grid={'random_state': range(10)},
                         scoring=make_scorer(accuracy_score), n_jobs=-1).fit(X=X_train, y=y_train)
    print(gs_cv.cv_results_)
    print("---------------------------------------")
    acc_score = accuracy_score(y_test, gs_cv.best_estimator_.predict(X_test))
    print(acc_score)
    print("---------------------------------------")
"""
constant_params = sequential_search.best_estimator_.get_params()
constant_params.pop('hidden_layer_size')
constant_params.pop('random_state')
constant_params.pop('predefined_input_weights')
param_grid = {'hidden_layer_size': [50, 100, 200, 400, 800, 1600],
              'random_state': range(1, 11)}

for params in ParameterGrid(param_grid):
    kmeans = MiniBatchKMeans(n_clusters=params['hidden_layer_size'], n_init=200, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=0, random_state=params['random_state'])
    t1 = time.time()
    kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
    w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
    t2 = time.time()
    km_esn = clone(sequential_search.best_estimator_)
    km_esn.input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T)
    km_esn.set_params(**constant_params, **params)
    km_esn.fit(X=X_train, y=y_train, n_jobs=8)
    score = accuracy_score(y_test, km_esn.predict(X_test))
    print("KM-ESN with params {0} achieved score of {1} and was trained in {2} seconds.".format(params, score, t2-t1))