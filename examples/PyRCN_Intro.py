#!/usr/bin/env python
# coding: utf-8
# Building blocks of Reservoir Computing
from pyrcn.base.blocks import InputToNode, BatchIntrinsicPlasticity
from pyrcn.base.blocks import NodeToNode, HebbianNodeToNode
from sklearn.datasets import make_blobs
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge as skRidge
from sklearn.pipeline import Pipeline, FeatureUnion
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.extreme_learning_machine import ELMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.datasets import mackey_glass
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer

import numpy as np
from sklearn.decomposition import PCA
from pyrcn.base.blocks import PredefinedWeightsNodeToNode

from pyrcn.echo_state_network import ESNClassifier
from pyrcn.metrics import accuracy_score
from pyrcn.datasets import load_digits


# Generate a toy dataset
U, y = make_blobs(n_samples=100, n_features=10)

# Input-to-Node
#      _ _ _ _ _ _ _ _
#     |               |
# ----| Input-to-Node |------
# u[n]|_ _ _ _ _ _ _ _|r'[n]
# U                    R_i2n

input_to_node = InputToNode(hidden_layer_size=50, k_in=5,
                            input_activation="tanh", input_scaling=1.0,
                            bias_scaling=0.1)

R_i2n = input_to_node.fit_transform(U)
print(U.shape, R_i2n.shape)

# Node-to-Node
#      _ _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |               |      |              |
# ----| Input-to-Node |------| Node-to-Node |------
# u[n]|_ _ _ _ _ _ _ _|r'[n] |_ _ _ _ _ _ _ |r[n]
# U                    R_i2n                 R_n2n

# Initialize, fit and apply NodeToNode
node_to_node = NodeToNode(hidden_layer_size=50, reservoir_activation="tanh",
                          spectral_radius=1.0, leakage=0.9,
                          bidirectional=False)
R_n2n = node_to_node.fit_transform(R_i2n)
print(U.shape, R_n2n.shape)

# Node-to-Output
#       _ _ _ _ _ _ _       _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |              |     |             |     |               |
# ----|Input-to-Node |-----|Node-to-Node |-----|Node-to-Output |
# u[n]| _ _ _ _ _ _ _|r'[n]|_ _ _ _ _ _ _|r[n] | _ _ _ _ _ _ _ |
# U                   R_i2n               R_n2n        |
# Initialize, fit and apply NodeToOutput
y_pred = Ridge().fit(R_n2n, y).predict(R_n2n)
print(y_pred.shape)

# Predicting the Mackey-Glass equation
# Load the dataset
X, y = mackey_glass(n_timesteps=5000)
# Define Train/Test lengths
trainLen = 1900
X_train, y_train = X[:trainLen], y[:trainLen]
X_test, y_test = X[trainLen:], y[trainLen:]

# Initialize and train an ELMRegressor and an ESNRegressor
esn = ESNRegressor().fit(X=X_train.reshape(-1, 1), y=y_train)
elm = ELMRegressor(
    regressor=skRidge()).fit(X=X_train.reshape(-1, 1), y=y_train)
print("Fitted models")

# Build Reservoir Computing Networks with PyRCN
U, y = make_blobs(n_samples=100, n_features=10)

# Vanilla ELM for regression tasks with input_scaling
#       _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |              |     |               |
# ----|Input-to-Node |-----|Node-to-Output |------
# u[n]| _ _ _ _ _ _ _|r'[n]| _ _ _ _ _ _ _ |y[n]
#                                           y_pred
#
vanilla_elm = ELMRegressor(input_scaling=0.9)
vanilla_elm.fit(U, y)
print(vanilla_elm.predict(U))

# Example of how to construct an ELM with a BIP "Input-to-Node" ELMs with PyRCN
# Custom ELM with BatchIntrinsicPlasticity
#       _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |              |     |               |
# ----|     BIP      |-----|Node-to-Output |------
# u[n]| _ _ _ _ _ _ _|r'[n]| _ _ _ _ _ _ _ |y[n]
#                                           y_pred
#
bip_elm = ELMRegressor(input_to_node=BatchIntrinsicPlasticity(),
                       regressor=Ridge(alpha=1e-5))
bip_elm.fit(U, y)
print(bip_elm.predict(U))

# Hierarchical or Ensemble ELMs can then be built using multiple
# "Input-to-Node" modules
# in parallel or in a cascade.
# ELM with cascaded InputToNode and default regressor
#       _ _ _ _ _ _ _        _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |     (bip)    |     |    (base)    |     |               |
# ----|Input-to-Node1|-----|Input-to-Node2|-----|Node-to-Output |
# u[n]| _ _ _ _ _ _ _|     | _ _ _ _ _ _ _|r'[n]| _ _ _ _ _ _ _ |
#                                                       |
#                                                       |
#                                                  y[n] | y_pred
#
i2n = Pipeline([('bip', BatchIntrinsicPlasticity()),
                ('base', InputToNode(bias_scaling=0.1))])
casc_elm = ELMRegressor(input_to_node=i2n).fit(U, y)

# Ensemble of InputToNode with activations
#             _ _ _ _ _ _ _
#           |      (i)     |
#      |----|Input-to-Node1|-----|
#      |    | _ _ _ _ _ _ _|     |       _ _ _ _ _ _ _
#      |                          -----|               |
# -----o                          r'[n]|Node-to-Output |------
# u[n] |      _ _ _ _ _ _ _      |-----| _ _ _ _ _ _ _ |y[n]
#      |    |     (th)     |     |                      y_pred
#      |----|Input-to-Node2|-----|
#           | _ _ _ _ _ _ _|
#
i2n = FeatureUnion([('i', InputToNode(input_activation="identity")),
                    ('th', InputToNode(input_activation="tanh"))])
ens_elm = ELMRegressor(input_to_node=i2n)
ens_elm.fit(U, y)
print(casc_elm, ens_elm)

# Echo State Networks
# Vanilla ESN for regression tasks with spectral_radius and leakage
#       _ _ _ _ _ _ _       _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |              |     |             |     |               |
# ----|Input-to-Node |-----|Node-to-Node |-----|Node-to-Output |
# u[n]| _ _ _ _ _ _ _|r'[n]|_ _ _ _ _ _ _|r[n] | _ _ _ _ _ _ _ |
#                                                      |
#                                                      |
#                                                 y[n] | y_pred
#
vanilla_esn = ESNRegressor(spectral_radius=1, leakage=0.9)
vanilla_esn.fit(U, y)
print(vanilla_esn.predict(U))

# Custom ESN with BatchIntrinsicPlasticity and HebbianNodeToNode
#       _ _ _ _ _ _ _       _ _ _ _ _ _ _        _ _ _ _ _ _ _
#     |     (bip)    |     |   (hebb)    |     |               |
# ----|Input-to-Node |-----|Node-to-Node |-----|Node-to-Output |
# u[n]| _ _ _ _ _ _ _|r'[n]|_ _ _ _ _ _ _|r[n] | _ _ _ _ _ _ _ |
#                                                      |
#                                                      |
#                                                 y[n] | y_pred
#
bip_esn = ESNRegressor(input_to_node=BatchIntrinsicPlasticity(),
                       node_to_node=HebbianNodeToNode(),
                       regressor=Ridge(alpha=1e-5))

bip_esn.fit(U, y)
print(bip_esn.predict(U))

# The "Deep ESN" can refer to different approaches of hierarchical ESN
# architectures:
# Multilayer ESN
#                  u[n]
#                   |
#                   |
#          _________o_________
#         |                   |
#   _ _ _ | _ _ _       _ _ _ | _ _ _
# |      (i)     |    |      (i)     |
# |Input-to-Node1|    |Input-to-Node2|
# | _ _ _ _ _ _ _|    | _ _ _ _ _ _ _|
#         |r1'[n]             | r2'[n]
#   _ _ _ | _ _ _       _ _ _ | _ _ _
# |     (th)     |    |     (th)     |
# | Node-to-Node1|    | Node-to-Node2|
# | _ _ _ _ _ _ _|    | _ _ _ _ _ _ _|
#         |r1[n]              | r2[n]
#         |_____         _____|
#               |       |
#             _ | _ _ _ | _
#           |               |
#           | Node-to-Node3 |
#           | _ _ _ _ _ _ _ |
#                   |
#              r3[n]|
#             _ _ _ | _ _ _
#           |               |
#           |Node-to-Output |
#           | _ _ _ _ _ _ _ |
#                   |
#               y[n]|

l1 = Pipeline([('i2n1', InputToNode(hidden_layer_size=100)),
               ('n2n1', NodeToNode(hidden_layer_size=100))])

l2 = Pipeline([('i2n2', InputToNode(hidden_layer_size=400)),
               ('n2n2', NodeToNode(hidden_layer_size=400))])

i2n = FeatureUnion([('l1', l1),
                    ('l2', l2)])
n2n = NodeToNode(hidden_layer_size=500)
layered_esn = ESNRegressor(input_to_node=i2n, node_to_node=n2n)

layered_esn.fit(U, y)
print(layered_esn.predict(U))

# Yet another example for a deep ESN
# Multiple small reservoirs with different leakages in parallel
res1 = FeatureUnion([
    ("lambda_0.1",
     Pipeline([('i2n', InputToNode(hidden_layer_size=10)),
               ('n2n', NodeToNode(hidden_layer_size=10,
                                  leakage=0.1))])),
    ("lambda_0.2",
     Pipeline([('i2n', InputToNode(hidden_layer_size=10)),
               ('n2n', NodeToNode(hidden_layer_size=10,
                                  leakage=0.2))])),
    ("lambda_0.3",
     Pipeline([('i2n', InputToNode(hidden_layer_size=10)),
               ('n2n', NodeToNode(hidden_layer_size=10,
                                  leakage=0.3))])),
    ("lambda_0.4",
     Pipeline([('i2n', InputToNode(hidden_layer_size=10)),
               ('n2n', NodeToNode(hidden_layer_size=10,
                                  leakage=0.4))])),])

pca = PCA(n_components=10)

res2 = Pipeline([("i2n", InputToNode(hidden_layer_size=100)),
                 ("n2n", NodeToNode(hidden_layer_size=100))])

i2n = FeatureUnion([("path1",
                     Pipeline([("res1", res1), ("pca", pca),
                               ("res2", res2)])),
                    ("path2", res1)])

n2n = PredefinedWeightsNodeToNode(
    predefined_recurrent_weights=np.eye(40+100),
    spectral_radius=0, leakage=1)

deep_esn = ESNRegressor(input_to_node=i2n, node_to_node=n2n)
deep_esn.fit(U, y)
print(deep_esn.predict(U))

# Complex example: Optimize the hyper-parameters of RCNs
# Load the dataset
X, y = mackey_glass(n_timesteps=5000)
X_train, X_test = X[:1900], X[1900:]
y_train, y_test = y[:1900], y[1900:]

# Define initial ESN model
esn = ESNRegressor(bias_scaling=0, spectral_radius=0, leakage=1,
                   requires_sequence=False)

# Define optimization workflow
scorer = make_scorer(mean_squared_error, greater_is_better=False)
step_1_params = {
    'input_scaling': uniform(loc=1e-2, scale=1),
    'spectral_radius': uniform(loc=0, scale=2)
}
kwargs_1 = {
    'n_iter': 200, 'n_jobs': -1, 'scoring': scorer,
    'cv': TimeSeriesSplit()
}
step_2_params = {'leakage': [0.2, 0.4, 0.7, 0.9, 1.0]}
kwargs_2 = {
    'verbose': 5, 'scoring': scorer, 'n_jobs': -1,
    'cv': TimeSeriesSplit()
}

searches = [('step1', RandomizedSearchCV, step_1_params, kwargs_1),
            ('step2', GridSearchCV, step_2_params, kwargs_2)]

# Perform the search
esn_opti = SequentialSearchCV(esn, searches).fit(X_train.reshape(-1, 1),
                                                 y_train)
print(esn_opti)

# Programming pattern for sequence processing
# Load the dataset
X, y = load_digits(return_X_y=True, as_sequence=True)
print("Number of digits: {0}".format(len(X)))
print("Shape of digits {0}".format(X[0].shape))
# Divide the dataset into training and test subsets
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)
print("Number of digits in training set: {0}".format(len(X_tr)))
print("Shape of the first digit: {0}".format(X_tr[0].shape))
print("Number of digits in test set: {0}".format(len(X_te)))
print("Shape of the first digit: {0}".format(X_te[0].shape))

# These parameters were optimized using SequentialSearchCV
esn_params = {
    'input_scaling': 0.05077514155476392,
    'spectral_radius': 1.1817858863764836,
    'input_activation': 'identity',
    'k_in': 5,
    'bias_scaling': 1.6045393364745582,
    'reservoir_activation': 'tanh',
    'leakage': 0.03470266988650412,
    'k_rec': 10,
    'alpha': 3.0786517836196185e-05,
    'decision_strategy': "winner_takes_all"
}

b_esn = ESNClassifier(**esn_params)

param_grid = {
    'hidden_layer_size': [50, 100, 200, 400, 500],
    'bidirectional': [False, True]
}

for params in ParameterGrid(param_grid):
    esn_cv = cross_validate(clone(b_esn).set_params(**params), X=X_tr, y=y_tr,
                            scoring=make_scorer(accuracy_score))
    esn = clone(b_esn).set_params(**params).fit(X_tr, y_tr, n_jobs=-1)
    acc_score = accuracy_score(y_te, esn.predict(X_te))
