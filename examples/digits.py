#!/usr/bin/env python
# coding: utf-8

"""
Recognizing hand-written digits
-------------------------------

This notebook adapts the existing example of applying support vector
classification from scikit-learn to PyRCN to demonstrate, how PyRCN can be used
to classify hand-written digits.

The tutorial is based on numpy, scikit-learn and PyRCN.
"""
import numpy as np
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    ParameterGrid, RandomizedSearchCV, cross_validate)
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from sklearn.metrics import make_scorer

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.metrics import accuracy_score
from pyrcn.datasets import load_digits


# Load the dataset (part of scikit-learn) and consists of 1797 8x8 images.
# We are using our dataloader that is derived from scikit-learns dataloader and
# returns arrays of 8x8 sequences and corresponding labels.
X, y = load_digits(return_X_y=True, as_sequence=True)
print("Number of digits: {0}".format(len(X)))
print("Shape of digits {0}".format(X[0].shape))

# Split dataset in training and test
# Afterwards, we split the dataset into training and test sets.
# We train the ESN using 80% of the digits and test it using the remaining
# images.
stratify = np.asarray([np.unique(yt) for yt in y]).flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=stratify, random_state=42)
X_tr = np.copy(X_train)
y_tr = np.copy(y_train)
X_te = np.copy(X_test)
y_te = np.copy(y_test)
for k, _ in enumerate(y_tr):
    y_tr[k] = np.repeat(y_tr[k], 8, 0)
for k, _ in enumerate(y_te):
    y_te[k] = np.repeat(y_te[k], 8, 0)

print("Number of digits in training set: {0}".format(len(X_train)))
print("Shape of digits in training set: {0}".format(X_train[0].shape))
print("Number of digits in test set: {0}".format(len(X_test)))
print("Shape of digits in test set: {0}".format(X_test[0].shape))

# Set up a ESN
# To develop an ESN model for digit recognition, we need to tune several
# hyper-parameters, e.g., input_scaling, spectral_radius, bias_scaling and
# leaky integration.
#
# We follow the way proposed in the introductory paper of PyRCN to optimize
# hyper-parameters sequentially.
#
# We define the search spaces for each step together with the type of search
# (a grid search in this context).
#
# At last, we initialize an ESNClassifier with the desired output strategy
# and with the initially fixed parameters.


initially_fixed_params = {'hidden_layer_size': 50,
                          'input_activation': 'identity',
                          'k_in': 5,
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': 1.0,
                          'bidirectional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'decision_strategy': "winner_takes_all"}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': uniform(loc=0, scale=2)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e0)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': make_scorer(accuracy_score)
                }
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': make_scorer(accuracy_score)
                }
kwargs_step3 = {'verbose': 1, 'n_jobs': -1,
                'scoring': make_scorer(accuracy_score)
                }
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': make_scorer(accuracy_score)
                }

# The searches are defined similarly to the steps of a
# sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', RandomizedSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = ESNClassifier(**initially_fixed_params)


# Optimization
# We provide a SequentialSearchCV that basically iterates through the list of
# searches that we have defined before. It can be combined with any model
# selection tool from
# scikit-learn.
sequential_search = SequentialSearchCV(base_esn,
                                       searches=searches).fit(X_tr, y_tr)


# Use the ESN with final hyper-parameters
#
# After the optimization, we extract the ESN with final hyper-parameters as the
# result # of the optimization.
base_esn = sequential_search.best_estimator_


# Test the ESN
# Finally, we increase the reservoir size and compare the impact of uni- and
# bidirectional ESNs. Notice that the ESN strongly benefit from both,
# increasing the reservoir size and from the bi-directional working mode.
param_grid = {'hidden_layer_size': [50, 100, 200, 400, 500],
              'bidirectional': [False, True]}

print("CV results\tFit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in ParameterGrid(param_grid):
    esn_cv = cross_validate(clone(base_esn).set_params(**params), X=X_train,
                            y=y_train, scoring=make_scorer(accuracy_score),
                            n_jobs=-1)
    t1 = time.time()
    esn = clone(base_esn).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    t1 = time.time()
    esn_par = clone(base_esn).set_params(**params).fit(X_train, y_train,
                                                       n_jobs=-1)
    t_fit_par = time.time() - t1
    mem_size = esn.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, esn.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(esn_cv, t_fit, t_inference,
                                           acc_score, mem_size))
