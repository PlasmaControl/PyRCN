#!/usr/bin/env python
# coding: utf-8

# # Prediction of musical notes
#
# ## Introduction
#
# This notebook adapts one reference experiment for note prediction using ESNs
# from ([https://arxiv.org/abs/1812.11527](https://arxiv.org/abs/1812.11527))
# to PyRCN and shows that introducing bidirectional ESNs significantly
# improves the results
# in terms of Accuracy, already for rather small networks.
#
# The tutorial is based on numpy, scikit-learn, joblib and PyRCN.
# We are using the ESNRegressor, because we further process the outputs of the
# ESN.
# Note that the same can also be done using the ESNClassifier.
import numpy as np
import os
from joblib import load
from sklearn.base import clone
from sklearn.model_selection import (ParameterGrid, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

from pyrcn.echo_state_network import ESNClassifier
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.metrics import mean_squared_error, accuracy_score


# ## Load the dataset, which is available at
# http://www-etud.iro.umontreal.ca/~boulanni/icml2012.
dataset_path = os.path.normpath(r"E:\MusicPrediction\Piano-midi.de.pickle")
dataset = load(dataset_path)
X_train = np.empty(shape=(len(dataset['train']) + len(dataset['valid']),),
                   dtype=object)
y_train = np.empty(shape=(len(dataset['train']) + len(dataset['valid']),),
                   dtype=object)

X_test = np.empty(shape=(len(dataset['test']),), dtype=object)
y_test = np.empty(shape=(len(dataset['test']),), dtype=object)
print("Number of sequences in the training and test set: {0}, {1}"
      .format(len(X_train), len(X_test)))

# Prepare the dataset
#
# We use the MultiLabelBinarizer to transform the sequences of MIDI pitches
# into one-hot encoded vectors. The piano is restricted to 88 keys.
mlb = MultiLabelBinarizer(classes=range(128))
for k, X in enumerate(dataset['train'] + dataset['valid']):
    X_train[k] = mlb.fit_transform(X[:-1])
    y_train[k] = mlb.fit_transform(X[1:])
for k, X in enumerate(dataset['test']):
    X_test[k] = mlb.fit_transform(X[:-1])
    y_test[k] = mlb.fit_transform(X[1:])
print("Shape of first sequences in the training and test set: {0}, {1}"
      .format(X_train[0].shape, X_test[0].shape))

# ## Set up a basic ESN
#
# To develop an ESN model, we need to tune several hyper-parameters,
# e.g., input_scaling, spectral_radius, bias_scaling and leaky integration.
#
# We follow the way proposed in the introductory paper of PyRCN
# to optimize hyper-parameters sequentially.
initially_fixed_params = {
    'hidden_layer_size': 50,
    'input_activation': 'identity',
    'k_in': 10,
    'input_scaling': 0.4,
    'bias_scaling': 0.0,
    'spectral_radius': 0.0,
    'reservoir_activation': 'tanh',
    'leakage': 1.0,
    'bidirectional': False,
    'k_rec': 10,
    'alpha': 1e-3,
    'random_state': 42
}

step1_esn_params = {
    'input_scaling': uniform(loc=1e-2, scale=1),
    'spectral_radius': uniform(loc=0, scale=2)
}
step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {
    'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(mean_squared_error, greater_is_better=False,
                           needs_proba=True)
}
kwargs_step2 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(mean_squared_error, greater_is_better=False,
                           needs_proba=True)
}
kwargs_step3 = {
    'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(mean_squared_error, greater_is_better=False,
                           needs_proba=True)
}
kwargs_step4 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(mean_squared_error, greater_is_better=False,
                           needs_proba=True)
}

searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = ESNClassifier(**initially_fixed_params)
sequential_search = \
    SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)

# ## Test the ESN
#
# In the test case, we train the ESN using the entire training set as seen
# before. Next, we compute the predicted outputs on the training and test set
# and fix a threshold of 0.5, above a note is assumed to be predicted.
#
# We report the accuracy score for each frame in order to follow the reference
# paper.
param_grid = {'hidden_layer_size': [500, 1000, 2000, 4000, 5000]}
base_esn = sequential_search.best_estimator_

for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn).set_params(**params)
    esn.fit(X_train, y_train)
    training_score = accuracy_score(y_train, esn.predict(X_train))
    test_score = accuracy_score(y_test, esn.predict(X_test))
    print('{0}\t{1}'.format(training_score, test_score))
