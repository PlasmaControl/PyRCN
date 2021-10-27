"""
Testing for Echo State Network module (pyrcn.echo_state_network)
"""
import scipy
import numpy as np
import time

from joblib import Parallel, delayed

import pytest

from pyrcn.datasets import mackey_glass, load_digits
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNClassifier, ESNRegressor
from pyrcn.metrics import mean_squared_error


def test_esn_regressor_jobs():
    print('\ntest_esn_regressor_jobs():')
    X, y= mackey_glass(n_timesteps=8000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    param_grid = {
        'input_to_node': [InputToNode(bias_scaling=.1, hidden_layer_size=10, input_activation='identity', random_state=42),
                          InputToNode(bias_scaling=.1, hidden_layer_size=50, input_activation='identity', random_state=42)],
        'node_to_node': [NodeToNode(spectral_radius=1., hidden_layer_size=10, random_state=42),
                         NodeToNode(spectral_radius=1., hidden_layer_size=50, random_state=42)],
        'regressor': [IncrementalRegression(alpha=.0001), IncrementalRegression(alpha=.01)],
        'random_state': [42]
    }
    esn = GridSearchCV(ESNRegressor(), param_grid)
    esn.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_esn = esn.predict(X_test.reshape(-1, 1))
    print("tests - esn:\n sin | cos \n {0}".format(y_test-y_esn))
    print("best_params_: ".format(esn.best_params_))
    print("best_score: ".format(esn.best_score_))
    np.testing.assert_allclose(y_test[:-1], y_esn[:-1], atol=1e-1)


def test_esn_regressor_requires_no_sequence():
    print('\ntest_esn_regressor_requires_sequence():')
    X, y= mackey_glass(n_timesteps=8000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    param_grid = {
            'hidden_layer_size': [20, 50],
            'input_scaling': [1.],
            'bias_scaling': [10.],
            'input_activation': ['identity'],
            'random_state': [42],
            'spectral_radius': [0.],
            'reservoir_activation': ['tanh'],
            'alpha': [1e-2, 1e-5],
    }
    esn = GridSearchCV(ESNRegressor(), param_grid)
    esn.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    np.testing.assert_equal(esn.best_estimator_.requires_sequence, False)


def test_esn_regressor_requires_sequence():
    print('\ntest_esn_regressor_requires_sequence():')
    X, y= mackey_glass(n_timesteps=8000)
    X_train = np.empty(shape=(10, ), dtype=object)
    y_train = np.empty(shape=(10, ), dtype=object)
    X_test = np.empty(shape=(10, ), dtype=object)
    y_test = np.empty(shape=(10, ), dtype=object)
    splitter = TimeSeriesSplit(n_splits=10)
    for k, (train_index, test_index) in enumerate(splitter.split(X, y)):
        X_train[k] = X[train_index].reshape(-1, 1)
        y_train[k] = y[train_index]
        X_test[k] = X[test_index].reshape(-1, 1)
        y_test[k] = y[test_index]
    param_grid = {
            'hidden_layer_size': [20, 50],
            'input_scaling': [1.],
            'bias_scaling': [10.],
            'input_activation': ['identity'],
            'random_state': [42],
            'spectral_radius': [0.],
            'reservoir_activation': ['tanh'],
            'alpha': [1e-2, 1e-5],
    }
    esn = GridSearchCV(ESNRegressor(), param_grid, scoring=make_scorer(mean_squared_error, greater_is_better=False))
    esn.fit(X_train, y_train, n_jobs=2)
    np.testing.assert_equal(esn.best_estimator_.requires_sequence, True)


def test_digits_ensemble_iterative_regression():
    print('\ntest_digits_ensemble_iterative_regression():')
    pass
    """ TODO: continue from here!!!
    X, y= load_digits(as_sequence=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    cls = ESNClassifier(
        input_to_node=FeatureUnion([
            ('tanh', InputToNode(hidden_layer_size=10, random_state=42, input_activation='identity')),
            ('bounded_relu', InputToNode(hidden_layer_size=10, random_state=42, input_activation='bounded_relu'))],),
        node_to_node=NodeToNode(hidden_layer_size=20, spectral_radius=0.0),
        regressor=IncrementalRegression(alpha=.01), requires_sequence=False)

    for X, y in zip(X_train, y_train):
        cls.partial_fit(X_train[samples, :].reshape(-1, 1), y_train[samples], classes=np.arange(3, dtype=int))
    y_predicted = cls.predict(X_test)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))

    print('score: {0}'.format(cls.score(X_test, y_test)))
    print('proba: {0}'.format(cls.predict_proba(X_test)))
    print('log_proba: {0}'.format(cls.predict_log_proba(X_test)))
    assert cls.score(X_test, y_test) >= 4./5.
    """


if __name__ == "__main__":
    test_esn_regressor_jobs()
    test_esn_regressor_requires_no_sequence()
    test_esn_regressor_requires_sequence()
    test_digits_ensemble_iterative_regression()
