"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np

import pytest

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from pyrcn.base.blocks import InputToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor


X_iris, y_iris = load_iris(return_X_y=True)


def test_elm_regressor_jobs():
    print('\ntest_elm_regressor_jobs():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    param_grid = {
        'input_to_node': [
            [('default', InputToNode(bias_scaling=10., hidden_layer_size=20, random_state=42))],
            [('default', InputToNode(bias_scaling=10., hidden_layer_size=50, random_state=42))]],
        'regressor': [
            IncrementalRegression(alpha=.0001),
            IncrementalRegression(alpha=.01)],
        'random_state': [42]
    }
    elm = GridSearchCV(ELMRegressor(), param_grid)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print("tests - elm:\n sin | cos \n {0}".format(y_test-y_elm))
    print("best_params_: ".format(elm.best_params_))
    print("best_score: ".format(elm.best_score_))
    np.testing.assert_allclose(y_test, y_elm, atol=1e-1)


def test_elm_regressor_chunk():
    print('\ntest_elm_regressor_chunk():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    param_grid = {
            'input_to_node__hidden_layer_size': [20, 50],
            'input_to_node__input_scaling': [1.],
            'input_to_node__bias_scaling': [10.],
            'input_to_node__activation': ['tanh'],
            'input_to_node__random_state': [42],
            'chunk_size': [500],
            'regressor__alpha': [1e-2, 1e-5],
            'random_state': [42]
    }
    elm = GridSearchCV(ELMRegressor(), param_grid)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print("tests - elm:\n sin | cos \n {0}".format(y_test-y_elm))
    print("best_params_: ".format(elm.best_params_))
    print("best_score: ".format(elm.best_score_))
    np.testing.assert_allclose(y_test, y_elm, atol=1e-1)


def test_iris_ensemble_iterative_regression():
    print('\ntest_iris_ensemble_iterative_regression():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5, random_state=42)

    cls = ELMClassifier(
        input_to_node=FeatureUnion([('tanh',InputToNode(hidden_layer_size=10, random_state=42, input_activation='tanh')), 
                                    ('bounded_relu', InputToNode(hidden_layer_size=10, random_state=42, input_activation='bounded_relu'))]),
        regressor=IncrementalRegression(alpha=.01),
        random_state=42)

    for samples in np.split(np.arange(0, X_train.shape[0]), 5):
        cls.partial_fit(X_train[samples, :], y_train[samples], classes=np.arange(3, dtype=int))
    y_predicted = cls.predict(X_test)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))

    print('score: {0}'.format(cls.score(X_test, y_test)))
    print('proba: {0}'.format(cls.predict_proba(X_test)))
    print('log_proba: {0}'.format(cls.predict_log_proba(X_test)))
    assert cls.score(X_test, y_test) >= 4./5.


if __name__ == "__main__":
    test_elm_regressor_jobs()
    test_elm_regressor_chunk()
    test_iris_ensemble_iterative_regression()
