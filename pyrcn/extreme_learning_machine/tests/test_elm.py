"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np

import pytest

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from pyrcn.base import InputToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor


X_iris, y_iris = load_iris(return_X_y=True)


def test_elm_regressor_jobs():
    print('\ntest_elm_regressor_sine():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    elm = ELMRegressor(
        input_to_nodes=[('default', InputToNode(bias_scaling=10.))],
        regressor=Ridge(alpha=.0001),
        random_state=42)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print("tests: {0} train: {1}".format(y_test, y_elm))
    print(elm.get_params())
    np.testing.assert_allclose(y_test, y_elm, rtol=1e-2)


def test_iris_ensemble_iterative_regression():
    print('\ntest_iris_ensemble_iterative_regression():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5, random_state=42)
    cls = ELMClassifier(
        input_to_nodes=[
            ('tanh', InputToNode(hidden_layer_size=10, random_state=42, activation='tanh')),
            ('bounded_relu', InputToNode(hidden_layer_size=10, random_state=42, activation='bounded_relu'))],
        regressor=IncrementalRegression(alpha=.01),
        random_state=42)

    for samples in np.split(np.arange(0, X_train.shape[0]), 5):
        cls.partial_fit(X_train[samples, :], y_train[samples])
    y_predicted = cls.predict(X_test)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))

    print('score: %f' % cls.score(X_test, y_test))
    assert cls.score(X_test, y_test) >= 4./5.
