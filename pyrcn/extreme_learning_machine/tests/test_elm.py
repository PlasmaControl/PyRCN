"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np

import pytest

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor, InputToNode


X_iris, y_iris = load_iris(return_X_y=True)


def test_input_to_node_dense():
    print('\ntest_input_to_node_dense():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights)
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_input_to_node_sparse():
    print('\ntest_input_to_node_sparse():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5, activation='tanh', input_scaling=1., bias_scaling=1.,
                      random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights.toarray())
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_transform_bounded_relu():
    print('\ntest_transform_bounded_relu():')
    rs = np.random.RandomState(42)
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='bounded_relu', input_scaling=1., bias_scaling=1.,
                      random_state=rs)
    X = rs.uniform(low=-1., high=1., size=(10, 3))
    i2n.fit(X)
    y = i2n.transform(X)
    print('test bounded relu')
    print(y)
    assert y.shape == (10, 5)


def test_input_to_node_type():
    print('\ntest_input_to_node_type():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='bounded_relu', input_scaling=1., bias_scaling=1.,
                      random_state=42)


def test_elm_regressor_linear():
    print('\ntest_elm_regressor_linear():')
    X_train, X_test, y_train, y_test =\
        train_test_split(np.linspace(0, 10, 200), np.linspace(0, 10, 200)*3-2, test_size=10)
    elm = ELMRegressor(input_to_nodes=[('default', InputToNode(bias_scaling=10.))], random_state=42)
    elm.fit(X_train.reshape(-1, 1), y_train)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    # print("test: {0} train: {1}".format(y_test, y_elm))
    np.testing.assert_allclose(y_test, y_elm, rtol=1e-2)


def test_elm_regressor_sine():
    print('\ntest_elm_regressor_sine():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    elm = ELMRegressor(input_to_nodes=[('default', InputToNode(bias_scaling=10.))], random_state=42)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    # print("test: {0} train: {1}".format(y_test, y_elm))
    # print(elm.get_params())
    np.testing.assert_allclose(y_test, y_elm, rtol=1e-2)


def test_iris():
    print('\ntest_iris():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5, random_state=42)
    lb = LabelBinarizer()
    y_train_numeric = lb.fit_transform(y_train)
    cls = ELMClassifier([('default', InputToNode(hidden_layer_size=10, random_state=42))], random_state=42)
    cls.fit(X_train, y_train_numeric)
    y_predicted = cls.predict(X_test)
    # y_predicted = lb.inverse_transform(y_predicted_numeric)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))

    assert cls.score(X_test, y_test) >= 4./5.


def test_iris_ensemble_iterative_regression():
    print('\ntest_iris_ensemble_iterative_regression():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5, random_state=42)
    lb = LabelBinarizer()
    y_train_numeric = lb.fit_transform(y_train)
    cls = ELMClassifier(
        input_to_nodes=[
            ('tanh', InputToNode(hidden_layer_size=10, random_state=42, activation='tanh')),
            ('bounded_relu', InputToNode(hidden_layer_size=10, random_state=42, activation='bounded_relu'))],
        regressor=IncrementalRegression(alpha=.01),
        random_state=42)
    cls.fit(X_train, y_train_numeric)
    y_predicted = cls.predict(X_test)
    # y_predicted = lb.inverse_transform(y_predicted_numeric)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))

    assert cls.score(X_test, y_test) >= 4./5.
