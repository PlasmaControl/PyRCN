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

from pyrcn import extreme_learning_machine
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor, InputToNode


X_iris, y_iris = load_iris(return_X_y=True)


def test_input_to_node_dense():
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='tanh', input_scaling=1., bias=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights)
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_input_to_node_sparse():
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5, activation='tanh', input_scaling=1., bias=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights.toarray())
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_transform_bounded_relu():
    rs = np.random.RandomState(42)
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='bounded_relu', input_scaling=1., bias=1., random_state=rs)
    X = rs.uniform(low=-1., high=1., size=(10, 3))
    y = i2n.fit_transform(X)
    print(y)
    assert y.shape == (10, 5)


def test_hidden_layer_size():
    cls = ELMClassifier(hidden_layer_size=499, random_state=42).fit(np.array([1]).reshape(-1, 1), np.array([1]))
    assert cls.input_weights_.shape[0] == cls.get_params()['hidden_layer_size']


def test_iris():
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5, random_state=42)
    lb = LabelBinarizer()
    y_train_numeric = lb.fit_transform(y_train)
    classifier = ELMClassifier(hidden_layer_size=10, random_state=42)
    classifier.fit(X_train, y_train_numeric)
    y_predicted_numeric = classifier.predict(X_test)
    y_predicted = lb.inverse_transform(y_predicted_numeric)

    for record in range(len(y_test)):
        # print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))
        assert y_predicted[record] == y_test[record]  # this combination fits coincidental - not a general assumption
