"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np

import pytest

from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.base import InputToNode, NodeToNode


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


def test_node_to_node_dense():
    print('\ntest_node_to_node_dense():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    n2n = NodeToNode(hidden_layer_size=5, sparsity=1., activation='tanh', spectral_radius=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    print(n2n.transform(i2n.transform(X)))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)


def test_node_to_node_sparse():
    print('\ntest_node_to_node_sparse():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5, activation='tanh', input_scaling=1., bias_scaling=1.,
                      random_state=42)
    n2n = NodeToNode(hidden_layer_size=5, sparsity=2/5, activation='tanh', spectral_radius=1., bias_scaling=1.,
                     random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    print(n2n.transform(i2n.transform(X)))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)


def test_transform_bounded_relu():
    print('\ntest_transform_bounded_relu():')
    rs = np.random.RandomState(42)
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., activation='bounded_relu', input_scaling=1., bias_scaling=1.,
                      random_state=rs)
    X = rs.uniform(low=-1., high=1., size=(10, 3))
    i2n.fit(X)
    y = i2n.transform(X)
    print('tests bounded relu')
    print(y)
    assert y.shape == (10, 5)
