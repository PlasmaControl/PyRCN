"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pytest

from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.base import InputToNode, NodeToNode, BatchIntrinsicPlasticity, HebbianNodeToNode


def test_input_to_node_dense():
    print('\ntest_input_to_node_dense():')
    i2n = InputToNode(
        hidden_layer_size=5, sparsity=1., input_activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights)
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_input_to_node_sparse():
    print('\ntest_input_to_node_sparse():')
    i2n = InputToNode(
        hidden_layer_size=5, sparsity=2/5, input_activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights.toarray())
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)


def test_node_to_node_dense():
    print('\ntest_node_to_node_dense():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., input_activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    n2n = NodeToNode(hidden_layer_size=5, sparsity=1., reservoir_activation='tanh', spectral_radius=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    i2n_hidden = i2n.transform(X)
    print(n2n.transform(i2n_hidden))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)


def test_node_to_node_sparse():
    print('\ntest_node_to_node_sparse():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5, input_activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    n2n = NodeToNode(hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh', spectral_radius=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    i2n_hidden = i2n.transform(X)
    print(n2n.transform(i2n_hidden))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)


def test_transform_bounded_relu():
    print('\ntest_transform_bounded_relu():')
    rs = np.random.RandomState(42)
    i2n = InputToNode(hidden_layer_size=5, sparsity=1., input_activation='bounded_relu', input_scaling=1., bias_scaling=1.,
                      random_state=rs)
    X = rs.uniform(low=-1., high=1., size=(10, 3))
    i2n.fit(X)
    y = i2n.transform(X)
    print('tests bounded relu')
    print(y)
    assert y.shape == (10, 5)


def test_bip():
    print('\ntest_bip()')
    rs = np.random.RandomState(42)
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs, distribution='uniform', algorithm='dresden')
    X = rs.normal(size=(1000, 1))
    i2n.fit(X[:1000, :])
    y = i2n.transform(X)
    y_test = y[(y > -.75) & (y < .75)] / 1.5 + .5

    statistic, pvalue = scipy.stats.ks_1samp(y_test, scipy.stats.uniform.cdf)
    assert statistic < pvalue
    print("Kolmogorov-Smirnov does not reject H_0: y is uniformly distributed in [-.75, .75]")

def test_node_to_node_hebbian():
    print('\ntest_node_to_node_hebbian():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5, input_activation='tanh', input_scaling=1., bias_scaling=1., random_state=42)
    n2n = HebbianNodeToNode(hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh', spectral_radius=1., random_state=42, learning_rate=0.01)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    i2n_hidden = i2n.transform(X)
    print(n2n.transform(i2n_hidden))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)


if __name__ == "__main__":
    test_input_to_node_dense()
    test_input_to_node_sparse()
    test_node_to_node_dense()
    test_node_to_node_sparse()
    test_node_to_node_hebbian()
    test_transform_bounded_relu()
    test_bip()
