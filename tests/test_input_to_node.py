"""Testing for blocks.input_to_node module."""
import scipy
import numpy as np
import pytest
from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.base.blocks import (InputToNode, PredefinedWeightsInputToNode,
                               BatchIntrinsicPlasticity)


def test_input_to_node_invalid_bias_scaling() -> None:
    print('\ntest_input_to_node_invalid_bias_scaling():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        i2n = InputToNode(bias_scaling=-1e-5)
        i2n.fit(X)


def test_input_to_node_invalid_input_scaling() -> None:
    print('\ntest_input_to_node_invalid_input_scaling():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        i2n = InputToNode(input_scaling=0)
        i2n.fit(X)


def test_input_to_node_invalid_activation() -> None:
    print('\ntest_input_to_node_invalid_activation():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        i2n = InputToNode(input_activation="test")
        i2n.fit(X)


def test_input_to_node_invalid_hls() -> None:
    print('\ntest_input_to_node_invalid_hls():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        i2n = InputToNode(hidden_layer_size=0)
        X = np.zeros(shape=(10, 3))
        i2n.fit(X)


def test_input_to_node_invalid_sparsity() -> None:
    print('\ntest_input_to_node_invalid_sparsity():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        i2n = InputToNode(sparsity=1.1)
        i2n.fit(X)
    with pytest.raises(ValueError):
        i2n = InputToNode(sparsity=0.0)
        i2n.fit(X)
    with pytest.raises(ValueError):
        i2n = InputToNode(k_in=-1)
        i2n.fit(X)
    with pytest.raises(ValueError):
        i2n = InputToNode(k_in=500)
        i2n.fit(X)


def test_predefined_weights_input_to_node() -> None:
    print('\ntest_predefined_weights_input_to_node():')
    X = np.zeros(shape=(10, 3))
    weights = np.random.rand(5, 5)
    with pytest.raises(ValueError):
        i2n = PredefinedWeightsInputToNode(
            predefined_input_weights=weights, input_activation='tanh',
            input_scaling=1., bias_scaling=1., random_state=42)
        i2n.fit(X)
    weights = np.random.rand(5, )
    with pytest.raises(ValueError):
        i2n = PredefinedWeightsInputToNode(
            predefined_input_weights=weights, input_activation='tanh',
            input_scaling=1., bias_scaling=1., random_state=42)
        i2n.fit(X)
    weights = np.random.rand(3, 5)
    i2n = PredefinedWeightsInputToNode(
        predefined_input_weights=weights, input_activation='tanh',
        input_scaling=1., bias_scaling=1., random_state=42)
    i2n.fit(X)
    print(i2n._input_weights)
    assert i2n._input_weights.shape == (3, 5)
    assert i2n.__sizeof__() != 0
    assert i2n.input_weights is not None
    assert i2n.bias_weights is not None


def test_bip_dresden() -> None:
    print('\ntest_bip_dresden()')
    rs = np.random.RandomState(42)
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='uniform', algorithm='dresden')
    X = rs.normal(size=(1000, 1))
    i2n.fit(X[:1000, :])
    y = i2n.transform(X)
    y_test = y[(y > -.75) & (y < .75)] / 1.5 + .5

    statistic, pvalue = scipy.stats.ks_1samp(y_test, scipy.stats.uniform.cdf)
    assert statistic < pvalue
    print("Kolmogorov-Smirnov does not reject H_0:"
          "y is uniformly distributed in [-.75, .75]")


def test_bip_run_neumann() -> None:
    print('\ntest_bip_run_neumann()')
    rs = np.random.RandomState(42)
    X = rs.normal(size=(1000, 1))
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='uniform', algorithm='neumann')
    i2n.fit(X.reshape(-1, 1))
    i2n.transform(X.reshape(-1, 1))


def test_bip_invalid_params() -> None:
    print('\ntest_bip_invalid_params()')
    rs = np.random.RandomState(42)
    X = rs.normal(size=(1000, 1))
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='test', algorithm='neumann')
    with pytest.raises(ValueError):
        i2n.fit(X.reshape(-1, 1))
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='relu', random_state=rs,
        distribution='uniform', algorithm='dresden')
    with pytest.raises(ValueError):
        i2n.fit(X.reshape(-1, 1))
    i2n = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='uniform', algorithm='test')
    with pytest.raises(ValueError):
        i2n.fit(X.reshape(-1, 1))


def test_input_to_node_dense() -> None:
    print('\ntest_input_to_node_dense():')
    i2n = InputToNode(
        hidden_layer_size=5, sparsity=1., input_activation='tanh',
        input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights)
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)
    assert i2n.__sizeof__() != 0
    assert i2n.input_weights is not None
    assert i2n.bias_weights is not None


def test_input_to_node_sparse() -> None:
    print('\ntest_input_to_node_sparse():')
    i2n = InputToNode(
        hidden_layer_size=5, sparsity=2/5, input_activation='tanh',
        input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights.toarray())
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)
    i2n = InputToNode(
        hidden_layer_size=5, k_in=2, input_activation='tanh',
        input_scaling=1., bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    print(i2n._input_weights.toarray())
    assert i2n._input_weights.shape == (3, 5)
    assert safe_sparse_dot(X, i2n._input_weights).shape == (10, 5)
    assert i2n.__sizeof__() != 0
    assert i2n.input_weights is not None
    assert i2n.bias_weights is not None
