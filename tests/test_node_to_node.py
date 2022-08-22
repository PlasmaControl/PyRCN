"""Testing for blocks.node_to_node module."""
import numpy as np
import pytest
from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.base.blocks import (InputToNode, NodeToNode,
                               AttentionWeightsNodeToNode,
                               PredefinedWeightsNodeToNode, HebbianNodeToNode)


def test_input_to_node_invalid_spectral_radius() -> None:
    print('\ntest_input_to_node_invalid_spectral_radius():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        n2n = NodeToNode(spectral_radius=-1e-5)
        n2n.fit(X)


def test_node_to_node_invalid_activation() -> None:
    print('\ntest_node_to_node_invalid_activation():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        n2n = NodeToNode(reservoir_activation="test")
        n2n.fit(X)


def test_node_to_node_invalid_hls() -> None:
    print('\ntest_node_to_node_invalid_hls():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        n2n = NodeToNode(hidden_layer_size=0)
        X = np.zeros(shape=(10, 3))
        n2n.fit(X)


def test_node_to_node_invalid_sparsity() -> None:
    print('\ntest_node_to_node_invalid_sparsity():')
    X = np.zeros(shape=(10, 500))
    with pytest.raises(ValueError):
        n2n = NodeToNode(sparsity=1.1)
        n2n.fit(X)
    with pytest.raises(ValueError):
        n2n = NodeToNode(sparsity=0.0)
        n2n.fit(X)
    with pytest.raises(ValueError):
        n2n = NodeToNode(k_rec=-1)
        n2n.fit(X)
    with pytest.raises(ValueError):
        n2n = NodeToNode(k_rec=500)
        n2n.fit(X)


def test_attention_weights_node_to_node() -> None:
    print('\ntest_attention_weights_node_to_node():')
    X = np.zeros(shape=(10, 3))
    weights = np.random.rand(3, 5)
    with pytest.raises(ValueError):
        n2n = AttentionWeightsNodeToNode(
            recurrent_attention_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(5, 3)
    with pytest.raises(ValueError):
        n2n = AttentionWeightsNodeToNode(
            recurrent_attention_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(5, )
    with pytest.raises(ValueError):
        n2n = AttentionWeightsNodeToNode(
            recurrent_attention_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(3, 3)
    n2n = AttentionWeightsNodeToNode(
        recurrent_attention_weights=weights, reservoir_activation='tanh',
        spectral_radius=1.)
    n2n.fit(X)
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (3, 3)
    assert n2n.__sizeof__() != 0
    assert n2n.recurrent_weights is not None


def test_predefined_weights_node_to_node() -> None:
    print('\ntest_predefined_weights_node_to_node():')
    X = np.zeros(shape=(10, 3))
    weights = np.random.rand(3, 5)
    with pytest.raises(AssertionError):
        n2n = PredefinedWeightsNodeToNode(
            predefined_recurrent_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(5, 3)
    with pytest.raises(AssertionError):
        n2n = PredefinedWeightsNodeToNode(
            predefined_recurrent_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(5, )
    with pytest.raises(ValueError):
        n2n = PredefinedWeightsNodeToNode(
            predefined_recurrent_weights=weights, reservoir_activation='tanh',
            spectral_radius=1.)
        n2n.fit(X)
    weights = np.random.rand(3, 3)
    n2n = PredefinedWeightsNodeToNode(
        predefined_recurrent_weights=weights, reservoir_activation='tanh',
        spectral_radius=1.)
    n2n.fit(X)
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (3, 3)
    assert n2n.__sizeof__() != 0
    assert n2n.recurrent_weights is not None


def test_node_to_node_dense() -> None:
    print('\ntest_node_to_node_dense():')
    n2n = NodeToNode(
        hidden_layer_size=5, sparsity=1., reservoir_activation='tanh',
        spectral_radius=1., random_state=42)
    X = np.zeros(shape=(10, 5))
    n2n.fit(X)
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert n2n.__sizeof__() != 0
    assert n2n.recurrent_weights is not None


def test_node_to_node_sparse() -> None:
    print('\ntest_node_to_node_sparse():')
    X = np.zeros(shape=(10, 5))
    n2n = NodeToNode(
        hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh',
        spectral_radius=1., random_state=42)
    n2n.fit(X)
    assert n2n._recurrent_weights.shape == (5, 5)
    n2n = NodeToNode(
        hidden_layer_size=5, k_rec=2, reservoir_activation='tanh',
        spectral_radius=1., random_state=42)
    n2n.fit(X)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert n2n.__sizeof__() != 0
    assert n2n.recurrent_weights is not None


def test_node_to_node_bidirectional() -> None:
    print('\ntest_node_to_node_bidirectional():')
    X = np.zeros(shape=(10, 5))
    with pytest.raises(ValueError):
        n2n = NodeToNode(
            hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh',
            spectral_radius=1., bidirectional="True", random_state=42)
        n2n.fit(X)
    n2n = NodeToNode(
        hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh',
        spectral_radius=1., bidirectional=True, random_state=42)
    n2n.fit(X)
    n2n.transform(X)
    assert n2n._recurrent_weights.shape == (5, 5)


def test_node_to_node_invalid_leakage() -> None:
    print('\ntest_node_to_node_bidirectional():')
    X = np.zeros(shape=(10, 5))
    with pytest.raises(ValueError):
        n2n = NodeToNode(
            hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh',
            spectral_radius=1., leakage=1.1, random_state=42)
        n2n.fit(X)
    with pytest.raises(ValueError):
        n2n = NodeToNode(
            hidden_layer_size=5, sparsity=2/5, reservoir_activation='tanh',
            spectral_radius=1., leakage=0, random_state=42)
        n2n.fit(X)


def test_node_to_node_hebbian() -> None:
    print('\ntest_node_to_node_hebbian():')
    i2n = InputToNode(hidden_layer_size=5, sparsity=2/5,
                      input_activation='tanh', input_scaling=1.,
                      bias_scaling=1., random_state=42)
    X = np.zeros(shape=(10, 3))
    i2n.fit(X)
    n2n = HebbianNodeToNode(hidden_layer_size=5, sparsity=2/5,
                            reservoir_activation='tanh', spectral_radius=1.,
                            random_state=42, learning_rate=0.01)
    n2n.fit(i2n.transform(X))
    n2n = HebbianNodeToNode(hidden_layer_size=5, sparsity=2/5,
                            reservoir_activation='tanh', spectral_radius=1.,
                            random_state=42, learning_rate=0.01,
                            training_method="anti_hebbian")
    n2n.fit(i2n.transform(X))
    n2n = HebbianNodeToNode(hidden_layer_size=5, sparsity=2/5,
                            reservoir_activation='tanh', spectral_radius=1.,
                            random_state=42, learning_rate=0.01,
                            training_method="oja")
    n2n.fit(i2n.transform(X))
    n2n = HebbianNodeToNode(hidden_layer_size=5, sparsity=2/5,
                            reservoir_activation='tanh', spectral_radius=1.,
                            random_state=42, learning_rate=0.01,
                            training_method="anti_oja")
    n2n.fit(i2n.transform(X))
    i2n_hidden = i2n.transform(X)
    print(n2n.transform(i2n_hidden))
    print(n2n._recurrent_weights)
    assert n2n._recurrent_weights.shape == (5, 5)
    assert safe_sparse_dot(
        i2n.transform(X), n2n._recurrent_weights).shape == (10, 5)
