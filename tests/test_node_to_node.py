"""Testing for blocks.node_to_node module."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence
from sklearn.exceptions import NotFittedError
from sklearn.utils.extmath import safe_sparse_dot

from pyrcn.base._base import _unitary_spectral_radius
from pyrcn.base.blocks import (EulerNodeToNode, HebbianNodeToNode, InputToNode,
                               NodeToNode, PredefinedWeightsNodeToNode)


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


def test_euler_node_to_node_k_rec() -> None:
    print('\ntest_euler_node_to_node_k_rec():')
    X = np.zeros(shape=(10, 5))
    n2n = EulerNodeToNode(hidden_layer_size=5, k_rec=2, random_state=42)
    n2n.fit(X)
    assert n2n._recurrent_weights.shape == (5, 5)
    out = n2n.transform(X)
    assert out.shape == (10, 5)


def test_euler_node_to_node_not_fitted() -> None:
    print('\ntest_euler_node_to_node_not_fitted():')
    n2n = EulerNodeToNode(hidden_layer_size=5, random_state=42)
    with pytest.raises(NotFittedError):
        n2n.transform(np.zeros(shape=(10, 5)))


_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_euler_node_to_node_parity() -> None:
    """P6/P8: EulerNodeToNode.transform is bit-identical after hoisting the
    effective-matrix rebuild out of the loop and using np.dot for dense."""
    print('\ntest_euler_node_to_node_parity():')
    ref = dict(np.load(os.path.join(_FIXTURES, "blocks_parity_p6.npz")))
    i = 0
    for hls in (5, 20, 50):
        for rs_ in (1, 42):
            for act in ('tanh', 'identity', 'relu'):
                for scal in (0.5, 1.0):
                    for gamma in (0.001, 0.01):
                        for eps in (0.01, 0.1):
                            rng = np.random.RandomState(rs_)
                            X = rng.normal(size=(15, hls))
                            n2n = EulerNodeToNode(
                                hidden_layer_size=hls,
                                reservoir_activation=act,
                                recurrent_scaling=scal, gamma=gamma,
                                epsilon=eps, random_state=rs_)
                            n2n.fit(X)
                            assert np.array_equal(n2n.transform(X),
                                                  ref[f"o{i}"])
                            i += 1
    assert i == len(ref)


def test_node_to_node_parity() -> None:
    """P8: NodeToNode.transform is bit-identical when using np.dot for dense
    recurrent weights (and safe_sparse_dot for the sparse path)."""
    print('\ntest_node_to_node_parity():')
    ref = dict(np.load(os.path.join(_FIXTURES, "blocks_parity_p8.npz")))
    i = 0
    for hls in (5, 20, 50):
        for sparsity in (1.0, 0.4):
            for rs_ in (1, 42):
                for act in ('tanh', 'identity', 'relu'):
                    for sr in (0.5, 0.9):
                        for leak in (1.0, 0.5):
                            for bidir in (False, True):
                                rng = np.random.RandomState(rs_)
                                X = rng.normal(size=(15, hls))
                                n2n = NodeToNode(
                                    hidden_layer_size=hls, sparsity=sparsity,
                                    reservoir_activation=act,
                                    spectral_radius=sr, leakage=leak,
                                    bidirectional=bidir, random_state=rs_)
                                n2n.fit(X)
                                assert np.array_equal(n2n.transform(X),
                                                      ref[f"o{i}"])
                                i += 1
    assert i == len(ref)


def test_node_to_node_dense_dot_matches_safe_sparse_dot() -> None:
    """P8: np.dot equals safe_sparse_dot exactly for dense weights."""
    print('\ntest_node_to_node_dense_dot_matches_safe_sparse_dot():')
    rng = np.random.RandomState(0)
    v = rng.normal(size=(50,))
    W = rng.normal(size=(50, 50))
    assert np.array_equal(safe_sparse_dot(v, W), np.dot(v, W))


def test_unitary_spectral_radius_no_convergence() -> None:
    print('\ntest_unitary_spectral_radius_no_convergence():')
    rs = np.random.RandomState(42)
    weights = rs.normal(size=(10, 10))
    exc = ArpackNoConvergence(
        "no convergence", np.array([2.0, -1.0]), np.zeros((10, 0)))
    with patch("pyrcn.base._base.eigens", side_effect=exc):
        result = _unitary_spectral_radius(weights, rs)
    np.testing.assert_allclose(result, weights / 2.0)
