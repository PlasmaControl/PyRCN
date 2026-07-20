"""The bridge builds torch modules from fitted blocks that reproduce the
block's NumPy ``transform``, and correctly classifies fast-path membership.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from pyrcn.nn._bridge import (
    build_input_map, build_readout, build_reservoir, input_is_backable,
    node_is_backable, regressor_is_backable)
from pyrcn.base.blocks import (BatchIntrinsicPlasticity, EulerNodeToNode,
                               HebbianNodeToNode, InputToNode, NodeToNode,
                               PredefinedWeightsInputToNode,
                               PredefinedWeightsNodeToNode)
from pyrcn.linear_model import IncrementalRegression

RTOL, ATOL = 1e-8, 1e-11


def _seq(length: int, features: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).normal(size=(length, features)) * 0.3


def test_input_is_backable_classification() -> None:
    assert input_is_backable(InputToNode())
    assert input_is_backable(
        PredefinedWeightsInputToNode(
            predefined_input_weights=np.zeros((3, 5))))
    # bespoke transform -> fallback
    assert not input_is_backable(BatchIntrinsicPlasticity())
    assert not input_is_backable(FeatureUnion([("a", InputToNode())]))


def test_node_is_backable_classification() -> None:
    assert node_is_backable(NodeToNode())
    assert node_is_backable(EulerNodeToNode())
    assert node_is_backable(HebbianNodeToNode())
    assert node_is_backable(
        PredefinedWeightsNodeToNode(
            predefined_recurrent_weights=np.zeros((5, 5))))
    assert not node_is_backable(InputToNode())


def test_regressor_is_backable_classification() -> None:
    assert regressor_is_backable(IncrementalRegression())
    assert not regressor_is_backable(IncrementalRegression(normalize=True))
    assert not regressor_is_backable(Ridge())


def test_build_input_map_matches_transform() -> None:
    i2n = InputToNode(
        hidden_layer_size=20, input_scaling=0.5, input_shift=0.2,
        bias_scaling=2.0, bias_shift=-0.1, input_activation="tanh",
        random_state=42)
    X = _seq(15, 6, 0)
    i2n.fit(X)
    expected = i2n.transform(X)

    fm = build_input_map(i2n, dtype=torch.float64)
    got = fm(torch.as_tensor(X, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_build_reservoir_matches_transform_leaky() -> None:
    hidden = 20
    n2n = NodeToNode(
        hidden_layer_size=hidden, spectral_radius=0.9, leakage=0.6,
        reservoir_activation="tanh", random_state=42)
    n2n.fit(np.zeros((2 * hidden, hidden)))
    X = _seq(30, hidden, 1)
    expected = n2n.transform(X)

    res = build_reservoir(n2n, dtype=torch.float64)
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))
    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_build_reservoir_matches_transform_bidirectional() -> None:
    hidden = 20
    n2n = NodeToNode(
        hidden_layer_size=hidden, spectral_radius=0.9, leakage=0.6,
        reservoir_activation="tanh", bidirectional=True, random_state=3)
    n2n.fit(np.zeros((2 * hidden, hidden)))
    X = _seq(30, hidden, 1)
    expected = n2n.transform(X)                      # (30, 2*hidden)

    res = build_reservoir(n2n, dtype=torch.float64)
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))
    assert states.shape == (1, 30, 2 * hidden)
    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_build_reservoir_matches_transform_euler() -> None:
    hidden = 20
    euler = EulerNodeToNode(
        hidden_layer_size=hidden, recurrent_scaling=0.5, gamma=0.01,
        epsilon=0.1, reservoir_activation="tanh", random_state=42)
    euler.fit(np.zeros((2 * hidden, hidden)))
    X = _seq(30, hidden, 1)
    expected = euler.transform(X)

    res = build_reservoir(euler, dtype=torch.float64)
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))
    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_build_readout_matches_incremental_regression() -> None:
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(40, 8))
    y = rng.normal(size=(40, 2))
    reg = IncrementalRegression(alpha=1e-3, fit_intercept=True)
    reg.fit(Z, y)
    expected = reg.predict(Z)

    readout = build_readout(reg, dtype=torch.float64)
    readout.fit(torch.as_tensor(Z, dtype=torch.float64),
                torch.as_tensor(y, dtype=torch.float64))
    got = readout.predict(torch.as_tensor(Z, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)
