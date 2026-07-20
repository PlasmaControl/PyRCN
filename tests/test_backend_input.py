"""Parity: torch InputFeatureMap vs the legacy InputToNode.transform.

Input weights and bias come from a real ``InputToNode`` and are injected into
the torch module, so this compares the feature-map computation (scaling/shift +
activation) only. float64, tight tolerance. Plus torch-native init checks.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.nn import InputFeatureMap
from pyrcn.nn.init import (bernoulli_input_weights, uniform_bias_weights,
                           uniform_input_weights)
from pyrcn.base.blocks import InputToNode

RTOL, ATOL = 1e-8, 1e-11


def _fit_i2n(in_features: int, hidden_size: int, *, input_scaling: float,
             input_shift: float, bias_scaling: float, bias_shift: float,
             activation: str, random_state: int) -> InputToNode:
    i2n = InputToNode(
        hidden_layer_size=hidden_size, input_scaling=input_scaling,
        input_shift=input_shift, bias_scaling=bias_scaling,
        bias_shift=bias_shift, input_activation=activation,
        random_state=random_state)
    i2n.fit(np.zeros((3, in_features)))
    return i2n


def _dense(W: object) -> np.ndarray:
    return W.toarray() if hasattr(W, "toarray") else np.asarray(W)


@pytest.mark.parametrize("activation",
                         ["tanh", "identity", "relu", "logistic",
                          "bounded_relu"])
@pytest.mark.parametrize(
    "input_scaling,input_shift,bias_scaling,bias_shift",
    [(1.0, 0.0, 1.0, 0.0), (0.5, 0.2, 2.0, -0.1)])
def test_input_parity(activation: str, input_scaling: float,
                      input_shift: float, bias_scaling: float,
                      bias_shift: float) -> None:
    in_features, hidden_size, length = 6, 20, 15
    i2n = _fit_i2n(
        in_features, hidden_size, input_scaling=input_scaling,
        input_shift=input_shift, bias_scaling=bias_scaling,
        bias_shift=bias_shift, activation=activation, random_state=42)
    X = np.random.RandomState(0).normal(size=(length, in_features)) * 0.3

    expected = i2n.transform(X)

    fm = InputFeatureMap(
        in_features, hidden_size, input_scaling=input_scaling,
        input_shift=input_shift, bias_scaling=bias_scaling,
        bias_shift=bias_shift, activation=activation, dtype=torch.float64)
    fm.set_input_weights(_dense(i2n._input_weights), i2n._bias_weights)
    got = fm(torch.as_tensor(X, dtype=torch.float64))

    np.testing.assert_allclose(got.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_uniform_input_weights_shape_and_range() -> None:
    W = uniform_input_weights(
        6, 20, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert W.shape == (6, 20)
    assert float(W.max()) < 1.0 and float(W.min()) >= -1.0


def test_sparse_input_weights_fan_in() -> None:
    W = uniform_input_weights(
        10, 20, fan_in=3, generator=torch.Generator().manual_seed(0),
        dtype=torch.float64)
    nonzeros_per_column = (W != 0).sum(dim=0)
    assert int(nonzeros_per_column.max()) == 3
    assert int(nonzeros_per_column.min()) == 3


def test_bernoulli_input_weights_are_signed_constants() -> None:
    W = bernoulli_input_weights(
        8, 20, value=0.5, generator=torch.Generator().manual_seed(0),
        dtype=torch.float64)
    assert set(torch.unique(W).tolist()) <= {0.5, -0.5}


def test_uniform_bias_shape_and_range() -> None:
    b = uniform_bias_weights(
        20, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert b.shape == (20,)
    assert float(b.max()) < 1.0 and float(b.min()) >= -1.0


def test_input_feature_map_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        InputFeatureMap(6, 20, activation="nope", dtype=torch.float64)


def test_set_input_trainable_toggles() -> None:
    fm = InputFeatureMap(6, 20, dtype=torch.float64)
    assert not fm.weight.requires_grad and not fm.bias.requires_grad
    fm.set_input_trainable(True)
    assert fm.weight.requires_grad and fm.bias.requires_grad
    fm.set_input_trainable(False)
    assert not fm.weight.requires_grad and not fm.bias.requires_grad
