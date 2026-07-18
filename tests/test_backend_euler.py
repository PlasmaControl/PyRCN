"""Parity: torch Euler reservoir vs the legacy EulerNodeToNode.

Recurrent weights come from a real ``EulerNodeToNode`` (antisymmetric-uniform,
possibly sparse) and are injected into the torch module, so initializations are
identical and this compares the Euler recurrence only. float64, tight tol.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.backend import EulerReservoir, antisymmetric_recurrent_weights
from pyrcn.base.blocks import EulerNodeToNode

RTOL, ATOL = 1e-8, 1e-11


def _fit_euler(hidden_size: int, *, recurrent_scaling: float, gamma: float,
               epsilon: float, activation: str, random_state: int,
               k_rec: int | None = None) -> EulerNodeToNode:
    euler = EulerNodeToNode(
        hidden_layer_size=hidden_size, recurrent_scaling=recurrent_scaling,
        gamma=gamma, epsilon=epsilon, reservoir_activation=activation,
        k_rec=k_rec, random_state=random_state)
    euler.fit(np.zeros((2 * hidden_size, hidden_size)))
    return euler


def _dense(W: object) -> np.ndarray:
    return W.toarray() if hasattr(W, "toarray") else np.asarray(W)


@pytest.mark.parametrize("activation", ["tanh", "identity", "relu"])
@pytest.mark.parametrize("recurrent_scaling,gamma,epsilon",
                         [(1.0, 0.001, 0.01), (0.8, 0.01, 0.05)])
def test_euler_parity(activation: str, recurrent_scaling: float, gamma: float,
                      epsilon: float) -> None:
    hidden_size, length = 20, 40
    euler = _fit_euler(
        hidden_size, recurrent_scaling=recurrent_scaling, gamma=gamma,
        epsilon=epsilon, activation=activation, random_state=42)
    X = np.random.RandomState(0).normal(size=(length, hidden_size)) * 0.3

    expected = euler.transform(X)

    res = EulerReservoir(
        hidden_size=hidden_size, recurrent_scaling=recurrent_scaling,
        gamma=gamma, epsilon=epsilon, activation=activation,
        dtype=torch.float64)
    res.set_recurrent_weights(_dense(euler._recurrent_weights))
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))

    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_antisymmetric_init_is_antisymmetric() -> None:
    W = antisymmetric_recurrent_weights(
        20, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert W.shape == (20, 20)
    assert torch.allclose(W, -W.T)


def test_antisymmetric_init_reproducible() -> None:
    W1 = antisymmetric_recurrent_weights(
        16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    W2 = antisymmetric_recurrent_weights(
        16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    assert torch.equal(W1, W2)
