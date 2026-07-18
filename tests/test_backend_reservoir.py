"""Parity: the torch Reservoir must equal the legacy NumPy NodeToNode.

The recurrent weights are produced by PyRCN's own ``NodeToNode`` initialization
(dense and sparse ``k_rec`` variants, already spectral-radius-normalized by
PyRCN) and injected verbatim into the torch module. The initializations are
therefore identical by construction, so this compares the *compute* (the
recurrence) only. Run in float64 for a tight tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.backend import Reservoir
from pyrcn.base.blocks import HebbianNodeToNode, NodeToNode

RTOL, ATOL = 1e-8, 1e-11


def _fit_nodetonode(hidden_size: int, *, spectral_radius: float,
                    leakage: float, activation: str, random_state: int,
                    k_rec: int | None = None,
                    bidirectional: bool = False) -> NodeToNode:
    """A real NodeToNode with PyRCN-initialized recurrent weights."""
    n2n = NodeToNode(
        hidden_layer_size=hidden_size, spectral_radius=spectral_radius,
        leakage=leakage, reservoir_activation=activation, k_rec=k_rec,
        bidirectional=bidirectional, random_state=random_state)
    # NodeToNode's input dimension equals hidden_size (the InputToNode output).
    n2n.fit(np.zeros((2 * hidden_size, hidden_size)))
    return n2n


def _dense(W: object) -> np.ndarray:
    return W.toarray() if hasattr(W, "toarray") else np.asarray(W)


@pytest.mark.parametrize("activation",
                         ["tanh", "identity", "relu", "logistic",
                          "bounded_relu"])
@pytest.mark.parametrize("k_rec", [None, 5])          # dense vs sparse init
@pytest.mark.parametrize("spectral_radius,leakage", [(1.0, 1.0), (0.9, 0.5)])
def test_reservoir_parity(activation: str, k_rec: int | None,
                          spectral_radius: float, leakage: float) -> None:
    hidden_size, length = 20, 40
    n2n = _fit_nodetonode(
        hidden_size, spectral_radius=spectral_radius, leakage=leakage,
        activation=activation, random_state=42, k_rec=k_rec)
    X = np.random.RandomState(0).normal(size=(length, hidden_size)) * 0.3

    expected = n2n._pass_through_recurrent_weights(X)

    res = Reservoir(
        hidden_size=hidden_size, spectral_radius=spectral_radius,
        leakage=leakage, activation=activation, dtype=torch.float64)
    res.set_recurrent_weights(_dense(n2n._recurrent_weights))
    states, final = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))

    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        final.squeeze(0).numpy(), expected[-1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("training_method",
                         ["hebbian", "anti_hebbian", "oja", "anti_oja"])
def test_hebbian_learned_weights_parity(training_method: str) -> None:
    # Hebbian only *learns* the weights (in fit); transform is the standard
    # recurrence, so the torch form is the standard Reservoir with the
    # NumPy-learned weights injected.
    hidden_size = 20
    hebbian = HebbianNodeToNode(
        hidden_layer_size=hidden_size, spectral_radius=0.9, leakage=0.8,
        reservoir_activation="tanh", random_state=42, learning_rate=1e-3,
        epochs=2, training_method=training_method)
    X = np.random.RandomState(0).normal(size=(30, hidden_size)) * 0.3
    hebbian.fit(X)
    expected = hebbian.transform(X)

    res = Reservoir(hidden_size=hidden_size, spectral_radius=0.9, leakage=0.8,
                    activation="tanh", dtype=torch.float64)
    res.set_recurrent_weights(_dense(hebbian._recurrent_weights))
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))
    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_reservoir_bidirectional_parity() -> None:
    hidden_size = 20
    n2n = _fit_nodetonode(
        hidden_size, spectral_radius=0.9, leakage=0.6, activation="tanh",
        random_state=3, bidirectional=True)
    X = np.random.RandomState(0).normal(size=(30, hidden_size)) * 0.3
    expected = n2n.transform(X)                      # (30, 2*hidden_size)

    res = Reservoir(hidden_size=hidden_size, spectral_radius=0.9, leakage=0.6,
                    activation="tanh", bidirectional=True, dtype=torch.float64)
    res.set_recurrent_weights(_dense(n2n._recurrent_weights))
    states, _ = res(torch.as_tensor(X, dtype=torch.float64).unsqueeze(0))

    assert states.shape == (1, 30, 2 * hidden_size)
    np.testing.assert_allclose(
        states.squeeze(0).numpy(), expected, rtol=RTOL, atol=ATOL)


def test_reservoir_batched_matches_per_sequence() -> None:
    hidden_size = 20
    n2n = _fit_nodetonode(
        hidden_size, spectral_radius=0.9, leakage=0.6, activation="tanh",
        random_state=7)
    rng = np.random.RandomState(1)
    X1 = rng.normal(size=(30, hidden_size)) * 0.3
    X2 = rng.normal(size=(30, hidden_size)) * 0.3
    e1 = n2n._pass_through_recurrent_weights(X1)
    e2 = n2n._pass_through_recurrent_weights(X2)

    res = Reservoir(hidden_size=hidden_size, spectral_radius=0.9, leakage=0.6,
                    activation="tanh", dtype=torch.float64)
    res.set_recurrent_weights(_dense(n2n._recurrent_weights))
    Xb = torch.as_tensor(np.stack([X1, X2]), dtype=torch.float64)
    states, _ = res(Xb)

    np.testing.assert_allclose(states[0].numpy(), e1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(states[1].numpy(), e2, rtol=RTOL, atol=ATOL)
