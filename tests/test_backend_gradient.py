"""Gradient training core (B1): the trainable LinearReadout + train_readout.

Consistency check (D9): with a fixed set of features, a gradient-trained
linear readout converges to the closed-form ridge (IncrementalRidge) solution.
The features are whitened (orthonormal columns) so the objective is
well-conditioned and gradient descent converges tightly in few epochs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.nn import IncrementalRidge, LinearReadout, train_readout


def _whitened(n: int, p: int, t: int, seed: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, p)))       # orthonormal columns
    W_true = rng.normal(size=(p, t))
    y = Q @ W_true + 0.001 * rng.normal(size=(n, t))
    return (torch.as_tensor(Q, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64))


def test_gradient_readout_matches_closed_form() -> None:
    Z, y = _whitened(300, 8, 2, seed=0)
    alpha = 1e-8
    expected = IncrementalRidge(
        alpha=alpha, fit_intercept=False, dtype=torch.float64
    ).fit(Z, y).predict(Z).numpy()

    torch.manual_seed(0)
    readout = LinearReadout(8, 2, fit_intercept=False, dtype=torch.float64)
    train_readout(readout, Z, y, optimizer="adam", learning_rate=0.05,
                  epochs=600, weight_decay=alpha)
    got = readout.predict(Z).numpy()

    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-4)


def test_train_readout_reduces_loss() -> None:
    Z, y = _whitened(200, 6, 1, seed=1)
    torch.manual_seed(0)
    readout = LinearReadout(6, 1, fit_intercept=True, dtype=torch.float64)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        before = loss_fn(readout(Z), y).item()
    train_readout(readout, Z, y, optimizer="adam", learning_rate=0.05,
                  epochs=200)
    with torch.no_grad():
        after = loss_fn(readout(Z), y).item()
    assert after < before * 1e-2


def test_linear_readout_predict_shape_and_no_grad() -> None:
    readout = LinearReadout(4, 3, dtype=torch.float64)
    pred = readout.predict(torch.zeros(10, 4, dtype=torch.float64))
    assert pred.shape == (10, 3)
    assert not pred.requires_grad


def test_train_readout_unknown_optimizer_raises() -> None:
    readout = LinearReadout(4, 1, dtype=torch.float64)
    Z = torch.zeros(5, 4, dtype=torch.float64)
    y = torch.zeros(5, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        train_readout(readout, Z, y, optimizer="rmsprop")


def test_train_readout_unknown_loss_raises() -> None:
    readout = LinearReadout(4, 1, dtype=torch.float64)
    Z = torch.zeros(5, 4, dtype=torch.float64)
    y = torch.zeros(5, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        train_readout(readout, Z, y, loss="huber")
