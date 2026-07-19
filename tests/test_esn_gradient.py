"""Phase B1 gradient solver for the ESN estimators (plumbing + contracts).

The gradient solver keeps the reservoir fixed, computes its states once
(dropping ``washout`` per sequence) and trains a ``LinearReadout`` with an
optimizer loop instead of the closed-form ridge solve. These tests check that
the gradient path is wired correctly (runs, learns, right shapes, right
guards) rather than asserting tight closed-form equivalence.

A torch seed is set per gradient test because the readout's initial weights
come from torch's global RNG (gradient reproducibility w.r.t. ``random_state``
is a later refinement).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode
from pyrcn.datasets import load_digits, mackey_glass
from pyrcn.echo_state_network import ESNClassifier, ESNRegressor


def test_esn_gradient_nonsequence_learns() -> None:
    torch.manual_seed(0)
    X, y = mackey_glass(n_timesteps=500)
    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y)
    esn = ESNRegressor(
        hidden_layer_size=50, spectral_radius=0.9, leakage=0.7,
        alpha=1e-8, solver="gradient", optimizer="adam",
        learning_rate=0.1, epochs=400, random_state=42).fit(X, y)

    assert esn._use_torch is True
    pred = esn.predict(X)
    assert pred.shape == y.shape          # 1-D target -> 1-D prediction
    assert esn.score(X, y) > 0.3          # gradient training learns


def test_esn_gradient_sequence_runs() -> None:
    torch.manual_seed(0)
    Xs, ys = mackey_glass(n_timesteps=600)
    Xs = np.asarray(Xs).reshape(-1, 1)
    ys = np.asarray(ys)
    X = np.empty(shape=(3,), dtype=object)
    y = np.empty(shape=(3,), dtype=object)
    for k in range(3):
        X[k] = Xs[k * 200:(k + 1) * 200]
        y[k] = ys[k * 200:(k + 1) * 200]

    esn = ESNRegressor(
        hidden_layer_size=50, spectral_radius=0.9, leakage=0.7,
        alpha=1e-8, solver="gradient", optimizer="adam",
        learning_rate=0.1, epochs=200, random_state=42).fit(X, y)

    assert esn._use_torch is True
    pred = esn.predict(X)
    assert pred.dtype == object
    assert len(pred) == 3
    for k in range(3):
        assert np.asarray(pred[k]).ndim == 1


def test_esn_gradient_requires_backable() -> None:
    X, y = mackey_glass(n_timesteps=200)
    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y)
    esn = ESNRegressor(
        solver="gradient",
        input_to_node=FeatureUnion([("x", InputToNode())]))
    with pytest.raises(NotImplementedError):
        esn.fit(X, y)


def test_esn_invalid_solver() -> None:
    X, y = mackey_glass(n_timesteps=200)
    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y)
    with pytest.raises(ValueError):
        ESNRegressor(solver="bogus").fit(X, y)


def test_esn_gradient_classifier() -> None:
    torch.manual_seed(0)
    X, y = load_digits(return_X_y=True, as_sequence=True)
    X, y = X[:40], y[:40]
    clf = ESNClassifier(
        hidden_layer_size=50, solver="gradient", optimizer="adam",
        learning_rate=0.05, epochs=200, random_state=42).fit(X, y)

    assert clf._use_torch is True
    pred = clf.predict(X)
    assert len(pred) == len(X)
