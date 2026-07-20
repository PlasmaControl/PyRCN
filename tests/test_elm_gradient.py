"""Phase B1 gradient solver for the ELM estimators (plumbing + contracts).

The gradient solver trains a ``LinearReadout`` with an optimizer loop instead
of the closed-form ridge solve. These tests check that the gradient path is
wired correctly (runs, learns, right shapes, right guards). The tight
gradient-vs-closed-form numerical equivalence is proven separately on
well-conditioned features in ``test_backend_gradient.py``; through the raw ELM
feature map gradient descent converges only slowly, so here we assert that
learning happens rather than exact agreement.

A torch seed is set per test because the readout's initial weights come from
torch's global RNG (gradient reproducibility w.r.t. ``random_state`` is a
later refinement).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor


def test_elm_gradient_learns_and_matches_shapes() -> None:
    torch.manual_seed(0)
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.sin(X).ravel()
    grad = ELMRegressor(
        hidden_layer_size=50, alpha=1e-8, solver="gradient",
        optimizer="adam", learning_rate=0.2, epochs=400,
        random_state=42).fit(X, y)

    assert grad._use_torch is True
    pred = grad.predict(X)
    assert pred.shape == y.shape          # 1-D target -> 1-D prediction
    assert grad.score(X, y) > 0.4         # gradient training learns


def test_elm_gradient_multitarget_shape() -> None:
    torch.manual_seed(0)
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.column_stack([np.sin(X).ravel(), np.cos(X).ravel()])
    grad = ELMRegressor(
        hidden_layer_size=50, solver="gradient", learning_rate=0.1,
        epochs=200, random_state=42).fit(X, y)
    assert grad.predict(X).shape == (200, 2)


def test_elm_gradient_requires_backable() -> None:
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel()
    elm = ELMRegressor(
        solver="gradient",
        input_to_node=FeatureUnion([("x", InputToNode())]))
    with pytest.raises(NotImplementedError):
        elm.fit(X, y)


def test_elm_invalid_solver() -> None:
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel()
    with pytest.raises(ValueError):
        ELMRegressor(solver="bogus").fit(X, y)


def test_elm_gradient_reproducible() -> None:
    # No torch.manual_seed: the estimator seeds its readout from random_state.
    X = np.linspace(0, 10, 150).reshape(-1, 1)
    y = np.sin(X).ravel()

    def _pred() -> np.ndarray:
        elm = ELMRegressor(
            hidden_layer_size=40, solver="gradient", optimizer="adam",
            learning_rate=0.1, epochs=150, random_state=42)
        return elm.fit(X, y).predict(X)

    np.testing.assert_array_equal(_pred(), _pred())


def test_elm_gradient_classifier() -> None:
    torch.manual_seed(0)
    X, y = load_digits(return_X_y=True)
    X, y = X[:400], y[:400]
    clf = ELMClassifier(
        hidden_layer_size=100, solver="gradient", optimizer="adam",
        learning_rate=0.1, epochs=200, random_state=42).fit(X, y)
    assert clf._use_torch is True
    assert clf.score(X, y) > 0.8
