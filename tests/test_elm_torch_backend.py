"""Torch-backend fast-path parity tests for the ELM estimators."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor
from pyrcn.linear_model import IncrementalRegression


def _regression_data() -> tuple:
    rng = np.random.RandomState(0)
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.01 * rng.randn(X.shape[0])
    return X, y


def test_elm_uses_torch_when_backable() -> None:
    X, y = _regression_data()
    elm = ELMRegressor(hidden_layer_size=25, random_state=42)
    elm.fit(X, y)
    assert elm._use_torch is True


def test_elm_falls_back_for_featureunion() -> None:
    X, y = _regression_data()
    elm = ELMRegressor(
        input_to_node=FeatureUnion([
            ("x", InputToNode(hidden_layer_size=20, random_state=42))]))
    elm.fit(X, y)
    assert getattr(elm, "_use_torch", False) is False
    y_pred = elm.predict(X)
    assert y_pred.shape[0] == X.shape[0]


def test_elm_falls_back_for_ridge() -> None:
    X, y = _regression_data()
    elm = ELMRegressor(
        input_to_node=InputToNode(hidden_layer_size=20, random_state=42),
        regressor=Ridge())
    elm.fit(X, y)
    assert getattr(elm, "_use_torch", False) is False


def test_elm_fastpath_matches_numpy_reference() -> None:
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    X_test = np.linspace(0, 10, 40).reshape(-1, 1)
    targets = {
        "1d": np.sin(X).ravel(),
        "2d": np.hstack((np.sin(X), np.cos(X))),
    }
    for y in targets.values():
        i2n = InputToNode(hidden_layer_size=30, bias_scaling=1.0,
                          random_state=42)
        i2n.fit(X)
        H = i2n.transform(X)
        reg = IncrementalRegression(alpha=1e-3)
        reg.fit(H, y)
        expected = reg.predict(i2n.transform(X_test))

        elm = ELMRegressor(
            input_to_node=InputToNode(hidden_layer_size=30,
                                      bias_scaling=1.0, random_state=42),
            regressor=IncrementalRegression(alpha=1e-3))
        elm.fit(X, y)
        assert elm._use_torch is True
        got = elm.predict(X_test)
        np.testing.assert_allclose(
            got, expected, rtol=1e-6, atol=1e-7)


def test_elm_classifier_fastpath() -> None:
    X, y = load_digits(return_X_y=True)
    elm = ELMClassifier(hidden_layer_size=500, random_state=42)
    elm.fit(X, y)
    assert elm._use_torch is True
    assert elm.score(X, y) > 0.9
