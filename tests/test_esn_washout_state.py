"""A4b: ESN washout (training-only) and predict initial_state / return_state.

Defaults (washout=0, initial_state=None, return_state=False) are behavior-
preserving; the new capabilities require the torch backend and raise on the
numpy fallback.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression

HID = 30


def _data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 1)) * 0.3
    y = np.sin(np.linspace(0, 8, n))
    return X, y


def _numpy_reference(X: np.ndarray, y: np.ndarray, X_test: np.ndarray,
                     washout: int) -> np.ndarray:
    """Legacy numpy pipeline with the first ``washout`` states dropped."""
    i2n = InputToNode(hidden_layer_size=HID, random_state=42)
    i2n.fit(X)
    n2n = NodeToNode(hidden_layer_size=HID, spectral_radius=0.9, leakage=0.6,
                     random_state=42)
    n2n.fit(i2n.transform(X))
    states = n2n.transform(i2n.transform(X))
    reg = IncrementalRegression(alpha=1e-3)
    reg.fit(states[washout:], y[washout:])
    test_states = n2n.transform(i2n.transform(X_test))
    return reg.predict(test_states)


def _esn(washout: int = 0) -> ESNRegressor:
    return ESNRegressor(
        hidden_layer_size=HID, spectral_radius=0.9, leakage=0.6, alpha=1e-3,
        washout=washout, random_state=42)


def test_washout_zero_is_behavior_preserving() -> None:
    X, y = _data()
    expected = _numpy_reference(X, y, X, washout=0)
    got = _esn(washout=0).fit(X, y).predict(X)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_washout_matches_numpy_reference() -> None:
    X, y = _data()
    expected = _numpy_reference(X, y, X, washout=10)
    got = _esn(washout=10).fit(X, y).predict(X)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_washout_changes_output() -> None:
    X, y = _data()
    y0 = _esn(washout=0).fit(X, y).predict(X)
    y1 = _esn(washout=25).fit(X, y).predict(X)
    assert not np.allclose(y0, y1)


def test_washout_negative_raises() -> None:
    X, y = _data()
    with pytest.raises(ValueError):
        _esn(washout=-1).fit(X, y)


def test_washout_fallback_raises() -> None:
    X, y = _data()
    esn = ESNRegressor(regressor=Ridge(), washout=3, hidden_layer_size=HID)
    with pytest.raises(NotImplementedError):
        esn.fit(X, y)


def test_initial_state_zero_matches_default() -> None:
    X, y = _data()
    esn = _esn().fit(X, y)
    y_none = esn.predict(X)
    y_zero = esn.predict(X, initial_state=np.zeros(HID))
    np.testing.assert_allclose(y_zero, y_none, rtol=1e-10, atol=1e-12)


def test_initial_state_changes_output() -> None:
    X, y = _data()
    esn = _esn().fit(X, y)
    y_none = esn.predict(X)
    y_seed = esn.predict(X, initial_state=np.full(HID, 0.5))
    assert not np.allclose(y_none, y_seed)


def test_return_state_nonsequence() -> None:
    X, y = _data()
    esn = _esn().fit(X, y)
    y_only = esn.predict(X)
    y_out, final = esn.predict(X, return_state=True)
    assert final.shape == (HID,)
    np.testing.assert_allclose(y_out, y_only, rtol=1e-12, atol=1e-12)


def test_return_state_carry_continues_trajectory() -> None:
    # Splitting a sequence and carrying the final state reproduces running
    # the whole sequence in one pass.
    X, y = _data(n=120)
    esn = _esn().fit(X, y)
    whole = esn.predict(X)
    first, second = X[:60], X[60:]
    y1, state = esn.predict(first, return_state=True)
    y2 = esn.predict(second, initial_state=state)
    np.testing.assert_allclose(
        np.concatenate([y1, y2]), whole, rtol=1e-8, atol=1e-9)


def test_initial_state_fallback_raises() -> None:
    X, y = _data()
    esn = ESNRegressor(regressor=Ridge(), hidden_layer_size=HID).fit(X, y)
    with pytest.raises(NotImplementedError):
        esn.predict(X, initial_state=np.zeros(HID))
