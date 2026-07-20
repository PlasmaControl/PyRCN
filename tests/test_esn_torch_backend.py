"""Parity tests for the ESN torch backend fast path."""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.util import concatenate_sequences


def _make_i2n(**kwargs: object) -> InputToNode:
    params = dict(hidden_layer_size=50, input_activation="identity",
                  bias_scaling=0.1, input_scaling=1.0, random_state=42)
    params.update(kwargs)
    return InputToNode(**params)


def _make_n2n(**kwargs: object) -> NodeToNode:
    params = dict(hidden_layer_size=50, spectral_radius=0.9,
                  reservoir_activation="tanh", random_state=42)
    params.update(kwargs)
    return NodeToNode(**params)


def _nonsequence_data() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(200)
    return X[:150], X[150:], y[:150], y[150:]


def _sequence_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = np.empty(shape=(4,), dtype=object)
    y = np.empty(shape=(4,), dtype=object)
    for k in range(4):
        length = 30 + k
        X[k] = rng.standard_normal((length, 1))
        y[k] = np.cos(X[k][:, 0])
    return X, y


def test_esn_uses_torch_when_backable() -> None:
    X_train, X_test, y_train, y_test = _nonsequence_data()
    esn = ESNRegressor(input_to_node=_make_i2n(), node_to_node=_make_n2n())
    esn.fit(X_train, y_train)
    assert esn._use_torch is True


def test_esn_falls_back_for_featureunion() -> None:
    X_train, X_test, y_train, y_test = _nonsequence_data()
    union = FeatureUnion([
        ("a", _make_i2n(hidden_layer_size=20)),
        ("b", _make_i2n(hidden_layer_size=20, random_state=7))])
    esn = ESNRegressor(input_to_node=union,
                       node_to_node=_make_n2n(hidden_layer_size=40))
    esn.fit(X_train, y_train)
    assert esn._use_torch is False
    y_pred = esn.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]


def test_esn_falls_back_for_ridge() -> None:
    X_train, X_test, y_train, y_test = _nonsequence_data()
    esn = ESNRegressor(input_to_node=_make_i2n(), node_to_node=_make_n2n(),
                       regressor=Ridge(alpha=1e-3))
    esn.fit(X_train, y_train)
    assert esn._use_torch is False
    y_pred = esn.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]


def _fit_numpy_reference(i2n: InputToNode, n2n: NodeToNode,
                         reg: IncrementalRegression, X: np.ndarray,
                         y: np.ndarray) -> np.ndarray:
    i2n.fit(X)
    n2n.fit(i2n.transform(X))
    states = n2n.transform(i2n.transform(X))
    reg.fit(states, y)
    return states


def test_esn_fastpath_matches_numpy_reference_nonsequence() -> None:
    X_train, X_test, y_train, y_test = _nonsequence_data()
    for bidirectional in (False, True):
        i2n = _make_i2n()
        n2n = _make_n2n(bidirectional=bidirectional)
        reg = IncrementalRegression(alpha=1e-3)
        _fit_numpy_reference(i2n, n2n, reg, X_train, y_train)
        s_test = n2n.transform(i2n.transform(X_test))
        expected = reg.predict(s_test)

        esn = ESNRegressor(
            input_to_node=_make_i2n(),
            node_to_node=_make_n2n(bidirectional=bidirectional),
            regressor=IncrementalRegression(alpha=1e-3))
        esn.fit(X_train, y_train)
        assert esn._use_torch is True
        got = esn.predict(X_test)
        assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_esn_fastpath_matches_numpy_reference_sequence() -> None:
    X_train, y_train = _sequence_data()
    X_test, _ = _sequence_data()

    i2n = _make_i2n()
    n2n = _make_n2n()
    reg = IncrementalRegression(alpha=1e-3)
    X_cat, y_cat, ranges = concatenate_sequences(X_train, y_train)
    i2n.fit(X_cat)
    n2n.fit(i2n.transform(X_cat))
    n_seq = len(ranges)
    for i, (start, stop) in enumerate(ranges):
        states = n2n.transform(i2n.transform(X_cat[start:stop]))
        reg.partial_fit(states, y_cat[start:stop], reset=(i == 0),
                        postpone_inverse=(i < n_seq - 1))
    expected = []
    for seq in X_test:
        expected.append(reg.predict(n2n.transform(i2n.transform(seq))))

    esn = ESNRegressor(input_to_node=_make_i2n(), node_to_node=_make_n2n(),
                       regressor=IncrementalRegression(alpha=1e-3))
    esn.fit(X_train, y_train)
    assert esn._use_torch is True
    got = esn.predict(X_test)
    assert len(got) == len(expected)
    for k in range(len(expected)):
        assert_allclose(got[k], expected[k], rtol=1e-5)
