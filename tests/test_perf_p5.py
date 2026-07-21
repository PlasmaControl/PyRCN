"""P5 parity: dropping the discarded input_to_node.transform on the backable
path must not change any fitted state or prediction.

``NodeToNode.fit`` uses only the column count of its argument (plus
``hidden_layer_size`` + ``random_state``); on the torch-backable path the
``input_to_node.transform(X)`` fed into it is otherwise discarded. This module
captures the CURRENT outputs (predictions + node_to_node recurrent weights) to
a fixture BEFORE the source edit, then asserts bit-identical equality AFTER.

Run ``python tests/test_perf_p5.py`` once on the pre-edit source to (re)create
the fixture; pytest then compares against it.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.datasets import load_digits
from pyrcn.echo_state_network import ESNClassifier, ESNRegressor

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "esn_p5_parity.npz")


def _i2n(hidden: int, seed: int, **kw: object) -> InputToNode:
    params = dict(hidden_layer_size=hidden, input_activation="identity",
                  bias_scaling=0.1, input_scaling=1.0, random_state=seed)
    params.update(kw)
    return InputToNode(**params)


def _n2n(hidden: int, seed: int, **kw: object) -> NodeToNode:
    params = dict(hidden_layer_size=hidden, spectral_radius=0.9,
                  reservoir_activation="tanh", random_state=seed)
    params.update(kw)
    return NodeToNode(**params)


def _reg_nonseq(seed: int, hidden: int) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((160, 1))
    y = np.sin(X[:, 0])
    esn = ESNRegressor(input_to_node=_i2n(hidden, seed),
                       node_to_node=_n2n(hidden, seed))
    return esn, X[:120], y[:120], X[120:]


def _reg_seq(seed: int, hidden: int) -> tuple:
    rng = np.random.default_rng(seed)
    X = np.empty(shape=(4,), dtype=object)
    y = np.empty(shape=(4,), dtype=object)
    for k in range(4):
        length = 30 + k
        X[k] = rng.standard_normal((length, 1))
        y[k] = np.cos(X[k][:, 0])
    esn = ESNRegressor(input_to_node=_i2n(hidden, seed),
                       node_to_node=_n2n(hidden, seed))
    return esn, X, y, X


def _clf_seq() -> tuple:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    esn = ESNClassifier(input_to_node=_i2n(50, 42), node_to_node=_n2n(50, 42))
    return esn, X[:80], y[:80], X[80:100]


def _clf_nonseq() -> tuple:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((150, 4))
    y = rng.integers(0, 3, size=150)
    esn = ESNClassifier(input_to_node=_i2n(40, 5), node_to_node=_n2n(40, 5))
    return esn, X[:120], y[:120], X[120:]


def _fallback_ridge() -> tuple:
    rng = np.random.default_rng(11)
    X = rng.standard_normal((160, 1))
    y = np.sin(X[:, 0])
    esn = ESNRegressor(input_to_node=_i2n(50, 11), node_to_node=_n2n(50, 11),
                       regressor=Ridge(alpha=1e-3))
    return esn, X[:120], y[:120], X[120:]


def _fallback_featureunion() -> tuple:
    rng = np.random.default_rng(13)
    X = rng.standard_normal((160, 1))
    y = np.sin(X[:, 0])
    union = FeatureUnion([("a", _i2n(20, 13)), ("b", _i2n(20, 7))])
    esn = ESNRegressor(input_to_node=union, node_to_node=_n2n(40, 13))
    return esn, X[:120], y[:120], X[120:]


CASES = {
    "reg_nonseq_a": lambda: _reg_nonseq(42, 50),
    "reg_nonseq_b": lambda: _reg_nonseq(7, 30),
    "reg_seq": lambda: _reg_seq(1, 50),
    "clf_seq": _clf_seq,
    "clf_nonseq": _clf_nonseq,
    "fallback_ridge": _fallback_ridge,
    "fallback_featureunion": _fallback_featureunion,
}


def _flatten(pred: object) -> np.ndarray:
    arr = np.asarray(pred, dtype=object) if isinstance(pred, np.ndarray) \
        and pred.dtype == object else None
    if arr is not None:
        parts = [np.asarray(p).ravel().astype(np.float64) for p in arr]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    return np.asarray(pred).ravel().astype(np.float64)


def _recur(esn: ESNRegressor) -> np.ndarray:
    w = esn._node_to_node.recurrent_weights
    if hasattr(w, "toarray"):
        w = w.toarray()
    return np.asarray(w).ravel().astype(np.float64)


def _run(name: str) -> tuple[np.ndarray, np.ndarray, bool]:
    esn, X, y, X_test = CASES[name]()
    esn.fit(X, y)
    used_torch = bool(getattr(esn, "_use_torch", False))
    return _flatten(esn.predict(X_test)), _recur(esn), used_torch


def _capture() -> dict:
    out: dict = {}
    for name in CASES:
        pred, recur, used_torch = _run(name)
        out[f"{name}__pred"] = pred
        out[f"{name}__recur"] = recur
        out[f"{name}__torch"] = np.array([used_torch])
    return out


# --- pytest ---------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.parametrize("name", list(CASES))
def test_p5_bit_identical(name: str) -> None:
    """Predictions + node_to_node weights match the pre-edit fixture."""
    ref = np.load(FIXTURE, allow_pickle=False)
    pred, recur, _ = _run(name)
    assert np.array_equal(pred, ref[f"{name}__pred"]), f"pred mismatch {name}"
    assert np.array_equal(recur, ref[f"{name}__recur"]), \
        f"recurrent-weights mismatch {name}"


def test_p5_fixture_covers_both_paths() -> None:
    """The fixture must exercise both the torch and the numpy-fallback path."""
    ref = np.load(FIXTURE, allow_pickle=False)
    torch_flags = {n: bool(ref[f"{n}__torch"][0]) for n in CASES}
    assert any(torch_flags.values()), "no torch/backable case captured"
    assert not all(torch_flags.values()), "no numpy-fallback case captured"


if __name__ == "__main__":
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez(FIXTURE, **_capture())
    print(f"wrote fixture {FIXTURE}")
