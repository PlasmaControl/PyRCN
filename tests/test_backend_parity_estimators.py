"""Backend parity harness for the pyrcn estimators.

This module proves that the PyTorch "fast path" and the legacy numpy
"fallback" path compute the *same* thing for an identical estimator
configuration.

Technique
---------
For each configuration we build two twin estimators:

* ``native`` -- every sub-estimator is a native pyrcn block
  (``InputToNode`` / ``NodeToNode`` + ``IncrementalRegression``), so the
  estimator selects the torch backend and ``_use_torch`` becomes ``True``.
* ``fallback`` -- identical, except the input stage is wrapped in a
  single-transformer ``FeatureUnion([("x", InputToNode(...))])``. A
  one-transformer ``FeatureUnion`` emits exactly the inner
  ``InputToNode.transform`` output, and with the same ``random_state`` and
  data the weights are identical. The estimator can no longer recognise a
  native input block, so it drops to the numpy path
  (``_use_torch`` is ``False``) while computing the same reservoir input,
  states and readout.

Because the two twins are mathematically identical and differ only in the
compute backend, their predictions must match. Any divergence larger than
the acceptance bar (rtol=1e-4, atol=1e-5) signals a real fast-path vs.
fallback discrepancy.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.datasets import load_digits, mackey_glass
from pyrcn.echo_state_network import ESNClassifier, ESNRegressor
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor
from pyrcn.linear_model import IncrementalRegression

RTOL = 1e-4
ATOL = 1e-5


def _wrap(i2n: InputToNode) -> FeatureUnion:
    """Force the numpy fallback by hiding the native input block."""
    return FeatureUnion([("x", i2n)])


def _build_esn_regressor(
    forced_fallback: bool, *, hidden_layer_size: int,
    spectral_radius: float, leakage: float, alpha: float,
    bidirectional: bool = False,
) -> ESNRegressor:
    """Build an ESN regressor twin (native or forced-fallback)."""
    i2n = InputToNode(hidden_layer_size=hidden_layer_size, random_state=42)
    n2n = NodeToNode(
        hidden_layer_size=hidden_layer_size,
        spectral_radius=spectral_radius, leakage=leakage,
        bidirectional=bidirectional, random_state=42)
    return ESNRegressor(
        input_to_node=_wrap(i2n) if forced_fallback else i2n,
        node_to_node=n2n,
        regressor=IncrementalRegression(alpha=alpha),
        verbose=False)


def _build_esn_classifier(
    forced_fallback: bool, *, hidden_layer_size: int,
    spectral_radius: float, leakage: float, alpha: float,
) -> ESNClassifier:
    """Build an ESN classifier twin (native or forced-fallback)."""
    i2n = InputToNode(hidden_layer_size=hidden_layer_size, random_state=42)
    n2n = NodeToNode(
        hidden_layer_size=hidden_layer_size,
        spectral_radius=spectral_radius, leakage=leakage,
        random_state=42)
    return ESNClassifier(
        input_to_node=_wrap(i2n) if forced_fallback else i2n,
        node_to_node=n2n,
        regressor=IncrementalRegression(alpha=alpha),
        verbose=False)


def _build_elm_regressor(
    forced_fallback: bool, *, hidden_layer_size: int, alpha: float,
) -> ELMRegressor:
    """Build an ELM regressor twin (native or forced-fallback)."""
    i2n = InputToNode(hidden_layer_size=hidden_layer_size, random_state=42)
    return ELMRegressor(
        input_to_node=_wrap(i2n) if forced_fallback else i2n,
        regressor=IncrementalRegression(alpha=alpha),
        verbose=False)


def _build_elm_classifier(
    forced_fallback: bool, *, hidden_layer_size: int, alpha: float,
) -> ELMClassifier:
    """Build an ELM classifier twin (native or forced-fallback)."""
    i2n = InputToNode(hidden_layer_size=hidden_layer_size, random_state=42)
    return ELMClassifier(
        input_to_node=_wrap(i2n) if forced_fallback else i2n,
        regressor=IncrementalRegression(alpha=alpha),
        verbose=False)


def _assert_backends(native: object, fallback: object) -> None:
    """Assert the twins really used different compute backends."""
    assert native._use_torch is True
    assert getattr(fallback, "_use_torch", False) is False


def _max_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (max abs diff, max rel diff) for reporting."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / (np.abs(b) + 1e-12)
    return float(abs_diff.max()), float(rel_diff.max())


def _mackey_glass_2d() -> tuple[np.ndarray, np.ndarray]:
    """Return a small non-sequence mackey-glass dataset."""
    X, y = mackey_glass(n_timesteps=600)
    return X.reshape(-1, 1), y


def test_esn_regressor_nonsequence_parity() -> None:
    """ESN regressor parity on 2-D (non-sequence) input."""
    X, y = _mackey_glass_2d()
    X_train, X_test = X[:-100], X[-100:]
    y_train = y[:-100]

    cfg = dict(hidden_layer_size=50, spectral_radius=0.9,
               leakage=0.7, alpha=1e-3)
    native = _build_esn_regressor(False, **cfg)
    fallback = _build_esn_regressor(True, **cfg)
    native.fit(X_train, y_train)
    fallback.fit(X_train, y_train)
    _assert_backends(native, fallback)

    got = native.predict(X_test)
    ref = fallback.predict(X_test)
    print("esn_nonsequence max abs/rel:", _max_diff(got, ref))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_esn_regressor_bidirectional_parity() -> None:
    """ESN regressor parity with a bidirectional reservoir."""
    X, y = _mackey_glass_2d()
    X_train, X_test = X[:-100], X[-100:]
    y_train = y[:-100]

    cfg = dict(hidden_layer_size=50, spectral_radius=0.9,
               leakage=0.7, alpha=1e-3, bidirectional=True)
    native = _build_esn_regressor(False, **cfg)
    fallback = _build_esn_regressor(True, **cfg)
    native.fit(X_train, y_train)
    fallback.fit(X_train, y_train)
    _assert_backends(native, fallback)

    got = native.predict(X_test)
    ref = fallback.predict(X_test)
    print("esn_bidirectional max abs/rel:", _max_diff(got, ref))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_esn_regressor_sequence_parity() -> None:
    """ESN regressor parity on an object-array of sequences."""
    X_flat, y_flat = mackey_glass(n_timesteps=600)
    n_seq = 4
    length = 120
    X = np.empty(shape=(n_seq,), dtype=object)
    y = np.empty(shape=(n_seq,), dtype=object)
    for k in range(n_seq):
        start = k * length
        stop = start + length
        X[k] = X_flat[start:stop].reshape(-1, 1)
        y[k] = y_flat[start:stop]

    cfg = dict(hidden_layer_size=50, spectral_radius=0.9,
               leakage=0.7, alpha=1e-3)
    native = _build_esn_regressor(False, **cfg)
    fallback = _build_esn_regressor(True, **cfg)
    native.fit(X, y)
    fallback.fit(X, y)
    _assert_backends(native, fallback)

    got = native.predict(X)
    ref = fallback.predict(X)
    assert len(got) == len(ref)
    worst = (0.0, 0.0)
    for k in range(len(ref)):
        worst = tuple(np.maximum(worst, _max_diff(got[k], ref[k])))
        np.testing.assert_allclose(got[k], ref[k], rtol=RTOL, atol=ATOL)
    print("esn_sequence max abs/rel:", worst)


def test_esn_classifier_sequence_to_value_parity() -> None:
    """ESN classifier parity for the sequence-to-value task."""
    X, y = load_digits(return_X_y=True, as_sequence=True)
    X, y = X[:40], y[:40]

    cfg = dict(hidden_layer_size=50, spectral_radius=0.9,
               leakage=0.7, alpha=1e-3)
    native = _build_esn_classifier(False, **cfg)
    fallback = _build_esn_classifier(True, **cfg)
    native.fit(X, y)
    fallback.fit(X, y)
    _assert_backends(native, fallback)

    got = native.predict(X)
    ref = fallback.predict(X)
    assert len(got) == len(ref)
    for k in range(len(ref)):
        assert np.array_equal(got[k], ref[k])

    got_p = native.predict_proba(X)
    ref_p = fallback.predict_proba(X)
    worst = (0.0, 0.0)
    for k in range(len(ref_p)):
        worst = tuple(np.maximum(worst, _max_diff(got_p[k], ref_p[k])))
        np.testing.assert_allclose(got_p[k], ref_p[k], rtol=RTOL,
                                   atol=ATOL)
    print("esn_classifier_seq proba max abs/rel:", worst)


def test_elm_regressor_parity() -> None:
    """ELM regressor parity on a 2-output regression target."""
    X = np.linspace(0, 10, 400).reshape(-1, 1)
    y = np.hstack((np.sin(X), np.cos(X)))

    cfg = dict(hidden_layer_size=50, alpha=1e-3)
    native = _build_elm_regressor(False, **cfg)
    fallback = _build_elm_regressor(True, **cfg)
    native.fit(X, y)
    fallback.fit(X, y)
    _assert_backends(native, fallback)

    got = native.predict(X)
    ref = fallback.predict(X)
    print("elm_regressor max abs/rel:", _max_diff(got, ref))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_elm_classifier_parity() -> None:
    """ELM classifier parity on the iris dataset."""
    X, y = load_iris(return_X_y=True)

    cfg = dict(hidden_layer_size=100, alpha=1e-3)
    native = _build_elm_classifier(False, **cfg)
    fallback = _build_elm_classifier(True, **cfg)
    native.fit(X, y)
    fallback.fit(X, y)
    _assert_backends(native, fallback)

    got = native.predict(X)
    ref = fallback.predict(X)
    assert np.array_equal(got, ref)

    got_p = native.predict_proba(X)
    ref_p = fallback.predict_proba(X)
    print("elm_classifier proba max abs/rel:", _max_diff(got_p, ref_p))
    np.testing.assert_allclose(got_p, ref_p, rtol=RTOL, atol=ATOL)
