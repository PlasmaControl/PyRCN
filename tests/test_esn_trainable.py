"""Phase B2: trainable reservoir (end-to-end gradient training via BPTT).

With ``trainable_reservoir=True`` (and ``solver='gradient'``) the reservoir's
recurrent weights are optimized jointly with the readout by backpropagating
through the recurrence. The RC initialization is the starting point. These
tests check the plumbing: the weights become trainable and actually move, the
fit learns, the invalid ``trainable + closed_form`` combo is rejected, and
fits are reproducible via ``random_state``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyrcn.datasets import load_digits
from pyrcn.echo_state_network import ESNClassifier, ESNRegressor


def _sin_next_step(n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    x = np.sin(np.linspace(0, 8 * np.pi, n)).reshape(-1, 1)
    return x, np.roll(x.ravel(), -1)


def _reg(**kw: object) -> ESNRegressor:
    params: dict = dict(
        hidden_layer_size=20, spectral_radius=0.9, leakage=0.7,
        solver="gradient", optimizer="adam", learning_rate=0.05,
        epochs=120, random_state=42)
    params.update(kw)
    return ESNRegressor(**params)


def test_trainable_reservoir_fits_and_flags() -> None:
    X, y = _sin_next_step()
    esn = _reg(trainable_reservoir=True).fit(X, y)
    assert esn._use_torch is True
    assert esn._torch_reservoir.cell.weight_hh.requires_grad is True
    assert esn.score(X, y) > 0.6           # end-to-end training learns


def test_trainable_reservoir_updates_recurrent_weights() -> None:
    X, y = _sin_next_step()
    fixed = _reg(trainable_reservoir=False, epochs=60).fit(X, y)
    trained = _reg(trainable_reservoir=True, epochs=60).fit(X, y)
    w_fixed = fixed._torch_reservoir.cell.weight_hh.detach().numpy()
    w_trained = trained._torch_reservoir.cell.weight_hh.detach().numpy()
    # the fixed path keeps the RC init; the trainable path moved away from it
    assert not np.allclose(w_fixed, w_trained)


def test_trainable_requires_gradient_solver() -> None:
    with pytest.raises(ValueError):
        ESNRegressor(trainable_reservoir=True, solver="closed_form").fit(
            np.zeros((10, 1)), np.zeros(10))


def test_trainable_reproducible() -> None:
    X, y = _sin_next_step(120)

    def _pred() -> np.ndarray:
        return _reg(trainable_reservoir=True, epochs=60).fit(X, y).predict(X)

    np.testing.assert_array_equal(_pred(), _pred())


def test_trainable_classifier_runs() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    X, y = X[:30], y[:30]
    clf = ESNClassifier(
        hidden_layer_size=25, solver="gradient", trainable_reservoir=True,
        optimizer="adam", learning_rate=0.05, epochs=60,
        random_state=42).fit(X, y)
    assert clf._use_torch is True
    assert clf._torch_reservoir.cell.weight_hh.requires_grad is True
    assert len(clf.predict(X)) == len(X)


def test_trainable_input_learns_and_updates_weights() -> None:
    X, y = _sin_next_step()
    fixed = _reg(trainable_input=False, epochs=80).fit(X, y)
    trained = _reg(trainable_input=True, epochs=80).fit(X, y)
    assert all(p.requires_grad
               for p in trained._torch_input_map.parameters())
    w_fixed = fixed._torch_input_map.weight.detach().numpy()
    w_trained = trained._torch_input_map.weight.detach().numpy()
    assert not np.allclose(w_fixed, w_trained)   # input weights moved
    assert trained.score(X, y) > 0.6


def test_trainable_input_and_reservoir_together() -> None:
    X, y = _sin_next_step()
    esn = _reg(trainable_input=True, trainable_reservoir=True,
               epochs=80).fit(X, y)
    assert all(p.requires_grad for p in esn._torch_input_map.parameters())
    assert esn._torch_reservoir.cell.weight_hh.requires_grad is True
    assert esn.score(X, y) > 0.6


def test_trainable_input_requires_gradient_solver() -> None:
    with pytest.raises(ValueError):
        ESNRegressor(trainable_input=True, solver="closed_form").fit(
            np.zeros((10, 1)), np.zeros(10))


def _sequences(n_seq: int = 6, length: int = 100
               ) -> tuple[np.ndarray, np.ndarray]:
    flat = np.sin(np.linspace(0, 12 * np.pi, n_seq * length))
    X = np.empty(n_seq, dtype=object)
    y = np.empty(n_seq, dtype=object)
    for k in range(n_seq):
        seg = flat[k * length:(k + 1) * length].reshape(-1, 1)
        X[k] = seg
        y[k] = np.roll(seg.ravel(), -1)
    return X, y


def test_bptt_minibatching_runs_and_is_reproducible() -> None:
    X, y = _sequences()

    def _pred(batch_size: int) -> np.ndarray:
        esn = ESNRegressor(
            hidden_layer_size=20, spectral_radius=0.9, leakage=0.7,
            solver="gradient", trainable_reservoir=True, optimizer="adam",
            learning_rate=0.05, epochs=40, batch_size=batch_size,
            random_state=42)
        return esn.fit(X, y).predict(X)

    out = _pred(2)                                   # 2 sequences / step
    assert len(out) == 6
    a, b = _pred(3), _pred(3)
    assert all(np.array_equal(a[k], b[k]) for k in range(6))
