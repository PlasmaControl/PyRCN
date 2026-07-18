"""Parity: torch IncrementalRidge vs the legacy IncrementalRegression.

The legacy readout and the torch readout are fed the *identical* feature
matrix Z and targets y; both compute the closed-form ridge solution
``W = (K + alpha I)^-1 xTy`` over accumulated statistics. float64, tight
tolerance. Plus torch-specific property tests (accumulation, merge, postpone).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.backend import IncrementalRidge
from pyrcn.linear_model import IncrementalRegression

RTOL, ATOL = 1e-7, 1e-9


def _legacy(alpha: float, fit_intercept: bool) -> IncrementalRegression:
    return IncrementalRegression(alpha=alpha, fit_intercept=fit_intercept)


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n_targets", [1, 3])
def test_fit_parity_single_batch(fit_intercept: bool, n_targets: int) -> None:
    n_samples, n_features, alpha = 40, 8, 1e-3
    Z = _rng(0).normal(size=(n_samples, n_features))
    y = _rng(1).normal(size=(n_samples, n_targets))

    expected = _legacy(alpha, fit_intercept).fit(Z, y).predict(Z)

    ridge = IncrementalRidge(
        alpha=alpha, fit_intercept=fit_intercept, dtype=torch.float64)
    ridge.fit(torch.as_tensor(Z, dtype=torch.float64),
              torch.as_tensor(y, dtype=torch.float64))
    got = ridge.predict(torch.as_tensor(Z, dtype=torch.float64))

    np.testing.assert_allclose(got.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_fit_parity_1d_target() -> None:
    n_samples, n_features, alpha = 30, 6, 1e-4
    Z = _rng(2).normal(size=(n_samples, n_features))
    y = _rng(3).normal(size=(n_samples,))

    expected = _legacy(alpha, True).fit(Z, y).predict(Z)

    ridge = IncrementalRidge(alpha=alpha, fit_intercept=True,
                             dtype=torch.float64)
    ridge.fit(torch.as_tensor(Z, dtype=torch.float64),
              torch.as_tensor(y, dtype=torch.float64))
    got = ridge.predict(torch.as_tensor(Z, dtype=torch.float64))

    assert got.ndim == 1
    np.testing.assert_allclose(got.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_sequence_postpone_parity() -> None:
    """Accumulate several batches with postpone_inverse, solve once at end.

    Mirrors the ESN/ELM sequence-fit flow: partial_fit per sequence with
    postpone_inverse=True, final partial_fit solves. Equivalent to a single
    closed-form solve over the concatenated data.
    """
    alpha, n_features, n_targets = 1e-3, 5, 2
    batches = [(_rng(10 + i).normal(size=(12, n_features)),
                _rng(50 + i).normal(size=(12, n_targets))) for i in range(4)]

    legacy = _legacy(alpha, True)
    for i, (Z, y) in enumerate(batches):
        legacy.partial_fit(Z, y, reset=(i == 0),
                           postpone_inverse=(i < len(batches) - 1))

    ridge = IncrementalRidge(alpha=alpha, fit_intercept=True,
                             dtype=torch.float64)
    for i, (Z, y) in enumerate(batches):
        ridge.partial_fit(
            torch.as_tensor(Z, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64), reset=(i == 0),
            postpone_inverse=(i < len(batches) - 1))

    Z_test = _rng(99).normal(size=(15, n_features))
    expected = legacy.predict(Z_test)
    got = ridge.predict(torch.as_tensor(Z_test, dtype=torch.float64))
    np.testing.assert_allclose(got.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_merge_equals_concatenated_fit() -> None:
    """Merged readouts (summed stats) == one readout on concatenated data."""
    alpha, n_features, n_targets = 1e-3, 5, 2
    Za = _rng(20).normal(size=(18, n_features))
    ya = _rng(21).normal(size=(18, n_targets))
    Zb = _rng(22).normal(size=(14, n_features))
    yb = _rng(23).normal(size=(14, n_targets))

    ra = IncrementalRidge(alpha=alpha, dtype=torch.float64)
    ra.partial_fit(torch.as_tensor(Za, dtype=torch.float64),
                   torch.as_tensor(ya, dtype=torch.float64),
                   reset=True, postpone_inverse=True)
    rb = IncrementalRidge(alpha=alpha, dtype=torch.float64)
    rb.partial_fit(torch.as_tensor(Zb, dtype=torch.float64),
                   torch.as_tensor(yb, dtype=torch.float64),
                   reset=True, postpone_inverse=True)
    merged = ra + rb
    merged.solve()

    whole = IncrementalRidge(alpha=alpha, dtype=torch.float64)
    whole.fit(torch.as_tensor(np.vstack([Za, Zb]), dtype=torch.float64),
              torch.as_tensor(np.vstack([ya, yb]), dtype=torch.float64))

    np.testing.assert_allclose(
        merged.output_weights.numpy(), whole.output_weights.numpy(),
        rtol=RTOL, atol=ATOL)


def test_predict_before_fit_raises() -> None:
    ridge = IncrementalRidge(dtype=torch.float64)
    with pytest.raises(RuntimeError):
        ridge.predict(torch.zeros(3, 4, dtype=torch.float64))
