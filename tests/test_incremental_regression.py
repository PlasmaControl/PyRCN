"""Testing for Linear model module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_regressor
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from pyrcn.linear_model import IncrementalRegression

X_diabetes, y_diabetes = load_diabetes(return_X_y=True)


def test_normalize() -> None:
    print('\ntest_normalize():')
    rs = np.random.RandomState(42)
    X = np.hstack((np.linspace(0., 10., 1000).reshape(-1, 1),
                   np.linspace(-1., 1., 1000).reshape(-1, 1),
                   rs.random(1000).reshape(-1, 1)))
    transformation = rs.random(size=(3, 2))
    y = np.matmul(X, transformation)
    reg = IncrementalRegression(normalize=True)
    reg.fit(X, y)


def test_postpone_inverse() -> None:
    print('\ntest_postpone_inverse():')
    rs = np.random.RandomState(42)
    index = range(1000)
    X = np.hstack((np.linspace(0., 10., 1000).reshape(-1, 1),
                   np.linspace(-1., 1., 1000).reshape(-1, 1),
                   rs.random(1000).reshape(-1, 1)))
    transformation = rs.random(size=(3, 2))
    y = np.matmul(X, transformation)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10,
                                                        random_state=42)
    reg = IncrementalRegression()
    assert is_regressor(reg)

    for prt in np.array_split(index, 3):
        reg.partial_fit(X[prt, :], y[prt, :], postpone_inverse=True)

    with pytest.raises(NotFittedError):
        y_reg = reg.predict(X_test)

    reg.partial_fit(X, y)
    y_reg = reg.predict(X_test)
    print(f"tests: {y_test}\nregr: {y_reg}")
    np.testing.assert_allclose(y_reg, y_test, rtol=.01, atol=.15)


def test_linear() -> None:
    print('\ntest_linear():')
    rs = np.random.RandomState(42)
    index = range(1000)
    X = np.hstack((np.linspace(0., 10., 1000).reshape(-1, 1),
                   np.linspace(-1., 1., 1000).reshape(-1, 1),
                   rs.random(1000).reshape(-1, 1)))
    transformation = rs.random(size=(3, 2))
    y = np.matmul(X, transformation)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10,
                                                        random_state=42)
    reg = IncrementalRegression()
    assert is_regressor(reg)

    for prt in np.array_split(index, 3):
        reg.partial_fit(X[prt, :], y[prt, :])

    y_reg = reg.predict(X_test)
    print(f"tests: {y_test}\nregr: {y_reg}")
    np.testing.assert_allclose(y_reg, y_test, rtol=.01, atol=.15)


def test_compare_ridge() -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_diabetes, y_diabetes, test_size=10, random_state=42)

    i_reg = IncrementalRegression(alpha=.01).fit(X_train, y_train)
    ridge = Ridge(alpha=.01, solver='svd').fit(X_train, y_train)

    print(f"incremental: {i_reg.coef_} ridge: {ridge.coef_}")
    np.testing.assert_allclose(i_reg.coef_, ridge.coef_, rtol=.0001)


def test_incremental_partial_normalize() -> None:
    print('\ntest_incremental_partial_normalize():')
    rs = np.random.RandomState(42)
    X = rs.normal(size=(20, 3))
    y = rs.normal(size=(20, 2))
    reg = IncrementalRegression(normalize=True)
    reg.partial_fit(X, y)
    reg.predict(X)


def test_incremental_coef_intercept_2d() -> None:
    print('\ntest_incremental_coef_intercept_2d():')
    rs = np.random.RandomState(42)
    X = rs.normal(size=(20, 3))
    y = rs.normal(size=(20, 2))
    reg = IncrementalRegression(fit_intercept=True).fit(X, y)
    assert reg.coef_.shape == (2, 3)
    assert reg.intercept_.shape == (2,)


def test_incremental_no_intercept() -> None:
    print('\ntest_incremental_no_intercept():')
    rs = np.random.RandomState(42)
    X = rs.normal(size=(20, 3))
    y = rs.normal(size=(20, 2))
    reg = IncrementalRegression(fit_intercept=False).fit(X, y)
    assert reg.coef_.shape == (2, 3)
    assert reg.intercept_.size == 0


def test_incremental_coef_not_fitted() -> None:
    print('\ntest_incremental_coef_not_fitted():')
    reg = IncrementalRegression()
    assert reg.coef_.shape == ()
    assert reg.intercept_.size == 0


# --- P3 parity: np.linalg.solve must match the legacy inv(A) @ rhs formula ---

# Sizes: (n_samples, n_features, n_targets), incl. multi-target & single-target
_P3_SIZES = [
    (40, 8, 1),
    (40, 8, 3),
    (60, 12, 2),
    (30, 5, 1),
]


def _reference_incremental_weights(batches, alpha, fit_intercept, postpone):
    """Replicate IncrementalRegression.partial_fit using the LEGACY inv(A).

    This is the frozen reference formula (explicit matrix inverse) that the
    ``np.linalg.solve`` implementation must reproduce bit-tight. Covers both
    the main solve and the incremental residual branch. ``normalize=False``.
    """
    K = None
    xTy = None
    w = None
    for (X, y), pp in zip(batches, postpone):
        if fit_intercept:
            Xp = np.hstack((X, np.ones(shape=(X.shape[0], 1))))
        else:
            Xp = X
        gram = np.matmul(Xp.T, Xp)
        rhs = np.matmul(Xp.T, y)
        K = gram if K is None else K + gram
        xTy = rhs if xTy is None else xTy + rhs
        if pp and w is None:
            continue
        A = K + alpha * np.identity(Xp.shape[1])
        P = np.linalg.inv(A)
        if w is None:
            w = np.matmul(P, xTy)
        else:
            w = w + np.matmul(P, np.matmul(Xp.T, y - np.matmul(Xp, w)))
    return w


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("size", _P3_SIZES)
def test_p3_solve_parity_main(fit_intercept, size) -> None:
    """Single fit(): solve() output identical to inv(K + aI) @ xTy."""
    n_samples, n_features, n_targets = size
    rs = np.random.RandomState(hash(size) % (2 ** 31))
    X = rs.normal(size=(n_samples, n_features))
    y = rs.normal(size=(n_samples, n_targets))
    alpha = 1e-3

    reg = IncrementalRegression(
        alpha=alpha, fit_intercept=fit_intercept).fit(X, y)

    ref = _reference_incremental_weights(
        [(X, y)], alpha, fit_intercept, [False])

    assert np.allclose(
        reg._output_weights, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("size", _P3_SIZES)
def test_p3_solve_parity_residual_branch(fit_intercept, size) -> None:
    """Multiple partial_fit() calls exercise the incremental residual path."""
    n_samples, n_features, n_targets = size
    rs = np.random.RandomState((hash(size) ^ 0x5bd1e995) % (2 ** 31))
    X = rs.normal(size=(n_samples, n_features))
    y = rs.normal(size=(n_samples, n_targets))
    alpha = 1e-3

    idx = np.array_split(np.arange(n_samples), 3)
    batches = [(X[prt, :], y[prt, :]) for prt in idx]

    reg = IncrementalRegression(alpha=alpha, fit_intercept=fit_intercept)
    for i, (Xb, yb) in enumerate(batches):
        reg.partial_fit(Xb, yb, partial_normalize=False, reset=(i == 0))

    ref = _reference_incremental_weights(
        batches, alpha, fit_intercept, [False] * len(batches))

    assert np.allclose(
        reg._output_weights, ref, atol=1e-12, rtol=1e-12)


def test_p3_solve_parity_postpone_then_residual() -> None:
    """postpone_inverse batches then non-postponed calls (both branches)."""
    n_samples, n_features, n_targets = 60, 10, 2
    rs = np.random.RandomState(7)
    X = rs.normal(size=(n_samples, n_features))
    y = rs.normal(size=(n_samples, n_targets))
    alpha = 1e-3

    idx = np.array_split(np.arange(n_samples), 4)
    batches = [(X[prt, :], y[prt, :]) for prt in idx]
    # first two batches postpone, last two trigger main solve then residual
    postpone = [True, True, False, False]

    reg = IncrementalRegression(alpha=alpha, fit_intercept=True)
    for i, (Xb, yb) in enumerate(batches):
        reg.partial_fit(Xb, yb, partial_normalize=False, reset=(i == 0),
                        postpone_inverse=postpone[i])

    ref = _reference_incremental_weights(batches, alpha, True, postpone)

    assert np.allclose(
        reg._output_weights, ref, atol=1e-12, rtol=1e-12)
