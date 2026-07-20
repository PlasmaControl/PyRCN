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
