"""Testing for Linear model module."""

import numpy as np
import pytest

from sklearn.base import is_regressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from pyrcn.linear_model import IncrementalRegression
from sklearn.linear_model import Ridge


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
    print("tests: {0}\nregr: {1}".format(y_test, y_reg))
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
    print("tests: {0}\nregr: {1}".format(y_test, y_reg))
    np.testing.assert_allclose(y_reg, y_test, rtol=.01, atol=.15)


def test_compare_ridge() -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_diabetes, y_diabetes, test_size=10, random_state=42)

    i_reg = IncrementalRegression(alpha=.01).fit(X_train, y_train)
    ridge = Ridge(alpha=.01, solver='svd').fit(X_train, y_train)

    print("incremental: {0} ridge: {1}".format(i_reg.coef_, ridge.coef_))
    np.testing.assert_allclose(i_reg.coef_, ridge.coef_, rtol=.0001)
