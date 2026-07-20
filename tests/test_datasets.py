"""Testing for datasets module (pyrcn.datasets)."""

from __future__ import annotations

import numpy as np
import pytest

from pyrcn.datasets import mackey_glass, lorenz, load_digits


def test_mackey_glass_default() -> None:
    print('\ntest_mackey_glass_default():')
    X, y = mackey_glass(n_timesteps=50)
    assert X.shape == (50,)
    assert y.shape == (50,)
    assert np.all(np.isfinite(X))


def test_mackey_glass_random_state_instance() -> None:
    print('\ntest_mackey_glass_random_state_instance():')
    rs = np.random.RandomState(1234)
    X, y = mackey_glass(n_timesteps=30, random_state=rs)
    assert X.shape == (30,)
    assert y.shape == (30,)


def test_mackey_glass_random_state_none() -> None:
    print('\ntest_mackey_glass_random_state_none():')
    X, y = mackey_glass(n_timesteps=30, random_state=None)
    assert X.shape == (30,)
    assert y.shape == (30,)


def test_mackey_glass_reproducible_seed() -> None:
    print('\ntest_mackey_glass_reproducible_seed():')
    X1, _ = mackey_glass(n_timesteps=40, random_state=7)
    X2, _ = mackey_glass(n_timesteps=40, random_state=7)
    np.testing.assert_array_equal(X1, X2)


def test_mackey_glass_tau_zero() -> None:
    print('\ntest_mackey_glass_tau_zero():')
    X, y = mackey_glass(n_timesteps=20, tau=0)
    assert X.shape == (20,)
    assert np.all(np.isfinite(X))


def test_mackey_glass_n_future() -> None:
    print('\ntest_mackey_glass_n_future():')
    X, y = mackey_glass(n_timesteps=25, n_future=3)
    assert X.shape == (25,)
    assert y.shape == (25,)
    np.testing.assert_array_equal(X[3:], y[:-3])


def test_lorenz_default_x0() -> None:
    print('\ntest_lorenz_default_x0():')
    X, y = lorenz(n_timesteps=30)
    assert X.shape == (30, 3)
    assert y.shape == (30, 3)
    assert np.all(np.isfinite(X))


def test_lorenz_custom_x0() -> None:
    print('\ntest_lorenz_custom_x0():')
    X, y = lorenz(n_timesteps=15, x_0=[2.0, 1.0, 1.0])
    assert X.shape == (15, 3)
    assert y.shape == (15, 3)


def test_load_digits_bunch() -> None:
    print('\ntest_load_digits_bunch():')
    data = load_digits(n_class=3)
    assert data.data.shape[1] == 64
    assert set(np.unique(data.target)) == {0, 1, 2}


def test_load_digits_return_x_y() -> None:
    print('\ntest_load_digits_return_x_y():')
    X, y = load_digits(n_class=2, return_X_y=True)
    assert X.shape[1] == 64
    assert set(np.unique(y)) == {0, 1}


def test_load_digits_as_sequence() -> None:
    print('\ntest_load_digits_as_sequence():')
    X, y = load_digits(n_class=2, return_X_y=True, as_sequence=True)
    assert X.dtype == object
    assert y.dtype == object
    assert X[0].shape == (8, 8)
    assert y[0].shape == (1,)


@pytest.mark.parametrize("kwargs", [
    dict(return_X_y=True),
    dict(as_frame=True),
    dict(as_frame=True, return_X_y=True),
])
def test_load_digits_not_as_sequence(kwargs: dict) -> None:
    print('\ntest_load_digits_not_as_sequence():')
    # as_sequence defaults to False so the else-branch is used.
    result = load_digits(n_class=2, as_sequence=True, **kwargs)
    assert result is not None
