"""Testing for Metrics module."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics
from sklearn.datasets import make_regression

import pyrcn.metrics

rng_true = np.random.RandomState(42)
rng_pred = np.random.RandomState(1234)
y_true_mono = np.empty(shape=(10,), dtype=object)
y_pred_mono = np.empty(shape=(10,), dtype=object)
sample_weight = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_mono[k] = make_regression(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_true)
    _, y_pred_mono[k] = make_regression(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_pred)
    sample_weight[k] = np.ones_like(y_true_mono[k])

y_true_multi = np.empty(shape=(10,), dtype=object)
y_pred_multi = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_multi[k] = make_regression(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_true)
    _, y_pred_multi[k] = make_regression(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_pred)

rng_pos = np.random.RandomState(24)
y_true_pos = np.empty(shape=(10,), dtype=object)
y_pred_pos = np.empty(shape=(10,), dtype=object)
sample_weight_pos = np.empty(shape=(10,), dtype=object)
for k in range(10):
    n = 10 * (k + 1)
    y_true_pos[k] = rng_pos.uniform(0.5, 2.0, size=n)
    y_pred_pos[k] = rng_pos.uniform(0.5, 2.0, size=n)
    sample_weight_pos[k] = np.ones(n)


def test_mean_absolute_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_error(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_absolute_error(
            y_true=y_true_mono, y_pred=y_pred_mono), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_error(
            y_true=y_true_mono, y_pred=y_pred_mono,
            sample_weight=sample_weight),
        sklearn.metrics.mean_absolute_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_error(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_absolute_error(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_error(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.mean_absolute_error(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_absolute_error(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_mean_absolute_percentage_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_mono, y_pred=y_pred_mono), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_mono, y_pred=y_pred_mono,
            sample_weight=sample_weight),
        sklearn.metrics.mean_absolute_percentage_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.mean_absolute_percentage_error(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_absolute_percentage_error(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_mean_squared_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_squared_error(
            y_true=y_true_mono, y_pred=y_pred_mono), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_mono, y_pred=y_pred_mono,
            sample_weight=sample_weight),
        sklearn.metrics.mean_squared_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_squared_error(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.mean_squared_error(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_median_absolute_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.median_absolute_error(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.median_absolute_error(
            y_true=y_true_mono, y_pred=y_pred_mono), 0), True)
    """This line needs to be verified"""
    np.testing.assert_almost_equal(
        pyrcn.metrics.median_absolute_error(
            y_true=y_true_mono, y_pred=y_pred_mono,
            sample_weight=sample_weight),
        sklearn.metrics.median_absolute_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)), -1)

    np.testing.assert_equal(
        pyrcn.metrics.median_absolute_error(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.median_absolute_error(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.median_absolute_error(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.median_absolute_error(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.median_absolute_error(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_explained_variance_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.explained_variance_score(
            y_true=y_true_mono, y_pred=y_true_mono), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.explained_variance_score(
            y_true=y_true_mono, y_pred=y_pred_mono), 1), True)
    np.testing.assert_equal(pyrcn.metrics.explained_variance_score(
        y_true=y_true_mono, y_pred=y_pred_mono, sample_weight=sample_weight),
        sklearn.metrics.explained_variance_score(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.explained_variance_score(
            y_true=y_true_multi, y_pred=y_true_multi), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.explained_variance_score(
            y_true=y_true_multi, y_pred=y_pred_multi), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.explained_variance_score(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.explained_variance_score(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.explained_variance_score(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_r2_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.r2_score(
            y_true=y_true_mono, y_pred=y_true_mono), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.r2_score(
            y_true=y_true_mono, y_pred=y_pred_mono), 1), True)
    np.testing.assert_equal(pyrcn.metrics.r2_score(
        y_true=y_true_mono, y_pred=y_pred_mono, sample_weight=sample_weight),
        sklearn.metrics.r2_score(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.r2_score(y_true=y_true_multi, y_pred=y_true_multi), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.r2_score(
            y_true=y_true_multi, y_pred=y_pred_multi), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.r2_score(y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.r2_score(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.r2_score(y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_max_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.max_error(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(np.greater(
        pyrcn.metrics.max_error(
            y_true=y_true_mono, y_pred=y_pred_mono), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.max_error(
            y_true=y_true_mono, y_pred=y_pred_mono),
        sklearn.metrics.max_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.max_error(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.max_error(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.max_error(y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.max_error(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.max_error(y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_mean_tweedie_deviance() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_mono, y_pred=y_true_mono), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_mono, y_pred=y_pred_mono), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_mono, y_pred=y_pred_mono,
            sample_weight=sample_weight),
        sklearn.metrics.mean_tweedie_deviance(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))

    np.testing.assert_equal(
        pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_multi, y_pred=y_true_multi), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_multi, y_pred=y_pred_multi), 0), True)
    np.testing.assert_equal(
        pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_multi, y_pred=y_pred_multi),
        sklearn.metrics.mean_tweedie_deviance(
            y_true=np.concatenate(y_true_multi),
            y_pred=np.concatenate(y_pred_multi)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_tweedie_deviance(
            y_true=y_true_multi[0], y_pred=y_pred_multi[0])


def test_root_mean_squared_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_mono, y_pred=y_true_mono, squared=False), 0)
    np.testing.assert_almost_equal(
        pyrcn.metrics.mean_squared_error(
            y_true=y_true_mono, y_pred=y_pred_mono, squared=False),
        sklearn.metrics.root_mean_squared_error(
            y_true=np.concatenate(y_true_mono),
            y_pred=np.concatenate(y_pred_mono)))


def test_mean_squared_log_error() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_squared_log_error(
            y_true=y_true_pos, y_pred=y_true_pos), 0)
    np.testing.assert_equal(
        np.greater(pyrcn.metrics.mean_squared_log_error(
            y_true=y_true_pos, y_pred=y_pred_pos), 0), True)
    np.testing.assert_almost_equal(
        pyrcn.metrics.mean_squared_log_error(
            y_true=y_true_pos, y_pred=y_pred_pos,
            sample_weight=sample_weight_pos),
        sklearn.metrics.mean_squared_log_error(
            y_true=np.concatenate(y_true_pos),
            y_pred=np.concatenate(y_pred_pos)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_squared_log_error(
            y_true=y_true_pos[0], y_pred=y_pred_pos[0])


def test_mean_poisson_deviance() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_poisson_deviance(
            y_true=y_true_pos, y_pred=y_true_pos), 0)
    np.testing.assert_almost_equal(
        pyrcn.metrics.mean_poisson_deviance(
            y_true=y_true_pos, y_pred=y_pred_pos,
            sample_weight=sample_weight_pos),
        sklearn.metrics.mean_poisson_deviance(
            y_true=np.concatenate(y_true_pos),
            y_pred=np.concatenate(y_pred_pos)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_poisson_deviance(
            y_true=y_true_pos[0], y_pred=y_pred_pos[0])


def test_mean_gamma_deviance() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.mean_gamma_deviance(
            y_true=y_true_pos, y_pred=y_true_pos), 0)
    np.testing.assert_almost_equal(
        pyrcn.metrics.mean_gamma_deviance(
            y_true=y_true_pos, y_pred=y_pred_pos,
            sample_weight=sample_weight_pos),
        sklearn.metrics.mean_gamma_deviance(
            y_true=np.concatenate(y_true_pos),
            y_pred=np.concatenate(y_pred_pos)))
    with pytest.raises(TypeError):
        pyrcn.metrics.mean_gamma_deviance(
            y_true=y_true_pos[0], y_pred=y_pred_pos[0])
