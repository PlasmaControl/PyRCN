"""Testing for postprocessing module (pyrcn.postprocessing)."""

from __future__ import annotations

import numpy as np

from pyrcn.postprocessing import NormalDistribution


def test_normal_distribution_init_defaults() -> None:
    print('\ntest_normal_distribution_init_defaults():')
    nd = NormalDistribution()
    assert nd._mean == 0
    assert nd._std == 0
    assert nd._size == 1


def test_normal_distribution_init_size() -> None:
    print('\ntest_normal_distribution_init_size():')
    nd = NormalDistribution(size=5)
    assert nd._size == 5


def test_normal_distribution_transform_shape() -> None:
    print('\ntest_normal_distribution_transform_shape():')
    nd = NormalDistribution(size=7)
    out = nd.transform(np.zeros((3, 2)))
    assert out.shape == (7,)
    # With mean=0 and std=0 the variates collapse onto the mean.
    np.testing.assert_array_equal(out, np.zeros(7))


def test_normal_distribution_transform_scalar_size() -> None:
    print('\ntest_normal_distribution_transform_scalar_size():')
    nd = NormalDistribution()
    out = nd.transform(np.zeros((4, 4)))
    assert out.shape == (1,)


def test_normal_distribution_fit_1d() -> None:
    print('\ntest_normal_distribution_fit_1d():')
    data = np.random.RandomState(0).normal(loc=2.0, scale=3.0, size=500)
    nd = NormalDistribution()
    returned = nd.fit(data)
    assert returned is nd
    # scipy.stats.norm.fit uses the MLE, whose std matches the population
    # std (ddof=0).
    assert np.isclose(nd._mean, data.mean())
    assert np.isclose(nd._std, data.std())


def test_normal_distribution_fit_2d() -> None:
    print('\ntest_normal_distribution_fit_2d():')
    data = np.random.RandomState(1).normal(loc=-1.0, scale=2.0, size=(40, 10))
    nd = NormalDistribution()
    returned = nd.fit(data)
    assert returned is nd
    assert np.isclose(nd._mean, data.mean())
    assert np.isclose(nd._std, data.std())


def test_normal_distribution_fit_transform() -> None:
    print('\ntest_normal_distribution_fit_transform():')
    data = np.random.RandomState(2).normal(size=100)
    nd = NormalDistribution(size=7)
    out = nd.fit_transform(data)
    assert isinstance(out, np.ndarray)
    assert out.shape == (7,)
