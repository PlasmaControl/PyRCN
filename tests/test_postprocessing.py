"""Testing for postprocessing module (pyrcn.postprocessing)."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_normal_distribution_fit_raises() -> None:
    print('\ntest_normal_distribution_fit_raises():')
    nd = NormalDistribution()
    # scipy.stats.norm.fit expects positional ``data``; the estimator
    # forwards keyword ``X``/``y`` instead, so fit raises TypeError.
    with pytest.raises(TypeError):
        nd.fit(np.random.RandomState(0).rand(20))


def test_normal_distribution_fit_transform_raises() -> None:
    print('\ntest_normal_distribution_fit_transform_raises():')
    nd = NormalDistribution()
    with pytest.raises(TypeError):
        nd.fit_transform(np.random.RandomState(0).rand(20))
