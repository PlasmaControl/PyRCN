"""Testing for coates preprocessing module (pyrcn.preprocessing.coates)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from pyrcn.preprocessing import Coates
from pyrcn.preprocessing._coates import (inplace_pool_average,
                                         inplace_pool_max, inplace_pool_mean,
                                         inplace_pool_min)
from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)


def test_image_transform() -> None:
    rs = np.random.RandomState(42)
    test_image = rs.randint(0, 255, size=(2, 7, 11))
    test_array = test_image.reshape((2, 7 * 11))
    np.testing.assert_array_equal(Coates._reshape_images_to_arrays(
        test_image, (7, 11))[0, :], test_array[0, :])
    np.testing.assert_array_equal(Coates._reshape_arrays_to_images(
        test_array, (7, 11))[0, :], test_image[0, :])


def test_fit() -> None:
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        n_patches=200,
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    trf.fit(X_digits)
    assert len(trf.clusterer.cluster_centers_) == 20


def test_pooling_functions() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert inplace_pool_max(X) == 4.0
    assert inplace_pool_min(X) == 1.0
    assert inplace_pool_average(X) == 2.5
    assert inplace_pool_mean(X) == 2.5
    np.testing.assert_array_equal(inplace_pool_max(X, axis=0),
                                  np.array([3.0, 4.0]))
    np.testing.assert_array_equal(inplace_pool_min(X, axis=1),
                                  np.array([1.0, 3.0]))
    np.testing.assert_array_equal(inplace_pool_average(X, axis=0),
                                  np.array([2.0, 3.0]))
    np.testing.assert_array_equal(inplace_pool_mean(X, axis=1),
                                  np.array([1.5, 3.5]))


def test_fit_default_clusterer() -> None:
    # clusterer is None -> KMeans() is set inside fit (line 172)
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        n_patches=200,
        random_state=42)
    trf.fit(X_digits)
    assert isinstance(trf.clusterer, KMeans)
    assert trf.clusterer.cluster_centers_.shape[0] == 8


def test_validate_bad_patch_size() -> None:
    trf = Coates(image_size=(8, 8), patch_size=(3,),
                 clusterer=KMeans(n_clusters=4, random_state=42),
                 random_state=42)
    with pytest.raises(ValueError, match='patch_size has invalid format'):
        trf.fit(X_digits)


def test_validate_stride_smaller_than_patch() -> None:
    trf = Coates(image_size=(8, 8), patch_size=(3, 3),
                 stride_size=(1, 1),
                 clusterer=KMeans(n_clusters=4, random_state=42),
                 random_state=42)
    with pytest.raises(ValueError, match='stride_size must be greater'):
        trf.fit(X_digits)


def test_validate_bad_pooling_func() -> None:
    trf = Coates(image_size=(8, 8), patch_size=(3, 3),
                 stride_size=(3, 3), pooling_func='invalid',
                 clusterer=KMeans(n_clusters=4, random_state=42),
                 random_state=42)
    with pytest.raises(ValueError, match='is not supported'):
        trf.fit(X_digits)


def test_validate_pooling_size_too_large() -> None:
    trf = Coates(image_size=(8, 8), patch_size=(3, 3),
                 stride_size=(3, 3), pooling_func='max',
                 pooling_size=(5, 5),
                 clusterer=KMeans(n_clusters=4, random_state=42),
                 random_state=42)
    with pytest.raises(ValueError, match='#patches must be greater'):
        trf.fit(X_digits)


def test_validate_not_a_clusterer() -> None:
    trf = Coates(image_size=(8, 8), patch_size=(3, 3),
                 stride_size=(3, 3), pooling_func='max',
                 clusterer=StandardScaler(),
                 random_state=42)
    with pytest.raises(TypeError, match='clusterer must be of type'):
        trf.fit(X_digits)


def test_transform() -> None:
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        stride_size=(3, 3),
        n_patches=200,
        pooling_func='max',
        pooling_size=(2, 2),
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    trf.fit(X_digits)
    features = trf.transform(X_digits)
    assert features.shape == (X_digits.shape[0], 20)


def test_transform_all_pooling_funcs() -> None:
    for pooling_func in ('max', 'min', 'average', 'mean'):
        trf = Coates(
            image_size=(8, 8),
            patch_size=(3, 3),
            stride_size=(3, 3),
            n_patches=200,
            pooling_func=pooling_func,
            pooling_size=(2, 2),
            clusterer=KMeans(n_clusters=10, random_state=42),
            random_state=42)
        trf.fit(X_digits)
        features = trf.transform(X_digits)
        assert features.shape == (X_digits.shape[0], 10)


def test_inverse_transform() -> None:
    # normalize/whiten disabled so inverse_preprocessing is the identity
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        stride_size=(3, 3),
        n_patches=200,
        normalize=False,
        whiten=False,
        pooling_func='max',
        pooling_size=(2, 2),
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    trf.fit(X_digits)
    features = trf.transform(X_digits)
    patches = trf.inverse_transform(features)
    assert patches.shape == (X_digits.shape[0], 1, 3, 3)


@pytest.mark.parametrize('normalize, whiten', [
    (False, False), (True, False), (False, True), (True, True)])
def test_inverse_transform_with_preprocessing(normalize: bool,
                                              whiten: bool) -> None:
    # Regression test for bug #4: inverse_transform fed a 3-D array into the
    # sklearn inverse_transform calls (which require <=2-D), and
    # _inverse_preprocessing reversed the forward operations in the wrong
    # order. Both must now work for every normalize/whiten combination.
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        stride_size=(3, 3),
        n_patches=200,
        normalize=normalize,
        whiten=whiten,
        pooling_func='max',
        pooling_size=(2, 2),
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    trf.fit(X_digits)
    features = trf.transform(X_digits)
    patches = trf.inverse_transform(features)
    assert patches.shape == (X_digits.shape[0], 1, 3, 3)


@pytest.mark.parametrize('normalize, whiten', [
    (False, False), (True, False), (False, True), (True, True)])
def test_preprocessing_roundtrip_allclose(normalize: bool,
                                          whiten: bool) -> None:
    # Bug #4: a forward preprocessing pass followed by its inverse must
    # recover the original 2-D patch array (correct inverse order).
    trf = Coates(normalize=normalize, whiten=whiten)
    rs = np.random.RandomState(42)
    P = rs.rand(50, 9)
    restored = trf._inverse_preprocessing(trf._preprocessing(P))
    assert np.allclose(restored, P, atol=1e-8)


def test_inverse_preprocessing_roundtrip() -> None:
    trf = Coates(normalize=True, whiten=True)
    rs = np.random.RandomState(42)
    Y = rs.rand(20, 5)
    trf._normalizer.fit(Y)
    trf._whitener = trf._whitener.fit(Y)
    transformed = trf._whitener.transform(trf._normalizer.transform(Y))
    restored = trf._inverse_preprocessing(transformed)
    assert restored.shape == Y.shape


def test_inverse_preprocessing_normalizer_not_fitted() -> None:
    trf = Coates(normalize=True, whiten=False)
    trf._normalizer = None
    rs = np.random.RandomState(42)
    with pytest.raises(NotFittedError, match='normalizer has not been'):
        trf._inverse_preprocessing(rs.rand(4, 3))


def test_inverse_preprocessing_whitener_not_fitted() -> None:
    trf = Coates(normalize=False, whiten=True)
    trf._whitener = None
    rs = np.random.RandomState(42)
    with pytest.raises(NotFittedError, match='whitener has not been'):
        trf._inverse_preprocessing(rs.rand(4, 3))
