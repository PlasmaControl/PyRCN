"""
Testing for coates preprocessing module (pyrcn.coates)
"""
import scipy
import numpy as np

import pytest

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

from pyrcn.preprocessing import Coates


X_digits, y_digits = load_digits(return_X_y=True)


def test_image_transform():
    rs = np.random.RandomState(42)
    test_image = rs.randint(0, 255, size=(2, 7, 11))
    test_array = test_image.reshape((2, 7 * 11))
    np.testing.assert_array_equal(Coates._reshape_images_to_arrays(test_image, (7, 11))[0, :], test_array[0, :])
    np.testing.assert_array_equal(Coates._reshape_arrays_to_images(test_array, (7, 11))[0, :], test_image[0, :])

"""
def test_extract_random_patches():
    test_pictures = np.arange(2 * 7 * 11).reshape((2, 7, 11))
    indices_patch = np.mod(np.arange(0, 8), 4) + (np.arange(0, 8) // 4) * 11
    random_patches = Coates._extract_random_patches(
        test_pictures, image_size=(7, 11), patch_size=(2, 4), n_patches=2, random_state=42)
    np.testing.assert_array_equal(random_patches[0, :] - np.min(random_patches[0, :]).astype(int), indices_patch)


def test_extract_equidistant_patches():
    test_pictures = np.arange(2*7*11).reshape((2, 7, 11))
    patches = Coates._extract_equidistant_patches_2d(
        test_pictures, image_size=(7, 11), patch_size=(2, 4), stride_size=(2, 5))
    np.testing.assert_array_equal(Coates._reshape_arrays_to_images(patches, (6, 2, 4))[0, -1, :, :],
                                  test_pictures[0, 4:6, 5:9])

"""
def test_fit():
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        n_patches=200,
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    trf.fit(X_digits)
    assert len(trf.clusterer.cluster_centers_) == 20

"""
def test_transform():
    trf = Coates(
        image_size=(8, 8),
        patch_size=(3, 3),
        stride_size=(4, 3),
        n_patches=200,
        clusterer=KMeans(n_clusters=20, random_state=42),
        random_state=42)
    features = trf.fit_transform(X_digits)
    assert len(trf.clusterer.cluster_centers_) == 20
"""


if __name__ == "__main__":
    test_image_transform()
    test_extract_random_patches()
    test_extract_equidistant_patches()
    test_fit()
    test_transform()
