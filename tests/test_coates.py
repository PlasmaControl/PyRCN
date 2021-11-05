"""Testing for coates preprocessing module (pyrcn.preprocessing.coates)."""
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

from pyrcn.preprocessing import Coates


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
