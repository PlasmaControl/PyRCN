import scipy
import numpy as np

import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, PatchExtractor


class Coates(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            image_size=(),
            patch_size=(),
            stride_size = (),
            n_patches: int = 200,
            normalize=True,
            whiten=True,
            clusterer=KMeans(),
            random_state=None):
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.n_patches = n_patches
        self.normalize = normalize
        self.whiten = whiten
        self.clusterer = clusterer
        self.random_state = check_random_state(random_state)
        self._normalizer = None
        self._whitener = None

    def fit(self, X, y=None):
        self._validate_hyperparameters()
        self.clusterer.fit(self._preprocessing(Coates._extract_random_patches_2d(
            X,
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_patches=self.n_patches,
            random_state=self.random_state)))
        return self

    def transform(self, X, y=None):
        patches = Coates._extract_equidistant_patches_2d(
            X, image_size=self.image_size, patch_size=self.patch_size, stride_size=self.stride_size)
        patches_2d = patches.reshape((patches.shape[0] * patches.shape[1], np.prod(self.patch_size)))
        patches_2d_preprocessed = self._preprocessing(patches_2d)
        patches_preprocessed = patches_2d_preprocessed.reshape(patches.shape)
        return np.dot(patches_2d_preprocessed, self.clusterer.cluster_centers_.T)

    def _validate_hyperparameters(self):
        """
        Validate the hyperparameter. Ensure that the parameter ranges and dimensions are valid.
        Returns
        -------

        """
        if len(self.patch_size) != 2:
            raise ValueError('patch_size has not a valid format, got {0}'.format(self.patch_size))

        if len(self.stride_size) != 2:
            print('stride_size has not a valid format, got {0}. Set stride_size = patch_size '.format(self.stride_size))
            self.stride_size = self.patch_size

        if any(stride < patch for stride, patch in zip(self.stride_size, self.patch_size)):
            raise ValueError('stride_size must be greater or equal than patch_size, '
                             'got stride_size = {0}, patch_size = {1}'.format(self.stride_size, self.patch_size))

        if getattr(self.clusterer, "_estimator_type", None) != "clusterer":
            raise TypeError('clusterer must be of type clusterer, got {0}'.format(self.clusterer))

    @staticmethod
    def _reshape_arrays_to_images(X, image_size):
        if (X.ndim == 3 or X.ndim == 4) and (X.shape[0], ) + image_size == X.shape:
            images = X
        else:
            if len(image_size) == 2 or len(image_size) == 3:
                new_size = (X.shape[0], ) + image_size
                images = X.reshape(new_size)
            else:
                raise ValueError('image_size is not a valid image size format, got {0}'.format(image_size))
        return images

    @staticmethod
    def _reshape_images_to_arrays(X, image_size):
        return X.reshape((X.shape[0], np.prod(image_size)))

    @staticmethod
    def _extract_random_patches_2d(X, image_size, patch_size, n_patches, random_state=None):
        rs = check_random_state(random_state)

        random_images = Coates._reshape_arrays_to_images(X[rs.randint(0, high=X.shape[0], size=n_patches)], image_size)

        random_patches = PatchExtractor(patch_size=patch_size, max_patches=1, random_state=rs).transform(random_images)

        return Coates._reshape_images_to_arrays(random_patches, patch_size)

    @staticmethod
    def _extract_equidistant_patches_2d(X, image_size, patch_size, stride_size):
        nm_patches = (image_size[0] // stride_size[0], image_size[1] // stride_size[1])

        # e.g. s=(2, 3), i=(5, 6) => [0, 1, 4, 5, 9, 10]
        indices_patch = np.mod(np.arange(0, np.prod(patch_size)), patch_size[1]) \
            + (np.arange(0, np.prod(patch_size)) // patch_size[1]) * image_size[1]

        # preallocate
        indices_patches = np.zeros(shape=(nm_patches + (np.prod(patch_size), )), dtype=int)

        # set indices
        for i in range(nm_patches[0]):  # stride_size[0]/image_size[0]
            for j in range(nm_patches[1]):  # stride_size[1]/image_size[1]
                indices_patches[i, j, :] = i * stride_size[0] * image_size[1] + j * stride_size[1] + indices_patch

        # indices matrix to array
        indices_patches = indices_patches.reshape((np.prod(nm_patches) * np.prod(patch_size), )).astype(int)

        images = Coates._reshape_images_to_arrays(X, image_size=image_size)
        patches = images[:, np.array(indices_patches, dtype=int)]

        return patches.reshape((X.shape[0], np.prod(nm_patches), np.prod(patch_size)))

    def _preprocessing(self, X):
        X_preprocessed = X

        if self.normalize:
            if self._normalizer is None:
                self._normalizer = StandardScaler().fit(X_preprocessed)

            X_preprocessed = self._normalizer.transform(X_preprocessed)

        if self.whiten:
            if self._whitener is None:
                self._whitener = PCA(whiten=True).fit(X_preprocessed)

            X_preprocessed = self._whitener.transform(X_preprocessed)

        return X_preprocessed

