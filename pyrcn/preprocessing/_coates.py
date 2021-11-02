"""The :mod:`coates` contains the Coates preprocessing."""

import sys
if sys.version_info >= (3, 8):
    from typing import Union, Callable, Dict, Tuple, Literal
else:
    from typing_extensions import Literal
    from typing import Union, Callable, Dict, Tuple
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import PatchExtractor


def inplace_pool_max(X: np.ndarray,
                     axis: Union[None, int, np.integer] = None) \
                         -> Union[float, np.ndarray]:
    """
    Apply max-Pooling on an array.

    Parameters
    ----------
    X : np.ndarray
        The patch on which to apply max pooling
    axis : Union[None, int, np.integer], default=None
        Axis along which to apply max-Pooling. This is important for batch processing.

    Returns
    -------
    y : Union[float, np.ndarray]
        The pooling value or array of pooling values in case of batch processing
    """
    return np.max(X, axis=axis)


def inplace_pool_min(X: np.ndarray,
                     axis: Union[None, int, np.integer] = None) \
                         -> Union[float, np.ndarray]:
    """
    Apply min-Pooling on an array.

    Parameters
    ----------
    X : np.ndarray
        The patch on which to apply min pooling
    axis : Union[None, int, np.integer], default=None
        Axis along which to apply min-Pooling. This is important for batch processing.

    Returns
    -------
    y : Union[float, np.ndarray]
        The pooling value or array of pooling values in case of batch processing
    """
    return np.min(X, axis=axis)


def inplace_pool_average(X: np.ndarray,
                         axis: Union[None, int, np.integer] = None) \
                             -> Union[float, np.ndarray]:
    """
    Apply average-Pooling on an array.

    Parameters
    ----------
    X : np.ndarray
        The patch on which to apply average pooling
    axis : Union[None, int, np.integer], default=None
        Axis along which to apply average-Pooling.
        This is important for batch processing.

    Returns
    -------
    y : Union[float, np.ndarray]
        The pooling value or array of pooling values in case of batch processing
    """
    return np.average(X, axis=axis)


def inplace_pool_mean(X: np.ndarray, axis: None = None) -> np.number:
    """
    Apply mean-Pooling on an array.

    Parameters
    ----------
    X : np.ndarray
        The patch on which to apply mean pooling
    axis : None
        Axis along which to apply mean-Pooling. This is important for batch processing.

    Returns
    -------
    y : Union[float, np.ndarray]
        The pooling value or array of pooling values in case of batch processing
    """
    return np.mean(X, axis=axis)


POOLINGS: Dict[str, Callable] = {'max': inplace_pool_max,
                                 'min': inplace_pool_min,
                                 'average': inplace_pool_average,
                                 'mean': inplace_pool_mean}


class Coates(BaseEstimator, TransformerMixin):
    """
    Coates Preprocessing.

    Parameters
    ----------
    image_size : Tuple, default=()
    patch_size : Tuple, default=()
    stride_size : Tuple, default=(),
    n_patches : Union[int, np.integer], default=200,
    normalize : bool, default=True,
    whiten : bool, default=True,
    clusterer : Callable, default=KMeans(),
    pooling_func : Literal['max', 'min', 'average', 'mean'], default='max',
    pooling_size : Tuple, default=(),
    random_state : Union[None, int, np.random.RandomState], default=None
    """

    def __init__(self, image_size: Tuple = (), patch_size: Tuple = (),
                 stride_size: Tuple = (), n_patches: Union[int, np.integer] = 200,
                 normalize: bool = True, whiten: bool = True,
                 clusterer: ClusterMixin = KMeans(),
                 pooling_func: Literal['max', 'min', 'average', 'mean'] = 'max',
                 pooling_size: Tuple = (),
                 random_state: Union[None, int, np.random.RandomState] = None):
        """Construct the Coates."""
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.n_patches = n_patches
        self.normalize = normalize
        self.whiten = whiten
        self.clusterer = clusterer
        self.pooling_func = pooling_func
        self.pooling_size = pooling_size
        self.random_state = check_random_state(random_state)
        self._normalizer = StandardScaler()
        self._whitener = PCA(whiten=True)

    def fit(self, X: np.ndarray, y: None = None) -> TransformerMixin:
        """
        Fit the Coates.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : None
            ignored

        Returns
        -------
        self : returns a trained Coates.
        """
        self._validate_hyperparameters()
        self.clusterer.fit(self._preprocessing(Coates._extract_random_patches(
            X,
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_patches=self.n_patches,
            random_state=self.random_state)))
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Transform with the fitted clusterer.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : None
            ignored

        Returns
        -------
        features : returns the transformed features.
        """
        # patches[#samples][#patches][#features]
        patches = Coates._extract_equidistant_patches(
            X, image_size=self.image_size, patch_size=self.patch_size,
            stride_size=self.stride_size)

        # preprocessing
        patches_2d = patches.reshape([patches.shape[0] * patches.shape[1],
                                      int(np.prod(self.patch_size, axis=None))])
        patches_2d_preprocessed = self._preprocessing(patches_2d)
        patches_preprocessed = patches_2d_preprocessed.reshape(patches.shape)

        # feature mapping
        features = Coates._feature_mapping(patches_preprocessed,
                                           self.clusterer.cluster_centers_)

        # pooling
        return self._pooling(features)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        inverse transform with the fitted clusterer.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        patches : returns the original features.
        """
        patch_array = Coates._reshape_arrays_to_images(X, image_size=(
            int(X.shape[-1] / self.clusterer.cluster_centers_.shape[0]),
            self.clusterer.cluster_centers_.shape[0]))

        patches = Coates._reshape_arrays_to_images(
            self._inverse_preprocessing(self._inverse_feature_mapping(
                patch_array, self.clusterer.cluster_centers_)),
            image_size=self.patch_size)

        return patches

    def _validate_hyperparameters(self) -> None:
        """
        Validate the hyperparameters.

        Ensure that the parameter ranges and dimensions are valid.

        """
        if len(self.patch_size) not in {2, 3}:
            raise ValueError('patch_size has invalid format, got {0}'
                             .format(self.patch_size))

        if len(self.stride_size) != len(self.patch_size):
            print('stride_size has invalid format, got {0}. '
                  'Set stride_size = patch_size '.format(self.stride_size))
            self.stride_size = self.patch_size

        if any(stride < patch for stride, patch in zip(self.stride_size,
                                                       self.patch_size)):
            raise ValueError('stride_size must be greater or equal than patch_size, '
                             'got stride_size = {0}, patch_size = {1}'
                             .format(self.stride_size, self.patch_size))

        if self.pooling_func not in POOLINGS:
            raise ValueError("The pooling_func '%s' is not supported. Supported "
                             "activations are %s." % (self.pooling_func, POOLINGS))

        if any(patches < pool for patches, pool in
               zip(Coates._patches_per_image(self.image_size, self.stride_size),
                   self.pooling_size)):
            raise ValueError('#patches must be greater or equal than pooling_size, '
                             'got patch_size = {0}, pooling_size = {1}'
                             .format(self.patch_size, self.pooling_size))

        if getattr(self.clusterer, "_estimator_type", None) != "clusterer":
            raise TypeError('clusterer must be of type clusterer, got {0}'
                            .format(self.clusterer))

    @staticmethod
    def _reshape_arrays_to_images(X: np.ndarray, image_size: Tuple) -> np.ndarray:
        """
        Reshape an array to image.

        Parameters
        ----------
        X : ndarray of shape (..., n_image_array)
        param image_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)

        Returns
        -------
        y : ndarray of shape (..., n_height, n_width)
        or (..., n_height, n_width, n_channels)
        """
        index_dimensions = X.shape[:-1]
        return X.reshape(index_dimensions + image_size)

    @staticmethod
    def _reshape_images_to_arrays(X: np.ndarray, image_size: Tuple) -> np.ndarray:
        """
        Reshape an image to array.

        Parameters
        ----------
        X : ndarray of shape (..., n_height, n_width)
        or (..., n_height, n_width, n_channels)
        image_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)

        Returns
        -------
        y : ndarray of shape (..., n_image_array)
        """
        index_dimensions = X.shape[:-len(image_size)]
        return X.reshape(index_dimensions + (int(np.prod(image_size)), ))

    @staticmethod
    def _patches_per_image(image_size: Tuple, stride_size: Tuple) -> Tuple:
        """
        Compute tuple strides fitting in image.

        Parameters
        ----------
        image_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)
        stride_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)

        Returns
        -------
        Tuple, strides in x and y directions
        """
        return image_size[0] // stride_size[0], image_size[1] // stride_size[1]

    @staticmethod
    def _extract_random_patches(X: np.ndarray, image_size: Tuple,
                                patch_size: Tuple, n_patches: Union[int, np.integer],
                                random_state: Union[None, int,
                                                    np.random.RandomState] = None) \
            -> np.ndarray:
        """
        Extract random patches from image array.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_image_array)
        image_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)
            Image size
        patch_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)
            Patch size, features to extract
        n_patches : Union[int, np.integer]
            Number of patches to extract
        random_state : Union[None, int, np.random.RandomState], default=None

        Returns
        -------
        ndarray of shape (n_samples, n_patch_features)
        """
        rs = check_random_state(random_state)
        random_images = Coates._reshape_arrays_to_images(
            X[rs.randint(0, high=X.shape[0], size=n_patches)], image_size)
        random_patches = PatchExtractor(patch_size=(patch_size[0], patch_size[1]),
                                        max_patches=1,
                                        random_state=rs).transform(random_images)

        return Coates._reshape_images_to_arrays(random_patches, patch_size)

    @staticmethod
    def _extract_equidistant_patches(X: np.ndarray, image_size: Tuple,
                                     patch_size: Tuple,
                                     stride_size: Tuple) -> np.ndarray:
        """
        Extract equidistant patches from image array.

        Parameters
        ----------
        X : np.ndarray of shape (..., n_image_array)
        image_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)
        patch_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)
        stride_size : Tuple (n_height, n_width) or (n_height, n_width, n_channels)

        Returns
        -------
        ndarray of shape (..., n_patches, n_features)
        """
        nm_patches = Coates._patches_per_image(image_size=image_size,
                                               stride_size=stride_size)

        # single patch indices (aka single patch connectivity map)
        indices_patch = np.mod(np.arange(0, np.prod(patch_size)), patch_size[1]) \
            + (np.arange(0, np.prod(patch_size)) // patch_size[1]) * image_size[1]

        # preallocate memory for patches indices (aka connectivity map)
        indices_patches = np.zeros(
            shape=(nm_patches + (np.prod(patch_size), )), dtype=int)

        for y in range(nm_patches[0]):
            for x in range(nm_patches[1]):
                indices_patches[y, x, :] = y * stride_size[0] * image_size[1] \
                    + x * stride_size[1] + indices_patch

        # indices matrix to array
        indices_patches = indices_patches.reshape(
            [int(np.prod(nm_patches, axis=None))
             * int(np.prod(patch_size, axis=None)), ]).astype(int)

        patches = X[..., np.array(indices_patches, dtype=int)]

        return Coates._reshape_arrays_to_images(
            patches, image_size=(np.prod(nm_patches), np.prod(patch_size)))

    @staticmethod
    def _feature_mapping(patches: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Feature Mapping from patches to a feature matrix.

        Parameters
        ----------
        patches : ndarray of shape (..., n_patch_features)
        centroids : ndarray of shape (n_features, n_patch_features)

        Returns
        -------
        features : ndarray (..., n_features)
        """
        return np.dot(patches, centroids.T)

    @staticmethod
    def _inverse_feature_mapping(features: np.ndarray,
                                 centroids: np.ndarray) -> np.ndarray:
        """
        Inverse Mapping from features back to patches.

        Parameters
        ----------
        features : ndarray of shape (..., n_features)
        centroids : ndarray of shape (n_features, n_patch_features)

        Returns
        -------
        patches : ndarray of shape (..., n_patch_features)
        """
        return np.dot(features, np.linalg.pinv(centroids.T))

    def _pooling(self, X: np.ndarray) -> np.ndarray:
        """
        Pool patch features by pooling function.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        y : np.ndarray
        """
        nm_patches = Coates._patches_per_image(image_size=self.image_size,
                                               stride_size=self.stride_size)

        # feature_pools[#samples][#features][#pools][#pool_features]
        feature_pools = Coates._extract_equidistant_patches(
            np.transpose(X, axes=(0, 2, 1)),
            image_size=nm_patches,
            patch_size=self.pooling_size,
            stride_size=self.pooling_size)

        # pooled[#samples][#features][#pools]
        pooled = POOLINGS[self.pooling_func](feature_pools, axis=feature_pools.ndim - 1)
        return Coates._reshape_images_to_arrays(pooled, image_size=(
            self.clusterer.cluster_centers_.shape[0],
            np.prod(Coates._patches_per_image(nm_patches, self.pooling_size))
        ))

    def _preprocessing(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing on input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
        """
        X_preprocessed = X

        if self.normalize:
            self._normalizer.fit(X_preprocessed)
            X_preprocessed = self._normalizer.transform(X_preprocessed)

        if self.whiten:
            self._whitener = PCA(whiten=True).fit(X_preprocessed)
            X_preprocessed = self._whitener.transform(X_preprocessed)

        return X_preprocessed

    def _inverse_preprocessing(self, X: np.ndarray) -> np.ndarray:
        """
        Apply inverse preprocessing on input data.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
        """
        X_preprocessed = X
        if self.normalize:
            if self._normalizer is None:
                raise NotFittedError('normalizer has not been fitted!')
            X_preprocessed = self._normalizer.inverse_transform(X_preprocessed)
        if self.whiten:
            if self._whitener is None:
                raise NotFittedError('whitener has not been fitted!')
            X_preprocessed = self._whitener.inverse_transform(X_preprocessed)
        return X_preprocessed
