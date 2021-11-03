"""The KCluster algorithm, a comparable algorithm to K-Means."""

import sys
if sys.version_info >= (3, 8):
    from typing import Union, Callable, Tuple, Literal
else:
    from typing_extensions import Literal
    from typing import Union, Callable, Tuple

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler


class KCluster(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    KCluster -- an extension of the KMeans algorithm.

    Parameters
    ----------
    n_clusters : Union[int, np.integer], default=10
            The number of clusters to form and the number of centroids to generate.
    metric : Union[Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                           'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                           'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                           'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                           'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                           'wminkowski', 'yule'],
                   callable], default = 'cityblock'.
        Distance metric to be used.
    init : Union[Literal['random'], np.ndarray], default = 'random'
        Method for initialization:
            - 'random', choose ```n_clusters``` observations (rows) at random from data
            for the initial centroids.
            - If an ```np.ndarray``` is passed, it should be of shape
            ```(n_clusters, n_features)``` and gives the initial centers.
            - TODO: Add callable here!
    n_inits :  Union[int, np.integer], default = 10.
        Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of ```n_init``` consecutive runs
        in terms of inertia.
    random_state : Optional[int, np.random.RandomState], default = None
        Scales the input bias of the activation.
    max_iter : Union[int, np.integer], default = 300.
        Maximum number of iterations of the cluster algorithm for a single run.
    """

    def __init__(self,
                 n_clusters: int = 10,
                 metric: Union[Literal['braycurtis', 'canberra', 'chebyshev',
                                       'cityblock', 'correlation', 'cosine', 'dice',
                                       'euclidean', 'hamming', 'jaccard',
                                       'jensenshannon', 'kulsinski', 'mahalanobis',
                                       'matching', 'minkowski', 'rogerstanimoto',
                                       'russellrao', 'seuclidean', 'sokalmichener',
                                       'sokalsneath', 'sqeuclidean', 'wminkowski',
                                       'yule'], Callable] = 'cityblock',
                 init: Union[Literal['random'], np.ndarray] = 'random',
                 n_inits:  int = 10,
                 random_state: Union[int, np.random.RandomState, None] = None,
                 max_iter: int = 300) -> None:
        """Construct the KCluster."""
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.n_inits = n_inits
        self.random_state = check_random_state(random_state)
        self.max_iter = max_iter
        self.cluster_centers_: np.ndarray = np.ndarray([])
        self.labels_: np.ndarray = np.ndarray([])
        self.inertia_ = np.nan
        self.n_iter_: int
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> ClusterMixin:
        """
        Fit the KCluster. Compute K clusters from the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self : returns a Fitted estimator.
        """
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        if self.init == 'random':
            for init in self.random_state.randint(low=1e6, size=self.n_inits):
                centroids, inertia, labels, iterations = KCluster._calculate_centroids(
                    X,
                    cluster_centers_init=X[np.random.RandomState(init).randint(
                        low=0, high=X.shape[0], size=self.n_clusters), ...],
                    metric=self.metric, max_iter=self.max_iter)
                if self.inertia_ > inertia:
                    self.cluster_centers_ = centroids
                    self.labels_ = labels
                    self.inertia_ = inertia
                    self.n_iter_ = iterations
        elif (isinstance(self.init, np.ndarray)
              and self.init.shape == (self.n_clusters, X.shape[1])):
            self.cluster_centers_ = self.init
            self.cluster_centers_, self.inertia_, self.labels_, self.n_iter_ = \
                KCluster._calculate_centroids(X, cluster_centers_init=self.init,
                                              metric=self.metric,
                                              max_iter=self.max_iter)
        else:
            raise ValueError('invalid init value')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, ```cluster_centers_``` is called the code
        book and each value returned by predict is the index of the closest code in the
        code book.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return scipy.spatial.distance.cdist(X, self.cluster_centers_,
                                            metric=self.metric).argmin(axis=1)

    @staticmethod
    def _calculate_centroids(
            X: np.ndarray, cluster_centers_init: np.ndarray,
            metric: Union[Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                  'correlation', 'cosine', 'dice', 'euclidean',
                                  'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                                  'mahalanobis', 'matching', 'minkowski',
                                  'rogerstanimoto', 'russellrao', 'seuclidean',
                                  'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                  'wminkowski', 'yule'], Callable],
            max_iter: int) -> Tuple[np.ndarray, float, np.ndarray, int]:
        """
        Calculate the centroids iteratively based on the selected metric.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data
        cluster_centers_init : np.ndarray of shape (n_clusters, n_features)
            The initialized cluster centers
        metric : Union[Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                               'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                               'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                               'seuclidean', 'sokalmichener', 'sokalsneath',
                               'sqeuclidean', 'wminkowski', 'yule'], callable]
            The chosen metric for cluster centroid computation
        max_iter : int
            Maximum number of iterations after which the centroid computation
            is interrupted.

        Returns
        -------
        cluster_centers : ndarray of shape (n_clusters, n_features)
            The final cluster centers
        inertia : float
            Sum of squared distances of samples to their closest cluster center.
        labels : ndarray of shape (n_samples, )
            Labels of each point
        n_iter : Union[int, np.integer]
            Number of iterations run
        """
        cluster_centers = cluster_centers_init
        iteration: int = max_iter

        distance = scipy.spatial.distance.cdist(X, cluster_centers, metric=metric)
        # labels = (previous labels, next labels)
        labels = np.zeros((X.shape[0], )), distance.argmin(axis=1)

        while iteration > 0:
            if len(np.unique(labels[1])) < cluster_centers.shape[0] \
                    or (labels[0] == labels[1]).all():
                # inertia = np.linalg.norm(np.take_along_axis(distance,
                #                                             labels[1].reshape(-1, 1),
                #                                             axis=1),
                #                          ord=2)
                inertia = np.sum(np.take_along_axis(distance, labels[1].reshape(-1, 1),
                                                    axis=1))
                return cluster_centers, inertia, labels[1], max_iter - iteration

            for idx in range(cluster_centers.shape[0]):
                cluster_centers[idx, ...] = np.mean(X[labels[1] == idx, ...], axis=0)

            distance = scipy.spatial.distance.cdist(X, cluster_centers, metric=metric)

            labels = labels[1], distance.argmin(axis=1)
            iteration -= 1

        inertia = np.sum(np.take_along_axis(distance, labels[1].reshape(-1, 1), axis=1))
        return cluster_centers, inertia, labels[1], max_iter - iteration
