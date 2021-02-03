import numpy as np
import scipy

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler


class KCluster(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=10, metric='cityblock', init='random', n_inits=10, random_state=None, max_iter=300):
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.n_inits = n_inits
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.scaler = None
        return

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        if self.init == 'random':
            self.random_state = check_random_state(self.random_state)

            for init in self.random_state.randint(low=1e6, size=self.n_inits):
                centroids, inertia, labels, iterations = KCluster._calculate_centroids(
                    X,
                    cluster_centers_init=X[np.random.RandomState(init).randint(low=0, high=X.shape[0], size=self.n_clusters), ...],
                    # cluster_centers_init=np.random.RandomState(init).uniform(low=np.min(X), high=np.max(X), size=(self.n_clusters, X.shape[1])),
                    metric=self.metric,
                    max_iter=self.max_iter)

                if self.inertia_ is None or self.inertia_ > inertia:
                    self.cluster_centers_ = centroids
                    self.labels_ = labels
                    self.inertia_ = inertia
                    self.n_iter_ = iterations
        elif self.init.shape == (self.n_clusters, X.shape[1]):
            self.cluster_centers_ = self.init
            self.cluster_centers_, self.inertia_, self.labels_, self.n_iter_ = KCluster._calculate_centroids(
                X,
                cluster_centers_init=self.init,
                metric=self.metric,
                max_iter=self.max_iter)
        else:
            raise ValueError('invalid init value')

        return self

    def predict(self, X):
        return scipy.spatial.distance.cdist(X, self.cluster_centers_, metric=self.metric).argmin(axis=1)

    @staticmethod
    def _calculate_centroids(X, cluster_centers_init, metric, max_iter):
        cluster_centers = cluster_centers_init
        iteration = max_iter

        distance = scipy.spatial.distance.cdist(X, cluster_centers, metric=metric)
        # assignment = (previous assignment, next assignment)
        assignment = np.zeros((X.shape[0], )), distance.argmin(axis=1)

        while iteration > 0:
            if len(np.unique(assignment[1])) < cluster_centers.shape[0] or (assignment[0] == assignment[1]).all():
                # inertia = np.linalg.norm(np.take_along_axis(distance, assignment[1].reshape(-1, 1), axis=1), ord=2)
                inertia = np.sum(np.take_along_axis(distance, assignment[1].reshape(-1, 1), axis=1))
                return cluster_centers, inertia, assignment[1], max_iter - iteration

            for idx in range(cluster_centers.shape[0]):
                cluster_centers[idx, ...] = np.mean(X[assignment[1] == idx, ...], axis=0)

            distance = scipy.spatial.distance.cdist(X, cluster_centers, metric=metric)

            assignment = assignment[1], distance.argmin(axis=1)

            iteration -= 1

        inertia = np.sum(np.take_along_axis(distance, assignment[1].reshape(-1, 1), axis=1))
        return cluster_centers, inertia, assignment[1], max_iter - iteration
