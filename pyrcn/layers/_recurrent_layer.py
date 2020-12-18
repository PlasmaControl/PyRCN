import numpy as np
import scipy.sparse

import warnings
from abc import ABCMeta, abstractmethod

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, check_is_fitted
from ..activation_functions._activation_functions import ACTIVATIONS


class BaseFeedForwardLayer(TransformerMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, n_components: int = 500, dense_output: bool = True, input_scaling: float = 1.0, k_in: int = 10,
                 bias_scaling: float = 0.0, activation_function: str = 'tanh', random_state=None):
        """
        Initialize a BaseFeedForwardLayer without any specific functionality.

        Parameter
        ---------
        n_components : int
            Dimensionality of the layer

        random_state : int
            Random state for reproduceable results
        """
        self.n_components = n_components
        self.dense_output = dense_output
        self.input_scaling = input_scaling
        self.k_in = k_in
        self.bias_scaling = bias_scaling
        self.activation_function = activation_function
        self.random_state = random_state

    def _validate_parameters(self, n_features):
        if self.k_in < 1 and self.k_in != -1:
            raise ValueError("k_in must be -1 or greater than 0, got %s" % self.k_in)
        if self.k_in > n_features:
            raise ValueError("k_in must not be larger than %s, got %s" % (n_features, self.k_in))
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0, got %s" % self.n_components)
        elif self.n_components <= n_features:
            warnings.warn("The number of components is smaller than the number of features: n_components < n_features"
                          "(%s < %s)" % (self.n_components, n_features))

        if self.activation_function not in ACTIVATIONS:
            raise ValueError("The reservoir_activation '%s' is not supported. Supported activations are %s."
                             % (self.activation_function, ACTIVATIONS))

    @abstractmethod
    def _initialize_weight_matrices(self, n_components, n_features):
        """
        Generate a random projection matrix.

        Parameters
        ----------
        n_components : int
            Dimensionality of the target projection space
        n_features : int
            Dimensionality of the original feature space

        Returns
        -------
        feedforward_weights : np.array or CSR matrix [n_components, n_features]
            The feedforward weight matrix.
        bias_weights: np.array [n_components, ]
            The bias weight matrix.

        """

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=['csr', 'csc'])

        n_samples, n_features = X.shape

        self._validate_parameters(n_features=n_features)

        self.n_components_ = self.n_components

        # Generate a feedforward weight matrix of size [n_components, n_features]
        # and a a bias weight matrix of size [n_components, n_features]
        self.input_weights_, self.bias_weights_ = self._initialize_random_matrices(self.n_components_, n_features)

        # Check contract
        assert self.components_.shape == (self.n_components_, n_features), \
            "An error has occurred the self.components_ matrix has not the proper shape."

        return self

    def transform(self, X):
        """
        Project the data by using matrix product with the random matrix
        Parameters
        ----------
        X : numpy array or scipy.sparse of shape [n_samples, n_features]
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : numpy array or scipy sparse of shape [n_samples, n_components]
            Projected array.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'])

        check_is_fitted(self)

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("Impossible to perform projection:"
                             "X at fit stage had a different number of features. "
                             "(%s != %s)" % (X.shape[1], self.components_.shape[1]))

        X_new = safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
        return X_new


class FeedForwardLayer(BaseFeedForwardLayer):

    def __init__(self, n_components: int = 500, dense_output=True, random_state=None, input_scaling: float = 1.0,
                 k_in: int = 10, bias_scaling: float = 0.0):
        super().__init__(n_components=n_components, dense_output=dense_output, input_scaling=input_scaling, k_in=k_in,
                         bias_scaling=bias_scaling, random_state=random_state)

    def _initialize_weight_matrices(self, n_components, n_features):
        random_state = check_random_state(self.random_state)

        idx_co = 0

        if self.k_in == -1:
            feedforward_weight_matrix = random_state.rand(self.n_components_, n_features) * 2 - 1
        else:
            nr_entries = np.int32(self.n_components_*self.k_in)
            ij = np.zeros((2, nr_entries), dtype=int)
            data_vec = random_state.rand(nr_entries) * 2 - 1
            for en in range(self.n_components_):
                per = random_state.permutation(n_features)[:self.k_in]
                ij[0][idx_co:idx_co+self.k_in] = en
                ij[1][idx_co:idx_co+self.k_in] = per
                idx_co = idx_co + self.k_in
            feedforward_weight_matrix = scipy.sparse.csc_matrix((data_vec, ij),
                                                                shape=(self.n_components_, n_features), dtype='float64')
        bias_weight_matrix = (random_state.rand(self.n_components_) * 2 - 1)
        return feedforward_weight_matrix, bias_weight_matrix
