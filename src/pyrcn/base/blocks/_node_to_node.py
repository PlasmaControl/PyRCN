"""The :mod:`node_to_node` contains NodeToNode classes and derivatives."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

import sys

import scipy
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigs as eigens
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

import numpy as np
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import NotFittedError

from ...base import ACTIVATIONS, _normal_random_recurrent_weights

if sys.version_info >= (3, 8):
    from typing import Union, Literal
else:
    from typing_extensions import Literal
    from typing import Union


class NodeToNode(BaseEstimator, TransformerMixin):
    """
    NodeToNode class for reservoir computing modules.

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default = 1.
        Quotient of recurrent weights per node (k_rec)
        and number of input features (n_features)
    reservoir_activation : Literal['tanh', 'identity', 'logistic', 'relu',
    'bounded_relu'], default = 'tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck,
            returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1/(1+exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function,
            returns f(x) = min(max(x, 0),1)
    spectral_radius : float, default = 1.
        Scales the recurrent weight matrix.
    leakage : float, default = 1.
        parameter to determine the degree of leaky integration.
    bidirectional : bool, default = False.
        Whether to work bidirectional.
    k_rec : Union[int, np.integer, None], default = None.
        recurrent weights per node. By default, it is None.
        If set, it overrides sparsity.
    random_state : Union[int, np.random.RandomState, None], default = 42
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 hidden_layer_size: int = 500,
                 sparsity: float = 1.,
                 reservoir_activation: Literal['tanh', 'identity', 'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 spectral_radius: float = 1.,
                 leakage: float = 1.,
                 bidirectional: bool = False,
                 k_rec: Union[int, np.integer, None] = None,
                 random_state: Union[int, np.random.RandomState, None] = 42) -> None:
        """Construct the NodeToNode."""
        self.hidden_layer_size = hidden_layer_size
        self.sparsity = sparsity
        self.reservoir_activation = reservoir_activation
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.bidirectional = bidirectional
        self.k_rec = k_rec
        self.random_state = random_state

        self._recurrent_weights: np.ndarray = np.ndarray([])
        self._hidden_layer_state: np.ndarray = np.ndarray([])

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the NodeToNode. Initialize recurrent weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : returns a trained NodeToNode.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)

        if self.k_rec is not None:
            self.sparsity = float(self.k_rec) / float(X.shape[1])
        self._recurrent_weights = _normal_random_recurrent_weights(
            hidden_layer_size=self.hidden_layer_size,
            fan_in=int(np.rint(self.hidden_layer_size * self.sparsity)),
            random_state=self._random_state)
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Transform the input matrix X.

        Parameters
        ----------
        X : ndarray of size (n_samples, n_features)

        Returns
        -------
        Y: ndarray of size (n_samples, hidden_layer_size)
        """
        if self._recurrent_weights.shape == ():
            raise NotFittedError(self)

        if self.bidirectional:
            _hidden_layer_state_fw = self._pass_through_recurrent_weights(
                X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
                self.spectral_radius, self.leakage, self.reservoir_activation)
            _hidden_layer_state_bw = np.flipud(
                self._pass_through_recurrent_weights(np.flipud(X),
                                                     self.hidden_layer_size,
                                                     self.bidirectional,
                                                     self._recurrent_weights,
                                                     self.spectral_radius, self.leakage,
                                                     self.reservoir_activation))
            self._hidden_layer_state = np.concatenate((_hidden_layer_state_fw,
                                                       _hidden_layer_state_bw), axis=1)
        else:
            self._hidden_layer_state = self._pass_through_recurrent_weights(
                X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
                self.spectral_radius, self.leakage, self.reservoir_activation)
        return self._hidden_layer_state

    @staticmethod
    def _pass_through_recurrent_weights(X: np.ndarray,
                                        hidden_layer_size: int, bidirectional: bool,
                                        recurrent_weights: Union[np.ndarray,
                                                                 csr_matrix],
                                        spectral_radius: float, leakage: float,
                                        reservoir_activation:
                                            Literal['tanh', 'identity',
                                                    'logistic', 'relu',
                                                    'bounded_relu']) -> np.ndarray:
        """
        Return the reservoir state matrix.

        Parameters
        ----------
        X : ndarray of size (n_samples, n_features)
        hidden_layer_size : Union[int, np.integer].
        bidirectional : bool
        recurrent_weights : Union[np.ndarray, scipy.sparse.csr.csr_matrix]
        spectral_radius : float
        leakage : float
        reservoir_activation : Literal['tanh', 'identity', 'logistic', 'relu',
        'bounded_relu']

        Returns
        -------
        hidden_layer_state : ndarray of size (n_samples, hidden_layer_size)
        """
        hidden_layer_state = np.zeros(shape=[X.shape[0]+1, hidden_layer_size])
        for sample in range(X.shape[0]):
            a = X[sample, :]
            b = safe_sparse_dot(hidden_layer_state[sample, :], recurrent_weights) * \
                spectral_radius
            pre_activation = a + b
            ACTIVATIONS[reservoir_activation](pre_activation)
            hidden_layer_state[sample+1, :] = pre_activation
            hidden_layer_state[sample + 1, :] = \
                (1 - leakage) * hidden_layer_state[sample, :] + \
                leakage * hidden_layer_state[sample + 1, :]
        return hidden_layer_state[1:, :]

    def _validate_hyperparameters(self) -> None:
        """Validate the hyperparameters."""
        self._random_state = check_random_state(self.random_state)

        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got {0}%s."
                             .format(self.hidden_layer_size))
        if self.sparsity <= 0. or self.sparsity > 1.:
            raise ValueError("sparsity must be between 0. and 1., got {0}."
                             .format(self.sparsity))
        if self.reservoir_activation not in ACTIVATIONS:
            raise ValueError("The activation_function {0} is not supported. "
                             "Supported activations are {1}."
                             .format(self.reservoir_activation, ACTIVATIONS))
        if self.spectral_radius < 0.:
            raise ValueError("spectral_radius must be >= 0, got {0}."
                             .format(self.spectral_radius))
        if self.leakage <= 0. or self.leakage > 1.:
            raise ValueError("leakage must be between 0. and 1., got {0}."
                             .format(self.leakage))
        if self.bidirectional not in [False, True]:
            raise ValueError("bidirectional must be either False or True,, got {0}."
                             .format(self.bidirectional))
        if self.k_rec is not None and (self.k_rec <= 0
                                       or self.k_rec >= self.hidden_layer_size):
            raise ValueError("k_rec must be > 0, got {0}.".format(self.k_rec))

    def __sizeof__(self) -> int:
        """
        Return the size of the object in bytes.

        Returns
        -------
        size : int
        Object memory in bytes.
        """
        if scipy.sparse.issparse(self._recurrent_weights):
            return object.__sizeof__(self) + \
                np.asarray(self._recurrent_weights).nbytes + \
                self._hidden_layer_state.nbytes + sys.getsizeof(self.random_state)
        else:
            return object.__sizeof__(self) + self._recurrent_weights.nbytes + \
                self._hidden_layer_state.nbytes + sys.getsizeof(self.random_state)

    @property
    def recurrent_weights(self) -> Union[np.ndarray, csr_matrix]:
        """
        Return the recurrent weights.

        Returns
        -------
        recurrent_weights : Union[np.ndarray, scipy.sparse.csr.csr_matrix]
        of size (hidden_layer_size, hidden_layer_size)
        """
        return self._recurrent_weights


class PredefinedWeightsNodeToNode(NodeToNode):
    """
    PredefinedWeightsNodeToNode class for reservoir computing modules.

    Parameters
    ----------
    predefined_recurrent_weights : np.ndarray
        A set of predefined recurrent weights.
    hidden_layer_size : Union[int, np.integer], default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    reservoir_activation : Literal['tanh', 'identity', 'logistic', 'relu',
    'bounded_relu'], default = 'tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck,
            returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1/(1+exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function,
            returns f(x) = min(max(x, 0),1)
    spectral_radius :  float, default = 1.
        Scales the recurrent weight matrix.
    leakage : float, default = 1.
        parameter to determine the degree of leaky integration.
    bidirectional : bool, default = False.
        Whether to work bidirectional.
    """

    @_deprecate_positional_args
    def __init__(self,
                 predefined_recurrent_weights: np.ndarray, *,
                 reservoir_activation: Literal['tanh', 'identity', 'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 spectral_radius: float = 1.,
                 leakage: float = 1.,
                 bidirectional: bool = False) -> None:
        """Construct the PredefinedWeightsNodeToNode."""
        if predefined_recurrent_weights.ndim != 2:
            raise ValueError('predefined_recurrent_weights has not the '
                             'expected ndim {0}, given 2.'
                             .format(predefined_recurrent_weights.shape))
        super().__init__(hidden_layer_size=predefined_recurrent_weights.shape[0],
                         reservoir_activation=reservoir_activation,
                         spectral_radius=spectral_radius,
                         leakage=leakage,
                         bidirectional=bidirectional)
        self.predefined_recurrent_weights = predefined_recurrent_weights

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the PredefinedWeightsNodeToNode. Sets the recurrent weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : returns a trained PredefinedWeightsNodeToNode.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)

        if self.predefined_recurrent_weights.shape[0] != X.shape[1]:
            raise ValueError('X has not the expected shape {0}, given {1}.'.format(
                self.predefined_recurrent_weights.shape[0], X.shape[1]))

        if (self.predefined_recurrent_weights.shape[0]
                != self.predefined_recurrent_weights.shape[1]):
            raise ValueError('Recurrent weights need to be a squared matrix, given {0}.'
                             .format(self.predefined_recurrent_weights.shape))

        self._recurrent_weights = self.predefined_recurrent_weights
        return self


class HebbianNodeToNode(NodeToNode):
    """
    HebbianNodeToNode for reservoir computing modules (e.g. ESN).

    Applies the hebbian rule to a given set of randomly initialized reservoir weights.

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default = 1.
        Quotient of recurrent weights per node (k_rec)
        and number of input features (n_features)
    reservoir_activation : Literal['tanh', 'identity', 'logistic', 'relu',
    'bounded_relu'], default = 'tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck,
            returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1/(1+exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function,
            returns f(x) = min(max(x, 0),1)
    spectral_radius :  float, default = 1.
        Scales the recurrent weight matrix.
    leakage : float, default = 1.
        parameter to determine the degree of leaky integration.
    bidirectional : bool, default = False.
        Whether to work bidirectional.
    k_rec : Union[int, np.integer, None], default = None.
        recurrent weights per node. By default, it is None.
        If set, it overrides sparsity.
    random_state : Union[int, np.random.RandomState, None], default = 42
    learning_rate : float, default = 0.01.
        Determines how fast the weight values are updated.
    epochs : Union[int, np.integer], default=100
        Number of training epochs
    training_method : Literal['hebbian', 'anti_hebbian', 'oja', 'anti_oja'],
    default = "hebbian"
        Method used to fit the recurrent weights.
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 hidden_layer_size: int = 500,
                 sparsity: float = 1.,
                 reservoir_activation: Literal['tanh', 'identity', 'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 spectral_radius: float = 1.,
                 leakage: float = 1.,
                 bidirectional: bool = False,
                 k_rec: Union[int, np.integer, None] = None,
                 random_state: Union[int, np.random.RandomState, None] = 42,
                 learning_rate: float = 0.01,
                 epochs: Union[int, np.integer] = 100,
                 training_method:  Literal['hebbian', 'anti_hebbian', 'oja',
                                           'anti_oja'] = 'hebbian'):
        """Construct the HebbianNodeToNode."""
        super().__init__(hidden_layer_size=hidden_layer_size,
                         sparsity=sparsity,
                         reservoir_activation=reservoir_activation,
                         spectral_radius=spectral_radius,
                         leakage=leakage,
                         bidirectional=bidirectional,
                         k_rec=k_rec,
                         random_state=random_state)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_method = training_method

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the HebbianNodeToNode. Initialize recurrent weights and do Hebbian learning.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : None
            ignored

        Returns
        -------
        self
        """
        super().fit(X=X, y=y)
        for k in range(self.epochs):
            if self.training_method == 'hebbian':
                self._hebbian_learning(X=X, y=y)
            elif self.training_method == 'anti_hebbian':
                self._anti_hebbian_learning(X=X, y=y)
            elif self.training_method == 'oja':
                self._oja_learning(X=X, y=y)
            elif self.training_method == 'anti_oja':
                self._anti_oja_learning(X=X, y=y)

            try:
                we = eigens(self._recurrent_weights,
                            k=np.minimum(10, self.hidden_layer_size - 2),
                            which='LM', return_eigenvectors=False,
                            v0=self._random_state.normal(loc=0., scale=1.,
                                                         size=self.hidden_layer_size)
                            )
            except ArpackNoConvergence:
                print("WARNING: No convergence! Returning possibly invalid values!!!")
                we = ArpackNoConvergence.eigenvalues
            self._recurrent_weights = self._recurrent_weights / np.amax(np.absolute(we))
        return self

    def _hebbian_learning(self, X: np.ndarray, y: None = None) -> None:
        """Use the Hebbian learning rule."""
        hidden_layer_state = self._pass_through_recurrent_weights(
            X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
            self.spectral_radius, self.leakage, self.reservoir_activation)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= self.learning_rate * \
                safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T,
                                hidden_layer_state[k:k+1, :]) * self._recurrent_weights

    def _anti_hebbian_learning(self, X: np.ndarray, y: None = None) -> None:
        """Use the Anti-Hebbian learning rule."""
        hidden_layer_state = self._pass_through_recurrent_weights(
            X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
            self.spectral_radius, self.leakage, self.reservoir_activation)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= -self.learning_rate * \
                safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T,
                                hidden_layer_state[k:k+1, :]) * self._recurrent_weights

    def _oja_learning(self, X: np.ndarray, y: None = None) -> None:
        """Use the Oja learning rule."""
        hidden_layer_state = self._pass_through_recurrent_weights(
            X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
            self.spectral_radius, self.leakage, self.reservoir_activation)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= \
                self.learning_rate * (safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T,
                                                      hidden_layer_state[k:k+1, :])
                                      - safe_sparse_dot(
                                          hidden_layer_state[k+1:k+2, :].T,
                                          hidden_layer_state[k+1:k+2, :])) \
                * self._recurrent_weights

    def _anti_oja_learning(self, X: np.ndarray, y: None = None) -> None:
        """Use the Anti-Oja learning rule."""
        hidden_layer_state = self._pass_through_recurrent_weights(
            X, self.hidden_layer_size, self.bidirectional, self._recurrent_weights,
            self.spectral_radius, self.leakage, self.reservoir_activation)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= \
                self.learning_rate * (-safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T,
                                                       hidden_layer_state[k:k+1, :])
                                      - safe_sparse_dot(
                                          hidden_layer_state[k+1:k+2, :].T,
                                          hidden_layer_state[k+1:k+2, :])) \
                * self._recurrent_weights
