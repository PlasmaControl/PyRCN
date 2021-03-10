"""
The :mod:`pyrcn.base`contains utilities for the reservoir computing modules
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import sys

import scipy
import numpy as np

from pkg_resources import parse_version
import warnings

import sklearn
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import NotFittedError

from sklearn.preprocessing import StandardScaler

if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens


if parse_version(sklearn.__version__) < parse_version('0.24.0'):
    from sklearn.utils import check_array

    def validate_data(self, X, y=None, *args, **kwargs):
        warnings.warn('Due to scikit version, _validate_data(X, y) returns check_array(X), y.', DeprecationWarning)
        if y is not None:
            if 'accept_sparse' in kwargs:
                del kwargs['accept_sparse']
            if 'multi_output' in kwargs:
                del kwargs['multi_output']
            return check_array(X, accept_sparse=True, **kwargs), y
        else:
            if 'accept_sparse' in kwargs:
                del kwargs['accept_sparse']
            return check_array(X, accept_sparse=True, **kwargs)

    setattr(BaseEstimator, '_validate_data', validate_data)


def inplace_bounded_relu(X):
    """
    Compute the bounded rectified linear unit function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.minimum(np.maximum(X, 0, out=X), 1, out=X)


def inplace_tanh_inverse(X):
    """
    Compute the tanh inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.arctanh(X, out=X)


def inplace_identity_inverse(X):
    """
    Compute the identity inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['identity'](X)


def inplace_logistic_inverse(X):
    """
    Compute the logistic inverse function inplace.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    np.negative(np.log(1 - X, out=X), out=X)


def inplace_relu_inverse(X):
    """
    Compute the relu inverse function inplace.

    The relu function is not invertible!
    This is an approximation assuming $x = f^{-1}(y=0) = 0$. It is valid in $x \in [0, \infty]$.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['relu'](X)


def inplace_bounded_relu_inverse(X):
    """
    Compute the bounded relu inverse function inplace.

    The bounded relu function is not invertible!
    This is an approximation assuming $x = f^{-1}(y=0) = 0$ and $x = f^{-1}(y=1) = 1$. It is valid in $x \in [0, 1]$.

    Parameters
    ----------
    X : Union(array-like, sparse matrix), shape (n_samples, n_features)
        The input data.
    """
    ACTIVATIONS['bounded_relu'](X)


ACTIVATIONS.update({'bounded_relu': inplace_bounded_relu})

ACTIVATIONS_INVERSE = {
    'tanh': inplace_tanh_inverse,
    'identity': inplace_identity_inverse,
    'logistic': inplace_logistic_inverse,
    'relu': inplace_relu_inverse,
    'bounded_relu': inplace_bounded_relu_inverse
}

ACTIVATIONS_INVERSE_BOUNDS = {
    'tanh': [-.99, .99],
    'identity': [-np.inf, np.inf],
    'logistic': [0.01, .99],
    'relu': [0, np.inf],
    'bounded_relu': [0, 1]
}


class InputToNode(BaseEstimator, TransformerMixin):
    """
    InputToNode class for reservoir computing modules (e.g. ELM)

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default=1.
        Quotient of input weights per node (k_in) and number of input features (n_features)
    activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    input_scaling : float, default=1.
        Scales the input weight matrix.
    bias_scaling : float, default=1.
        Scales the input bias of the activation.
    k_in : int, default=None.
        input weights per node. By default, it is None. If set, it overrides sparsity.
    random_state : {None, int, RandomState}, default=None
    """
    def __init__(self,
                 hidden_layer_size=500,
                 sparsity=1.,
                 activation='tanh',
                 input_scaling=1.,
                 bias_scaling=1.,
                 k_in=None,
                 random_state=None):
        self.hidden_layer_size = hidden_layer_size
        self.sparsity = sparsity
        self.activation = activation
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.k_in = k_in
        self.random_state = check_random_state(random_state)

        self._input_weights = None
        self._bias = None
        self._hidden_layer_state = None

    def fit(self, X, y=None):
        """
        Fit the InputToNode. Initialize input weights and bias.

        Parameters
        ----------
        X : Union(ndarray, sparse matrix) of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)
        if self.k_in is not None:
            self.sparsity = self.k_in / X.shape[1]
        self._input_weights = self._uniform_random_input_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size * self.sparsity).astype(int),
            random_state=self.random_state)
        self._bias = self._uniform_random_bias(
            hidden_layer_size=self.hidden_layer_size,
            random_state=self.random_state)
        return self

    def transform(self, X):
        """
        Transforms the input matrix X.

        Parameters
        ----------
        X : Union(ndarray, sparse matrix) of size (n_samples, n_features)

        Returns
        -------
        Y: ndarray of size (n_samples, hidden_layer_size)
        """
        if self._input_weights is None or self._bias is None:
            raise NotFittedError(self)

        self._hidden_layer_state = InputToNode._node_inputs(
            X, self._input_weights, self.input_scaling, self._bias, self.bias_scaling)
        ACTIVATIONS[self.activation](self._hidden_layer_state)
        return self._hidden_layer_state

    @staticmethod
    def _uniform_random_input_weights(n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
        """
        Returns uniform random input weights in range [-1, 1]

        Parameters
        ----------
        n_features_in : int
        hidden_layer_size : int
        fan_in : int
            Determines how many features are mapped to one neuron.
        random_state : numpy.RandomState

        Returns
        -------
        uniform_random_input_weights : ndarray of size (n_features, hidden_layer_size)
        """
        nr_entries = np.int32(n_features_in * fan_in)
        weights_array = random_state.uniform(low=-1., high=1., size=nr_entries)

        if fan_in < hidden_layer_size:
            indices = np.zeros(shape=nr_entries, dtype=int)
            indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in, step=fan_in)

            for en in range(0, n_features_in * fan_in, fan_in):
                indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
            return scipy.sparse.csr_matrix(
                (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
        else:
            return weights_array.reshape((n_features_in, hidden_layer_size))

    @staticmethod
    def _uniform_random_bias(hidden_layer_size: int, random_state):
        """
        Returns uniform random bias in range [-1, 1].

        Parameters
        ----------
        hidden_layer_size : int
        random_state : numpy.RandomState

        Returns
        -------
        uniform_random_bias : ndarray of size (hidden_layer_size)
        """
        return random_state.uniform(low=-1., high=1., size=hidden_layer_size)

    @staticmethod
    def _node_inputs(X, input_weights, input_scaling, bias, bias_scaling):
        """
        Returns the node inputs scaled by input_scaling, multiplied by input_weights and bias added.

        Parameters
        ----------
        X : ndarray of size (n_samples, n_features)
        input_weights : ndarray of size (n_features, hidden_layer_size)
        input_scaling : float
        bias : ndarray of size (hidden_layer_size)
        bias_scaling : float

        Returns
        -------
        node_inputs : ndarray of size (n_samples, hidden_layer_size)
        """
        return safe_sparse_dot(X, input_weights) * input_scaling + np.ones(shape=(X.shape[0], 1)) * bias * bias_scaling

    def _validate_hyperparameters(self):
        """
        Validates the hyperparameters.

        Returns
        -------

        """
        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got %s." % self.hidden_layer_size)
        if self.input_scaling <= 0.:
            raise ValueError("input_scaling must be > 0, got %s." % self.input_scaling)
        if self.sparsity <= 0. or self.sparsity > 1.:
            raise ValueError("sparsity must be between 0. and 1., got %s." % self.sparsity)
        if self.bias_scaling < 0:
            raise ValueError("bias must be > 0, got %s." % self.bias_scaling)
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.activation, ACTIVATIONS))
        if self.k_in is not None and self.k_in <= 0:
            raise ValueError("k_in must be > 0, got %d." % self.k_in)

    def __sizeof__(self):
        """
        Returns the size of the object in bytes.

        Returns
        -------
        size : int
        Object memory in bytes.
        """
        return object.__sizeof__(self) + \
            self._bias.nbytes + \
            self._input_weights.nbytes + \
            self._hidden_layer_state.nbytes + \
            sys.getsizeof(self.random_state)


class BatchIntrinsicPlasticity(InputToNode):
    def __init__(
            self,
            distribution: str = 'normal',
            algorithm: str = 'dresden',
            activation='tanh',
            hidden_layer_size=500,
            sparsity=1.,
            random_state=None):
        super().__init__(
            activation=activation,
            hidden_layer_size=hidden_layer_size,
            sparsity=sparsity,
            random_state=random_state)
        self.distribution = distribution
        self.algorithm = algorithm
        self._scaler = None
        self._m = 1
        self._c = 0

    IN_DISTRIBUTION_PARAMS = {
        'exponential': (-.5, -.5),
        'uniform': (.7, .0),
        'normal': (.3, .0)
    }

    OUT_DISTRIBUTION = {
        'exponential': lambda size: np.random.poisson(lam=1., size=size),
        'uniform': lambda size: np.random.uniform(low=-1., high=1., size=size),
        'normal': lambda size: np.random.normal(loc=0., scale=1., size=size)
    }

    def fit(self, X, y=None):
        self._validate_hyperparameters()

        if self.algorithm == 'neumann':
            self._fit_neumann(X, y=None)

        if self.algorithm == 'dresden':
            self._fit_dresden(X, y=None)
        return self

    def transform(self, X):
        if self.algorithm == 'neumann':
            return super().transform(X)

        if self.algorithm == 'dresden':
            s = BatchIntrinsicPlasticity._node_inputs(
                X, self._input_weights, self.input_scaling, self._bias, self.bias_scaling)
            np.add(np.multiply(self._scaler.transform(s), self._m), self._c, out=s)
            ACTIVATIONS[self.activation](s)
            return s

    def _fit_neumann(self, X, y=None):
        super().fit(X, y=None)

        s = np.sort(BatchIntrinsicPlasticity._node_inputs(
            X, self._input_weights, self.input_scaling, self._bias, self.bias_scaling), axis=0)

        phi = np.transpose(np.stack((s, np.ones(s.shape)), axis=2), axes=(1, 0, 2))

        if callable(BatchIntrinsicPlasticity.OUT_DISTRIBUTION[self.distribution]):
            t = BatchIntrinsicPlasticity.OUT_DISTRIBUTION[self.distribution](size=X.shape[0])
            t_min, t_max = np.min(t), np.max(t)

            if self.distribution in {'uniform'} and self.activation in {'tanh', 'logistic'}:
                bound_low = ACTIVATIONS_INVERSE_BOUNDS[self.activation][0] * .5
                bound_high = ACTIVATIONS_INVERSE_BOUNDS[self.activation][1] * .5
            else:
                bound_low = ACTIVATIONS_INVERSE_BOUNDS[self.activation][0]
                bound_high = ACTIVATIONS_INVERSE_BOUNDS[self.activation][1]

            if bound_low == np.inf:
                bound_low = t_min

            if bound_high == np.inf:
                bound_high = t_max

            t = (t - t_min) * (bound_high - bound_low) / (t_max - t_min) + bound_low

            t.sort()
            ACTIVATIONS_INVERSE[self.activation](t)
        else:
            raise ValueError('Not a valid activation inverse, got {0}'.format(self.distribution))

        v = safe_sparse_dot(np.linalg.pinv(phi), t)

        np.multiply(self._input_weights, v[:, 0], out=self._input_weights)
        self._bias += v[:, 1]
        return self

    def _fit_dresden(self, X, y=None):
        if self.activation != 'tanh':
            raise ValueError('This algorithm is working with tanh-activation only, got {0}'.format(self.activation))

        super().fit(X, y=None)

        s = BatchIntrinsicPlasticity._node_inputs(
            X, self._input_weights, self.input_scaling, self._bias, self.bias_scaling)

        self._scaler = StandardScaler().fit(s)

        if self.distribution:
            self._m, self._c = BatchIntrinsicPlasticity.IN_DISTRIBUTION_PARAMS[self.distribution]
        return self

    def _validate_hyperparameters(self):
        super()._validate_hyperparameters()

        if self.algorithm not in {'neumann', 'dresden'}:
            raise ValueError('The selected algorithm ist unknown, got {0}'.format(self.algorithm))


class NodeToNode(BaseEstimator, TransformerMixin):
    """
    NodeToNode class for reservoir computing modules (e.g. ESN)

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default=1.
        Quotient of input weights per node (k_in) and number of input features (n_features)
    activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    spectral_radius : float, default=1.
        Scales the input weight matrix.
    leakage : float, default=1.
        parameter to determine the degree of leaky integration.
    bias_scaling : float, default=1.
        Scales the input bias of the activation.
    bi_directional : bool, default=False
        Whether to work in bidirectional mode.
    k_rec : int, default=None.
        recurrent weights per node. By default, it is None. If set, it overrides sparsity.
    random_state : {None, int, RandomState}, default=None
    """
    def __init__(self,
                 hidden_layer_size=500,
                 sparsity=1.,
                 activation='tanh',
                 spectral_radius=1.,
                 leakage=1.,
                 bias_scaling=1.,
                 bi_directional=False,
                 k_rec=None,
                 random_state=None):
        self.hidden_layer_size = hidden_layer_size
        self.sparsity = sparsity
        self.activation = activation
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.bias_scaling = bias_scaling
        self.bi_directional = bi_directional
        self.k_rec = k_rec
        self.random_state = check_random_state(random_state)

        self._recurrent_weights = None
        self._bias = None
        self._hidden_layer_state = None

    def fit(self, X, y=None):
        """
        Fit the NodeToNode. Initialize input weights and bias.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)

        if self.k_rec is not None:
            self.sparsity = self.k_rec / X.shape[1]
        self._recurrent_weights = self._normal_random_recurrent_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size * self.sparsity).astype(int),
            random_state=self.random_state)
        self._bias = self._uniform_random_bias(
            hidden_layer_size=self.hidden_layer_size,
            random_state=self.random_state)
        return self

    def transform(self, X):
        """Transforms the input matrix X.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of size (n_samples, hidden_layer_size)

        Returns
        -------
        Y: ndarray of size (n_samples, hidden_layer_size)
        """
        if self._recurrent_weights is None or self._bias is None:
            raise NotFittedError(self)

        if self.bi_directional:
            _hidden_layer_state_fw = self._pass_through_recurrent_weights(X=X)
            _hidden_layer_state_bw = np.flipud(self._pass_through_recurrent_weights(X=np.flipud(X)))
            self._hidden_layer_state = np.concatenate((_hidden_layer_state_fw, _hidden_layer_state_bw), axis=1)
        else:
            self._hidden_layer_state = self._pass_through_recurrent_weights(X=X)
        return self._hidden_layer_state

    def _pass_through_recurrent_weights(self, X):
        hidden_layer_state = np.zeros(shape=(X.shape[0]+1, self.hidden_layer_size))
        for sample in range(X.shape[0]):
            a = X[sample, :]
            b = safe_sparse_dot(hidden_layer_state[sample, :], self._recurrent_weights) * self.spectral_radius
            c = self._bias * self.bias_scaling
            hidden_layer_state[sample+1, :] = ACTIVATIONS[self.activation](a + b + c)
            hidden_layer_state[sample + 1, :] = (1 - self.leakage) * hidden_layer_state[sample, :] \
                                                 + self.leakage * hidden_layer_state[sample + 1, :]
        return hidden_layer_state[1:, :]

    @staticmethod
    def _normal_random_recurrent_weights(n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
        if n_features_in != hidden_layer_size:
            raise ValueError("Dimensional mismatch: n_features must match hidden_layer_size, got %s !=%s." %
                             (n_features_in, hidden_layer_size))
        nr_entries = np.int32(n_features_in * fan_in)
        weights_array = random_state.normal(loc=0., scale=1., size=nr_entries)

        if fan_in < hidden_layer_size:
            indices = np.zeros(shape=nr_entries, dtype=int)
            indptr = np.arange(start=0, stop=(n_features_in + 1) * fan_in, step=fan_in)

            for en in range(0, n_features_in * fan_in, fan_in):
                indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
            recurrent_weights_init = scipy.sparse.csr_matrix(
                (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
        else:
            recurrent_weights_init = weights_array.reshape((n_features_in, hidden_layer_size))

        we = eigens(recurrent_weights_init, return_eigenvectors=False, k=np.minimum(10, hidden_layer_size - 2))

        return recurrent_weights_init / np.amax(np.absolute(we))

    @staticmethod
    def _uniform_random_bias(hidden_layer_size: int, random_state):
        return random_state.uniform(low=-1., high=1., size=hidden_layer_size)

    def _validate_hyperparameters(self):
        """
        Validates the hyperparameters.

        Returns
        -------

        """
        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got %s." % self.hidden_layer_size)
        if self.spectral_radius < 0.:
            raise ValueError("spectral_radius must be >= 0, got %s." % self.spectral_radius)
        if self.leakage <= 0. or self.leakage > 1.:
            raise ValueError("leakage must be between 0. and 1., got %s." % self.leakage)
        if self.sparsity <= 0. or self.sparsity > 1.:
            raise ValueError("sparsity must be between 0. and 1., got %s." % self.sparsity)
        if self.bias_scaling < 0:
            raise ValueError("bias must be > 0, got %s." % self.bias_scaling)
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.activation, ACTIVATIONS))
        if self.k_rec is not None and self.k_rec <= 0:
            raise ValueError("k_rec must be > 0, got %d." % self.k_rec)
