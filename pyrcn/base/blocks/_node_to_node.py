# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import sys

import scipy
import numpy as np

import sklearn
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import check_consistent_length, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import NotFittedError

from pyrcn.base import ACTIVATIONS

if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
    from scipy.sparse.linalg import ArpackNoConvergence
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence


class NodeToNode(BaseEstimator, TransformerMixin):
    """
    NodeToNode class for reservoir computing modules (e.g. ESN)

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default=1.
        Quotient of recurrent weights per node (k_rec) and number of input features (n_features)
    reservoir_activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    spectral_radius : float, default=1.
        Scales the recurrent weight matrix.
    leakage : float, default=1.
        parameter to determine the degree of leaky integration.
    bidirectional : bool, default=False
        Whether to work in bidirectional mode.
    k_rec : int, default=None.
        recurrent weights per node. By default, it is None. If set, it overrides sparsity.
    random_state : {None, int, RandomState}, default=42
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 hidden_layer_size=500,
                 sparsity=1.,
                 reservoir_activation='tanh',
                 spectral_radius=1.,
                 leakage=1.,
                 bidirectional=False,
                 k_rec=None,
                 random_state=42):
        self.hidden_layer_size = hidden_layer_size
        self.sparsity = sparsity
        self.reservoir_activation = reservoir_activation
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.bidirectional = bidirectional
        self.k_rec = k_rec
        self.random_state = random_state

        self._recurrent_weights = None
        self._hidden_layer_state = None

    def fit(self, X, y=None):
        """
        Fit the NodeToNode. Initialize recurrent weights.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        self._validate_hyperparameters()
        if y is None:
            self._validate_data(X, y)
        else:
            self._validate_data(X, y, multi_output=True)
        self._check_n_features(X, reset=True)

        if self.k_rec is not None:
            self.sparsity = self.k_rec / X.shape[1]
        self._recurrent_weights = self._normal_random_recurrent_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size * self.sparsity).astype(int),
            random_state=self._random_state)
        return self

    def transform(self, X, y=None):
        """Transforms the input matrix X.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of size (n_samples, hidden_layer_size)
        y : ignored

        Returns
        -------
        Y: ndarray of size (n_samples, hidden_layer_size)
        """
        if self._recurrent_weights is None:
            raise NotFittedError(self)

        if self.bidirectional:
            _hidden_layer_state_fw = self._pass_through_recurrent_weights(X=X)
            _hidden_layer_state_bw = np.flipud(self._pass_through_recurrent_weights(X=np.flipud(X)))
            self._hidden_layer_state = np.concatenate((_hidden_layer_state_fw, _hidden_layer_state_bw), axis=1)
        else:
            self._hidden_layer_state = self._pass_through_recurrent_weights(X=X, y=y)
        return self._hidden_layer_state

    def _pass_through_recurrent_weights(self, X, y=None):
        hidden_layer_state = np.zeros(shape=(X.shape[0]+1, self.hidden_layer_size))
        if self._hidden_layer_state is not None and not self.bidirectional:
            hidden_layer_state[0, :] = self._hidden_layer_state[-1, :]
        for sample in range(X.shape[0]):
            a = X[sample, :]
            b = safe_sparse_dot(hidden_layer_state[sample, :], self._recurrent_weights) * self.spectral_radius
            pre_activation = a + b
            ACTIVATIONS[self.reservoir_activation](pre_activation)
            hidden_layer_state[sample+1, :] = pre_activation
            hidden_layer_state[sample + 1, :] = (1 - self.leakage) * hidden_layer_state[sample, :] \
                                                 + self.leakage * hidden_layer_state[sample + 1, :]
        return hidden_layer_state[1:, :]

    @staticmethod
    def _normal_random_recurrent_weights(n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
        """
        Returns normally distributed random reservoir weights

        Parameters
        ----------
        n_features_in : int
        hidden_layer_size : int
        fan_in : int
            Determines how many features are mapped to one neuron.
        random_state : numpy.RandomState

        Returns
        -------
        normal_random_recurrent_weights : ndarray of size (hidden_layer_size, hidden_layer_size)
        """
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

        try:
            we = eigens(recurrent_weights_init, 
                        k=np.minimum(10, hidden_layer_size - 2), 
                        which='LM', 
                        return_eigenvectors=False, 
                        v0=random_state.normal(loc=0., scale=1., size=hidden_layer_size)
                        )
        except ArpackNoConvergence:
            print("WARNING: No convergence! Returning possibly invalid values!!!")
            we = ArpackNoConvergence.eigenvalues
        return recurrent_weights_init / np.amax(np.absolute(we))

    def _validate_hyperparameters(self):
        """
        Validates the hyperparameters.

        Returns
        -------

        """
        self._random_state = check_random_state(self.random_state)

        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got %s." % self.hidden_layer_size)
        if self.sparsity <= 0. or self.sparsity > 1.:
            raise ValueError("sparsity must be between 0. and 1., got %s." % self.sparsity)
        if self.reservoir_activation not in ACTIVATIONS:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.reservoir_activation, ACTIVATIONS))
        if self.spectral_radius < 0.:
            raise ValueError("spectral_radius must be >= 0, got %s." % self.spectral_radius)
        if self.leakage <= 0. or self.leakage > 1.:
            raise ValueError("leakage must be between 0. and 1., got %s." % self.leakage)
        if self.bidirectional not in [False, True]:
            raise ValueError("bidirectional must be either False or True,, got %s." % self.bidirectional)
        if self.k_rec is not None and self.k_rec <= 0:
            raise ValueError("k_rec must be > 0, got %d." % self.k_rec)

    def __sizeof__(self):
        """
        Returns the size of the object in bytes.

        Returns
        -------
        size : int
        Object memory in bytes.
        """
        if scipy.sparse.issparse(self._recurrent_weights):
            return object.__sizeof__(self) + \
                self._recurrent_weights.todense().nbytes + \
                self._hidden_layer_state.nbytes + \
                sys.getsizeof(self.random_state)
        else:
            return object.__sizeof__(self) + \
                self._recurrent_weights.nbytes + \
                self._hidden_layer_state.nbytes + \
                sys.getsizeof(self.random_state)

    @property
    def recurrent_weights(self):
        """Returns the recurrent weights.

        Returns
        -------
        recurrent_weights : ndarray of size (hidden_layer_size, hidden_layer_size)
        """
        return self._recurrent_weights


class PredefinedWeightsNodeToNode(NodeToNode):
    """
    PredefinedWeightsNodeToNode class for reservoir computing modules (e.g. ESN)

    Parameters
    ----------
    predefined_recurrent_weights : np.ndarray
        A set of predefined recurrent weights weights.
    reservoir_activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    spectral_radius : float, default=1.
        Scales the recurrent weight matrix.
    random_state : {None, int, RandomState}, default=42
    """
    @_deprecate_positional_args
    def __init__(self,
            predefined_recurrent_weights, *,
            reservoir_activation = 'tanh',
            spectral_radius = 1.,
            leakage=1.,
            bidirectional=False,
            random_state = 42):
        super().__init__(
            hidden_layer_size=predefined_recurrent_weights.shape[0],
            sparsity=1.,
            reservoir_activation = reservoir_activation,
            spectral_radius=spectral_radius,
            leakage=leakage,
            bidirectional=bidirectional,
            random_state=random_state)
        self.predefined_recurrent_weights = predefined_recurrent_weights

    def fit(self, X, y=None):
        """
        
        self._recurrent_weights = self._normal_random_recurrent_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size * self.sparsity).astype(int),
            random_state=self._random_state)
        return self
        """
        self._validate_hyperparameters()
        if y is None:
            self._validate_data(X, y)
        else:
            self._validate_data(X, y, multi_output=True)
        self._check_n_features(X, reset=True)

        if self.k_rec is not None:
            self.sparsity = self.k_rec / X.shape[1]
        if self.predefined_recurrent_weights is None:
            raise ValueError('predefined_recurrent_weights have to be defined, use NodeToNode class!')

        if self.predefined_recurrent_weights.shape[0] != X.shape[1]:
            raise ValueError('X has not the expected shape {0}, given {1}.'.format(
                self.predefined_recurrent_weights.shape[0], X.shape[1]))

        if self.predefined_recurrent_weights.shape[0] != self.predefined_recurrent_weights.shape[1]:
            raise ValueError('Recurrent weights need to be a squared matrix, given {1}.'.format(
                self.predefined_recurrent_weights.shape))

        self._recurrent_weights = self.predefined_recurrent_weights
        return self


class HebbianNodeToNode(NodeToNode):
    """
    HebbianNodeToNode for reservoir computing modules (e.g. ESN).

    Applies the hebbian rule to a given set of randomly initialized reservoir weights.
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 hidden_layer_size=500,
                 sparsity=1.,
                 reservoir_activation='tanh',
                 spectral_radius=1.,
                 leakage=1.,
                 bidirectional=False,
                 k_rec=None,
                 random_state=42,
                 learning_rate=0.01,
                 epochs=100,
                 training_method='hebbian'):
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

    def fit(self, X, y=None):
        """
        Fit the HebbianNodeToNode. Initialize recurrent weights and do Hebbian learning.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : ignored

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
                            which='LM', 
                            return_eigenvectors=False, 
                            v0=self._random_state.normal(loc=0., scale=1., size=self.hidden_layer_size)
                            )
            except ArpackNoConvergence:
                print("WARNING: No convergence! Returning possibly invalid values!!!")
                we = ArpackNoConvergence.eigenvalues
            self._recurrent_weights = self._recurrent_weights / np.amax(np.absolute(we))
        return self

    def _hebbian_learning(self, X, y=None):
        hidden_layer_state = self._pass_through_recurrent_weights(X=X, y=y)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= self.learning_rate * safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k:k+1, :]) * self._recurrent_weights

    def _anti_hebbian_learning(self, X, y=None):
        hidden_layer_state = self._pass_through_recurrent_weights(X=X, y=y)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= -self.learning_rate * safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k:k+1, :]) * self._recurrent_weights

    def _oja_learning(self, X, y=None):
        hidden_layer_state = self._pass_through_recurrent_weights(X=X, y=y)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= self.learning_rate * (safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k:k+1, :]) - safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k+1:k+2, :])) * self._recurrent_weights

    def _anti_oja_learning(self, X, y=None):
        hidden_layer_state = self._pass_through_recurrent_weights(X=X, y=y)
        for k in range(hidden_layer_state.shape[0] - 1):
            self._recurrent_weights -= self.learning_rate * (-safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k:k+1, :]) - safe_sparse_dot(hidden_layer_state[k+1:k+2, :].T, hidden_layer_state[k+1:k+2, :])) * self._recurrent_weights


class FeedbackNodeToNode(NodeToNode):
    """
    FeedbackNodeToNode class for reservoir computing modules (e.g. ESN)

    Parameters
    ----------
    hidden_layer_size : int, default=500
        Sets the number of nodes in hidden layer. Equals number of output features.
    sparsity : float, default=1.
        Quotient of recurrent weights per node (k_rec) and number of recurrent features (n_features)
    reservoir_activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    spectral_radius : float, default=1.
        Scales the recurrent weight matrix.
    leakage : float, default=1.
        parameter to determine the degree of leaky integration.
    teacher_scaling : float, default=1.
        Factor applied to the target signal
    teacher_shift: float, default=0.
        Additive term applied to the target signal
    bidirectional : bool, default=None
        Whether to work in bidirectional mode. This is ignored here.
    k_rec : int, default=None.
        recurrent weights per node. By default, it is None. If set, it overrides sparsity.
    output_activation : {'tanh', 'identity', 'logistic', 'relu', 'bounded_relu'}, default='tanh'
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the bounded rectified linear unit function, returns f(x) = min(max(x, 0),1)
    random_state : {None, int, RandomState}, default=42
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 hidden_layer_size=500,
                 sparsity=1.,
                 reservoir_activation='tanh',
                 spectral_radius=1.,
                 leakage=1.,
                 teacher_scaling=1.,
                 teacher_shift=1.,
                 k_rec=None,
                 output_activation='tanh',
                 random_state=42):
        super().__init__(
            hidden_layer_size=hidden_layer_size,
            sparsity=sparsity,
            reservoir_activation=reservoir_activation,
            spectral_radius=spectral_radius,
            leakage=leakage,
            bidirectional=False,
            k_rec=k_rec,
            random_state=random_state)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.output_activation = output_activation

        self._output_weights = None
        self._feedback_weights = None
        self._y_pred = None

    def fit(self, X, y=None):
        """
        Fit the NodeToNode. Initialize recurrent weights.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        super().fit(X=X, y=y)
        self._validate_hyperparameters()
        self._feedback_weights = self._uniform_random_feedback(
            hidden_layer_size=self.hidden_layer_size,
            output_size=y.shape[1],
            random_state=self._random_state)
        return self

    def _pass_through_recurrent_weights(self, X, y):
        hidden_layer_state = np.zeros(shape=(X.shape[0] + 1, self.hidden_layer_size))
        if y is not None:
            if self._hidden_layer_state is not None:
                hidden_layer_state[0, :] = self._hidden_layer_state[-1, :]
            for sample in range(X.shape[0]):
                a = X[sample, :]
                b = safe_sparse_dot(hidden_layer_state[sample, :], self._recurrent_weights) * self.spectral_radius
                c = safe_sparse_dot(self._feedback_weights, y[sample, :] * self.teacher_scaling + self.teacher_shift)
                pre_activation = a + b + c
                ACTIVATIONS[self.reservoir_activation](pre_activation)
                hidden_layer_state[sample+1, :] = pre_activation
                hidden_layer_state[sample + 1, :] = (1 - self.leakage) * hidden_layer_state[sample, :] \
                                                     + self.leakage * hidden_layer_state[sample + 1, :]
                self._last_output = y[sample, :] * self.teacher_scaling + self.teacher_shift
        else:
            hidden_layer_state[0, :] = self._hidden_layer_state[-1, :]
            self._y_pred = np.zeros(shape=(X.shape[0] + 1, self._feedback_weights.shape[1]))
            self._y_pred[0, :] = self._last_output
            for sample in range(X.shape[0]):
                a = X[sample, :]
                b = safe_sparse_dot(hidden_layer_state[sample, :], self._recurrent_weights) * self.spectral_radius
                c = safe_sparse_dot(self._feedback_weights, self._y_pred[sample, :])
                pre_activation = a + b + c
                ACTIVATIONS[self.reservoir_activation](pre_activation)
                hidden_layer_state[sample + 1, :] = pre_activation
                hidden_layer_state[sample + 1, :] = (1 - self.leakage) * hidden_layer_state[sample, :] \
                                                     + self.leakage * hidden_layer_state[sample + 1, :]
                _y_pred = safe_sparse_dot(np.hstack((hidden_layer_state[sample+1, :], 1)), self._output_weights)
                ACTIVATIONS[self.output_activation](_y_pred)
                self._y_pred[sample + 1, :] = _y_pred
                self._last_output = self._y_pred[sample, :]
        return hidden_layer_state[:-1, :]

    @staticmethod
    def _uniform_random_feedback(hidden_layer_size: int, output_size: int, random_state):
        return random_state.uniform(low=-1., high=1., size=(hidden_layer_size, output_size))

    def _validate_hyperparameters(self):
        """
        Validates the hyperparameters.

        Returns
        -------

        """
        super()._validate_hyperparameters()
        if self.output_activation not in ACTIVATIONS:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.output_activation, ACTIVATIONS))

    def __sizeof__(self):
        """
        Returns the size of the object in bytes.

        Returns
        -------
        size : int
        Object memory in bytes.
        """
        if scipy.sparse.issparse(self._recurrent_weights):
            return object.__sizeof__(self) + \
                self._recurrent_weights.todense().nbytes + \
                self._feedback_weights.nbytes + \
                self._hidden_layer_state.nbytes + \
                sys.getsizeof(self.random_state)
        else:
            return object.__sizeof__(self) + \
                self._recurrent_weights.nbytes + \
                self._feedback_weights.nbytes + \
                self._hidden_layer_state.nbytes + \
                sys.getsizeof(self.random_state)

    @property
    def recurrent_weights(self):
        """Returns the recurrent weights.

        Returns
        -------
        recurrent_weights : ndarray of size (hidden_layer_size, hidden_layer_size)
        """
        return self._recurrent_weights

    @property
    def feedback_weights(self):
        """Returns the feedback weights.

        Returns
        -------
        recurrent_weights : ndarray of size (hidden_layer_size)
        """
        return self._feedback_weights
