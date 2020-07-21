import scipy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y, column_or_1d, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.exceptions import NotFittedError

from joblib import Parallel, delayed

if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
    from scipy.sparse.linalg import ArpackNoConvergence
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

_OFFLINE_SOLVERS = ['pinv', 'ridge', 'lasso']


class BaseEchoStateNetwork(BaseEstimator):
    """Base class for ESN classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.00
    """

    def __init__(self, k_in: int = 2, input_scaling: float = 1., spectral_radius: float = 0., bias: float = 0.,
                 ext_bias: int = 0, leakage: float = 1., feedback_scaling: float = 0.,reservoir_size: int = 500,
                 k_res: int = 10, wash_out: int = 0, reservoir_activation: str = 'tanh', bi_directional: bool = False,
                 teacher_scaling: float = 1., teacher_shift: float = 0., solver: str = 'ridge', beta: float = 1e-6,
                 random_state: int = None):
        self.k_in = k_in
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.bias = bias
        self.ext_bias = ext_bias
        self.leakage = leakage
        self.feedback_scaling = feedback_scaling
        self.reservoir_size = reservoir_size
        self.k_res = k_res
        self.wash_out = wash_out
        self.reservoir_activation = reservoir_activation
        self.bi_directional = bi_directional
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.solver = solver
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN model.
        """
        self._validate_hyperparameters()
        X, y = self._validate_input(X, y)
        self._initialize(y=y, n_features=X.shape[1])
        return self._fit(X, y, update_output_weights=True, n_jobs=n_jobs)

    def finalize(self, n_jobs=0):
        """
        Finalize the training by solving the linear regression problem and deleting xTx and xTy attributes.

        Parameters
        ----------
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------

        """
        self._finalize(n_jobs=n_jobs)

    def _validate_input(self, X, y):
        """
        Ensure that the input and output is in a proper format and transform it if possible.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        """
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        y = self.teacher_scaling * y + self.teacher_shift
        return X, y

    def _validate_hyperparameters(self):
        """
        Validate the hyperparameter. Ensure that the parameter ranges and dimensions are valid.
        Returns
        -------

        """
        if self.reservoir_size <= 0:
            raise ValueError("reservoir_size must be > 0, got %s." % self.reservoir_size)
        if self.input_scaling <= 0:
            raise ValueError("input_scaling must be > 0, got %s." % self.input_scaling)
        if self.k_in <= 0:
            raise ValueError("k_in must be > 0, got %s." % self.k_in)
        if self.spectral_radius < 0:
            raise ValueError("spectral_radius must be >= 0, got %s." % self.spectral_radius)
        if self.k_res <= 0:
            raise ValueError("k_res must be > 0, got %s." % self.k_res)
        if self.bias < 0:
            raise ValueError("bias must be > 0, got %s." % self.bias)
        if self.leakage > 1 or self.leakage < 0:
            raise ValueError("leakage must be >= 0 and <= 1")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0, got %s." % self.beta)
        # raise ValueError if not registered
        supported_activations = ('identity', 'logistic', 'tanh', 'relu')
        if self.reservoir_activation not in supported_activations:
            raise ValueError("The reservoir_activation '%s' is not supported. Supported "
                             "activations are %s." % (self.reservoir_activation, supported_activations))
        supported_solvers = _OFFLINE_SOLVERS
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))

    def _initialize(self, y, n_features):
        """
        Initialize everything for the Echo State Network. Set all attributes, allocate weights.
        Parameters
        ----------
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        n_features : int
            The number of input features, e.g. the second dimension of input matrix X

        Returns
        -------

        """
        self._random_state = check_random_state(self.random_state)
        # Either one- or multi-dimensional output
        if y.ndim == 2:
            self.n_outputs_ = y.shape[1]
        else:
            self.n_outputs_ = 1

        # Initialize number of training samples
        self._n_samples = 0

        # initialize all weights the model consists of
        input_weights_init, reservoir_weights_init, bias_weights_init, feedback_weights_init, output_weights_init = \
            self._init_weights(n_features)
        self.input_weights_ = input_weights_init
        self.reservoir_weights_ = reservoir_weights_init
        self.bias_weights_ = bias_weights_init
        self.output_weights_ = output_weights_init
        self._init_state_collection_matrices()

    def _init_state_collection_matrices(self):
        # collect the mean and variances of all reservoir nodes. This is required for the dropout strategy.
        self._activations_mean = np.zeros(shape=(self.reservoir_size,))
        self._activations_var = np.zeros(shape=(self.reservoir_size,))
        # initialize xTx and xTy for linear regression. Will be deleted after the training is finalized.
        if self.bi_directional:
            self._xTx = np.zeros(shape=(2 * self.reservoir_size + 1, 2 * self.reservoir_size + 1))
            self._xTy = np.zeros(shape=(2 * self.reservoir_size + 1, self.n_outputs_))
        else:
            self._xTx = np.zeros(shape=(self.reservoir_size + 1, self.reservoir_size + 1))
            self._xTy = np.zeros(shape=(self.reservoir_size + 1, self.n_outputs_))

    def _init_weights(self, n_features):
        """
        Initialize all weight matrices, e.g. connections from the input to the reservoir, and recurrent connections
        inside the reservoir.
        Parameters
        ----------
        n_features : int
            The number of input features, e.g. the second dimension of input matrix X

        Returns
        -------

        """
        # Input-to-reservoir weights, drawn from uniform distribution.
        idx_co = 0
        nr_entries = np.int32(self.reservoir_size*self.k_in)
        ij = np.zeros((2, nr_entries), dtype=int)
        data_vec = self._random_state.rand(nr_entries) * 2 - 1
        for en in range(self.reservoir_size):
            per = self._random_state.permutation(n_features)[:self.k_in]
            ij[0][idx_co:idx_co+self.k_in] = en
            ij[1][idx_co:idx_co+self.k_in] = per
            idx_co = idx_co + self.k_in
        input_weights_init = scipy.sparse.csc_matrix((data_vec, ij),
                                                     shape=(self.reservoir_size, n_features), dtype='float64')
        # Recurrent weights inside the reservoir, drawn from a standard normal distribution.
        converged = False
        # Recurrent weights are normalized to a unitary spectral radius if possible.
        attempts = 50
        while not converged and attempts > 0:
            try:
                idx_co = 0
                nr_entries = np.int32(self.reservoir_size * self.k_res)
                ij = np.zeros((2, nr_entries), dtype=int)
                data_vec = self._random_state.randn(nr_entries)
                for en in range(self.reservoir_size):
                    per = self._random_state.permutation(self.reservoir_size)[:self.k_res]
                    ij[0][idx_co:idx_co + self.k_res] = en
                    ij[1][idx_co:idx_co + self.k_res] = per
                    idx_co += self.k_res

                reservoir_weights_init = scipy.sparse.csc_matrix((data_vec, ij),
                                                                 shape=(self.reservoir_size, self.reservoir_size),
                                                                 dtype='float64')
                we = eigens(reservoir_weights_init, return_eigenvectors=False, k=6)
                converged = True
            except ArpackNoConvergence:
                print("WARNING: No convergence! Redo {0} times...".format(attempts-1))
                attempts = attempts - 1
                if attempts == 0:
                    print("WARNING: Returning possibly invalid eigenvalues...")
                we = ArpackNoConvergence.eigenvalues
                pass

        reservoir_weights_init *= (1. / np.amax(np.absolute(we)))
        # Bias weights, fully connected bias for the reservoir nodes, drawn from uniform distribution.
        if self.ext_bias > 0:
            bias_weights_init = (self._random_state.rand(self.reservoir_size, self.ext_bias) * 2 - 1)
        else:
            bias_weights_init = (self._random_state.rand(self.reservoir_size) * 2 - 1)
        # Feedback weights, fully connected feedback from the output to the reservoir nodes
        # drawn from uniform distribution.
        feedback_weights_init = (self._random_state.rand(self.n_outputs_, self.reservoir_size) * 2 - 1)
        output_weights_init = None  # np.zeros(shape=(self.reservoir_size + 1, self.n_outputs_))
        return input_weights_init, reservoir_weights_init, bias_weights_init, feedback_weights_init, output_weights_init

    def _fit(self, X, y, incremental=False, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        incremental : bool, default False
            If True, the network can be fitted with data that does not fit into memory. After each call to fit,
            output weights are trained if update_output_weights == True
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN model.
        """
        n_samples, n_features = X.shape
        if self.ext_bias > 0:
            n_features = n_features - self.ext_bias
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.n_outputs_ = y.shape[1]
        if (not hasattr(self, 'input_weights_')) or (not hasattr(self, 'reservoir_weights_')) \
                or (not hasattr(self, 'bias_weights_')) or not incremental:
            # First time training the model
            self._initialize(y, n_features)

        # Run the offline optimization solver
        if self.solver in _OFFLINE_SOLVERS:
            self._fit_offline(X, y, incremental, update_output_weights=update_output_weights, n_jobs=n_jobs)
        self.is_fitted_ = True
        return self

    def _pass_through_reservoir(self, X):
        """
        Pass the data forward and, if required, backwards through the reservoir.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        Returns
        -------
        reservoir_state : ndarray of shape (n_samples, reservoir_size)
            The collected reservoir states
        """
        reservoir_state = self._forward_pass(reservoir_inputs=X)
        if self.bi_directional:
            reservoir_state = \
                np.concatenate((reservoir_state, np.flipud(self._forward_pass(reservoir_inputs=np.flipud(X)))), 1)
        reservoir_state = np.concatenate((np.ones((reservoir_state.shape[0], 1)), reservoir_state), 1)
        return reservoir_state

    def _fit_offline(self, X, y, incremental=False, update_output_weights=True, n_jobs: int = 0):
        """
        Do a single fit of the model on the entire dataset passed trough.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        incremental : bool, default False
            If True, the network can be fitted with data that does not fit into memory. After each call to fit,
            output weights are trained if update_output_weights == True
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------

        """
        n_samples = X.shape[0]
        self._n_samples = self._n_samples + n_samples - self.wash_out

        reservoir_state = self._pass_through_reservoir(X=X)

        if incremental:
            self._xTx = self._xTx + np.dot(reservoir_state[self.wash_out:, :].T, reservoir_state[self.wash_out:, :])
            self._xTy = self._xTy + np.dot(reservoir_state[self.wash_out:, :].T, y[self.wash_out:, :])
            if self.bi_directional:
                new_activations_mean = np.mean(reservoir_state[self.wash_out:, :], axis=0)[1:self.reservoir_size + 1]
                new_activations_var = np.var(reservoir_state[self.wash_out:, :], axis=0)[1:self.reservoir_size + 1]
            else:
                new_activations_mean = np.mean(reservoir_state[self.wash_out:, :], axis=0)[1:]
                new_activations_var = np.var(reservoir_state[self.wash_out:, :], axis=0)[1:]
            m = self._n_samples
            n = reservoir_state.shape[0]
            tmp_activations_mean = self._activations_mean
            self._activations_mean = m/(m+n)*tmp_activations_mean + n/(m+n)*new_activations_mean
            self._activations_var = m / (m + n) * self._activations_var + n / (m + n)*new_activations_var + \
                                    m * n / (m + n)**2 * (tmp_activations_mean - new_activations_mean)**2
        else:
            self._xTx = np.dot(reservoir_state[self.wash_out:, :].T, reservoir_state[self.wash_out:, :])
            self._xTy = np.dot(reservoir_state[self.wash_out:, :].T, y[self.wash_out:, :])
            if self.bi_directional:
                self.activations_mean = np.mean(reservoir_state[self.wash_out:, :], axis=0)[1:self.reservoir_size + 1]
                self.activations_var = np.var(reservoir_state[self.wash_out:, :], axis=0)[1:self.reservoir_size + 1]
            else:
                self.activations_mean = np.mean(reservoir_state[self.wash_out:, :], axis=0)[1:]
                self.activations_var = np.var(reservoir_state[self.wash_out:, :], axis=0)[1:]

        if update_output_weights:
            self._compute_output_weights(n_jobs=n_jobs)
        else:
            self.output_weights_ = None

        if not incremental:
            self._xTx = None
            self._xTy = None

    def _forward_pass(self, reservoir_inputs):
        """
        Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        reservoir_inputs : ndarray of shape (n_samples, n_features)
            The input data

        Returns
        -------

        """
        n_samples, n_features = reservoir_inputs.shape
        reservoir_state = np.zeros(shape=(n_samples+1, self.reservoir_size))
        if self.ext_bias > 0:
            for sample in range(n_samples):
                if scipy.sparse.issparse(self.input_weights_):
                    a = self.input_weights_ * reservoir_inputs[sample, :-self.ext_bias] * self.input_scaling
                else:
                    a = np.dot(self.input_weights_, reservoir_inputs[sample, :-self.ext_bias], self.input_scaling)
                if scipy.sparse.issparse(self.reservoir_weights_):
                    b = self.reservoir_weights_ * reservoir_state[sample, :] * self.spectral_radius
                else:
                    b = np.dot(self.reservoir_weights_, reservoir_state[sample, :], self.spectral_radius)

                reservoir_state[sample + 1, :] = ACTIVATIONS[self.reservoir_activation](
                    np.atleast_2d(a + b).T + self.bias_weights_ * self.bias * reservoir_inputs[sample, -self.ext_bias]).flatten()
                reservoir_state[sample + 1, :] = (1 - self.leakage) * reservoir_state[sample, :]\
                                                 + self.leakage * reservoir_state[sample + 1, :]
        else:
            for sample in range(n_samples):
                if scipy.sparse.issparse(self.input_weights_):
                    a = self.input_weights_ * reservoir_inputs[sample, :] * self.input_scaling
                else:
                    a = np.dot(self.input_weights_, reservoir_inputs[sample, :], self.input_scaling)
                if scipy.sparse.issparse(self.reservoir_weights_):
                    b = self.reservoir_weights_ * reservoir_state[sample, :] * self.spectral_radius
                else:
                    b = np.dot(self.reservoir_weights_, reservoir_state[sample, :], self.spectral_radius)

                reservoir_state[sample+1, :] = \
                    ACTIVATIONS[self.reservoir_activation](a + b + self.bias_weights_*self.bias)
                reservoir_state[sample+1, :] = \
                    (1 - self.leakage) * reservoir_state[sample, :] + self.leakage * reservoir_state[sample+1, :]
        return reservoir_state[1:, :]

    def partial_fit(self, X, y, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN model.
        """
        if self.solver not in _OFFLINE_SOLVERS:
            raise AttributeError('partial_fit is only available for offline optimizers, not for %s.' % self.solver)
        return self._partial_fit(X=X, y=y, update_output_weights=update_output_weights, n_jobs=n_jobs)

    def _partial_fit(self, X, y, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN model.
        """
        return self._fit(X, y, incremental=True, update_output_weights=update_output_weights, n_jobs=n_jobs)

    def drop_out(self, drop_out_rate=0.0):
        """
        Experimental dropout strategy for the ESN. After passing some data through the network and collecting reservoir
        activations, a percentile of nodes with very low activity can be removed. Note that this works best without
        bidirectional mode right now.

        Warnings : This is experimental and needs to be validated at first. Publications will follow!!!

        Parameters
        ----------
        drop_out_rate : double, default 0.0
            Determines the percentile of nodes to be removed

        Returns
        -------

        """
        self._drop_out(drop_out_rate=drop_out_rate)

    def _drop_out(self, drop_out_rate):
        """
        Experimental dropout strategy for the ESN. After passing some data through the network and collecting reservoir
        activations, a percentile of nodes with very low activity can be removed. Note that this works best without
        bidirectional mode right now.

        Warnings : This is experimental and needs to be validated at first. Publications will follow!!!

        Warnings : After dropout, the output weights need to be trained again, because several nodes have been cut off!

        Parameters
        ----------
        drop_out_rate : double, default 0.0
            Determines the percentile of nodes to be removed

        Returns
        -------

        """
        if drop_out_rate > 0.0:
            new_reservoir_size = int(drop_out_rate * self.reservoir_size)
            idx_to_drop_ = np.argsort(self._activations_var)[::-1][int(drop_out_rate * self.reservoir_size):]
            self.bias_weights_ = np.delete(self.bias_weights_, idx_to_drop_)
            self.input_weights_ = scipy.sparse.csc_matrix(
                np.delete(self.input_weights_.toarray(), idx_to_drop_, axis=0), dtype='float64')
            self.reservoir_weights_ = scipy.sparse.csc_matrix(
                np.delete(np.delete(self.reservoir_weights_.toarray(), idx_to_drop_, axis=0), idx_to_drop_, axis=1),
                dtype='float64')

            self._n_samples = 0

            self.reservoir_size = new_reservoir_size
            self._init_state_collection_matrices()

    def _finalize(self, n_jobs: int = 0):
        """
        This finalizes the training of a model. No more required attributes, such as activations, xTx, xTy will be
        removed.

        Warnings : The model cannot be improved with additional data afterwards!!!

        Parameters
        ----------
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------

        """
        if self.output_weights_ is None:
            self._compute_output_weights(n_jobs=n_jobs)

        self._xTx = None
        self._xTy = None
        self._activations_mean = None
        self._activations_var = None
        self.is_fitted_ = True

    def _compute_output_weights(self, n_jobs=0):
        """
        This is a helper function to compute the output weights using linear regression
        Parameters
        ----------
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------

        """
        if self.solver == 'pinv':
            inv_xTx = np.linalg.inv(self._xTx)
        elif self.solver == 'ridge':
            lmda = self.beta ** 2 * self._n_samples
            inv_xTx = np.linalg.inv(self._xTx + lmda * np.eye(self._xTx.shape[0]))
        else:
            print("Warning: Not implemented. Falling back to pinv solution")
            inv_xTx = np.linalg.inv(self._xTx)
        if n_jobs > 0:
            self.output_weights_ = Parallel(n_jobs=n_jobs)(
                delayed(np.dot)(inv_xTx, self._xTy[:, n]) for n in range(self.n_outputs_))
        else:
            self.output_weights_ = np.dot(inv_xTx, self._xTy)

    def predict(self, X, keep_reservoir_state=False):
        """
        Predict using the trained ESN model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values
        """
        check_is_fitted(self, ['input_weights_', 'reservoir_weights_', 'bias_weights_', 'output_weights_'])
        if not self.output_weights_.any():
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})
        X = check_array(X, accept_sparse=False)
        y_pred = self._predict(X=X, keep_reservoir_state=keep_reservoir_state)
        return y_pred

    def _predict(self, X, keep_reservoir_state=False):
        """
        Predict using the trained ESN model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values
        """
        reservoir_state = self._pass_through_reservoir(X=X)
        if keep_reservoir_state:
            self.reservoir_state = reservoir_state
        y_pred = safe_sparse_dot(reservoir_state, self.output_weights_)
        return y_pred


class ESNClassifier(BaseEchoStateNetwork, ClassifierMixin):
    """
    Echo State Network classifier.

    This model optimizes the mean squared error loss function using linear regression.

    .. versionadded:: 0.00

    Parameters
    ----------
    k_in : int, default 2
        This element represents the sparsity of the connections between the input and recurrent nodes.
        It determines the number of features that every node inside the reservoir receives.
    input_scaling : float, default 1.0
        This element represents the input scaling factor from the input to the reservoir. It is a global scaling factor
        for the input weight matrix.
    spectral_radius : float, default 0.0
        This element represents the spectral radius of the reservoir weights. It is a global scaling factor for the
        reservoir weight matrix.
    bias : float, default 0.0
        This element represents the bias scaling of the bias weights. It is a global scaling factor for the bias weight
        matrix.
    ext_bias : int, default 0
        This element represents the number of external biases to be used. If greater than zero, they are supposed to be
        appended to the input matrix.
    leakage : float, default 1.0
        This element represents the leakage of the reservoir. Depending on the value, it acts as a short- or long-term
        memory coefficient.
    reservoir_size : int, default 500
        This element represents the number of neurons in the reservoir.
    k_res : int, default 10
        This element represents the sparsity of the connections inside the reservoir. It determines the number of nodes
        that every node inside the reservoir is connected with.
    wash_out: int, default 0
        This element defines the number of first samples to be not regarded for training.
    reservoir_activation : {'tanh', 'identity', 'logistic', 'relu'}
        This element represents the activation function in the reservoir.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
    teacher_scaling : float, default 1.0
        This element represents the teacher scaling factor. In some cases, for example for imbalanced datasets, it might
        make sense to increase this hyperparameter.
    teacher_shift : float, default 0.0
        This element represents the teacher shift factor. In some cases, for example for imbalanced datasets, it might
        make sense to increase this hyperparameter.
    bi_directional : bool, default False
        If True, the input sequences are passed through the network two times, forward and backward.
    solver : {'ridge', 'pinv'}
        The solver for weight optimization.
        - 'pinv' uses the pseudoinverse solution of linear regression.
        - 'ridge' uses L2 penalty while computing the linear regression
    beta : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.
    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    TODO
    """
    def __init__(self, k_in: int = 2, input_scaling: float = 1., spectral_radius: float = 0., bias: float = 0.,
                 ext_bias: int = 0, leakage: float = 1., reservoir_size: int = 500, k_res: int = 10, wash_out: int = 0,
                 reservoir_activation: str = 'tanh', bi_directional: bool = False, teacher_scaling: float = 1.,
                 teacher_shift: float = 0., solver: str = 'ridge', beta: float = 1e-6, random_state: int = None):
        super().__init__(k_in=k_in, input_scaling=input_scaling, spectral_radius=spectral_radius, bias=bias,
                         ext_bias=ext_bias, leakage=leakage, reservoir_size=reservoir_size, k_res=k_res,
                         wash_out=wash_out, reservoir_activation=reservoir_activation, bi_directional=bi_directional,
                         teacher_scaling=teacher_scaling, teacher_shift=teacher_shift, solver=solver, beta=beta,
                         random_state=random_state)

    def _validate_input(self, X, y):
        """
        Ensure that the input and output is in a proper format and transform it if possible.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        """
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_
        y = self._label_binarizer.transform(y)
        y = self.teacher_scaling * y + self.teacher_shift
        return X, y

    def fit(self, X, y, n_jobs: int = 0):
        """
        Fit the model to the data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN model.
        """
        self._validate_hyperparameters()
        X, y = self._validate_input(X, y)
        if self.ext_bias:
            self._initialize(y=y, n_features=X.shape[1] - 1)
        else:
            self._initialize(y=y, n_features=X.shape[1])
        return self._fit(X, y, incremental=False, update_output_weights=True, n_jobs=n_jobs)

    def predict(self, X, keep_reservoir_state=False):
        """
        Predict the classes using the trained ESN classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes
        """
        check_is_fitted(self, ['input_weights_', 'reservoir_weights_', 'bias_weights_', 'output_weights_'])
        if self.output_weights_.size == 0:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})
        y_pred = super().predict(X, keep_reservoir_state=keep_reservoir_state)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        return self._label_binarizer.inverse_transform(y_pred)

    def partial_fit(self, X, y, classes=None, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        classes : ndarray of shape (class labels, )
            The class labels to be predicted!
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN classifier.
        """
        if self.solver not in _OFFLINE_SOLVERS:
            raise AttributeError('partial_fit is only available for offline optimizers, not for %s.' % self.solver)
        return self._partial_fit(X=X, y=y, classes=classes, update_output_weights=update_output_weights, n_jobs=n_jobs)

    def _partial_fit(self, X, y, classes=None, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN classifier.
        """
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer().fit(classes)
            if self.ext_bias:
                super()._initialize(y=y, n_features=X.shape[1] - 1)
            else:
                super()._initialize(y=y, n_features=X.shape[1])

        super()._partial_fit(X, y, update_output_weights=update_output_weights, n_jobs=n_jobs)
        return self

    def predict_proba(self, X, keep_reservoir_state=False):
        """
        Predict the probability estimates using the trained ESN classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted probability estimates
        """
        y_pred = super().predict(X, keep_reservoir_state=keep_reservoir_state)
        y_pred = np.maximum(y_pred, 1e-3)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def predict_log_proba(self, X, keep_reservoir_state=False):
        """
        Predict the logarithmic probability estimates using the trained ESN classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted logarithmic probability estimates
        """
        y_pred = self.predict_proba(X=X, keep_reservoir_state=keep_reservoir_state)
        return np.log(y_pred)


class ESNRegressor(BaseEchoStateNetwork, RegressorMixin):
    """
    Echo State Network regressor.

    This model optimizes the mean squared error loss function using linear regression.

    .. versionadded:: 0.00

    Parameters
    ----------
    k_in : int, default 2
        This element represents the sparsity of the connections between the input and recurrent nodes.
        It determines the number of features that every node inside the reservoir receives.
    input_scaling : float, default 1.0
        This element represents the input scaling factor from the input to the reservoir. It is a global scaling factor
        for the input weight matrix.
    spectral_radius : float, default 0.0
        This element represents the spectral radius of the reservoir weights. It is a global scaling factor for the
        reservoir weight matrix.
    bias : float, default 0.0
        This element represents the bias scaling of the bias weights. It is a global scaling factor for the bias weight
        matrix.
    ext_bias : int, default 0
        This element represents the number of external biases to be used. If greater than zero, they are supposed to be
        appended to the input matrix.
    leakage : float, default 1.0
        This element represents the leakage of the reservoir. Depending on the value, it acts as a short- or long-term
        memory coefficient.
    reservoir_size : int, default 500
        This element represents the number of neurons in the reservoir.
    k_res : int, default 10
        This element represents the sparsity of the connections inside the reservoir. It determines the number of nodes
        that every node inside the reservoir is connected with.
    wash_out: int, default 0
        This element defines the number of first samples to be not regarded for training.
    reservoir_activation : {'tanh', 'identity', 'logistic', 'relu'}
        This element represents the activation function in the reservoir.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
    teacher_scaling : float, default 1.0
        This element represents the teacher scaling factor. In some cases, for example for imbalanced datasets, it might
        make sense to increase this hyperparameter.
    teacher_shift : float, default 0.0
        This element represents the teacher shift factor. In some cases, for example for imbalanced datasets, it might
        make sense to increase this hyperparameter.
    bi_directional : bool, default False
        If True, the input sequences are passed through the network two times, forward and backward.
    solver : {'ridge', 'pinv'}
        The solver for weight optimization.
        - 'pinv' uses the pseudoinverse solution of linear regression.
        - 'ridge' uses L2 penalty while computing the linear regression
    beta : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.
    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    TODO

    Notes
    -----
    TODO

    References
    -----------
    TODO
    """
    def __init__(self, k_in: int = 2, input_scaling: float = 1., spectral_radius: float = 0., bias: float = 0.,
                 ext_bias: int = 0, leakage: float = 1., reservoir_size: int = 500, k_res: int = 10, wash_out: int = 0,
                 reservoir_activation: str = 'tanh', bi_directional: bool = False, teacher_scaling: float = 1.,
                 teacher_shift: float = 0., solver: str = 'ridge', beta: float = 1e-6, random_state: int = None):
        super().__init__(k_in=k_in, input_scaling=input_scaling, spectral_radius=spectral_radius, bias=bias,
                         ext_bias=ext_bias, leakage=leakage, reservoir_size=reservoir_size, k_res=k_res,
                         wash_out=wash_out, reservoir_activation=reservoir_activation, bi_directional=bi_directional,
                         teacher_scaling=teacher_scaling, teacher_shift=teacher_shift, solver=solver, beta=beta,
                         random_state=random_state)

    def fit(self, X, y, n_jobs=0):
        self._validate_hyperparameters()
        X, y = self._validate_input(X, y)
        self._initialize(y=y, n_features=X.shape[1])
        return self._fit(X, y, update_output_weights=True, n_jobs=n_jobs)

    def predict(self, X, keep_reservoir_state=False):
        """
        Predict the classes using the trained ESN regressor

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_reservoir_state : bool, default False
            If True, the reservoir state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes
        """
        y_pred = super().predict(X, keep_reservoir_state=keep_reservoir_state)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def partial_fit(self, X, y, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN classifier.
        """
        if self.solver not in _OFFLINE_SOLVERS:
            raise AttributeError("partial_fit is only available for offline optimizer. %s is not offline"
                                 % self.solver)
        return self._partial_fit(X=X, y=y, update_output_weights=update_output_weights, n_jobs=n_jobs)

    def _partial_fit(self, X, y, update_output_weights=True, n_jobs=0):
        """
        Fit the model to the data matrix X and target(s) y without finalizing it. This can be used to add more training
        data later.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        y : ndarray of shape (n_samples, ) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        update_output_weights : bool, default True
            If False, no output weights are computed after passing the current data through the network.
            This is computationally more efficient in case of a lot of outputs and a large dataset that is fitted
            incrementally.
        n_jobs : int, default: 0
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately
            using joblib.

        Returns
        -------
        self : returns a trained ESN classifier.
        """
        super()._partial_fit(X, y, update_output_weights=update_output_weights, n_jobs=n_jobs)
        return self
