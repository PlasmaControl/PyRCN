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


class BaseExtremeLearningMachine(BaseEstimator):
    """Base class for ELM classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.00
    """

    def __init__(self, k_in: int = -1, input_scaling: float = 1., bias: float = 0., hidden_layer_size: int = 500,
                 activation_function: str = 'tanh', solver: str = 'ridge', beta: float = 1e-6,
                 random_state: int = None):
        self.k_in = k_in
        self.input_scaling = input_scaling
        self.bias = bias
        self.hidden_layer_size = hidden_layer_size
        self.activation_function = activation_function
        self.solver = solver
        self.beta = beta
        self.random_state = random_state

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
        self : returns a trained ELM model.
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
        return X, y

    def _validate_hyperparameters(self):
        """
        Validate the hyperparameter. Ensure that the parameter ranges and dimensions are valid.
        Returns
        -------

        """
        if self.hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be > 0, got %s." % self.hidden_layer_size)
        if self.input_scaling <= 0:
            raise ValueError("input_scaling must be > 0, got %s." % self.input_scaling)
        if self.k_in <= 0 and self.k_in != -1:
            raise ValueError("k_in must be > 0 or -1 (all inputs are used by each neuron), got %s." % self.k_in)
        if self.bias < 0:
            raise ValueError("bias must be > 0, got %s." % self.bias)
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0, got %s." % self.beta)
        # raise ValueError if not registered
        supported_activations = ('identity', 'logistic', 'tanh', 'relu', 'bounded_relu')
        if self.activation_function not in supported_activations:
            raise ValueError("The activation_function '%s' is not supported. Supported "
                             "activations are %s." % (self.activation_function, supported_activations))
        supported_solvers = _OFFLINE_SOLVERS
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))

    def _initialize(self, y, n_features):
        """
        Initialize everything for the Extreme Learning Machine. Set all attributes, allocate weights.
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
        input_weights_init, bias_weights_init, output_weights_init = self._init_weights(n_features)
        self.input_weights_ = input_weights_init
        self.bias_weights_ = bias_weights_init
        self.output_weights_ = output_weights_init
        self._init_state_collection_matrices()

    def _init_state_collection_matrices(self):
        """
        Initialize all weight matrices, e.g. connections from the input to the hidden layer, and recurrent connections
        inside the hidden layer.
        Parameters
        ----------
        n_features : int
            The number of input features, e.g. the second dimension of input matrix X

        Returns
        -------

        """
        # collect the mean and variances of all hidden layer nodes. This is required for the dropout strategy.
        self._activations_mean = np.zeros(shape=(self.hidden_layer_size,))
        self._activations_var = np.zeros(shape=(self.hidden_layer_size,))
        # initialize xTx and xTy for linear regression. Will be deleted after the training is finalized.
        self._xTx = np.zeros(shape=(self.hidden_layer_size + 1, self.hidden_layer_size + 1))
        self._xTy = np.zeros(shape=(self.hidden_layer_size + 1, self.n_outputs_))

    def _init_weights(self, n_features):
        """
        Initialize all weight matrices, e.g. connections from the input to the hidden layer.
        Parameters
        ----------
        n_features : int
            The number of input features, e.g. the second dimension of input matrix X

        Returns
        -------

        """
        # Input-to-node weights, drawn from uniform distribution.
        idx_co = 0

        if self.k_in == -1:
            self.k_in = n_features

        nr_entries = np.int32(self.hidden_layer_size*self.k_in)
        ij = np.zeros((2, nr_entries), dtype=int)
        data_vec = self._random_state.rand(nr_entries) * 2 - 1
        for en in range(self.hidden_layer_size):
            per = self._random_state.permutation(n_features)[:self.k_in]
            ij[0][idx_co:idx_co+self.k_in] = en
            ij[1][idx_co:idx_co+self.k_in] = per
            idx_co = idx_co + self.k_in
        input_weights_init = scipy.sparse.csc_matrix((data_vec, ij),
                                                     shape=(self.hidden_layer_size, n_features), dtype='float64')
        # Bias weights, fully connected bias for the hidden layer nodes, drawn from uniform distribution.
        # TODO: Optionally set bias weights to zero (GBH Paper)
        bias_weights_init = (self._random_state.rand(self.hidden_layer_size) * 2 - 1)
        # Feedback weights, fully connected feedback from the output to the hidden layer nodes
        # drawn from uniform distribution.
        output_weights_init = None  # np.zeros(shape=(self.hidden_layer_size + 1, self.n_outputs_))
        return input_weights_init, bias_weights_init, output_weights_init

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
        self : returns a trained ELM model.
        """
        n_samples, n_features = X.shape
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))  # ANNOTATION: makes the y a column vector if it already is? 2d
        self.n_outputs_ = y.shape[1]
        if (not hasattr(self, 'input_weights_')) or (not hasattr(self, 'bias_weights_')) or not incremental:
            # First time training the model
            self._initialize(y, n_features)

        # Run the offline optimization solver
        if self.solver in _OFFLINE_SOLVERS:
            self._fit_offline(X, y, incremental, update_output_weights=update_output_weights, n_jobs=n_jobs)
        self.is_fitted_ = True
        return self

    def _input_to_node(self, X):
        """
        Pass the data through the hidden layer.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data
        Returns
        -------
        hidden_layer_state : ndarray of shape (n_samples, hidden_layer_size)
            The collected hidden layer states
        """
        hidden_layer_state = self._forward_pass(elm_inputs=X)
        hidden_layer_state = np.concatenate((np.ones((hidden_layer_state.shape[0], 1)), hidden_layer_state), 1)
        return hidden_layer_state

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
        self._n_samples = self._n_samples + n_samples

        hidden_layer_state = self._input_to_node(X=X)

        if incremental:
            self._xTx = self._xTx + np.dot(hidden_layer_state.T, hidden_layer_state)
            self._xTy = self._xTy + np.dot(hidden_layer_state.T, y)
            new_activations_mean = np.mean(hidden_layer_state, axis=0)[1:]
            new_activations_var = np.var(hidden_layer_state, axis=0)[1:]
            m = self._n_samples
            n = hidden_layer_state.shape[0]
            tmp_activations_mean = self._activations_mean
            self._activations_mean = m/(m+n)*tmp_activations_mean + n/(m+n)*new_activations_mean
            self._activations_var = m / (m + n) * self._activations_var + n / (m + n)*new_activations_var + \
                                    m * n / (m + n)**2 * (tmp_activations_mean - new_activations_mean)**2
        else:
            self._xTx = np.dot(hidden_layer_state.T, hidden_layer_state)
            self._xTy = np.dot(hidden_layer_state.T, y)
            self.activations_mean = np.mean(hidden_layer_state, axis=0)[1:]
            self.activations_var = np.var(hidden_layer_state, axis=0)[1:]

        if update_output_weights:
            self._compute_output_weights(n_jobs=n_jobs)
        else:
            self.output_weights_ = None

        if not incremental:
            self._xTx = None
            self._xTy = None

    def _forward_pass(self, elm_inputs):
        """
        Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        elm_inputs : ndarray of shape (n_samples, n_features)
            The input data

        Returns
        -------

        """
        n_samples, n_features = elm_inputs.shape
        hidden_layer_state = np.zeros(shape=(n_samples+1, self.hidden_layer_size))
        for sample in range(n_samples):
            if scipy.sparse.issparse(self.input_weights_):
                a = self.input_weights_ * elm_inputs[sample, :] * self.input_scaling
            else:
                a = np.dot(self.input_weights_, elm_inputs[sample, :] * self.input_scaling)  # Changed , to *

            # no bounded_relu support
            # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/neural_network/_base.py
            hidden_layer_state[sample+1, :] = ACTIVATIONS[self.activation_function](a + self.bias_weights_*self.bias)
        """This should be the same: 
        hidden_layer_state = ACTIVATIONS[self.activation_function](self.input_weights_ * elm_inputs * self.input_scaling + self.bias_weights_*self.bias)
        return hidden_layer_state
        """
        return hidden_layer_state[1:, :]

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
        self : returns a trained ELM model.
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
        self : returns a trained ELM model.
        """
        return self._fit(X, y, incremental=True, update_output_weights=update_output_weights, n_jobs=n_jobs)

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

    def predict(self, X, keep_hidden_layer_state=False):
        """
        Predict using the trained ELM model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values
        """
        check_is_fitted(self, ['input_weights_', 'bias_weights_', 'output_weights_'])  # , 'recurrent_weights_'
        if not self.output_weights_.any():
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})
        X = check_array(X, accept_sparse=False)
        y_pred = self._predict(X=X, keep_hidden_layer_state=keep_hidden_layer_state)
        return y_pred

    def _predict(self, X, keep_hidden_layer_state=False):
        """
        Predict using the trained ELM model

        TODO: remove keep_hidden_layer_state
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values
        """
        hidden_layer_state = self._input_to_node(X=X)
        if keep_hidden_layer_state:
            self.hidden_layer_state = hidden_layer_state
        y_pred = safe_sparse_dot(hidden_layer_state, self.output_weights_)
        return y_pred


class ELMClassifier(BaseExtremeLearningMachine, ClassifierMixin):
    """
    Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using linear regression.

    .. versionadded:: 0.00

    Parameters
    ----------
    k_in : int, default 2
        This element represents the sparsity of the connections between the input and recurrent nodes.
        It determines the number of features that every node inside the hidden layer receives.
    input_scaling : float, default 1.0
        This element represents the input scaling factor from the input to the hidden layer. It is a global scaling factor
        for the input weight matrix.
    bias : float, default 0.0
        This element represents the bias scaling of the bias weights. It is a global scaling factor for the bias weight
        matrix.
    hidden_layer_size : int, default 500
        This element represents the number of neurons in the hidden layer.
    activation_function : {'tanh', 'identity', 'logistic', 'relu'}
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
            - 'bounded_relu', the rectified linear unit function, returns f(x) = min(max(0, x),1)
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
    def __init__(self, k_in: int = -1, input_scaling: float = 1., bias: float = 0., hidden_layer_size: int = 500,
                 activation_function: str = 'tanh', solver: str = 'ridge', beta: float = 1e-6,
                 random_state: int = None):
        super().__init__(k_in=k_in, input_scaling=input_scaling, bias=bias, hidden_layer_size=hidden_layer_size,
                         activation_function=activation_function, solver=solver, beta=beta, random_state=random_state)

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
        self : returns a trained ELM model.
        """
        self._validate_hyperparameters()
        X, y = self._validate_input(X, y)
        self._initialize(y=y, n_features=X.shape[1])
        return self._fit(X, y, incremental=False, update_output_weights=True, n_jobs=n_jobs)

    def predict(self, X, keep_hidden_layer_state=False):
        """
        Predict the classes using the trained ELM classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes
        """
        check_is_fitted(self, ['input_weights_', 'bias_weights_', 'output_weights_'])  # , 'recurrent_weights_'
        if self.output_weights_.size == 0:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})
        y_pred = super().predict(X, keep_hidden_layer_state=keep_hidden_layer_state)

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
        self : returns a trained ELM classifier.
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
        self : returns a trained ELM classifier.
        """
        if _check_partial_fit_first_call(self, classes):
            super()._initialize(y=y, n_features=X.shape[1])

        super()._partial_fit(X, y, update_output_weights=update_output_weights, n_jobs=n_jobs)
        return self

    def predict_proba(self, X, keep_hidden_layer_state=False):
        """
        Predict the probability estimates using the trained ELM classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted probability estimates
        """
        y_pred = super().predict(X, keep_hidden_layer_state=keep_hidden_layer_state)
        y_pred = np.maximum(y_pred, 1e-3)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def predict_log_proba(self, X, keep_hidden_layer_state=False):
        """
        Predict the logarithmic probability estimates using the trained ELM classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted logarithmic probability estimates
        """
        y_pred = self.predict_proba(X=X, keep_hidden_layer_state=keep_hidden_layer_state)
        return np.log(y_pred)


class ELMRegressor(BaseExtremeLearningMachine, RegressorMixin):
    """
    Extreme Learning Machine regressor.

    This model optimizes the mean squared error loss function using linear regression.

    .. versionadded:: 0.00

    Parameters
    ----------
    k_in : int, default 2
        This element represents the sparsity of the connections between the input and recurrent nodes.
        It determines the number of features that every node inside the hidden layer receives.
    input_scaling : float, default 1.0
        This element represents the input scaling factor from the input to the hidden layer. It is a global scaling factor
        for the input weight matrix.
    bias : float, default 0.0
        This element represents the bias scaling of the bias weights. It is a global scaling factor for the bias weight
        matrix.
    hidden_layer_size : int, default 500
        This element represents the number of neurons in the hidden layer.
    activation_function : {'tanh', 'identity', 'logistic', 'relu'}
        This element represents the activation function in the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)
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
    def __init__(self, k_in: int = -1, input_scaling: float = 1., bias: float = 0., hidden_layer_size: int = 500,
                 activation_function: str = 'tanh', solver: str = 'ridge', beta: float = 1e-6,
                 random_state: int = None):
        super().__init__(k_in=k_in, input_scaling=input_scaling, bias=bias, hidden_layer_size=hidden_layer_size,
                         activation_function=activation_function, solver=solver, beta=beta, random_state=random_state)

    def fit(self, X, y, n_jobs=0):
        self._validate_hyperparameters()
        X, y = self._validate_input(X, y)
        self._initialize(y=y, n_features=X.shape[1])
        return self._fit(X, y, update_output_weights=True, n_jobs=n_jobs)

    def predict(self, X, keep_hidden_layer_state=False):
        """
        Predict the output value using the trained ELM regressor

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        keep_hidden_layer_state : bool, default False
            If True, the hidden layer state is kept and can be accessed from outside. This is useful for visualization
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes
        """
        y_pred = super().predict(X, keep_hidden_layer_state=keep_hidden_layer_state)

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
        self : returns a trained ELM classifier.
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
        self : returns a trained ELM classifier.
        """
        super()._partial_fit(X, y, update_output_weights=update_output_weights, n_jobs=n_jobs)
        return self
