import scipy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, MultiOutputMixin, is_regressor
from pyrcn.extreme_learning_machine._base import ACTIVATIONS
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline


from joblib import Parallel, delayed

if scipy.__version__ == '0.9.0' or scipy.__version__ == '0.10.1':
    from scipy.sparse.linalg import eigs as eigens
    from scipy.sparse.linalg import ArpackNoConvergence
else:
    from scipy.sparse.linalg.eigen.arpack import eigs as eigens
    from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

_OFFLINE_SOLVERS = ['pinv', 'ridge', 'lasso']


class InputToNode(BaseEstimator, TransformerMixin):
    """InputToNode class for ELM

    .. versionadded:: 0.00
    """
    def __init__(self,
                 hidden_layer_size=500,
                 sparsity=1.,
                 activation='tanh',
                 input_scaling=1.,
                 bias_scaling=1.,
                 random_state=None):
        self.hidden_layer_size = hidden_layer_size  # read only
        self.sparsity = sparsity  # read only
        self.activation = activation  # read/write
        self.input_scaling = input_scaling  # read/write
        self.bias_scaling = bias_scaling  # read/write
        self.random_state = check_random_state(random_state)  # read only

        self._input_weights = None
        self._bias = None
        self._hidden_layer_state = None

    def fit(self, X, y=None, n_jobs=None):
        """
        Fit the input_weights_matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        y : Ignored
        n_jobs: Ignored

        Returns
        -------
        self : returns a trained ELM model.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y)
        self._check_n_features(X, reset=True)
        self._set_uniform_random_input_weights(
            n_features_in=self.n_features_in_,
            hidden_layer_size=self.hidden_layer_size,
            fan_in=np.rint(self.hidden_layer_size*self.sparsity).astype(int),
            random_state=self.random_state)
        self._set_uniform_random_bias(
            hidden_layer_size=self.hidden_layer_size,
            random_state=self.random_state)
        return self

    def _set_uniform_random_input_weights(self, n_features_in: int, hidden_layer_size: int, fan_in: int, random_state):
        nr_entries = np.int32(n_features_in * fan_in)
        weights_array = random_state.uniform(low=-1., high=1., size=nr_entries)

        if fan_in < hidden_layer_size:
            indices = np.zeros(shape=nr_entries, dtype=int)
            indptr = np.arange(start=0, stop=(n_features_in + 1)*fan_in, step=fan_in)

            for en in range(0, n_features_in*fan_in, fan_in):
                indices[en: en + fan_in] = random_state.permutation(hidden_layer_size)[:fan_in].astype(int)
            self._input_weights = scipy.sparse.csr_matrix(
                (weights_array, indices, indptr), shape=(n_features_in, hidden_layer_size), dtype='float64')
        else:
            self._input_weights = weights_array.reshape((n_features_in, hidden_layer_size))

    def _set_uniform_random_bias(self, hidden_layer_size: int, random_state):
        self._bias = random_state.uniform(low=-1., high=1., size=hidden_layer_size)

    def transform(self, X):
        if self._input_weights is None or self._bias is None:
            raise NotFittedError(self)

        self._hidden_layer_state = safe_sparse_dot(X, self._input_weights) * self.input_scaling\
                                   + np.ones(shape=(X.shape[0], 1)) * self._bias * self.bias_scaling
        ACTIVATIONS[self.activation](self._hidden_layer_state)
        return self._hidden_layer_state

    def _validate_hyperparameters(self):
        """
        Validate the hyperparameter. Ensure that the parameter ranges and dimensions are valid.
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


class ELMRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
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
    random_state : int, RandomState instance, default=None
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
    def __init__(self, input_to_nodes, regressor=Ridge(alpha=.0001), random_state=None):
        self.input_to_nodes = input_to_nodes
        self.regressor = regressor
        self.random_state = check_random_state(random_state)
        self._input_to_node = None
        self._hidden_layer_state = None
        self._regressor = None

    def fit(self, X, y, n_jobs=1, transformer_weights=None):
        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        """
        # shorthand
        self._elm = Pipeline(steps=[
            ('input_to_node', FeatureUnion(
                transformer_list=self.input_to_nodes,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights)),
            ('regressor', self.regressor)])
        """
        self._input_to_node = FeatureUnion(
            transformer_list=self.input_to_nodes,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights)
        self._hidden_layer_state = self._input_to_node.fit_transform(X)

        self._regressor = self.regressor.fit(self._hidden_layer_state, y)
        return self

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
        # shorthand
        # check_is_fitted(self, ['_elm'])
        # return self._elm.predict(X)

        if self._input_to_node is None or self._regressor is None:
            raise NotFittedError(self)

        hidden_layer_state = self._input_to_node.transform(X)
        if not keep_hidden_layer_state:
            self._hidden_layer_state = hidden_layer_state

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self):
        if not self.input_to_nodes:
            self.input_to_nodes = [('default', InputToNode())]
        else:
            for n, t in self.input_to_nodes:
                if t is None:
                    continue
                if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform"):
                    raise TypeError("All input_to_nodes should be transformers "
                                    "and implement fit and transform "
                                    "'%s' (type %s) doesn't" % (t, type(t)))
        if not is_regressor(self.regressor):
            raise TypeError("The last step should be a regressor "
                            "and implement fit and predict"
                            "'%s' (type %s) doesn't" % (self.regressor, type(self.regressor)))


class ELMClassifier(ELMRegressor, ClassifierMixin):
    """
    Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using linear regression.

    .. versionadded:: 0.00

    Parameters
    ----------
    k_in : int, default -1
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
    beta : float, default 0.0001
        L2 penalty (regularization term) parameter.
    random_state : int, RandomState instance, default=None
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

    def __init__(self, input_to_nodes, regressor=Ridge(alpha=.0001), random_state=None):
        super().__init__(input_to_nodes=input_to_nodes, regressor=regressor, random_state=random_state)
        self._encoder = None

    def fit(self, X, y, n_jobs=1, transformer_weights=None):
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
        self._validate_data(X, y, multi_output=True)

        if y.shape[1] == 1 and np.unique(y) > 2:
            self._encoder = OneHotEncoder()
            return super().fit(self._encoder.fit_transform(X), y, n_jobs=1, transformer_weights=None)
        else:
            return super().fit(X, y, n_jobs=1, transformer_weights=None)

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
        if self._encoder is not None:
            return np.argmax(super().predict(self._encoder.transform(X), keep_hidden_layer_state=False), axis=1)
        else:
            return np.argmax(super().predict(X, keep_hidden_layer_state=False), axis=1)

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
        # for single dim proba use np.amax
        if self._encoder is not None:
            return super().predict(self._encoder.transform(X), keep_hidden_layer_state=keep_hidden_layer_state)
        else:
            return super().predict(X, keep_hidden_layer_state=keep_hidden_layer_state)

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
        return np.log(self.predict_proba(X=X, keep_hidden_layer_state=keep_hidden_layer_state))

    def _validate_hyperparameters(self):
        return super()._validate_hyperparameters()
