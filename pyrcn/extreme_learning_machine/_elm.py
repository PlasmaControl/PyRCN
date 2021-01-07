import scipy
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor
from pyrcn.extreme_learning_machine._base import InputToNode
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline


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
        This element represents the input scaling factor from the input to the hidden layer. It is a global scaling
        factor for the input weight matrix.
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
        if not self.input_to_nodes or self.input_to_nodes is None:
            self.input_to_nodes = [('default', InputToNode())]
        else:
            for n, t in self.input_to_nodes:
                if t == 'drop':
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
        This element represents the input scaling factor from the input to the hidden layer. It is a global scaling
        factor for the input weight matrix.
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
            If n_jobs is larger than 1, then the linear regression for each output dimension is computed separately.
        transformer_weights: float or ndarray of shape (n_samples,) weights the targets y individually.

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
