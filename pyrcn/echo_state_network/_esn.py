"""
The :mod:`echo_state_network` contains the ESNRegressor and the ESNClassifier
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

import sys

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion


class ESNRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Echo State Network regressor.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : iterable, default=[('default', InputToNode())]
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    node_to_node : iterable, default=[('default', NodeToNode())]
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    chunk_size : int, default=None
        if X.shape[0] > chunk_size, calculate results incrementally with partial_fit
    random_state : int, RandomState instance, default=None
    """
    def __init__(self,
                 input_to_node=InputToNode(),
                 node_to_node=NodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 random_state=None):
        self.input_to_node = input_to_node
        self.node_to_node = node_to_node
        self._regressor = regressor
        self._chunk_size = chunk_size
        self.random_state = random_state

    def partial_fit(self, X, y, n_jobs=None, transformer_weights=None, postpone_inverse=False):
        """
        Fits the regressor partially.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : ignored

        Returns
        -------
        self : Returns a traines ESNRegressor model.
        """
        if not hasattr(self._regressor, 'partial_fit'):
            raise BaseException('Regressor has no attribute partial_fit, got {0}'.format(self._regressor))

        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        # input_to_node
        try:
            hidden_layer_state = self._input_to_node.transform(X)
        except NotFittedError as e:
            print('input_to_node has not been fitted yet: {0}'.format(e))
            hidden_layer_state = self._input_to_node.fit_transform(X)
            pass

        # node_to_node
        try:
            hidden_layer_state = self._node_to_node.transform(hidden_layer_state)
        except NotFittedError as e:
            print('node_to_node has not been fitted yet: {0}'.format(e))
            hidden_layer_state = self._node_to_node.fit_transform(hidden_layer_state)
            pass

        # regression
        if self._regressor:
            self._regressor.partial_fit(hidden_layer_state, y, postpone_inverse=postpone_inverse)
        return self

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        """
        Fits the regressor.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : ignored

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)
        self._input_to_node.fit(X)
        self._node_to_node.fit(self._input_to_node.transform(X))
        self._regressor = self._regressor.__class__()

        if self._chunk_size is None or self._chunk_size > X.shape[0]:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)
            hidden_layer_state = self._node_to_node.transform(hidden_layer_state)

            # regression
            self._regressor.fit(hidden_layer_state, y)

        elif self._chunk_size < X.shape[0]:
            # setup chunk list
            chunks = list(range(0, X.shape[0], self._chunk_size))
            # postpone inverse calculation for chunks n-1
            for idx in chunks[:-1]:
                ESNRegressor.partial_fit(
                    self,
                    X=X[idx:idx + self._chunk_size, ...],
                    y=y[idx:idx + self._chunk_size, ...],
                    n_jobs=n_jobs,
                    transformer_weights=transformer_weights,
                    postpone_inverse=True
                )
            # last chunk, calculate inverse and bias
            ESNRegressor.partial_fit(
                self,
                X=X[chunks[-1]:, ...],
                y=y[chunks[-1]:, ...],
                n_jobs=n_jobs,
                transformer_weights=transformer_weights,
                postpone_inverse=False
            )
        else:
            raise ValueError('chunk_size invalid {0}'.format(self._chunk_size))
        return self

    def predict(self, X):
        """
        Predicts the targets using the trained ELM regressor.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
            The predicted targets
        """
        if self._input_to_node is None or self._regressor is None:
            raise NotFittedError(self)

        hidden_layer_state = self._input_to_node.transform(X)
        hidden_layer_state = self._node_to_node.transform(hidden_layer_state)

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self):
        """Validates the hyperparameters.
        Returns
        -------
        """
        self.random_state = check_random_state(self.random_state)

        if not (hasattr(self.input_to_node, "fit") and hasattr(self.input_to_node, "fit_transform") and hasattr(
                self.input_to_node, "transform")):
            raise TypeError("All input_to_node should be transformers "
                            "and implement fit and transform "
                            "'%s' (type %s) doesn't" % (self.input_to_node, type(self.input_to_node)))

        if not (hasattr(self.node_to_node, "fit") and hasattr(self.node_to_node, "fit_transform") and hasattr(
                self.node_to_node, "transform")):
            raise TypeError("All node_to_node should be transformers "
                            "and implement fit and transform "
                            "'%s' (type %s) doesn't" % (self.node_to_node, type(self.node_to_node)))

        if self._chunk_size is not None and (not isinstance(self._chunk_size, int) or self._chunk_size < 0):
            raise ValueError('Invalid value for chunk_size, got {0}'.format(self._chunk_size))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor "
                            "and implement fit and predict"
                            "'%s' (type %s) doesn't" % (self._regressor, type(self._regressor)))

    def __sizeof__(self):
        """Returns the size of the object in bytes.
        Returns
        -------
        size : int
        Object memory in bytes.
        """
        return object.__sizeof__(self) + \
            sys.getsizeof(self._input_to_node) + \
            sys.getsizeof(self._node_to_node) + \
            sys.getsizeof(self._regressor)

    @property
    def regressor(self):
        """Returns the chunk_size, in which X will be chopped.
        Returns
        -------
        chunk_size : int or None
        """
        return self._regressor

    @regressor.setter
    def regressor(self, regressor):
        """Sets the regressor.
        Parameters
        ----------
        regressor : regressor or None
        Returns
        -------
        """
        self._regressor = regressor

    @property
    def input_to_node(self):
        """Returns the input_to_node list or the input_to_node Transformer.
        Returns
        -------
        input_to_node : Transformer or [Transformer]
        """
        return self._input_to_node

    @input_to_node.setter
    def input_to_node(self, input_to_node, n_jobs=None, transformer_weights=None):
        """Sets the input_to_node list or the input_to_node Transformer.
        Parameters
        ----------
        input_to_node : Transformer or [Transformer]
        n_jobs : int, default=None
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in transformer_list.
        Returns
        -------
        """
        if hasattr(input_to_node, '__iter__'):
            # Feature Union of list of input_to_node
            self._input_to_node = FeatureUnion(
                transformer_list=input_to_node,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights)
        else:
            # single input_to_node
            self._input_to_node = input_to_node

    @property
    def node_to_node(self):
        """Returns the node_to_node list or the input_to_node Transformer.
        Returns
        -------
        input_to_node : Transformer or [Transformer]
        """
        return self._node_to_node

    @node_to_node.setter
    def node_to_node(self, node_to_node, n_jobs=None, transformer_weights=None):
        """Sets the input_to_node list or the input_to_node Transformer.
        Parameters
        ----------
        node_to_node : Transformer or [Transformer]
        n_jobs : int, default=None
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in transformer_list.
        Returns
        -------
        """
        if hasattr(node_to_node, '__iter__'):
            # Feature Union of list of input_to_node
            self._node_to_node = FeatureUnion(
                transformer_list=node_to_node,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights)
        else:
            # single input_to_node
            self._node_to_node = node_to_node

    @property
    def chunk_size(self):
        """Returns the chunk_size, in which X will be chopped.
        Returns
        -------
        chunk_size : int or None
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size):
        """Sets the chunk_size, in which X will be chopped.
        Parameters
        ----------
        chunk_size : int or None
        Returns
        -------
        """
        self._chunk_size = chunk_size


class ESNClassifier(ESNRegressor, ClassifierMixin):
    """Echo State Network classifier.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    node_to_node : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    random_state : int, RandomState instance, default=None
    """
    def __init__(self, input_to_node, node_to_node, regressor=IncrementalRegression(alpha=.0001), random_state=None):
        super().__init__(input_to_node=input_to_node, node_to_node=node_to_node, regressor=regressor,
                         random_state=random_state)
        self._encoder = None

    def partial_fit(self, X, y, classes=None, n_jobs=None, transformer_weights=None, postpone_inverse=False):
        """Fits the classifier partially.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            The targets to predict.
        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : ignored
        postpone_inverse : bool, default False

        Returns
        -------
        self : returns a traines ESNClassifier model
        """
        self._validate_data(X, y, multi_output=True)

        if self._encoder is None:
            self._encoder = LabelBinarizer().fit(classes)

        return super().partial_fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None,
                                   postpone_inverse=postpone_inverse)

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        """Fits the regressor.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : ignored

        Returns
        -------
        self : Returns a traines ESNClassifier model.
        """
        self._validate_data(X, y, multi_output=True)
        self._encoder = LabelBinarizer().fit(y)

        return super().fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None)

    def predict(self, X):
        """Predict the classes using the trained ESN classifier.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        return self._encoder.inverse_transform(super().predict(X), threshold=.0)

    def predict_proba(self, X):
        """Predict the probability estimated using the trained ESN classifier.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted probability estimated.
        """
        # for single dim proba use np.amax
        return np.maximum(super().predict(X), 1e-5)

    def predict_log_proba(self, X):
        """Predict the logarithmic probability estimated using the trained ESN classifier.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted logarithmic probability estimated.
        """
        return np.log(self.predict_proba(X=X))
