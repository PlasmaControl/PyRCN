"""
The :mod:`echo_state_network` contains the ESNRegressor and the ESNClassifier
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion


class ESNRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """Echo State Network regressor.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_nodes : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    nodes_to_nodes : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    random_state : int, RandomState instance, default=None
    """
    def __init__(self, input_to_nodes, nodes_to_nodes, regressor=IncrementalRegression(alpha=.0001), random_state=None):
        self.input_to_nodes = input_to_nodes
        self.nodes_to_nodes = nodes_to_nodes
        self.regressor = regressor
        self.random_state = check_random_state(random_state)
        self._input_to_node = None
        self._node_to_node = None
        self._regressor = None
        self.n_samples_ = 0

    def partial_fit(self, X, y, n_jobs=None, transformer_weights=None):
        """Fits the regressor partially.

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
        if not hasattr(self.regressor, 'partial_fit'):
            raise BaseException('Regressor has no attribute partial_fit, got {0}'.format(self.regressor))

        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        if self._input_to_node is None:
            self._input_to_node = FeatureUnion(
                transformer_list=self.input_to_nodes,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights).fit(X)

        if self._node_to_node is None:
            self._node_to_node = FeatureUnion(
                transformer_list=self.nodes_to_nodes,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights).fit(self._input_to_node.transform(X))

        forwarded_state = self._input_to_node.transform(X)
        hidden_layer_state = self._node_to_node.transform(forwarded_state)

        if self._regressor:
            self._regressor.partial_fit(hidden_layer_state, y)
        else:
            self._regressor = self.regressor.partial_fit(hidden_layer_state, y)
        return self

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        """Fits the regressor.

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
        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        self._input_to_node = FeatureUnion(
            transformer_list=self.input_to_nodes,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights).fit(X)

        self._node_to_node = FeatureUnion(
            transformer_list=self.nodes_to_nodes,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights).fit(self._input_to_node.transform(X))

        forwarded_state = self._input_to_node.transform(X)
        hidden_layer_state = self._node_to_node.transform(forwarded_state)

        self._regressor = self.regressor.fit(hidden_layer_state, y)
        return self

    def predict(self, X):
        """Predicts the targets using the trained ESN regressor.

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

        forwarded_state = self._input_to_node.transform(X)
        hidden_layer_state = self._node_to_node.transform(forwarded_state)

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self):
        """Validates the hyperparameters.

        Returns
        -------

        """
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

        if not self.nodes_to_nodes or self.nodes_to_nodes is None:
            self.nodes_to_nodes = [('default', NodeToNode())]
        else:
            for n, t in self.nodes_to_nodes:
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


class ESNClassifier(ESNRegressor, ClassifierMixin):
    """Echo State Network classifier.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_nodes : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    nodes_to_nodes : iterable
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    random_state : int, RandomState instance, default=None
    """
    def __init__(self, input_to_nodes, nodes_to_nodes, regressor=IncrementalRegression(alpha=.0001), random_state=None):
        super().__init__(input_to_nodes=input_to_nodes, nodes_to_nodes=nodes_to_nodes, regressor=regressor,
                         random_state=random_state)
        self._encoder = None

    def partial_fit(self, X, y, n_jobs=None, transformer_weights=None):
        """Fits the regressor partially.

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
        self : returns a traines ESNClassifier model
        """
        self._validate_data(X, y, multi_output=True)

        if self._encoder is None:
            self._encoder = LabelBinarizer().fit(y)

        return super().partial_fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None)

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
        return self._encoder.inverse_transform(np.maximum(super().predict(X), 1e-5), threshold=.5)

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
