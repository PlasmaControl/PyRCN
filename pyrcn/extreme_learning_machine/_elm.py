"""
The :mod:`extreme_learninc_machine` contains the ELMRegressor and the ELMClassifier
"""

# Author: Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor
from pyrcn.base import InputToNode
from pyrcn.linear_model import IncrementalRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion


class ELMRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """Extreme Learning Machine regressor.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_nodes : iterable, default=[('default', InputToNode())]
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    random_state : int, RandomState instance, default=None
    """
    def __init__(self,
                 input_to_nodes=[('default', InputToNode())],
                 regressor=IncrementalRegression(alpha=.0001),
                 random_state=None):
        self.input_to_nodes = input_to_nodes
        self.regressor = regressor
        self.random_state = random_state
        self._input_to_node = None
        self._regressor = None

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
        self : Returns a traines ELMRegressor model.
        """
        if not hasattr(self.regressor, 'partial_fit'):
            raise BaseException('regressor has no attribute partial_fit, got {0}'.format(self.regressor))

        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        if self._input_to_node is None:
            self._input_to_node = FeatureUnion(
                transformer_list=self.input_to_nodes,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights).fit(X)

        hidden_layer_state = self._input_to_node.transform(X)

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
        self : Returns a traines ELMRegressor model.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        self._input_to_node = FeatureUnion(
            transformer_list=self.input_to_nodes,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights)
        hidden_layer_state = self._input_to_node.fit_transform(X)

        self._regressor = self.regressor.fit(hidden_layer_state, y)
        return self

    def predict(self, X):
        """Predicts the targets using the trained ELM regressor.

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

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self):
        """Validates the hyperparameters.

        Returns
        -------

        """
        self.random_state = check_random_state(self.random_state)

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
    """Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_nodes : iterable, default=[('default', InputToNode())]
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    random_state : int, RandomState instance, default=None
    """
    def __init__(self,
                 input_to_nodes=[('default', InputToNode())],
                 regressor=IncrementalRegression(alpha=.0001),
                 random_state=None):
        super().__init__(input_to_nodes=input_to_nodes, regressor=regressor, random_state=random_state)
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
        self : returns a traines ELMClassifier model
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
        self : Returns a traines ELMClassifier model.
        """
        self._validate_data(X, y, multi_output=True)
        self._encoder = LabelBinarizer().fit(y)

        return super().fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None)

    def predict(self, X):
        """Predict the classes using the trained ELM classifier.

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
        """Predict the probability estimated using the trained ELM classifier.

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
        return self._encoder.inverse_transform(super().predict(X), threshold=.5)

    def predict_log_proba(self, X):
        """Predict the logarithmic probability estimated using the trained ELM classifier.

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
