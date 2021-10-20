"""
The :mod:`echo_state_network` contains the ESNRegressor and the ESNClassifier
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

import sys
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor, clone
from sklearn.exceptions import DataDimensionalityWarning
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.utils import stack_sequence
from pyrcn.linear_model import IncrementalRegression
from pyrcn.projection import MatrixToIndexProjection
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion

from joblib import Parallel, delayed


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
    requires_sequence : "auto" or bool
        If True, the input data is expected to be a sequence. 
        If "auto", tries to automatically estimate when calling ```fit``` for the first time
    kwargs : dict, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node=None,
                 node_to_node=None,
                 regressor=None,
                 requires_sequence="auto",
                 verbose=False,
                 **kwargs):
        if input_to_node is None:
            i2n_params = InputToNode()._get_param_names()
            self.input_to_node = InputToNode(**{ key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        else:
            i2n_params = input_to_node._get_param_names()
            self.input_to_node = input_to_node.set_params(**{ key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        if node_to_node is None:
            n2n_params = NodeToNode()._get_param_names()
            self.node_to_node = NodeToNode(**{ key: kwargs[key] for key in kwargs.keys() if key in n2n_params})
        else:
            n2n_params = node_to_node._get_param_names()
            self.node_to_node = node_to_node.set_params(**{ key: kwargs[key] for key in kwargs.keys() if key in n2n_params})
        if regressor is None:
            reg_params = IncrementalRegression()._get_param_names()
            self.regressor = IncrementalRegression(**{ key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        else:
            reg_params = regressor._get_param_names()
            self.regressor = regressor.set_params(**{ key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        self._requires_sequence = requires_sequence
        self.verbose=verbose

    def __add__(self, other):
        self.regressor._K = self.regressor._K + other.regressor._K
        self.regressor._xTy = self.regressor._xTy  + other.regressor._xTy
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_params(self, deep=True):
        if deep:
            return {**self.input_to_node.get_params(), **self.node_to_node.get_params(), **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node, "node_to_node": self.node_to_node, "regressor": self.regressor, "requires_sequence": self._requires_sequence}

    def set_params(self, **parameters):
        i2n_params = self.input_to_node._get_param_names()
        self.input_to_node = self.input_to_node.set_params(**{ key: parameters[key] for key in parameters.keys() if key in i2n_params})
        n2n_params = self.node_to_node._get_param_names()
        self.node_to_node = self.node_to_node.set_params(**{ key: parameters[key] for key in parameters.keys() if key in n2n_params})
        reg_params = self.regressor._get_param_names()
        self.regressor = self.regressor.set_params(**{ key: parameters[key] for key in parameters.keys() if key in reg_params})
        for parameter, value in parameters.items():
            if parameter in self.get_params(deep=False):
                setattr(self, parameter, value)

        return self

    def _check_if_sequence(self, X, y):
        if isinstance(X, list):
            lengths_X = np.unique([x.shape[0] for x in X])
            if len(lengths_X) == 1 and self.verbose:
                warnings.warn("Treat input as instance, not as sequences. If not desired,"
                              "explicitly pass requires_sequence=True.", DataDimensionalityWarning)
            X = np.asarray(X)
        if isinstance(y, list):
            lengths_y = np.unique([yt.shape[0] for yt in y])
            if len(lengths_y) == 1 and self.verbose:
                warnings.warn("Treat target as instance, not as sequences. If not desired,"
                              "explicitly pass requires_sequence=True.", DataDimensionalityWarning)
            y = np.asarray(y)
        if X.ndim > 2:
            raise ValueError("Could not determine a valid structure, because X has {0} dimensions."
                             "Only 1 or 2 dimensions are allowed."%X.ndim)
        if y.ndim > 2:
            raise ValueError("Could not determine a valid structure, because y has {0} dimensions."
                             "Only 1 or 2 dimensions are allowed."%X.ndim)
        self.requires_sequence = X.ndim == 1

    def partial_fit(self, X, y, transformer_weights=None, postpone_inverse=False):
        """
        Fits the regressor partially.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        y : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        transformer_weights : ignored
        postpone_inverse : bool, default=False
            If output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to IncrementalRegression for details.

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        if not hasattr(self._regressor, 'partial_fit'):
            raise BaseException('Regressor has no attribute partial_fit, got {0}'.format(self._regressor))

        self._validate_hyperparameters()
        self._validate_data(X=X, y=y, multi_output=True)

        # input_to_node
        try:
            hidden_layer_state = self._input_to_node.transform(X)
        except NotFittedError as e:
            if self.verbose:
                print('input_to_node has not been fitted yet: {0}'.format(e))
            hidden_layer_state = self._input_to_node.fit_transform(X)
            pass

        # node_to_node
        try:
            hidden_layer_state = self._node_to_node.transform(hidden_layer_state)
        except NotFittedError as e:
            if self.verbose:
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
        if self.requires_sequence is "auto":
            self._check_if_sequence(X, y)
        if self.requires_sequence:
            X, y, sequence_ranges = stack_sequence(X, y)
        else:
            self._validate_data(X, y, multi_output=True)
        self._input_to_node.fit(X)
        self._node_to_node.fit(self._input_to_node.transform(X))
        self._regressor = self._regressor.__class__()
        if not self.requires_sequence:
            return self._instance_fit(X, y)
        else:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)

    def _instance_fit(self, X, y):
        # input_to_node
        hidden_layer_state = self._input_to_node.transform(X)
        hidden_layer_state = self._node_to_node.transform(hidden_layer_state)
        # regression
        self._regressor.fit(hidden_layer_state, y)
        return self

    def _sequence_fit(self, X, y, sequence_ranges, n_jobs):
        reg = Parallel(n_jobs=n_jobs)(delayed(ESNRegressor.partial_fit)
                                      (clone(self), X[idx[0]:idx[1], ...], 
                                       y[idx[0]:idx[1], ...],
                                       postpone_inverse=True)
                                      for idx in sequence_ranges[:-1])
        self._regressor = sum(reg)._regressor
        # last sequence, calculate inverse and bias
        ESNRegressor.partial_fit(self,
                                 X=X[sequence_ranges[-1][0]:, ...], 
                                 y=y[sequence_ranges[-1][0]:, ...],
                                 postpone_inverse=False)
        return self

    def predict(self, X):
        """
        Predicts the targets using the trained ESN regressor.

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

        if self.requires_sequence is False:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)
            hidden_layer_state = self._node_to_node.transform(hidden_layer_state)
            # regression
            return self._regressor.predict(hidden_layer_state)
        else:
            y = np.empty(shape=X.shape, dtype=object)
            for k, seq in enumerate(X):
                # input_to_node
                hidden_layer_state = self._input_to_node.transform(seq)
                hidden_layer_state = self._node_to_node.transform(hidden_layer_state)
                # regression
                y[k] = self._regressor.predict(hidden_layer_state)
            return y

    def _validate_hyperparameters(self):
        """Validates the hyperparameters.

        Returns
        -------
        """
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

        if self._requires_sequence != "auto" and (not isinstance(self._requires_sequence, bool)):
            raise ValueError('Invalid value for requires_sequence, got {0}'.format(self._requires_sequence))

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
        """Returns the regressor.
        Returns
        -------
        regressor : Regressor
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
        node_to_node : Transformer or [Transformer]
        """
        return self._node_to_node

    @property
    def hidden_layer_state(self):
        """Returns the hidden_layer_state, e.g. the resevoir state over time.
        Returns
        -------
        hidden_layer_state : np.ndarray
        """
        return self._node_to_node._hidden_layer_state

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
    def requires_sequence(self):
        """Returns the requires_sequence parameter.
        Returns
        -------
        requires_sequence : "auto" or bool
        """
        return self._requires_sequence

    @requires_sequence.setter
    def requires_sequence(self, requires_sequence):
        """Sets the requires_sequence parameter.
        Parameters
        ----------
        requires_sequence : "auto" or bool
        Returns
        -------
        """
        self._requires_sequence = requires_sequence


class ESNClassifier(ESNRegressor, ClassifierMixin):
    """
    Echo State Network classifier.

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
    requires_sequence : "auto" or bool
        If True, the input data is expected to be a sequence. 
        If "auto", tries to automatically estimate when calling ```fit``` for the first time
    decision_strategy : str, one of {'winner_takes_all', 'median', 'weighted', 'last_value', 'mode'}, default='winner_takes_all'
        Decision strategy for sequence-to-label task. Ignored if the target output is a sequence
    kwargs : dict, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node=None,
                 node_to_node=None,
                 regressor=None,
                 requires_sequence="auto",
                 verbose=False,
                 decision_strategy="winner_takes_all",
                 **kwargs):
        super().__init__(input_to_node=input_to_node, node_to_node=node_to_node, regressor=regressor,
                         requires_sequence=requires_sequence, verbose=verbose, **kwargs)
        self._decision_strategy = decision_strategy
        self._encoder = None
        self._sequence_to_label = False

    def partial_fit(self, X, y, classes=None, transformer_weights=None, postpone_inverse=False):
        """
        Fits the classifier partially on a sequence of observations.

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
        transformer_weights : ignored
        postpone_inverse : bool, default=False
            If output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to IncrementalRegression for details.

        Returns
        -------
        self : returns a trained ESNClassifier model
        """
        self._validate_data(X, y, multi_output=True)

        if self._encoder is None:
            self._encoder = LabelBinarizer().fit(classes)

        return super().partial_fit(X, self._encoder.transform(y), transformer_weights=None,
                                   postpone_inverse=postpone_inverse)

    def _check_if_sequence_to_label(self, X, y):
        len_X = np.unique([x.shape[0] for x in X])
        len_y = np.unique([yt.shape[0] for yt in y])
        self._sequence_to_label = not len_X==len_y

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        """
        Fits the classifier.

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
        self : Returns a trained ESNClassifier model.
        """

        self._validate_hyperparameters()
        if self.requires_sequence is "auto":
            self._check_if_sequence(X, y)
        if self.requires_sequence:
            self._check_if_sequence_to_label(X, y)  # change this to "sequence_to_value
            X, y, sequence_ranges = stack_sequence(X, y, sequence_to_label=self._sequence_to_label)  # concatenate_sequences
            self._encoder = LabelBinarizer().fit(y)
            y = self._encoder.transform(y)
        else:
            self._validate_data(X, y, multi_output=True)
        self._input_to_node.fit(X)
        self._node_to_node.fit(self._input_to_node.transform(X))
        self._regressor = self._regressor.__class__()
        if not self.requires_sequence:
            return self._instance_fit(X, y)
        else:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)

    def predict(self, X):
        """
        Predict the classes using the trained ESN classifier.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        y = super().predict(X)
        if self.requires_sequence and self._sequence_to_label:
            for k, _ in enumerate(y):
                y[k] = MatrixToIndexProjection(output_strategy=self._decision_strategy).fit_transform(y[k])
            return y
        elif self.requires_sequence:
            for k, _ in enumerate(y):
                y[k] = self._encoder.inverse_transform(y[k], threshold=None)
            return y
        else:
            return self._encoder.inverse_transform(super().predict(X), threshold=None)

    def predict_proba(self, X):
        """
        Predict the probability estimated using the trained ESN classifier.

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
        # predicted_positive = np.subtract(predicted.T, np.min(predicted, axis=1))
        y = super().predict(X)
        if self.requires_sequence and self._sequence_to_label:
            for k, _ in enumerate(y):
                y[k] = MatrixToIndexProjection(output_strategy=self._decision_strategy,
                                               needs_proba=True).fit_transform(y[k])
                y[k] = np.clip(y[k], a_min=1e-5, a_max=None)
            return y
        elif self.requires_sequence:
            for k, _ in enumerate(y):
                y[k] = self._encoder.inverse_transform(y[k], threshold=None)
                y[k] = np.clip(y[k], a_min=1e-5, a_max=None)
            return y
        else:
            return self._encoder.inverse_transform(super().predict(X), threshold=None)

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
        if self.requires_sequence:
            y = self.predict_proba(X=X)
            for k, _ in enumerate(y):
                y[k] = np.log(y[k])
            return y
        else:
            return np.log(self.predict_proba(X=X))
