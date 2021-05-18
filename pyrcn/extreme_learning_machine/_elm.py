"""
The :mod:`extreme_learning_machine` contains the ELMRegressor and the ELMClassifier
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import sys

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin, is_regressor
from pyrcn.base import InputToNode
from pyrcn.linear_model import IncrementalRegression
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion


class ELMRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Extreme Learning Machine regressor.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : iterable, default=InputToNode()
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    chunk_size : int, default=None
        if X.shape[0] > chunk_size, calculate results incrementally with partial_fit
    kwargs : dict, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node=None,
                 regressor=None,
                 chunk_size=None,
                 verbose=False,
                 **kwargs):
        if input_to_node is None:
            i2n_params = InputToNode()._get_param_names()
            self.input_to_node = InputToNode(**{ key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        else:
            i2n_params = input_to_node._get_param_names()
            self.input_to_node = input_to_node.set_params(**{ key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        if regressor is None:
            reg_params = IncrementalRegression()._get_param_names()
            self.regressor = IncrementalRegression(**{ key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        else:
            reg_params = regressor._get_param_names()
            self.regressor = regressor.set_params(**{ key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        self._chunk_size = chunk_size
        self.verbose = verbose

    def get_params(self, deep=True):
        if deep:
            return {**self.input_to_node.get_params(), **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node, "regressor": self.regressor, "chunk_size": self.chunk_size}

    def set_params(self, **parameters):
        i2n_params = self.input_to_node._get_param_names()
        self.input_to_node = self.input_to_node.set_params(**{ key: parameters[key] for key in parameters.keys() if key in i2n_params})
        reg_params = self.regressor._get_param_names()
        self.regressor = self.regressor.set_params(**{ key: parameters[key] for key in parameters.keys() if key in reg_params})
        for parameter, value in parameters.items():
            if parameter in self.get_params(deep=False):
                setattr(self, parameter, value)

        return self

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
        postpone_inverse : bool, default=False
            If output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to IncrementalRegression for details.

        Returns
        -------
        self : Returns a trained ELMRegressor model.
        """
        if not hasattr(self._regressor, 'partial_fit'):
            raise BaseException('regressor has no attribute partial_fit, got {0}'.format(self._regressor))

        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        # input_to_node
        try:
            hidden_layer_state = self._input_to_node.transform(X)
        except NotFittedError as e:
            if self.verbose:
                print('input_to_node has not been fitted yet: {0}'.format(e))
            hidden_layer_state = self._input_to_node.fit_transform(X)
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
        self : Returns a trained ELMRegressor model.
        """
        self._validate_hyperparameters()
        self._validate_data(X, y, multi_output=True)

        self._input_to_node.fit(X)
        self._regressor = self._regressor.__class__()

        if self._chunk_size is None or self._chunk_size >= X.shape[0]:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)

            # regression
            self._regressor.fit(hidden_layer_state, y)
        elif self._chunk_size < X.shape[0]:
            # setup chunk list
            chunks = list(range(0, X.shape[0], self._chunk_size))
            # postpone inverse calculation for chunks n-1
            for idx in chunks[:-1]:
                ELMRegressor.partial_fit(
                    self,
                    X=X[idx:idx + self._chunk_size, ...],
                    y=y[idx:idx + self._chunk_size, ...],
                    n_jobs=n_jobs,
                    transformer_weights=transformer_weights,
                    postpone_inverse=True
                )
            # last chunk, calculate inverse and bias
            ELMRegressor.partial_fit(
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

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self):
        """
        Validates the hyperparameters.

        Returns
        -------
        """
        if not (hasattr(self.input_to_node, "fit") and hasattr(self.input_to_node, "fit_transform") and hasattr(
                self.input_to_node, "transform")):
            raise TypeError("All input_to_node should be transformers "
                            "and implement fit and transform "
                            "'%s' (type %s) doesn't" % (self.input_to_node, type(self.input_to_node)))

        if self._chunk_size is not None and (not isinstance(self._chunk_size, int) or self._chunk_size < 0):
            raise ValueError('Invalid value for chunk_size, got {0}'.format(self._chunk_size))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor "
                            "and implement fit and predict"
                            "'%s' (type %s) doesn't" % (self._regressor, type(self._regressor)))

    def __sizeof__(self):
        """
        Returns the size of the object in bytes.

        Returns
        -------
        size : int
        Object memory in bytes.
        """
        return object.__sizeof__(self) + \
            sys.getsizeof(self._input_to_node) + \
            sys.getsizeof(self._regressor)

    @property
    def regressor(self):
        """
        Returns the chunk_size, in which X will be chopped.

        Returns
        -------
        chunk_size : int or None
        """
        return self._regressor

    @regressor.setter
    def regressor(self, regressor):
        """
        Sets the regressor.

        Parameters
        ----------
        regressor : regressor or None

        Returns
        -------
        """
        self._regressor = regressor

    @property
    def input_to_node(self):
        """
        Returns the input_to_node list or the input_to_node Transformer.

        Returns
        -------
        input_to_node : Transformer or [Transformer]
        """
        return self._input_to_node

    @property
    def hidden_layer_state(self):
        """Returns the hidden_layer_state, e.g. the resevoir state over time.
        Returns
        -------
        hidden_layer_state : np.ndarray
        """
        return self._input_to_node._hidden_layer_state

    @input_to_node.setter
    def input_to_node(self, input_to_node, n_jobs=None, transformer_weights=None):
        """
        Sets the input_to_node list or the input_to_node Transformer.

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
    def chunk_size(self):
        """
        Returns the chunk_size, in which X will be chopped.

        Returns
        -------
        chunk_size : int or None
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size):
        """
        Sets the chunk_size, in which X will be chopped.

        Parameters
        ----------
        chunk_size : int or None

        Returns
        -------
        """
        self._chunk_size = chunk_size


class ELMClassifier(ELMRegressor, ClassifierMixin):
    """
    Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : iterable, default=[('default', InputToNode())]
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    regressor : object, default=IncrementalRegression(alpha=.0001)
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        regressor cannot be None, omit argument if in doubt
    chunk_size : int, default=None
        if X.shape[0] > chunk_size, calculate results incrementally with partial_fit
    kwargs : dict, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """
    def __init__(self, *,
                 input_to_node=None,
                 regressor=None,
                 chunk_size=None,
                 **kwargs):
        super().__init__(input_to_node=input_to_node, regressor=regressor, chunk_size=chunk_size, **kwargs)
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
        postpone_inverse : bool, default=False
            If output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to IncrementalRegression for details.

        Returns
        -------
        self : returns a trained ELMClassifier model
        """
        self._validate_data(X, y, multi_output=True)

        if self._encoder is None:
            self._encoder = LabelBinarizer().fit(classes)

        return super().partial_fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None,
                                   postpone_inverse=postpone_inverse)

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        """
        Fits the regressor.

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
        self : Returns a trained ELMClassifier model.
        """
        self._validate_data(X, y, multi_output=True)
        self._encoder = LabelBinarizer().fit(y)

        return super().fit(X, self._encoder.transform(y), n_jobs=n_jobs, transformer_weights=None)

    def predict(self, X):
        """
        Predict the classes using the trained ELM classifier.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        return self._encoder.inverse_transform(super().predict(X), threshold=None)

    def predict_proba(self, X):
        """
        Predict the probability estimated using the trained ELM classifier.

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
        predicted_positive = np.clip(super().predict(X), a_min=1e-5, a_max=None).T
        return np.divide(predicted_positive, np.sum(predicted_positive, axis=0)).T

    def predict_log_proba(self, X):
        """
        Predict the logarithmic probability estimated using the trained ELM classifier.

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
