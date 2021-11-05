"""The :mod:`extreme_learning_machine` contains the ELMRegressor and ELMClassifier."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause
import sys
from typing import Union, Any, Optional

import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin, MultiOutputMixin, is_regressor, clone)
from pyrcn.base.blocks import InputToNode
from pyrcn.linear_model import IncrementalRegression
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import NotFittedError

from joblib import Parallel, delayed


class ELMRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Extreme Learning Machine regressor.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : Union[InputToNode, TransformerMixin, None], default=None
        Any ```sklearn.base.TransformerMixin``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode()``` object is instantiated.
    regressor : Union[IncrementalRegression, RegressorMixin, None], default=None
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression()``` object is
        instantiated.
    chunk_size : Optional[int], default=None
         if X.shape[0] > chunk_size, calculate results incrementally with partial_fit
    decision_strategy : Literal['winner_takes_all', 'median', 'weighted', 'last_value',
                                'mode'], default='winner_takes_all'
        Decision strategy for sequence-to-label task. Ignored if the target output is
        a sequence
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node: Union[InputToNode, TransformerMixin, None] = None,
                 regressor: Union[IncrementalRegression, RegressorMixin, None] = None,
                 chunk_size: Optional[int] = None,
                 verbose: bool = False,
                 **kwargs: Any) -> None:
        """Construct the ELMRegressor."""
        if input_to_node is None:
            i2n_params = InputToNode()._get_param_names()
            self.input_to_node = InputToNode(
                **{key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        else:
            i2n_params = input_to_node._get_param_names()
            self.input_to_node = input_to_node.set_params(
                **{key: kwargs[key] for key in kwargs.keys() if key in i2n_params})
        if regressor is None:
            reg_params = IncrementalRegression()._get_param_names()
            self.regressor = IncrementalRegression(
                **{key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        else:
            reg_params = regressor._get_param_names()
            self.regressor = regressor.set_params(
                **{key: kwargs[key] for key in kwargs.keys() if key in reg_params})
        self._regressor = self.regressor
        self._chunk_size = chunk_size
        self.verbose = verbose

    def __add__(self, other: BaseEstimator) -> BaseEstimator:
        """
        Sum up two instances of an ```ELMRegressor```.

        We always need to update the correlation matrices of the regressor.

        Parameters
        ----------
        other : ELMRegressor
            ```ELMRegressor``` to be added to ```self```

        Returns
        -------
        self : returns the sum of two ```ELMRegressor``` instances.
        """
        self.regressor._K = self.regressor._K + other.regressor._K
        self.regressor._xTy = self.regressor._xTy + other.regressor._xTy
        return self

    def __radd__(self, other: BaseEstimator) -> BaseEstimator:
        """
        Sum up multiple instances of an ```ELMRegressor```.

        We always need to update the correlation matrices of the regressor.

        Parameters
        ----------
        other : ELMRegressor
            ```ELMRegressor``` to be added to ```self```

        Returns
        -------
        self : returns the sum of multiple ```ELMRegressor``` instances.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_params(self, deep: bool = True) -> dict:
        """Get all parameters of the ESNRegressor."""
        if deep:
            return {**self.input_to_node.get_params(),
                    **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node, "regressor": self.regressor,
                    "chunk_size": self.chunk_size}

    def set_params(self, **parameters: dict) -> BaseEstimator:
        """Set all possible parameters of the ESNRegressor."""
        i2n_params = self.input_to_node._get_param_names()
        self.input_to_node = self.input_to_node.set_params(
            **{key: parameters[key] for key in parameters.keys() if key in i2n_params})
        reg_params = self.regressor._get_param_names()
        self.regressor = self.regressor.set_params(
            **{key: parameters[key] for key in parameters.keys() if key in reg_params})
        for parameter, value in parameters.items():
            if parameter in self.get_params(deep=False):
                setattr(self, parameter, value)

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: Union[np.ndarray, None] = None,
                    postpone_inverse: bool = False) -> BaseEstimator:
        """
        Fit the regressor partially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        transformer_weights : Union[np.ndarray, None], default=None
            ignored
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to ```IncrementalRegression```
            for details.

        Returns
        -------
        self : Returns a trained ```ELMRegressor``` model.
        """
        if not hasattr(self._regressor, 'partial_fit'):
            raise BaseException('regressor has no attribute partial_fit, got {0}'
                                .format(self._regressor))
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
            self._regressor.partial_fit(hidden_layer_state, y,
                                        postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: Union[int, np.integer, None] = None,
            transformer_weights: Union[np.ndarray, None] = None) -> BaseEstimator:
        """
        Fit the regressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ```-1``` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights :  Union[np.ndarray, None], default=None
            ignored

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
            if n_jobs is None or n_jobs < 2:
                [ELMRegressor.partial_fit(self, X[idx:idx + self._chunk_size, ...],
                                          y[idx:idx + self._chunk_size, ...],
                                          transformer_weights=transformer_weights,
                                          postpone_inverse=True)
                 for idx in chunks[:-1]]
            else:
                reg = Parallel(n_jobs=n_jobs)(delayed(ELMRegressor.partial_fit)
                                              (clone(self),
                                               X[idx:idx + self._chunk_size, ...],
                                               y[idx:idx + self._chunk_size, ...],
                                               transformer_weights=transformer_weights,
                                               postpone_inverse=True)
                                              for idx in chunks[:-1])
                reg = sum(reg)
                self._regressor = reg._regressor
            # last chunk, calculate inverse and bias
            ELMRegressor.partial_fit(self, X=X[chunks[-1]:, ...], y=y[chunks[-1]:, ...],
                                     transformer_weights=transformer_weights,
                                     postpone_inverse=False)
        else:
            raise ValueError('chunk_size invalid {0}'.format(self._chunk_size))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the targets using the trained ```ELMRegressor```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of (n_samples,) or (n_samples, n_targets)
            The predicted targets
        """
        hidden_layer_state = self._input_to_node.transform(X)

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self) -> None:
        """Validate the hyperparameters."""
        if not (hasattr(self.input_to_node, "fit")
                and hasattr(self.input_to_node, "fit_transform")
                and hasattr(self.input_to_node, "transform")):
            raise TypeError("All input_to_node should be transformers and"
                            "implement fit and transform '{0}' (type {1}) doesn't"
                            .format(self.input_to_node, type(self.input_to_node)))

        if (self._chunk_size is not None
                and (not isinstance(self._chunk_size, int) or self._chunk_size < 0)):
            raise ValueError('Invalid value for chunk_size, got {0}'
                             .format(self._chunk_size))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor and"
                            "implement fit and predict '{0}' (type {1}) doesn't"
                            .format(self._regressor, type(self._regressor)))

    def __sizeof__(self) -> int:
        """
        Return the size of the object in bytes.

        Returns
        -------
        size : int
            Object memory in bytes.
        """
        return object.__sizeof__(self) + sys.getsizeof(self._input_to_node) + \
            sys.getsizeof(self._regressor)

    @property
    def regressor(self) -> RegressorMixin:
        """
        Return the regressor.

        Returns
        -------
        regressor : RegressorMixin
        """
        return self._regressor

    @regressor.setter
    def regressor(self, regressor: RegressorMixin) -> None:
        """
        Set the regressor.

        Parameters
        ----------
        regressor : RegressorMixin
        """
        self._regressor = regressor

    @property
    def input_to_node(self) -> Union[InputToNode, TransformerMixin]:
        """
        Return the input_to_node Transformer.

        Returns
        -------
        input_to_node : TransformerMixin
        """
        return self._input_to_node

    @input_to_node.setter
    def input_to_node(self, input_to_node: Union[InputToNode,
                                                 TransformerMixin]) -> None:
        """
        Set the input_to_node Transformer.

        Parameters
        ----------
        input_to_node : TransformerMixin
        """
        self._input_to_node = input_to_node

    @property
    def hidden_layer_state(self) -> np.ndarray:
        """
        Return the hidden_layer_state, e.g. the resevoir state over time.

        Returns
        -------
        hidden_layer_state : np.ndarray
        """
        return self._input_to_node._hidden_layer_state

    @property
    def chunk_size(self) -> Union[None, int, np.integer]:
        """
        Return the chunk_size, in which X will be chopped.

        Returns
        -------
        chunk_size : Union[int, np.integer]
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: Union[int, None]) -> None:
        """
        Set the chunk_size, in which X will be chopped.

        Parameters
        ----------
        chunk_size : Union[int, None]
        """
        self._chunk_size = chunk_size


class ELMClassifier(ELMRegressor, ClassifierMixin):
    """
    Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using linear regression.

    Parameters
    ----------
    input_to_node : Union[InputToNode, TransformerMixin, None], default=None
        Any ```sklearn.base.TransformerMixin``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode()``` object is instantiated.
    regressor : Union[IncrementalRegression, RegressorMixin, None], default=None
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression()``` object
        is instantiated.
    chunk_size : Union[int, None], default=None
         if X.shape[0] > chunk_size, calculate results incrementally with partial_fit
    decision_strategy : Literal['winner_takes_all', 'median', 'weighted', 'last_value',
    'mode'], default='winner_takes_all'
        Decision strategy for sequence-to-label task. Ignored if the target output
        is a sequence
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired, default=None
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node: Union[InputToNode, TransformerMixin, None] = None,
                 regressor: Union[IncrementalRegression, RegressorMixin, None] = None,
                 chunk_size: Union[int, None] = None, verbose: bool = False,
                 **kwargs: Any) -> None:
        """Construct the ELMClassifier."""
        super().__init__(input_to_node=input_to_node, regressor=regressor,
                         chunk_size=chunk_size, verbose=verbose, **kwargs)
        self._encoder = LabelBinarizer()

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: Optional[np.ndarray] = None,
                    postpone_inverse: bool = False,
                    classes: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        Fit the classifier partially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The targets to predict.
        classes : Optional[ndarray] of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        transformer_weights : Optional[ndarray], default=None
            ignored
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be hinted at
            postponing inverse calculation. Refer to ```IncrementalRegression```
            for details.

        Returns
        -------
        self : returns a trained ELMClassifier model
        """
        self._validate_data(X, y, multi_output=True)

        self._encoder.fit(classes)

        return super().partial_fit(X, self._encoder.transform(y),
                                   transformer_weights=None,
                                   postpone_inverse=postpone_inverse)

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: Union[int, np.integer, None] = None,
            transformer_weights: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        Fit the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The targets to predict.
        n_jobs : Union[int, np.integer, None], default=None
            The number of jobs to run in parallel. ```-1``` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : Optional[np.ndarray], default=None
            ignored

        Returns
        -------
        self : Returns a trained ELMClassifier model.
        """
        self._validate_data(X, y, multi_output=True)
        self._encoder = LabelBinarizer().fit(y)

        return super().fit(X, self._encoder.transform(y), n_jobs=n_jobs,
                           transformer_weights=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the classes using the trained ```ELMClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        return self._encoder.inverse_transform(super().predict(X), threshold=None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability estimated using the trained ```ELMClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted probability estimates.
        """
        predicted_positive = np.clip(super().predict(X), a_min=1e-5, a_max=None)
        return np.asarray(predicted_positive)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the log probability estimated using the trained ```ELMClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted logarithmic probability estimated.
        """
        return np.log(self.predict_proba(X=X))
