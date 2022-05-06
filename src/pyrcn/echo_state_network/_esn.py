"""The :mod:`echo_state_network` contains an ESNRegressor and ESNClassifier."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations
import sys
import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          MultiOutputMixin, is_regressor, clone)
from sklearn.linear_model._base import LinearModel

from ..base.blocks import InputToNode, NodeToNode
from ..util import concatenate_sequences
from ..linear_model import IncrementalRegression
from ..projection import MatrixToValueProjection
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.exceptions import NotFittedError

from joblib import Parallel, delayed

if sys.version_info >= (3, 8):
    from typing import Union, Dict, Any, Optional, Literal
else:
    from typing_extensions import Literal
    from typing import Union, Dict, Any, Optional


class ESNRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Echo State Network regressor.

    This model optimizes the mean squared error loss function
    using linear regression.

    Parameters
    ----------
    input_to_node : Optional[InputToNode], default=None
        Any ```InputToNode``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode```
        object is instantiated.
    node_to_node : Optional[NodeToNode], default=None
        Any ```NodeToNode``` object that transforms the outputs of
        ```input_to_node```.
        If ```None```, a ```pyrcn.base.blocks.NodeToNode```
        object is instantiated.
    regressor : Union[IncrementalRegression, LinearModel, None], default=None
        Regressor object such as derived from ``BaseEstimator``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression```
        object is instantiated.
    requires_sequence : Union[Literal["auto"], bool], default="auto"
        If True, the input data is expected to be a sequence.
        If "auto", tries to automatically estimate when calling ```fit```
        for the first time
    decision_strategy : Literal["winner_takes_all", "median", "last_value"],
    default='winner_takes_all'
        Decision strategy for sequence-to-label task. Ignored if the
        target output is a sequence
    verbose : bool = False
        Verbosity output
    kwargs : Any
        keyword arguments passed to the subestimators if this is desired,
        default=None
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node: Optional[InputToNode] = None,
                 node_to_node: Optional[NodeToNode] = None,
                 regressor: Union[IncrementalRegression,
                                  LinearModel, None] = None,
                 requires_sequence: Union[Literal["auto"], bool] = "auto",
                 decision_strategy: Literal["winner_takes_all", "median",
                                            "last_value"] = "winner_takes_all",
                 verbose: bool = True,
                 **kwargs: Any) -> None:
        """Construct the ESNRegressor."""
        if input_to_node is None:
            i2n_params = InputToNode()._get_param_names()
            self.input_to_node = InputToNode(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in i2n_params})
        else:
            i2n_params = input_to_node._get_param_names()
            self.input_to_node = input_to_node.set_params(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in i2n_params})
        if node_to_node is None:
            n2n_params = NodeToNode()._get_param_names()
            self.node_to_node = NodeToNode(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in n2n_params})
        else:
            n2n_params = node_to_node._get_param_names()
            self.node_to_node = node_to_node.set_params(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in n2n_params})
        if regressor is None:
            reg_params = IncrementalRegression()._get_param_names()
            self.regressor = IncrementalRegression(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in reg_params})
        else:
            reg_params = regressor._get_param_names()
            self.regressor = regressor.set_params(
                **{key: kwargs[key] for key in kwargs.keys()
                   if key in reg_params})
        self._regressor = self.regressor
        self._requires_sequence = requires_sequence
        self.verbose = verbose
        self.decision_strategy = decision_strategy

    def __add__(self, other: ESNRegressor) -> ESNRegressor:
        """
        Sum up two instances of an ```ESNRegressor```.

        We always need to update the correlation matrices of the regressor.

        Parameters
        ----------
        other : ESNRegressor
            ```ESNRegressor``` to be added to ```self```

        Returns
        -------
        self : returns the sum of two ```ESNRegressor``` instances.
        """
        self.regressor._K = self.regressor._K + other.regressor._K
        self.regressor._xTy = self.regressor._xTy + other.regressor._xTy
        return self

    def __radd__(self, other: ESNRegressor) -> ESNRegressor:
        """
        Sum up multiple instances of an ```ESNRegressor```.

        We always need to update the correlation matrices of the regressor.

        Parameters
        ----------
        other : ESNRegressor
            ```ESNRegressor``` to be added to ```self```

        Returns
        -------
        self : returns the sum of two ```ESNRegressor``` instances.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_params(self, deep: bool = True) -> Dict:
        """Get all parameters of the ESNRegressor."""
        if deep:
            return {**self.input_to_node.get_params(),
                    **self.node_to_node.get_params(),
                    **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node,
                    "node_to_node": self.node_to_node,
                    "regressor": self.regressor,
                    "requires_sequence": self._requires_sequence}

    def set_params(self, **parameters: dict) -> ESNRegressor:
        """Set all possible parameters of the ESNRegressor."""
        i2n_params = self.input_to_node._get_param_names()
        self.input_to_node = self.input_to_node.set_params(
            **{key: parameters[key] for key in parameters.keys()
               if key in i2n_params})
        n2n_params = self.node_to_node._get_param_names()
        self.node_to_node = self.node_to_node.set_params(
            **{key: parameters[key] for key in parameters.keys()
               if key in n2n_params})
        reg_params = self.regressor._get_param_names()
        self.regressor = self.regressor.set_params(
            **{key: parameters[key] for key in parameters.keys()
               if key in reg_params})
        for parameter, value in parameters.items():
            if parameter in self.get_params(deep=False):
                setattr(self, parameter, value)

        return self

    def _check_if_sequence(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validation of the training data.

        If X is a list and each member of the list has the same number of
        samples, we treat it as an array of instance (one sequence).

        If X or y have more than two dimensions, it is no valid data type.

        If the number of dimensions of X after converting it to a
        ```ndarray``` is one, the ESN runs in sequential mode.

        Parameters
        ----------
        X : np.ndarray
            The input data
        y : np.ndarray
            The target data
        """
        if X.ndim > 2 or y.ndim > 2:
            raise ValueError("Could not determine a valid structure,"
                             "because X has {0} and y has {1} dimensions."
                             "Only 1 or 2 dimensions allowed."
                             .format(X.ndim, y.ndim))
        self.requires_sequence = X.ndim == 1

    def _check_if_sequence_to_value(self,
                                    X: np.ndarray, y: np.ndarray) -> None:
        """
        Validation of the training data.

        If the numbers of samples in each element of (X, y) in sequential form
        are different, we assume to have a sequence-to-value problem,
        such as a seqence-to-label classification.

        Parameters
        ----------
        X : np.ndarray
            The input data
        y : np.ndarray
            The target data
        """
        len_X = np.unique([x.shape[0] for x in X])
        len_y = np.unique([yt.shape[0] for yt in y])
        self._sequence_to_value = not np.any(len_X == len_y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: Union[None, np.ndarray] = None,
                    postpone_inverse: bool = False) -> ESNRegressor:
        """
        Fit the regressor partially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        transformer_weights : ignored
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation. Refer to
            ```IncrementalRegression```
            for details.

        Returns
        -------
        self : Returns a trained ```ESNRegressor``` model.
        """
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
            hidden_layer_state = self._node_to_node.transform(
                hidden_layer_state)
        except NotFittedError as e:
            if self.verbose:
                print('node_to_node has not been fitted yet: {0}'.format(e))
            hidden_layer_state = self._node_to_node.fit_transform(
                hidden_layer_state)
            pass

        # regression
        if not hasattr(self._regressor, 'partial_fit') and postpone_inverse:
            raise BaseException('Regressor has no attribute partial_fit, got'
                                '{0}'.format(self._regressor))
        elif not hasattr(self._regressor, 'partial_fit') \
                and not postpone_inverse:
            self._regressor.fit(hidden_layer_state, y)
        elif hasattr(self._regressor, 'partial_fit'):
            self._regressor.partial_fit(
                hidden_layer_state, y, postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: Union[int, np.integer, None] = None,
            transformer_weights: Optional[np.ndarray] = None) -> ESNRegressor:
        """
        Fit the regressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or of shape (n_sequences,)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        or of shape (n_sequences)
            The targets to predict.
        n_jobs : Optional[int, np.integer], default=None
            The number of jobs to run in parallel. ```-1``` means using all
            processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : Optional[np.ndarray] = None
            ignored

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        self._validate_hyperparameters()
        if self.requires_sequence == "auto":
            self._check_if_sequence(X, y)
        if self.requires_sequence:
            self._input_to_node.fit(X[0])
            self._node_to_node.fit(self._input_to_node.transform(X[0]))
            X, y, sequence_ranges = concatenate_sequences(X, y)
        else:
            self._validate_data(X, y, multi_output=True)
            self._input_to_node.fit(X)
            self._node_to_node.fit(self._input_to_node.transform(X))
        # self._regressor = self._regressor.__class__()
        if self.requires_sequence:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)
        else:
            return self.partial_fit(X, y, postpone_inverse=False)

    def _sequence_fit(self, X: np.ndarray, y: np.ndarray,
                      sequence_ranges: np.ndarray,
                      n_jobs: Union[int, np.integer,
                                    None] = None) -> ESNRegressor:
        """
        Call partial_fit for each sequence. Runs parallel if more than one job.

        Parameters
        ----------
        X : ndarray of shape (samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        sequence_ranges : ndarray of shape (n_sequences, 2)
            The start and stop indices of each sequence are denoted here.
        n_jobs : Union[int, np.integer, None], default=None
            The number of jobs to run in parallel. ```-1``` means using all
            processors.
            See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        if n_jobs is not None and n_jobs > 1:
            reg = Parallel(n_jobs=n_jobs)(delayed(ESNRegressor.partial_fit)
                                          (clone(self), X[idx[0]:idx[1], ...],
                                           y[idx[0]:idx[1], ...],
                                           postpone_inverse=True)
                                          for idx in sequence_ranges[:-1])
            reg = sum(reg)
            self._regressor = reg._regressor
        else:
            [ESNRegressor.partial_fit(self,
                                      X[idx[0]:idx[1], ...],
                                      y[idx[0]:idx[1], ...],
                                      postpone_inverse=True)
             for idx in sequence_ranges[:-1]]

        # last sequence, calculate inverse and bias
        ESNRegressor.partial_fit(self, X=X[sequence_ranges[-1][0]:, ...],
                                 y=y[sequence_ranges[-1][0]:, ...],
                                 postpone_inverse=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the targets using the trained ```ESNRegressor```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of (n_samples,) or (n_samples, n_targets)
            The predicted targets
        """
        if self._input_to_node is None or self._regressor is None:
            raise NotFittedError(self)

        if self.requires_sequence is False:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)
            hidden_layer_state = self._node_to_node.transform(
                hidden_layer_state)
            # regression
            return self._regressor.predict(hidden_layer_state)
        else:
            y = np.empty(shape=X.shape, dtype=object)
            for k, seq in enumerate(X):
                # input_to_node
                hidden_layer_state = self._input_to_node.transform(seq)
                hidden_layer_state = self._node_to_node.transform(
                    hidden_layer_state)
                # regression
                y[k] = self._regressor.predict(hidden_layer_state)
            return y

    def _validate_hyperparameters(self) -> None:
        """Validate the hyperparameters."""
        if not (hasattr(self.input_to_node, "fit")
                and hasattr(self.input_to_node, "fit_transform")
                and hasattr(self.input_to_node, "transform")):
            raise TypeError("All input_to_node should be transformers and"
                            "implement fit and transform '{0}' (type {1}) "
                            "doesn't".format(self.input_to_node,
                                             type(self.input_to_node)))

        if not (hasattr(self.node_to_node, "fit")
                and hasattr(self.node_to_node, "fit_transform")
                and hasattr(self.node_to_node, "transform")):
            raise TypeError("All node_to_node should be transformers and"
                            "implement fit and transform '{0}' (type {1}) "
                            "doesn't".format(self.node_to_node,
                                             type(self.node_to_node)))

        if (self._requires_sequence != "auto"
                and not isinstance(self._requires_sequence, bool)):
            raise ValueError('Invalid value for requires_sequence, got {0}'
                             .format(self._requires_sequence))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor and "
                            "implement fit and predict '{0}' (type {1})"
                            "doesn't".format(self._regressor,
                                             type(self._regressor)))

    def __sizeof__(self) -> int:
        """
        Return the size of the object in bytes.

        Returns
        -------
        size : int
            Object memory in bytes.
        """
        return object.__sizeof__(self) + sys.getsizeof(self._input_to_node) + \
            sys.getsizeof(self._node_to_node) + sys.getsizeof(self._regressor)

    @property
    def regressor(self) -> Union[LinearModel, IncrementalRegression]:
        """
        Return the regressor.

        Returns
        -------
        regressor : LinearModel
        """
        return self._regressor

    @regressor.setter
    def regressor(self, regressor: Union[LinearModel,
                                         IncrementalRegression]) -> None:
        """
        Set the regressor.

        Parameters
        ----------
        regressor : LinearModel
        """
        self._regressor = regressor

    @property
    def input_to_node(self) -> InputToNode:
        """
        Return the input_to_node Transformer.

        Returns
        -------
        input_to_node : InputToNode
        """
        return self._input_to_node

    @input_to_node.setter
    def input_to_node(self, input_to_node: InputToNode) -> None:
        """
        Set the input_to_node Estimator.

        Parameters
        ----------
        input_to_node : InputToNode
        """
        self._input_to_node = input_to_node

    @property
    def node_to_node(self) -> NodeToNode:
        """
        Return the node_to_node Transformer.

        Returns
        -------
        node_to_node : NodeToNode
        """
        return self._node_to_node

    @node_to_node.setter
    def node_to_node(self, node_to_node: NodeToNode) -> None:
        """
        Set the node_to_node Transformer.

        Parameters
        ----------
        node_to_node : NodeToNode
        """
        self._node_to_node = node_to_node

    def hidden_layer_state(self, X: np.ndarray) -> np.ndarray:
        """
        Return the hidden_layer_state, e.g. the reservoir state over time.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        hidden_layer_state : ndarray of (n_samples,)
            The hidden_layer_state, e.g. the reservoir state over time.
        """
        if self._input_to_node is None:
            raise NotFittedError(self)

        if self.requires_sequence is False:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)
            hidden_layer_state = self._node_to_node.transform(
                hidden_layer_state)
        else:
            hidden_layer_state = np.empty(shape=X.shape, dtype=object)
            for k, seq in enumerate(X):
                # input_to_node
                hls = self._input_to_node.transform(seq)
                hls = self._node_to_node.transform(hls)
                hidden_layer_state[k] = hls
        return hidden_layer_state

    @property
    def sequence_to_value(self) -> bool:
        """
        Return the sequence_to_value parameter.

        Returns
        -------
        sequence_to_value : bool
        """
        return self._sequence_to_value

    @sequence_to_value.setter
    def sequence_to_value(self, sequence_to_value: bool) -> None:
        """
        Set the sequence_to_value parameter.

        Parameters
        ----------
        sequence_to_value : bool
        """
        self._sequence_to_value = sequence_to_value

    @property
    def decision_strategy(self) -> Literal["winner_takes_all",
                                           "median", "last_value"]:
        """
        Return the decision_strategy parameter.

        Returns
        -------
        decision_strategy : Literal["winner_takes_all", "median", "last_value"]
        """
        return self._decision_strategy

    @decision_strategy.setter
    def decision_strategy(self, decision_strategy: Literal["winner_takes_all",
                                                           "median",
                                                           "last_value"])\
            -> None:
        """
        Set the requires_sequence parameter.

        Parameters
        ----------
        decision_strategy : Literal["winner_takes_all", "median", "last_value"]
        """
        self._decision_strategy = decision_strategy

    @property
    def requires_sequence(self) -> Union[Literal["auto"], bool]:
        """
        Return the requires_sequence parameter.

        Returns
        -------
        requires_sequence : Union[Literal["auto"], bool]
        """
        return self._requires_sequence

    @requires_sequence.setter
    def requires_sequence(self,
                          requires_sequence: Union[Literal["auto"], bool])\
            -> None:
        """
        Set the requires_sequence parameter.

        Parameters
        ----------
        requires_sequence : Union[Literal["auto"], bool]

        """
        self._requires_sequence = requires_sequence


class ESNClassifier(ESNRegressor, ClassifierMixin):
    """
    Echo State Network classifier.

    This model optimizes the mean squared error loss function using
    linear regression.

    Parameters
    ----------
    input_to_node : Optional[InputToNode], default=None
        Any ```InputToNode``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode```
        object is instantiated.
    node_to_node : Optional[NodeToNode], default=None
        Any ```NodeToNode``` object that transforms the outputs of
        ```input_to_node```.
        If ```None```, a ```pyrcn.base.blocks.NodeToNode()```
        object is instantiated.
    regressor : Union[IncrementalRegression, LinearModel, None], default=None
        Regressor object such as derived from ``LinearModel``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression()```
        object is instantiated.
    requires_sequence : Union[Literal["auto"], bool], default="auto"
        If True, the input data is expected to be a sequence.
        If "auto", tries to automatically estimate when calling ```fit```
        for the first time
    decision_strategy : Literal["winner_takes_all", "median", "last_value"],
    default='winner_takes_all'
        Decision strategy for sequence-to-label task.
        Ignored if the target output is a sequence
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired.
    """

    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node: Optional[InputToNode] = None,
                 node_to_node: Optional[NodeToNode] = None,
                 regressor: Union[IncrementalRegression,
                                  LinearModel, None] = None,
                 requires_sequence: Union[Literal["auto"], bool] = "auto",
                 decision_strategy: Literal["winner_takes_all", "median",
                                            "last_value"] = "winner_takes_all",
                 verbose: bool = False,
                 **kwargs: Any) -> None:
        """Construct the ESNClassifier."""
        super().__init__(input_to_node=input_to_node,
                         node_to_node=node_to_node, regressor=regressor,
                         requires_sequence=requires_sequence, verbose=verbose,
                         **kwargs)
        self._decision_strategy = decision_strategy
        self._encoder = MultiLabelBinarizer()
        self._sequence_to_value = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: Optional[np.ndarray] = None,
                    postpone_inverse: bool = False,
                    classes: Optional[np.ndarray] = None) -> ESNClassifier:
        """
        Fit the regressor partially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        classes : Optional[np.ndarray], default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        transformer_weights : Optional[ndarray], default=None
            ignored
        postpone_inverse : bool, default=False
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation. Refer to
            IncrementalRegression for details.

        Returns
        -------
        self : returns a trained ESNClassifier model
        """
        self._validate_data(X, y, multi_output=True)
        self._encoder.fit(classes)
        super().partial_fit(X, self._encoder.transform(y),
                            transformer_weights=None,
                            postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: Union[int, np.integer, None] = None,
            transformer_weights: Union[None,
                                       np.ndarray] = None) -> ESNClassifier:
        """
        Fit the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or of shape (n_sequences,)
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        or of shape (n_sequences)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ```-1``` means using all
            processors.
            See :term:`Glossary <n_jobs>` for more details.
        transformer_weights : ignored

        Returns
        -------
        self : Returns a trained ESNClassifier model.
        """
        self._validate_hyperparameters()
        if self.requires_sequence == "auto":
            self._check_if_sequence(X, y)
        if self.requires_sequence:
            self._input_to_node.fit(X[0])
            self._node_to_node.fit(self._input_to_node.transform(X[0]))
            self._check_if_sequence_to_value(X, y)
            X, y, sequence_ranges = concatenate_sequences(
                X, y, sequence_to_value=self._sequence_to_value)
        else:
            self._validate_data(X, y, multi_output=True)
            self._input_to_node.fit(X)
            self._node_to_node.fit(self._input_to_node.transform(X))
        self._encoder = LabelBinarizer().fit(y)
        y = self._encoder.transform(y)
        # self._regressor = self._regressor.__class__()
        if self.requires_sequence:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)
        else:
            super().partial_fit(X, y)
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the classes using the trained ```ESNClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        y = super().predict(X)
        if self.requires_sequence and self._sequence_to_value:
            for k, _ in enumerate(y):
                y[k] = MatrixToValueProjection(
                    output_strategy=self._decision_strategy)\
                    .fit_transform(y[k])
            return y
        elif self.requires_sequence:
            for k, _ in enumerate(y):
                y[k] = self._encoder.inverse_transform(y[k], threshold=None)
            return y
        else:
            return self._encoder.inverse_transform(super().predict(X),
                                                   threshold=None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability estimated using a trained ```ESNClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted probability estimates.
        """
        y = super().predict(X)
        if self.requires_sequence and self._sequence_to_value:
            for k, _ in enumerate(y):
                y[k] = MatrixToValueProjection(
                    output_strategy=self._decision_strategy, needs_proba=True)\
                    .fit_transform(y[k])
                y[k] = np.clip(y[k], a_min=1e-5, a_max=None)
            return y
        elif self.requires_sequence:
            for k, _ in enumerate(y):
                y[k] = np.clip(y[k], a_min=1e-5, a_max=None)
            return y
        else:
            return np.asarray(np.clip(y, a_min=1e-5, a_max=None))

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the log probability estimated using a trained
        ```ESNClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
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
