"""The :mod:`extreme_learning_machine` contains the ELMRegressor and
ELMClassifier."""

from __future__ import annotations

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from sklearn.base import (BaseEstimator, ClassifierMixin, MultiOutputMixin,
                          RegressorMixin, is_regressor)
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import validate_data

from ..nn._bridge import (
    build_input_map, build_readout, input_is_backable, regressor_is_backable)
from ..nn._input import InputFeatureMap
from ..nn._readout import IncrementalRidge, LinearReadout
from ..nn._training import torch_generator, train_readout
from ..base.blocks import InputToNode
from ..linear_model import IncrementalRegression


class ELMRegressor(RegressorMixin, MultiOutputMixin, BaseEstimator):
    """
    Extreme Learning Machine regressor.

    This model optimizes the mean squared error loss function using
    linear regression.

    Parameters
    ----------
    input_to_node : Optional[InputToNode], default=None
        Any ```InputToNode``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode```
        object is instantiated.
    regressor : Union[IncrementalRegression, RegressorMixin, None],
    default=None
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression```
        object is instantiated.
    chunk_size : Optional[int], default=None
         if X.shape[0] > chunk_size, calculate results incrementally with
         partial_fit
    solver : {"closed_form", "gradient"}, default="closed_form"
        Readout training method. ``"closed_form"`` solves the ridge normal
        equations; ``"gradient"`` trains the readout with an optimizer loop.
    optimizer : {"adam", "sgd"}, default="adam"
        Optimizer used when ``solver="gradient"``.
    learning_rate : float, default=1e-3
        Learning rate used when ``solver="gradient"``.
    epochs : int, default=100
        Number of training epochs when ``solver="gradient"``.
    batch_size : Optional[int], default=None
        Mini-batch size when ``solver="gradient"`` (``None`` = full batch).
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired,
        default=None
    """

    def __init__(self, *,
                 input_to_node: InputToNode | None = None,
                 regressor: (IncrementalRegression |
                             RegressorMixin | None) = None,
                 chunk_size: int | None = None,
                 verbose: bool = False,
                 solver: str = "closed_form",
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int | None = None,
                 **kwargs: Any) -> None:
        """Construct the ELMRegressor."""
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
        self._chunk_size = chunk_size
        self.verbose = verbose
        self.solver = solver
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self._use_torch: bool = False
        self._target_1d: bool = False
        self._torch_input_map: InputFeatureMap
        self._torch_readout: IncrementalRidge | LinearReadout

    def get_params(self, deep: bool = True) -> dict:
        """Get all parameters of the ESNRegressor."""
        if deep:
            return {**self.input_to_node.get_params(),
                    **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node,
                    "regressor": self.regressor,
                    "chunk_size": self.chunk_size,
                    "solver": self.solver,
                    "optimizer": self.optimizer,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size}

    def set_params(self, **parameters: dict) -> ELMRegressor:
        """Set all possible parameters of the ELMRegressor."""
        i2n_params = self.input_to_node._get_param_names()
        self.input_to_node = self.input_to_node.set_params(
            **{key: parameters[key] for key in parameters.keys()
               if key in i2n_params})
        reg_params = self.regressor._get_param_names()
        self.regressor = self.regressor.set_params(
            **{key: parameters[key] for key in parameters.keys()
               if key in reg_params})
        for parameter, value in parameters.items():
            if parameter in self.get_params(deep=False):
                setattr(self, parameter, value)

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: np.ndarray | None = None,
                    postpone_inverse: bool = False) -> ELMRegressor:
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
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation. Refer to
            ```IncrementalRegression```
            for details.

        Returns
        -------
        self : Returns a trained ```ELMRegressor``` model.
        """
        if not hasattr(self._regressor, 'partial_fit'):
            raise TypeError(
                "regressor has no attribute partial_fit, "
                f"got {self._regressor}")
        self._validate_hyperparameters()
        validate_data(self, X, y, multi_output=True)
        self._use_torch = False

        # input_to_node
        try:
            hidden_layer_state = self._input_to_node.transform(X)
        except NotFittedError as e:
            if self.verbose:
                print(f'input_to_node has not been fitted yet: {e}')
            hidden_layer_state = self._input_to_node.fit_transform(X)

        # regression
        if self._regressor:
            self._regressor.partial_fit(hidden_layer_state, y,
                                        postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: int | np.integer | None = None,
            transformer_weights: (np.ndarray |
                                  None) = None) -> ELMRegressor:
        """
        Fit the regressor.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            The targets to predict.
        n_jobs : int, default=None
            The number of jobs to run in parallel. ```-1``` means using all
            processors.
            See the scikit-learn glossary for n_jobs.
        transformer_weights :  Union[np.ndarray, None], default=None
            ignored

        Returns
        -------
        self : Returns a trained ELMRegressor model.
        """
        self._validate_hyperparameters()
        validate_data(self, X, y, multi_output=True)

        self._target_1d = (np.asarray(y).ndim == 1)
        backable = (input_is_backable(self._input_to_node)
                    and regressor_is_backable(self._regressor))

        if self.solver == "gradient":
            if not backable:
                raise NotImplementedError(
                    "the gradient solver requires a torch-backable "
                    "input_to_node and an IncrementalRegression readout")
            self._input_to_node.fit(X)
            dtype = torch.float64
            gen = torch_generator(self._input_to_node.random_state)
            fm = build_input_map(self._input_to_node, dtype=dtype)
            feats = fm(torch.as_tensor(np.asarray(X), dtype=dtype))
            y2 = torch.as_tensor(
                np.asarray(y), dtype=dtype).reshape(feats.shape[0], -1)
            lin_readout = LinearReadout(
                feats.shape[1], y2.shape[1],
                fit_intercept=self._regressor.fit_intercept,
                generator=gen, dtype=dtype)
            train_readout(
                lin_readout, feats, y2, optimizer=self.optimizer,
                learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size,
                weight_decay=self._regressor.alpha, generator=gen)
            self._torch_input_map = fm
            self._torch_readout = lin_readout
            self._use_torch = True
            return self

        self._input_to_node.fit(X)
        self._use_torch = False

        if (input_is_backable(self._input_to_node)
                and regressor_is_backable(self._regressor)):
            dtype = torch.float64
            fm = build_input_map(self._input_to_node, dtype=dtype)
            Z = fm(torch.as_tensor(np.asarray(X), dtype=dtype))
            readout = build_readout(self._regressor, dtype=dtype)
            readout.fit(Z, torch.as_tensor(np.asarray(y), dtype=dtype))
            self._torch_input_map = fm
            self._torch_readout = readout
            self._use_torch = True
            return self

        if self._chunk_size is None or self._chunk_size >= X.shape[0]:
            # input_to_node
            hidden_layer_state = self._input_to_node.transform(X)

            # regression
            self._regressor.fit(hidden_layer_state, y)

        elif self._chunk_size < X.shape[0]:
            # setup chunk list
            chunks = list(range(0, X.shape[0], self._chunk_size))
            # n_jobs is accepted for API compatibility but ignored; chunks
            # are accumulated serially (the torch fast path batches instead).
            for idx in chunks[:-1]:
                ELMRegressor.partial_fit(
                    self, X[idx:idx + self._chunk_size, ...],
                    y[idx:idx + self._chunk_size, ...],
                    transformer_weights=transformer_weights,
                    postpone_inverse=True)
            # last chunk, calculate inverse and bias
            ELMRegressor.partial_fit(self, X=X[chunks[-1]:, ...],
                                     y=y[chunks[-1]:, ...],
                                     transformer_weights=transformer_weights,
                                     postpone_inverse=False)
        else:
            raise ValueError(f'chunk_size invalid {self._chunk_size}')
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
        if getattr(self, "_use_torch", False):
            feats = self._torch_input_map(
                torch.as_tensor(np.asarray(X), dtype=torch.float64))
            pred = self._torch_readout.predict(feats)
            if getattr(self, "_target_1d", False):
                pred = pred.squeeze(-1)
            return pred.numpy()

        hidden_layer_state = self._input_to_node.transform(X)

        return self._regressor.predict(hidden_layer_state)

    def _validate_hyperparameters(self) -> None:
        """Validate the hyperparameters."""
        if not (hasattr(self.input_to_node, "fit")
                and hasattr(self.input_to_node, "fit_transform")
                and hasattr(self.input_to_node, "transform")):
            raise TypeError("All input_to_node should be transformers and"
                            "implement fit and transform '{}' (type {})"
                            "doesn't".format(self.input_to_node,
                                             type(self.input_to_node)))

        if (self._chunk_size is not None
                and (not isinstance(self._chunk_size, int)
                     or self._chunk_size < 0)):
            raise ValueError('Invalid value for chunk_size, got {}'
                             .format(self._chunk_size))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor and"
                            "implement fit and predict '{}' (type {}) "
                            "doesn't".format(self._regressor,
                                             type(self._regressor)))

        if self.solver not in ("closed_form", "gradient"):
            raise ValueError('Invalid value for solver, got {}'
                             .format(self.solver))

        if self.optimizer not in ("adam", "sgd"):
            raise ValueError('Invalid value for optimizer, got {}'
                             .format(self.optimizer))

        if (not isinstance(self.epochs, int)
                or isinstance(self.epochs, bool)
                or self.epochs <= 0):
            raise ValueError('Invalid value for epochs, got {}'
                             .format(self.epochs))

        if (not isinstance(self.learning_rate, (int, float))
                or isinstance(self.learning_rate, bool)
                or self.learning_rate <= 0):
            raise ValueError('Invalid value for learning_rate, got {}'
                             .format(self.learning_rate))

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
    def regressor(self) -> RegressorMixin | IncrementalRegression:
        """
        Return the regressor.

        Returns
        -------
        regressor : RegressorMixin
        """
        return self._regressor

    @regressor.setter
    def regressor(self, regressor: (RegressorMixin |
                                    IncrementalRegression)) -> None:
        """
        Set the regressor.

        Parameters
        ----------
        regressor : RegressorMixin
        """
        self._regressor = regressor

    @property
    def input_to_node(self) -> InputToNode:
        """
        Return the input_to_node Estimator.

        Returns
        -------
        input_to_node : InputToNode
        """
        return self._input_to_node

    @input_to_node.setter
    def input_to_node(self, input_to_node: InputToNode) -> None:
        """
        Set the input_to_node Transformer.

        Parameters
        ----------
        input_to_node : InputToNode
        """
        self._input_to_node = input_to_node

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

        # input_to_node
        hidden_layer_state = self._input_to_node.transform(X)
        return hidden_layer_state

    @property
    def chunk_size(self) -> None | int | np.integer:
        """
        Return the chunk_size, in which X will be chopped.

        Returns
        -------
        chunk_size : Union[int, np.integer]
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: int | None) -> None:
        """
        Set the chunk_size, in which X will be chopped.

        Parameters
        ----------
        chunk_size : Union[int, None]
        """
        self._chunk_size = chunk_size


class ELMClassifier(ClassifierMixin, ELMRegressor):
    """
    Extreme Learning Machine classifier.

    This model optimizes the mean squared error loss function using
    linear regression.

    Parameters
    ----------
    input_to_node : Optional[InputToNode], default=None
        Any ```InputToNode``` object that transforms the inputs.
        If ```None```, a ```pyrcn.base.blocks.InputToNode```
        object is instantiated.
    regressor : Union[IncrementalRegression, RegressorMixin, None],
    default=None
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        If ```None```, a ```pyrcn.linear_model.IncrementalRegression``` object
        is instantiated.
    chunk_size : Optional[int], default=None
         if X.shape[0] > chunk_size, calculate results incrementally
         with partial_fit
    solver : {"closed_form", "gradient"}, default="closed_form"
        Readout training method. ``"closed_form"`` solves the ridge normal
        equations; ``"gradient"`` trains the readout with an optimizer loop.
    optimizer : {"adam", "sgd"}, default="adam"
        Optimizer used when ``solver="gradient"``.
    learning_rate : float, default=1e-3
        Learning rate used when ``solver="gradient"``.
    epochs : int, default=100
        Number of training epochs when ``solver="gradient"``.
    batch_size : Optional[int], default=None
        Mini-batch size when ``solver="gradient"`` (``None`` = full batch).
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired,
        default=None
    """

    def __init__(self, *,
                 input_to_node: InputToNode | None = None,
                 regressor: (IncrementalRegression |
                             RegressorMixin | None) = None,
                 chunk_size: int | None = None, verbose: bool = False,
                 solver: str = "closed_form",
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int | None = None,
                 **kwargs: Any) -> None:
        """Construct the ELMClassifier."""
        super().__init__(input_to_node=input_to_node, regressor=regressor,
                         chunk_size=chunk_size, verbose=verbose,
                         solver=solver, optimizer=optimizer,
                         learning_rate=learning_rate, epochs=epochs,
                         batch_size=batch_size, **kwargs)
        self._encoder = LabelBinarizer()

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: np.ndarray | None = None,
                    postpone_inverse: bool = False,
                    classes: np.ndarray | None = None) -> ELMClassifier:
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
            If the output weights have not been fitted yet, regressor might be
            hinted at postponing inverse calculation. Refer to
            ```IncrementalRegression```
            for details.

        Returns
        -------
        self : returns a trained ELMClassifier model
        """
        validate_data(self, X, y, multi_output=True)

        self._encoder.fit(classes)

        super().partial_fit(X, self._encoder.transform(y),
                            transformer_weights=None,
                            postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: int | np.integer | None = None,
            transformer_weights: np.ndarray | None = None) -> ELMClassifier:
        """
        Fit the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The targets to predict.
        n_jobs : Union[int, np.integer, None], default=None
            The number of jobs to run in parallel. ```-1``` means using all
            processors.
            See the scikit-learn glossary for n_jobs.
        transformer_weights : Optional[np.ndarray], default=None
            ignored

        Returns
        -------
        self : Returns a trained ELMClassifier model.
        """
        validate_data(self, X, y, multi_output=True)
        self._encoder = LabelBinarizer().fit(y)
        super().fit(X, self._encoder.transform(y), n_jobs=n_jobs,
                    transformer_weights=None)
        return self

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
        return self._encoder.inverse_transform(super().predict(X),
                                               threshold=None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability estimated using a trained ```ELMClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted probability estimates.
        """
        predicted_positive = np.clip(
            super().predict(X), a_min=1e-5, a_max=None)
        return np.asarray(predicted_positive)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the log probability estimated using a trained
        ```ELMClassifier```.

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
