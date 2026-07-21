"""The :mod:`echo_state_network` contains an ESNRegressor and ESNClassifier."""

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

import sys
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.base import (BaseEstimator, ClassifierMixin, MultiOutputMixin,
                          RegressorMixin, is_regressor)
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import validate_data

from ..nn._bridge import (
    build_input_map, build_readout, build_reservoir, input_is_backable,
    node_is_backable, regressor_is_backable)
from ..nn._input import InputFeatureMap
from ..nn._readout import IncrementalRidge, LinearReadout
from ..nn._reservoir import EulerReservoir, Reservoir
from ..nn._training import (LOSSES, OPTIMIZERS, torch_generator,
                            train_readout)
from ..base.blocks import InputToNode, NodeToNode
from ..linear_model import IncrementalRegression
from ..projection import MatrixToValueProjection
from ..util import concatenate_sequences


class ESNRegressor(RegressorMixin, MultiOutputMixin, BaseEstimator):
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
    regressor : Union[IncrementalRegression, RegressorMixin, None],
    default=None
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
    washout : int, default=0
        Number of initial reservoir states (and matching targets) to drop
        per sequence when fitting the readout, discarding the start
        transient. Training-only; requires the torch backend.
    solver : {"closed_form", "gradient"}, default="closed_form"
        Readout training method. ``"closed_form"`` solves the ridge normal
        equations (requires a fixed reservoir); ``"gradient"`` trains the
        readout with an optimizer loop and enables the ``trainable_*`` flags.
    optimizer : {"adam", "adamw", "sgd", "rmsprop", "adagrad"}, default="adam"
        Optimizer used when ``solver="gradient"``.
    learning_rate : float, default=1e-3
        Learning rate used when ``solver="gradient"``.
    epochs : int, default=100
        Number of training epochs when ``solver="gradient"``.
    batch_size : Optional[int], default=None
        Mini-batch size when ``solver="gradient"`` (``None`` means full
        batch). With a trainable reservoir/input, batches are over sequences.
    loss : {"mse", "mae", "huber"}, default="mse"
        Loss used when ``solver="gradient"``.
    trainable_reservoir : bool, default=False
        If True (requires ``solver="gradient"``), train the recurrent
        reservoir weights by backpropagation through the recurrence.
    trainable_input : bool, default=False
        If True (requires ``solver="gradient"``), train the input
        feature-map weights. Combine with ``trainable_reservoir`` for a
        fully trainable RNN.
    verbose : bool = False
        Verbosity output
    kwargs : Any
        keyword arguments passed to the subestimators if this is desired,
        default=None
    """

    def __init__(self, *,
                 input_to_node: InputToNode | None = None,
                 node_to_node: NodeToNode | None = None,
                 regressor: (IncrementalRegression |
                             RegressorMixin | None) = None,
                 requires_sequence: Literal["auto"] | bool = "auto",
                 decision_strategy: Literal["winner_takes_all", "median",
                                            "last_value"] = "winner_takes_all",
                 washout: int = 0,
                 verbose: bool = True,
                 solver: str = "closed_form",
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int | None = None,
                 trainable_reservoir: bool = False,
                 trainable_input: bool = False,
                 loss: str = "mse",
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
        self.washout = washout
        self.verbose = verbose
        self.decision_strategy = decision_strategy
        self.solver = solver
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.trainable_reservoir = trainable_reservoir
        self.trainable_input = trainable_input
        self.loss = loss
        self._use_torch: bool = False
        self._target_1d: bool = False
        self._torch_input_map: InputFeatureMap
        self._torch_reservoir: Reservoir | EulerReservoir
        self._torch_readout: IncrementalRidge | LinearReadout

    def get_params(self, deep: bool = True) -> dict:
        """Get all parameters of the ESNRegressor."""
        if deep:
            return {**self.input_to_node.get_params(),
                    **self.node_to_node.get_params(),
                    **{"alpha": self.regressor.get_params()["alpha"]}}
        else:
            return {"input_to_node": self.input_to_node,
                    "node_to_node": self.node_to_node,
                    "regressor": self.regressor,
                    "requires_sequence": self._requires_sequence,
                    "washout": self.washout,
                    "solver": self.solver,
                    "optimizer": self.optimizer,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "trainable_reservoir": self.trainable_reservoir,
                    "trainable_input": self.trainable_input,
                    "loss": self.loss}

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
                             "because X has {} and y has {} dimensions."
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
                    transformer_weights: None | np.ndarray = None,
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
        self._use_torch = False
        validate_data(self, X=X, y=y, multi_output=True)

        # input_to_node
        try:
            hidden_layer_state = self._input_to_node.transform(X)
        except NotFittedError as e:
            if self.verbose:
                print(f'input_to_node has not been fitted yet: {e}')
            hidden_layer_state = self._input_to_node.fit_transform(X)

        # node_to_node
        try:
            hidden_layer_state = self._node_to_node.transform(
                hidden_layer_state)
        except NotFittedError as e:
            if self.verbose:
                print(f'node_to_node has not been fitted yet: {e}')
            hidden_layer_state = self._node_to_node.fit_transform(
                hidden_layer_state)

        # regression
        if not hasattr(self._regressor, 'partial_fit') and postpone_inverse:
            raise TypeError(
                "Regressor has no attribute partial_fit, "
                f"got {self._regressor}")
        elif not hasattr(self._regressor, 'partial_fit') \
                and not postpone_inverse:
            self._regressor.fit(hidden_layer_state, y)
        elif hasattr(self._regressor, 'partial_fit'):
            self._regressor.partial_fit(
                hidden_layer_state, y, postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: int | np.integer | None = None,
            transformer_weights: np.ndarray | None = None) -> ESNRegressor:
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
            See the scikit-learn glossary for n_jobs.
        transformer_weights : Optional[np.ndarray] = None
            ignored

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        self._validate_hyperparameters()
        if self.requires_sequence == "auto":
            self._check_if_sequence(X, y)
        self._target_1d = (np.asarray(y).ndim == 1)
        backable = (
            input_is_backable(self._input_to_node)
            and node_is_backable(self._node_to_node)
            and regressor_is_backable(self._regressor))
        if self.solver == "gradient":
            if not backable:
                raise NotImplementedError(
                    "the gradient solver requires torch-backable "
                    "input_to_node, node_to_node and an "
                    "IncrementalRegression readout")
            if self.requires_sequence:
                X, y, sequence_ranges = concatenate_sequences(X, y)
                self._input_to_node.fit(X)
                self._fit_node_to_node(X, backable)
            else:
                validate_data(self, X, y, multi_output=True)
                self._input_to_node.fit(X)
                self._fit_node_to_node(X, backable)
            self._build_torch_backend(torch.float64)
            self._use_torch = True
            ranges = sequence_ranges if self.requires_sequence else None
            if self.trainable_reservoir or self.trainable_input:
                return self._torch_trainable_fit(X, y, ranges)
            return self._torch_gradient_fit(X, y, ranges)
        self._use_torch = backable
        if self.washout > 0 and not self._use_torch:
            raise NotImplementedError(
                "washout > 0 requires the torch backend (native "
                "input_to_node / node_to_node / regressor)")
        if self.requires_sequence:
            X, y, sequence_ranges = concatenate_sequences(X, y)
            self._input_to_node.fit(X)
            self._fit_node_to_node(X, backable)
        else:
            validate_data(self, X, y, multi_output=True)
            self._input_to_node.fit(X)
            self._fit_node_to_node(X, backable)
        if self._use_torch:
            self._build_torch_backend(torch.float64)
            if self.requires_sequence:
                return self._torch_sequence_fit(X, y, sequence_ranges)
            states, _ = self._torch_states(X)
            states = states[self.washout:]
            ys = torch.as_tensor(np.asarray(y), dtype=torch.float64)
            cast(IncrementalRidge, self._torch_readout).fit(
                states, ys[self.washout:])
            return self
        if self.requires_sequence:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)
        else:
            return self.partial_fit(X, y, postpone_inverse=False)

    def _fit_node_to_node(self, X: np.ndarray, backable: bool) -> None:
        """Fit ``node_to_node`` given the already-fitted ``input_to_node``.

        ``NodeToNode.fit`` reads only the *column count* of its argument (plus
        ``hidden_layer_size`` + ``random_state``); on the torch-backable path
        the reservoir recomputes the input map itself, so the full
        ``input_to_node.transform(X)`` fed here would be computed and then
        discarded. Pass a cheap shape-only zero array of the correct width
        (``InputToNode.hidden_layer_size``) instead. On the numpy-fallback
        path ``input_to_node`` may be an arbitrary estimator (e.g. a
        ``FeatureUnion`` with a different width) whose transform is genuinely
        used, so keep the real transform there.
        """
        if backable:
            width = int(self._input_to_node.hidden_layer_size)
            self._node_to_node.fit(np.zeros((X.shape[0], width)))
        else:
            self._node_to_node.fit(self._input_to_node.transform(X))

    def _build_torch_backend(self, dtype: torch.dtype) -> None:
        """Build the torch backend modules from the fitted blocks."""
        self._torch_input_map = build_input_map(
            self._input_to_node, dtype=dtype)
        self._torch_reservoir = build_reservoir(
            self._node_to_node, dtype=dtype)
        self._torch_readout = build_readout(self._regressor, dtype=dtype)

    def _torch_states(self, seq: np.ndarray,
                      initial_state: np.ndarray | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(states (L, hidden*dir), final_state (hidden*dir,))``.

        ``initial_state`` seeds the reservoir (zeros by default).
        """
        Z = self._torch_input_map(
            torch.as_tensor(np.asarray(seq), dtype=torch.float64))
        if initial_state is None:
            states, final = self._torch_reservoir(Z.unsqueeze(0))
        else:
            init = torch.as_tensor(
                np.asarray(initial_state), dtype=torch.float64).reshape(1, -1)
            states, final = self._torch_reservoir(Z.unsqueeze(0), init)
        return states.squeeze(0), final.squeeze(0)

    def _torch_sequence_fit(self, X_cat: np.ndarray, y_cat: np.ndarray,
                            sequence_ranges: np.ndarray) -> ESNRegressor:
        """Accumulate the readout over sequences with a fresh zero state.

        The first ``washout`` states (and matching targets) of each sequence
        are dropped so the initial transient does not train the readout.
        """
        n_seq = len(sequence_ranges)
        for i, (start, stop) in enumerate(sequence_ranges):
            states, _ = self._torch_states(X_cat[start:stop])
            ys = torch.as_tensor(
                np.asarray(y_cat[start:stop]), dtype=torch.float64)
            states, ys = states[self.washout:], ys[self.washout:]
            cast(IncrementalRidge, self._torch_readout).partial_fit(
                states, ys, reset=(i == 0),
                postpone_inverse=(i < n_seq - 1))
        return self

    def _torch_gradient_fit(self, X: np.ndarray, y: np.ndarray,
                            sequence_ranges: (np.ndarray | None)
                            ) -> ESNRegressor:
        """Train a fresh ``LinearReadout`` on the fixed reservoir states.

        States are computed once through the frozen reservoir, dropping the
        first ``washout`` states (and matching targets) per sequence, then a
        ``LinearReadout`` is trained with an optimizer loop.
        """
        if sequence_ranges is not None:
            states_list = []
            y_list = []
            for start, stop in sequence_ranges:
                states, _ = self._torch_states(X[start:stop])
                ys = torch.as_tensor(
                    np.asarray(y[start:stop]), dtype=torch.float64)
                states_list.append(states[self.washout:])
                y_list.append(ys[self.washout:])
            all_states = torch.cat(states_list, dim=0)
            all_y = torch.cat(y_list, dim=0)
        else:
            states, _ = self._torch_states(X)
            all_states = states[self.washout:]
            all_y = torch.as_tensor(
                np.asarray(y), dtype=torch.float64)[self.washout:]
        y2 = all_y.reshape(all_states.shape[0], -1)
        gen = torch_generator(self._input_to_node.random_state)
        readout = LinearReadout(
            all_states.shape[1], y2.shape[1],
            fit_intercept=self._regressor.fit_intercept, generator=gen,
            dtype=torch.float64)
        train_readout(
            readout, all_states, y2, optimizer=self.optimizer,
            learning_rate=self.learning_rate, epochs=self.epochs,
            batch_size=self.batch_size, loss=self.loss,
            weight_decay=self._regressor.alpha, generator=gen)
        self._torch_readout = readout
        return self

    def _torch_trainable_fit(self, X: np.ndarray, y: np.ndarray,
                             sequence_ranges: (np.ndarray | None)
                             ) -> ESNRegressor:
        """Jointly train input/reservoir weights and the readout (BPTT).

        Whichever of the input feature map and the reservoir is marked
        trainable is optimized together with the readout by backpropagating
        through the recurrence; each epoch recomputes the states. A fixed
        input feature map is applied once and detached (cheaper); a trainable
        one is recomputed each step so gradients reach its weights.
        Mini-batching (``batch_size``) is over sequences; ``None`` is
        full-batch. In non-sequence mode there is a single sequence, so it is
        always full-BPTT over that sequence.
        """
        dtype = torch.float64
        train_input = self.trainable_input
        if train_input:
            self._torch_input_map.set_input_trainable(True)
        if self.trainable_reservoir:
            self._torch_reservoir.set_recurrent_trainable(True)
        if sequence_ranges is not None:
            segments = [(X[a:b], y[a:b]) for a, b in sequence_ranges]
        else:
            segments = [(X, y)]
        inputs = [torch.as_tensor(np.asarray(xs), dtype=dtype)
                  for xs, _ in segments]
        targets = [torch.as_tensor(np.asarray(ys), dtype=dtype)[self.washout:]
                   for _, ys in segments]
        fixed_feats = (None if train_input
                       else [self._torch_input_map(x).detach()
                             for x in inputs])
        n_targets = 1 if targets[0].ndim == 1 else targets[0].shape[1]
        gen = torch_generator(self._input_to_node.random_state)
        with torch.no_grad():
            probe_feats = (self._torch_input_map(inputs[0])
                           if fixed_feats is None else fixed_feats[0])
            probe, _ = self._torch_reservoir(probe_feats.unsqueeze(0))
        readout = LinearReadout(
            probe.shape[-1], n_targets,
            fit_intercept=self._regressor.fit_intercept, generator=gen,
            dtype=dtype)
        params: list = []
        if train_input:
            params += [p for p in self._torch_input_map.parameters()
                       if p.requires_grad]
        if self.trainable_reservoir:
            params += [p for p in self._torch_reservoir.parameters()
                       if p.requires_grad]
        params += list(readout.parameters())
        optimizer = OPTIMIZERS[self.optimizer](
            params, lr=self.learning_rate, weight_decay=self._regressor.alpha)
        loss_fn = LOSSES[self.loss]()
        n_seq = len(inputs)
        step = n_seq if self.batch_size is None else min(
            int(self.batch_size), n_seq)
        readout.train()
        for _ in range(int(self.epochs)):
            order = torch.randperm(n_seq, generator=gen).tolist()
            for begin in range(0, n_seq, step):
                batch = order[begin:begin + step]
                optimizer.zero_grad()
                states_parts = []
                target_parts = []
                for j in batch:
                    feats = (self._torch_input_map(inputs[j])
                             if fixed_feats is None else fixed_feats[j])
                    states, _ = self._torch_reservoir(feats.unsqueeze(0))
                    states_parts.append(states.squeeze(0)[self.washout:])
                    target_parts.append(targets[j])
                predicted = readout(torch.cat(states_parts, dim=0))
                expected = torch.cat(target_parts, dim=0).reshape(
                    predicted.shape[0], -1)
                loss = loss_fn(predicted, expected)
                loss.backward()
                optimizer.step()
        readout.eval()
        self._torch_readout = readout
        return self

    def _sequence_fit(self, X: np.ndarray, y: np.ndarray,
                      sequence_ranges: np.ndarray,
                      n_jobs: (int | np.integer |
                               None) = None) -> ESNRegressor:
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
            See the scikit-learn glossary for n_jobs.

        Returns
        -------
        self : Returns a trained ESNRegressor model.
        """
        # n_jobs is accepted for API compatibility but ignored: the torch
        # fast path batches this, and the numpy fallback runs serially.
        for idx in sequence_ranges[:-1]:
            ESNRegressor.partial_fit(self, X[idx[0]:idx[1], ...],
                                     y[idx[0]:idx[1], ...],
                                     postpone_inverse=True)

        # last sequence, calculate inverse and bias
        ESNRegressor.partial_fit(self, X=X[sequence_ranges[-1][0]:, ...],
                                 y=y[sequence_ranges[-1][0]:, ...],
                                 postpone_inverse=False)
        return self

    def predict(self, X: np.ndarray, initial_state: np.ndarray | None = None,
                return_state: bool = False
                ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict the targets using the trained ```ESNRegressor```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        initial_state : ndarray of shape (hidden_layer_size,), default=None
            Reservoir state to start from (zeros by default). Applied to each
            sequence in sequence mode. Requires the torch backend.
        return_state : bool, default=False
            If True, also return the final reservoir state(s). Requires the
            torch backend.

        Returns
        -------
        y : ndarray of (n_samples,) or (n_samples, n_targets)
            The predicted targets. If ``return_state`` is True, a tuple
            ``(y, final_state)`` is returned instead.
        """
        if self._input_to_node is None or self._regressor is None:
            raise NotFittedError(self)

        if getattr(self, "_use_torch", False):
            squeeze = getattr(self, "_target_1d", False)
            # No grad in inference: a trainable reservoir has requires_grad
            # weights, so its forward would otherwise track gradients.
            with torch.no_grad():
                if self.requires_sequence is False:
                    states, final = self._torch_states(X, initial_state)
                    pred = self._torch_readout.predict(states)
                    if squeeze:
                        pred = pred.squeeze(-1)
                    y = pred.numpy()
                    return (y, final.numpy()) if return_state else y
                y = np.empty(shape=X.shape, dtype=object)
                finals = np.empty(shape=X.shape, dtype=object)
                for k, seq in enumerate(X):
                    states, final = self._torch_states(seq, initial_state)
                    pred = self._torch_readout.predict(states)
                    if squeeze:
                        pred = pred.squeeze(-1)
                    y[k] = pred.numpy()
                    finals[k] = final.numpy()
                return (y, finals) if return_state else y

        if initial_state is not None or return_state:
            raise NotImplementedError(
                "initial_state / return_state require the torch backend "
                "(native input_to_node / node_to_node / regressor)")

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
                            "implement fit and transform '{}' (type {}) "
                            "doesn't".format(self.input_to_node,
                                             type(self.input_to_node)))

        if not (hasattr(self.node_to_node, "fit")
                and hasattr(self.node_to_node, "fit_transform")
                and hasattr(self.node_to_node, "transform")):
            raise TypeError("All node_to_node should be transformers and"
                            "implement fit and transform '{}' (type {}) "
                            "doesn't".format(self.node_to_node,
                                             type(self.node_to_node)))

        if (self._requires_sequence != "auto"
                and not isinstance(self._requires_sequence, bool)):
            raise ValueError('Invalid value for requires_sequence, got {}'
                             .format(self._requires_sequence))

        if not isinstance(self.washout, int) or self.washout < 0:
            raise ValueError('Invalid value for washout, got {}'
                             .format(self.washout))

        if not is_regressor(self._regressor):
            raise TypeError("The last step should be a regressor and "
                            "implement fit and predict '{}' (type {})"
                            "doesn't".format(self._regressor,
                                             type(self._regressor)))

        if self.solver not in ("closed_form", "gradient"):
            raise ValueError('Invalid value for solver, got {}'
                             .format(self.solver))

        if self.optimizer not in OPTIMIZERS:
            raise ValueError('Invalid value for optimizer, got {}'
                             .format(self.optimizer))

        if self.loss not in LOSSES:
            raise ValueError('Invalid value for loss, got {}'
                             .format(self.loss))

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

        if ((self.trainable_reservoir or self.trainable_input)
                and self.solver != "gradient"):
            raise ValueError(
                "trainable_reservoir / trainable_input require "
                "solver='gradient' (trainable weights cannot use the "
                "closed-form solver)")

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
    def requires_sequence(self) -> Literal["auto"] | bool:
        """
        Return the requires_sequence parameter.

        Returns
        -------
        requires_sequence : Union[Literal["auto"], bool]
        """
        return self._requires_sequence

    @requires_sequence.setter
    def requires_sequence(self,
                          requires_sequence: Literal["auto"] | bool)\
            -> None:
        """
        Set the requires_sequence parameter.

        Parameters
        ----------
        requires_sequence : Union[Literal["auto"], bool]

        """
        self._requires_sequence = requires_sequence


class ESNClassifier(ClassifierMixin, ESNRegressor):
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
    regressor : Union[IncrementalRegression, RegressorMixin, None],
    default=None
        Regressor object such as derived from ``RegressorMixin``. This
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
    washout : int, default=0
        Number of initial reservoir states (and matching targets) to drop
        per sequence when fitting the readout, discarding the start
        transient. Training-only; requires the torch backend.
    solver : {"closed_form", "gradient"}, default="closed_form"
        Readout training method. ``"closed_form"`` solves the ridge normal
        equations (requires a fixed reservoir); ``"gradient"`` trains the
        readout with an optimizer loop and enables the ``trainable_*`` flags.
    optimizer : {"adam", "adamw", "sgd", "rmsprop", "adagrad"}, default="adam"
        Optimizer used when ``solver="gradient"``.
    learning_rate : float, default=1e-3
        Learning rate used when ``solver="gradient"``.
    epochs : int, default=100
        Number of training epochs when ``solver="gradient"``.
    batch_size : Optional[int], default=None
        Mini-batch size when ``solver="gradient"`` (``None`` means full
        batch). With a trainable reservoir/input, batches are over sequences.
    loss : {"mse", "mae", "huber"}, default="mse"
        Loss used when ``solver="gradient"``.
    trainable_reservoir : bool, default=False
        If True (requires ``solver="gradient"``), train the recurrent
        reservoir weights by backpropagation through the recurrence.
    trainable_input : bool, default=False
        If True (requires ``solver="gradient"``), train the input
        feature-map weights. Combine with ``trainable_reservoir`` for a
        fully trainable RNN.
    verbose : bool = False
        Verbosity output
    kwargs : Any, default = None
        keyword arguments passed to the subestimators if this is desired.
    """

    def __init__(self, *,
                 input_to_node: InputToNode | None = None,
                 node_to_node: NodeToNode | None = None,
                 regressor: (IncrementalRegression |
                             RegressorMixin | None) = None,
                 requires_sequence: Literal["auto"] | bool = "auto",
                 decision_strategy: Literal["winner_takes_all", "median",
                                            "last_value"] = "winner_takes_all",
                 washout: int = 0,
                 verbose: bool = False,
                 solver: str = "closed_form",
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int | None = None,
                 trainable_reservoir: bool = False,
                 trainable_input: bool = False,
                 loss: str = "mse",
                 **kwargs: Any) -> None:
        """Construct the ESNClassifier."""
        super().__init__(input_to_node=input_to_node,
                         node_to_node=node_to_node, regressor=regressor,
                         requires_sequence=requires_sequence, washout=washout,
                         verbose=verbose, solver=solver, optimizer=optimizer,
                         learning_rate=learning_rate, epochs=epochs,
                         batch_size=batch_size,
                         trainable_reservoir=trainable_reservoir,
                         trainable_input=trainable_input, loss=loss, **kwargs)
        self._decision_strategy = decision_strategy
        self._encoder = LabelBinarizer()
        self._sequence_to_value = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    transformer_weights: np.ndarray | None = None,
                    postpone_inverse: bool = False,
                    classes: np.ndarray | None = None) -> ESNClassifier:
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
        validate_data(self, X, y, multi_output=True)
        self._encoder.fit(classes)
        super().partial_fit(X, self._encoder.transform(y),
                            transformer_weights=None,
                            postpone_inverse=postpone_inverse)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: int | np.integer | None = None,
            transformer_weights: (None |
                                  np.ndarray) = None) -> ESNClassifier:
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
            See the scikit-learn glossary for n_jobs.
        transformer_weights : ignored

        Returns
        -------
        self : Returns a trained ESNClassifier model.
        """
        self._validate_hyperparameters()
        if self.requires_sequence == "auto":
            self._check_if_sequence(X, y)
        backable = (
            input_is_backable(self._input_to_node)
            and node_is_backable(self._node_to_node)
            and regressor_is_backable(self._regressor))
        if self.solver == "gradient":
            if not backable:
                raise NotImplementedError(
                    "the gradient solver requires torch-backable "
                    "input_to_node, node_to_node and an "
                    "IncrementalRegression readout")
        else:
            self._use_torch = backable
            if self.washout > 0 and not self._use_torch:
                raise NotImplementedError(
                    "washout > 0 requires the torch backend (native "
                    "input_to_node / node_to_node / regressor)")
        if self.requires_sequence:
            self._check_if_sequence_to_value(X, y)
            X, y, sequence_ranges = concatenate_sequences(
                X, y, sequence_to_value=self._sequence_to_value)
            self._input_to_node.fit(X)
            self._fit_node_to_node(X, backable)
        else:
            validate_data(self, X, y, multi_output=True)
            self._input_to_node.fit(X)
            self._fit_node_to_node(X, backable)
        self._encoder = LabelBinarizer().fit(y)
        y = self._encoder.transform(y)
        self._target_1d = (np.asarray(y).ndim == 1)
        if self.solver == "gradient":
            self._build_torch_backend(torch.float64)
            self._use_torch = True
            ranges = sequence_ranges if self.requires_sequence else None
            if self.trainable_reservoir or self.trainable_input:
                return self._torch_trainable_fit(X, y, ranges)
            return self._torch_gradient_fit(X, y, ranges)
        if self._use_torch:
            self._build_torch_backend(torch.float64)
            if self.requires_sequence:
                return self._torch_sequence_fit(X, y, sequence_ranges)
            states, _ = self._torch_states(X)
            states = states[self.washout:]
            ys = torch.as_tensor(np.asarray(y), dtype=torch.float64)
            cast(IncrementalRidge, self._torch_readout).fit(
                states, ys[self.washout:])
            return self
        if self.requires_sequence:
            return self._sequence_fit(X, y, sequence_ranges, n_jobs)
        else:
            super().partial_fit(X, y)
            return self

    def predict(self, X: np.ndarray, initial_state: np.ndarray | None = None,
                return_state: bool = False
                ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict the classes using the trained ```ESNClassifier```.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        initial_state : ndarray of shape (hidden_layer_size,), default=None
            Reservoir state to start from (zeros by default). Requires the
            torch backend.
        return_state : bool, default=False
            Not supported for classifiers.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        if return_state:
            raise NotImplementedError(
                "return_state is not supported for classifiers")
        y = cast(np.ndarray, super().predict(X, initial_state=initial_state))
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
            return self._encoder.inverse_transform(y, threshold=None)

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
        y = cast(np.ndarray, super().predict(X))
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
