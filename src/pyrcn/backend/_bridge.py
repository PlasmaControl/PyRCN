"""Bridge from the scikit-learn config blocks to the torch backend modules.

Given a *fitted* PyRCN block (``InputToNode`` / ``NodeToNode`` family) or a
readout config, build the equivalent torch backend module with the block's
weights injected, so the torch compute reproduces the block's NumPy
``transform`` within floating-point precision. This is what lets the estimators
run their standard configuration entirely on the torch backend (the fast path)
while still falling back to the legacy NumPy path for arbitrary sub-estimators
(e.g. a ``FeatureUnion`` input stage or an external ``Ridge`` readout).

Fast-path membership is decided by *exact* type: only blocks whose
``transform`` is the standard affine-plus-activation (input) or the standard
leaky/Euler recurrence (reservoir) qualify, so subclasses with bespoke
transforms (e.g. ``BatchIntrinsicPlasticity``) correctly take the fallback.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

import numpy as np
import torch

from ..base.blocks import (EulerNodeToNode, HebbianNodeToNode, InputToNode,
                           NodeToNode, PredefinedWeightsInputToNode,
                           PredefinedWeightsNodeToNode)
from ..linear_model import IncrementalRegression
from ._input import InputFeatureMap
from ._readout import IncrementalRidge
from ._reservoir import EulerReservoir, Reservoir

_INPUT_TYPES = (InputToNode, PredefinedWeightsInputToNode)
_LEAKY_TYPES = (NodeToNode, PredefinedWeightsNodeToNode, HebbianNodeToNode)
_EULER_TYPES = (EulerNodeToNode,)


def _dense(weights: object) -> np.ndarray:
    if hasattr(weights, "toarray"):
        return weights.toarray()            # type: ignore[union-attr]
    return np.asarray(weights)


def input_is_backable(block: object) -> bool:
    """True if ``block``'s transform is the standard affine + activation."""
    return type(block) in _INPUT_TYPES


def node_is_backable(block: object) -> bool:
    """True if ``block``'s transform is the standard reservoir recurrence."""
    return type(block) in _LEAKY_TYPES + _EULER_TYPES


def regressor_is_backable(regressor: object) -> bool:
    """True if ``regressor`` is the ridge readout with normalize off."""
    return (type(regressor) is IncrementalRegression
            and not regressor.normalize)


def build_input_map(block: InputToNode, *, dtype: torch.dtype | None = None,
                    device: torch.device | str | int | None = None
                    ) -> InputFeatureMap:
    """Torch ``InputFeatureMap`` reproducing a fitted ``InputToNode``."""
    weights = _dense(block.input_weights)
    bias = _dense(block.bias_weights)
    feature_map = InputFeatureMap(
        weights.shape[0], weights.shape[1],
        input_scaling=block.input_scaling, input_shift=block.input_shift,
        bias_scaling=block.bias_scaling, bias_shift=block.bias_shift,
        activation=block.input_activation, dtype=dtype, device=device)
    feature_map.set_input_weights(weights, bias)
    return feature_map


def build_reservoir(block: NodeToNode, *, dtype: torch.dtype | None = None,
                    device: torch.device | str | int | None = None
                    ) -> Reservoir | EulerReservoir:
    """Torch reservoir module reproducing a fitted ``NodeToNode``."""
    weights = _dense(block.recurrent_weights)
    hidden_size = weights.shape[0]
    reservoir: Reservoir | EulerReservoir
    if isinstance(block, EulerNodeToNode):
        reservoir = EulerReservoir(
            hidden_size, recurrent_scaling=block.recurrent_scaling,
            gamma=block.gamma, epsilon=block.epsilon,
            activation=block.reservoir_activation, dtype=dtype, device=device)
    else:
        reservoir = Reservoir(
            hidden_size, spectral_radius=block.spectral_radius,
            leakage=block.leakage, activation=block.reservoir_activation,
            bidirectional=block.bidirectional, dtype=dtype, device=device)
    reservoir.set_recurrent_weights(weights)
    return reservoir


def build_readout(regressor: IncrementalRegression, *,
                  dtype: torch.dtype | None = None,
                  device: torch.device | str | int | None = None
                  ) -> IncrementalRidge:
    """Torch closed-form ridge readout from an ``IncrementalRegression``."""
    return IncrementalRidge(
        alpha=regressor.alpha, fit_intercept=regressor.fit_intercept,
        dtype=dtype, device=device)
