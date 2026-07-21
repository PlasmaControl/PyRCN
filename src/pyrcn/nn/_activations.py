"""Activation functions shared by the torch backend modules."""

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

from collections.abc import Callable

import torch

#: Activations that ``nn.RNNCell`` provides as a fused single-step op.
FUSED = ("tanh", "relu")

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "logistic": torch.sigmoid,
    "identity": lambda t: t,
    "bounded_relu": lambda t: t.clamp(0.0, 1.0),
}

SUPPORTED = tuple(ACTIVATIONS)
