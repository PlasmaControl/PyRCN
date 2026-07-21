"""PyTorch input feature map built on ``nn.Linear``.

This is the torch companion of ``InputToNode``: an affine input projection
(input weights + bias) with PyRCN's scaling/shift and an activation. It is the
feature map for ELMs and the input stage for ESNs. ``input_scaling`` is folded
into ``weight`` and the shifts/``bias_scaling`` into ``bias``, so ``forward``
reduces to ``activation(super().forward(x))``.
"""

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

import torch
import torch.nn as nn

from ._activations import ACTIVATIONS, SUPPORTED


class InputFeatureMap(nn.Linear):
    """Affine input projection + activation, matching ``InputToNode``::

        s = f(input_scaling * (x @ W_in) + input_shift
              + bias_scaling * bias + bias_shift)

    Weights are assigned via :meth:`set_input_weights`; all parameters are
    frozen (``requires_grad=False``).
    """

    def __init__(self, in_features: int, out_features: int, *,
                 input_scaling: float = 1.0, input_shift: float = 0.0,
                 bias_scaling: float = 1.0, bias_shift: float = 0.0,
                 activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        if activation not in SUPPORTED:
            raise ValueError(
                f"unknown activation {activation!r}; supported: "
                f"{sorted(SUPPORTED)}")
        super().__init__(in_features, out_features, bias=True, device=device,
                         dtype=dtype)
        self.input_scaling = float(input_scaling)
        self.input_shift = float(input_shift)
        self.bias_scaling = float(bias_scaling)
        self.bias_shift = float(bias_shift)
        self.activation = activation
        for p in self.parameters():
            p.requires_grad_(False)

    def set_input_trainable(self, trainable: bool = True) -> None:
        """Enable/disable gradient training of the input weights and bias."""
        for p in self.parameters():
            p.requires_grad_(trainable)

    def set_input_weights(self, input_weights: object,
                          bias_weights: object) -> None:
        """Load input weights ``(in_features, out_features)`` and bias.

        ``input_scaling`` is folded into ``weight`` and the shifts /
        ``bias_scaling`` into ``bias`` so a plain ``nn.Linear`` reproduces
        PyRCN's affine.
        """
        w = torch.as_tensor(
            input_weights, dtype=self.weight.dtype, device=self.weight.device)
        b = torch.as_tensor(
            bias_weights, dtype=self.weight.dtype,
            device=self.weight.device).reshape(-1)
        with torch.no_grad():
            self.weight.copy_(self.input_scaling * w.T)
            self.bias.copy_(
                self.bias_scaling * b + (self.input_shift + self.bias_shift))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ACTIVATIONS[self.activation](super().forward(input))
