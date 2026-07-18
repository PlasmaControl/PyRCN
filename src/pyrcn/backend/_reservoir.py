"""PyTorch reservoir built on ``nn.RNNCell``.

Reuses torch's fused single-step cell op for the ``tanh``/``relu`` activations
and adds only what torch lacks: leaky integration and the extra PyRCN
activations (``logistic``/``identity``/``bounded_relu``). ``weight_ih`` is a
frozen identity (the reservoir input is added directly; input weights belong to
``InputToNode``) and ``spectral_radius`` is folded into ``weight_hh``.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

_FUSED = ("tanh", "relu")
_EXTRA_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "logistic": torch.sigmoid,
    "identity": lambda t: t,
    "bounded_relu": lambda t: t.clamp(0.0, 1.0),
}
_SUPPORTED = _FUSED + tuple(_EXTRA_ACTIVATIONS)


class LeakyESNCell(nn.RNNCell):
    """One leaky reservoir step::

        h' = (1 - leakage) * h + leakage * f(x + spectral_radius * (h @ W))

    ``tanh``/``relu`` reuse ``nn.RNNCell``'s fused op; others use
    ``affine + activation``. ``spectral_radius`` folds into ``weight_hh`` and
    ``weight_ih`` is a frozen identity, so ``x`` is added unchanged.
    """

    def __init__(self, hidden_size: int, spectral_radius: float = 1.0,
                 leakage: float = 1.0, activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        if activation not in _SUPPORTED:
            raise ValueError(
                f"unknown activation {activation!r}; supported: "
                f"{sorted(_SUPPORTED)}")
        nonlinearity = activation if activation in _FUSED else "tanh"
        super().__init__(input_size=hidden_size, hidden_size=hidden_size,
                         bias=False, nonlinearity=nonlinearity, device=device,
                         dtype=dtype)
        self.spectral_radius = float(spectral_radius)
        self.leakage = float(leakage)
        self.activation = activation
        with torch.no_grad():
            self.weight_ih.copy_(torch.eye(
                hidden_size, device=device, dtype=self.weight_ih.dtype))
        for p in self.parameters():
            p.requires_grad_(False)

    def set_recurrent_weights(self, weights: object) -> None:
        """Load the ``(hidden_size, hidden_size)`` recurrent matrix.

        Stored transposed and pre-scaled by ``spectral_radius`` so both paths
        yield ``spectral_radius * (h @ weights)``.
        """
        w = torch.as_tensor(
            weights, dtype=self.weight_hh.dtype, device=self.weight_hh.device)
        if w.shape != (self.hidden_size, self.hidden_size):
            raise ValueError(
                f"expected weights of shape "
                f"{(self.hidden_size, self.hidden_size)}, got "
                f"{tuple(w.shape)}")
        with torch.no_grad():
            self.weight_hh.copy_(self.spectral_radius * w.T)

    def forward(self, input: torch.Tensor,
                hx: torch.Tensor | None = None) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(input.shape[0], self.hidden_size,
                             dtype=input.dtype, device=input.device)
        if self.activation in _FUSED:
            updated = super().forward(input, hx)            # fused cell op
        else:
            pre = input + F.linear(hx, self.weight_hh)      # weight_ih is I
            updated = _EXTRA_ACTIVATIONS[self.activation](pre)
        return (1.0 - self.leakage) * hx + self.leakage * updated


class Reservoir(nn.Module):
    """Run a :class:`LeakyESNCell` over batched (padded) sequences.

    ``forward`` takes ``x`` of shape ``(n_sequences, length, hidden_size)`` and
    an optional ``initial_state`` ``(n_sequences, hidden_size)`` (zeros by
    default), returning ``(states, final_state)``. With ``bidirectional=True``
    the same cell is run forward and over the time-reversed input (shared
    weights, as ``NodeToNode`` does); the two state sequences are concatenated
    on the feature axis, so ``states`` has ``2 * hidden_size`` features.
    """

    def __init__(self, hidden_size: int, spectral_radius: float = 1.0,
                 leakage: float = 1.0, activation: str = "tanh",
                 bidirectional: bool = False,
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.cell = LeakyESNCell(
            hidden_size, spectral_radius=spectral_radius, leakage=leakage,
            activation=activation, device=device, dtype=dtype)

    def set_recurrent_weights(self, weights: object) -> None:
        """Load the recurrent weight matrix into the cell."""
        self.cell.set_recurrent_weights(weights)

    def _zeros(self, n_sequences: int, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            n_sequences, self.hidden_size, dtype=x.dtype, device=x.device)

    def _run(self, x: torch.Tensor,
             h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h

    def forward(self, x: torch.Tensor,
                initial_state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        n_sequences = x.shape[0]
        if not self.bidirectional:
            if initial_state is None:
                initial_state = self._zeros(n_sequences, x)
            return self._run(x, initial_state)
        if initial_state is not None:
            raise ValueError(
                "initial_state is not supported with bidirectional=True")
        states_fw, final_fw = self._run(x, self._zeros(n_sequences, x))
        reversed_states, final_bw = self._run(
            torch.flip(x, dims=[1]), self._zeros(n_sequences, x))
        states = torch.cat(
            [states_fw, torch.flip(reversed_states, dims=[1])], dim=-1)
        return states, torch.cat([final_fw, final_bw], dim=-1)
