"""PyTorch reservoir module (batched leaky-integrator recurrence)."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "logistic": torch.sigmoid,
    "identity": lambda t: t,
    "bounded_relu": lambda t: t.clamp(0.0, 1.0),
}


class Reservoir(nn.Module):
    """Leaky-integrator reservoir as a batched PyTorch module.

    Computes, per time step, the same recurrence as the legacy NumPy
    ``NodeToNode``::

        h_t = (1 - leakage) * h_{t-1}
              + leakage * f(x_t + spectral_radius * (h_{t-1} @ W_hh))

    The recurrent weights ``W_hh`` are a dense ``Parameter`` with
    ``requires_grad=False`` by default; they are assigned externally (e.g. from
    any of PyRCN's initialization strategies) via ``set_recurrent_weights``.

    Parameters
    ----------
    hidden_size : int
        Reservoir size (and input feature size at each step).
    spectral_radius : float, default=1.0
        Scales the recurrent contribution.
    leakage : float, default=1.0
        Leaky-integration factor in ``(0, 1]``.
    activation : str, default="tanh"
        One of ``{"tanh", "relu", "logistic", "identity", "bounded_relu"}``.
    device, dtype : optional
        Passed through to the parameter tensor.
    """

    def __init__(self, hidden_size: int, spectral_radius: float = 1.0,
                 leakage: float = 1.0, activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; supported: "
                f"{sorted(_ACTIVATIONS)}")
        self.hidden_size = hidden_size
        self.spectral_radius = float(spectral_radius)
        self.leakage = float(leakage)
        self.activation = activation
        self.weight_hh = nn.Parameter(
            torch.zeros(hidden_size, hidden_size, device=device,
                        dtype=dtype),
            requires_grad=False)

    def set_recurrent_weights(self, weights: object) -> None:
        """Assign the (hidden_size, hidden_size) recurrent weight matrix."""
        w = torch.as_tensor(
            weights, dtype=self.weight_hh.dtype, device=self.weight_hh.device)
        if w.shape != (self.hidden_size, self.hidden_size):
            raise ValueError(
                f"expected weights of shape "
                f"{(self.hidden_size, self.hidden_size)}, got "
                f"{tuple(w.shape)}")
        with torch.no_grad():
            self.weight_hh.copy_(w)

    def forward(self, x: torch.Tensor,
                initial_state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence over a batch of (padded) sequences.

        Parameters
        ----------
        x : torch.Tensor of shape (n_sequences, length, hidden_size)
            Per-step reservoir inputs.
        initial_state : torch.Tensor (n_sequences, hidden_size), or None
            Initial hidden state; zeros when ``None``.

        Returns
        -------
        states : torch.Tensor of shape (n_sequences, length, hidden_size)
        final_state : torch.Tensor of shape (n_sequences, hidden_size)
        """
        n_sequences, length, _ = x.shape
        if initial_state is None:
            h = torch.zeros(
                n_sequences, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h = initial_state
        act = _ACTIVATIONS[self.activation]
        outputs = []
        for t in range(length):
            pre = x[:, t, :] + self.spectral_radius * (h @ self.weight_hh)
            h = (1.0 - self.leakage) * h + self.leakage * act(pre)
            outputs.append(h)
        states = torch.stack(outputs, dim=1)
        return states, h
