"""PyTorch reservoir cells and layers built on ``nn.RNNCell``.

Reuses torch's fused single-step cell op for ``tanh``/``relu`` and adds only
what torch lacks: leaky integration (:class:`LeakyESNCell`), Euler / EuSN
integration (:class:`EulerESNCell`), and the extra PyRCN activations
(``logistic``/``identity``/``bounded_relu``). In every cell ``weight_ih`` is a
frozen identity (the reservoir input is added directly; input weights belong to
``InputToNode``); the effective recurrent matrix is folded into ``weight_hh``.

The whole-sequence forward uses a two-tier dispatch (same maths, no per-step
Python overhead):

* **Fused sub-case** -- plain (non-Euler) reservoir with ``leakage == 1`` and a
  ``tanh``/``relu`` activation whose recurrent weights are *not* being trained:
  ATen's fused whole-sequence :func:`torch.rnn_tanh` / :func:`torch.rnn_relu`
  (``weight_ih`` is the identity, so the reservoir input is added directly and
  the recurrence equals the per-step cell). Bidirectional reuses the *same*
  weights on the time-flipped input, matching ``NodeToNode``.
* **General case** -- everything else (leaky integration, the Euler variant,
  ``logistic``/``identity``/``bounded_relu``, and any config whose recurrent
  weights require gradients): a :func:`torch.jit.script` compiled loop running
  the identical recurrence ``h' = a*h + b*f(x + h @ weight_hh.T)``.
"""

# Authors: Peter Steiner <peter.steiner@princeton.edu>
# License: BSD 3 clause

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._activations import ACTIVATIONS, FUSED, SUPPORTED

#: Integer id per activation for the (scriptable) general-path loop.
_ACT_IDS = {"tanh": 0, "relu": 1, "logistic": 2, "identity": 3,
            "bounded_relu": 4}


class _ReservoirCell(nn.RNNCell):
    """Shared machinery for the reservoir cells.

    Sets ``weight_ih`` to a frozen identity, freezes all parameters, and
    provides :meth:`_activate` = ``f(x + h @ weight_hh)`` (reusing the fused op
    for ``tanh``/``relu``). Subclasses fold their effective recurrent matrix
    into ``weight_hh`` and combine ``_activate`` into their state update.
    """

    def __init__(self, hidden_size: int, activation: str,
                 device: torch.device | str | int | None,
                 dtype: torch.dtype | None) -> None:
        if activation not in SUPPORTED:
            raise ValueError(
                f"unknown activation {activation!r}; supported: "
                f"{sorted(SUPPORTED)}")
        nonlinearity = activation if activation in FUSED else "tanh"
        super().__init__(input_size=hidden_size, hidden_size=hidden_size,
                         bias=False, nonlinearity=nonlinearity, device=device,
                         dtype=dtype)
        self.activation = activation
        with torch.no_grad():
            self.weight_ih.copy_(torch.eye(
                hidden_size, device=device, dtype=self.weight_ih.dtype))
        for p in self.parameters():
            p.requires_grad_(False)

    def _as_square(self, weights: object) -> torch.Tensor:
        w = torch.as_tensor(
            weights, dtype=self.weight_hh.dtype, device=self.weight_hh.device)
        if w.shape != (self.hidden_size, self.hidden_size):
            raise ValueError(
                f"expected weights of shape "
                f"{(self.hidden_size, self.hidden_size)}, got "
                f"{tuple(w.shape)}")
        return w

    def _zeros_like_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)

    def _activate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.activation in FUSED:
            return super().forward(x, h)                    # fused cell op
        pre = x + F.linear(h, self.weight_hh)               # weight_ih is I
        return ACTIVATIONS[self.activation](pre)


class LeakyESNCell(_ReservoirCell):
    """Leaky reservoir step.

    ``h' = (1 - leakage) * h + leakage * f(x + spectral_radius * (h @ W))``
    """

    def __init__(self, hidden_size: int, spectral_radius: float = 1.0,
                 leakage: float = 1.0, activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__(hidden_size, activation, device, dtype)
        self.spectral_radius = float(spectral_radius)
        self.leakage = float(leakage)

    def set_recurrent_weights(self, weights: object) -> None:
        """Load ``(hidden_size, hidden_size)`` weights (transposed, scaled)."""
        w = self._as_square(weights)
        with torch.no_grad():
            self.weight_hh.copy_(self.spectral_radius * w.T)

    def forward(self, input: torch.Tensor,
                hx: torch.Tensor | None = None) -> torch.Tensor:
        if hx is None:
            hx = self._zeros_like_state(input)
        return (1.0 - self.leakage) * hx + self.leakage * self._activate(
            input, hx)


class EulerESNCell(_ReservoirCell):
    """Euler (EuSN) reservoir step.

    ``h' = h + epsilon * f(x + h @ (recurrent_scaling * W + gamma * I))``
    """

    def __init__(self, hidden_size: int, recurrent_scaling: float = 1.0,
                 gamma: float = 0.001, epsilon: float = 0.01,
                 activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__(hidden_size, activation, device, dtype)
        self.recurrent_scaling = float(recurrent_scaling)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    def set_recurrent_weights(self, weights: object) -> None:
        """Fold ``recurrent_scaling * W + gamma * I`` into ``weight_hh``."""
        w = self._as_square(weights)
        identity = torch.eye(
            self.hidden_size, dtype=w.dtype, device=w.device)
        effective = self.recurrent_scaling * w + self.gamma * identity
        with torch.no_grad():
            self.weight_hh.copy_(effective.T)

    def forward(self, input: torch.Tensor,
                hx: torch.Tensor | None = None) -> torch.Tensor:
        if hx is None:
            hx = self._zeros_like_state(input)
        return hx + self.epsilon * self._activate(input, hx)


@torch.jit.script
def _iterate_general(x: torch.Tensor, h: torch.Tensor,
                     weight_hh: torch.Tensor, coeff_prev: float,
                     coeff_new: float, activation: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scripted whole-sequence loop of ``h' = a*h + b*f(x + h @ W.T)``.

    ``coeff_prev``/``coeff_new`` are ``(1 - leakage, leakage)`` for the leaky
    reservoir and ``(1.0, epsilon)`` for the Euler variant; ``weight_hh`` holds
    the (already scaled/folded) recurrent matrix, so ``F.linear`` reproduces
    the per-step cell exactly. Bit-exact with the original Python loop.
    """
    outputs: List[torch.Tensor] = []
    length = x.shape[1]
    for t in range(length):
        pre = x[:, t, :] + F.linear(h, weight_hh)
        if activation == 0:
            a = torch.tanh(pre)
        elif activation == 1:
            a = torch.relu(pre)
        elif activation == 2:
            a = torch.sigmoid(pre)
        elif activation == 3:
            a = pre
        else:
            a = torch.clamp(pre, 0.0, 1.0)
        h = coeff_prev * h + coeff_new * a
        outputs.append(h)
    return torch.stack(outputs, dim=1), h


def _iterate_fused(x: torch.Tensor, initial_state: torch.Tensor,
                   weight_ih: torch.Tensor, weight_hh: torch.Tensor,
                   activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused whole-sequence ``tanh``/``relu`` recurrence via ATen.

    Uses :func:`torch.rnn_tanh` / :func:`torch.rnn_relu` (single layer,
    unidirectional, no biases). ``weight_ih`` is the frozen identity, so this
    computes ``h' = f(x + h @ weight_hh.T)`` (the ``leakage == 1`` case).
    """
    hx = initial_state.unsqueeze(0)
    params = [weight_ih, weight_hh]
    if activation == "relu":
        out, hn = torch.rnn_relu(
            x, hx, params, False, 1, 0.0, False, False, True)
    else:
        out, hn = torch.rnn_tanh(
            x, hx, params, False, 1, 0.0, False, False, True)
    return out, hn.squeeze(0)


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

    def set_recurrent_trainable(self, trainable: bool = True) -> None:
        """Enable/disable gradient training of the recurrent weights."""
        self.cell.weight_hh.requires_grad_(trainable)

    def _run(self, x: torch.Tensor, h0: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        cell = self.cell
        use_fused = (cell.activation in FUSED and cell.leakage == 1.0
                     and not cell.weight_hh.requires_grad)
        if use_fused:
            return _iterate_fused(
                x, h0, cell.weight_ih, cell.weight_hh, cell.activation)
        return _iterate_general(
            x, h0, cell.weight_hh, 1.0 - cell.leakage, cell.leakage,
            _ACT_IDS[cell.activation])

    def forward(self, x: torch.Tensor,
                initial_state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.bidirectional:
            if initial_state is None:
                initial_state = self.cell._zeros_like_state(x)
            return self._run(x, initial_state)
        if initial_state is not None:
            raise ValueError(
                "initial_state is not supported with bidirectional=True")
        states_fw, final_fw = self._run(x, self.cell._zeros_like_state(x))
        reversed_states, final_bw = self._run(
            torch.flip(x, dims=[1]), self.cell._zeros_like_state(x))
        states = torch.cat(
            [states_fw, torch.flip(reversed_states, dims=[1])], dim=-1)
        return states, torch.cat([final_fw, final_bw], dim=-1)


class EulerReservoir(nn.Module):
    """Run an :class:`EulerESNCell` over batched sequences (unidirectional).

    ``forward`` matches :class:`Reservoir` (``(states, final_state)``);
    ``EulerNodeToNode`` is single-direction.
    """

    def __init__(self, hidden_size: int, recurrent_scaling: float = 1.0,
                 gamma: float = 0.001, epsilon: float = 0.01,
                 activation: str = "tanh",
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = EulerESNCell(
            hidden_size, recurrent_scaling=recurrent_scaling, gamma=gamma,
            epsilon=epsilon, activation=activation, device=device, dtype=dtype)

    def set_recurrent_weights(self, weights: object) -> None:
        """Load the recurrent weight matrix into the cell."""
        self.cell.set_recurrent_weights(weights)

    def set_recurrent_trainable(self, trainable: bool = True) -> None:
        """Enable/disable gradient training of the recurrent weights."""
        self.cell.weight_hh.requires_grad_(trainable)

    def forward(self, x: torch.Tensor,
                initial_state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state is None:
            initial_state = self.cell._zeros_like_state(x)
        # Euler always takes the general path: h' = h + epsilon * f(...).
        return _iterate_general(
            x, initial_state, self.cell.weight_hh, 1.0, self.cell.epsilon,
            _ACT_IDS[self.cell.activation])
