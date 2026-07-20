"""Weight initializers for the :mod:`pyrcn.nn` reservoir components.

Random and sparse recurrent designs are normalized to unit spectral radius;
a reservoir's ``spectral_radius`` then scales them at runtime. The
minimum-complexity topologies (Rodan & Tino, 2011) are deterministic
structured matrices whose scale is the ``forward_weight`` (and, for the
delay-line-with-feedback design, the ``feedback_weight``). Input designs
cover dense/sparse uniform weights, uniform bias, and signed-constant
(Bernoulli) input weights.

All initializers accept an optional :class:`torch.Generator` for
reproducibility and the usual ``dtype`` / ``device`` placement arguments.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

import torch


def spectral_normalize(weights: torch.Tensor,
                       radius: float = 1.0) -> torch.Tensor:
    """Scale a square matrix to the given maximum absolute eigenvalue."""
    eigenvalues = torch.linalg.eigvals(weights)
    return weights * (radius / eigenvalues.abs().max())


def normal_recurrent_weights(
        hidden_size: int, *, fan_in: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Normally distributed recurrent weights, unit spectral radius.

    If ``fan_in`` is given and smaller than ``hidden_size``, exactly ``fan_in``
    entries are kept per column (the rest zeroed) before normalization.
    """
    weights = torch.randn(hidden_size, hidden_size, generator=generator,
                          dtype=dtype, device=device)
    if fan_in is not None and fan_in < hidden_size:
        for column in range(hidden_size):
            order = torch.randperm(
                hidden_size, generator=generator, device=device)
            weights[order[fan_in:], column] = 0.0
    return spectral_normalize(weights)


def antisymmetric_recurrent_weights(
        hidden_size: int, *, fan_in: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Antisymmetric uniform recurrent weights ``U - U.T`` (used by EuSN).

    ``U`` is uniform in ``[-1, 1)`` (optionally sparsified to ``fan_in``
    entries per column before antisymmetrization).
    """
    u = torch.rand(hidden_size, hidden_size, generator=generator, dtype=dtype,
                   device=device) * 2.0 - 1.0
    if fan_in is not None and fan_in < hidden_size:
        for column in range(hidden_size):
            order = torch.randperm(
                hidden_size, generator=generator, device=device)
            u[order[fan_in:], column] = 0.0
    return u - u.T


def uniform_input_weights(
        n_features: int, hidden_size: int, *, fan_in: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Uniform input weights in ``[-1, 1)`` of shape ``(n_features, hidden)``.

    If ``fan_in`` is given, exactly ``fan_in`` entries are kept per column
    (hidden unit).
    """
    weights = torch.rand(n_features, hidden_size, generator=generator,
                         dtype=dtype, device=device) * 2.0 - 1.0
    if fan_in is not None and fan_in < n_features:
        for column in range(hidden_size):
            order = torch.randperm(
                n_features, generator=generator, device=device)
            weights[order[fan_in:], column] = 0.0
    return weights


def uniform_bias_weights(
        hidden_size: int, *, generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Uniform bias in ``[-1, 1)`` of shape ``(hidden_size,)``."""
    return torch.rand(hidden_size, generator=generator, dtype=dtype,
                      device=device) * 2.0 - 1.0


def bernoulli_input_weights(
        n_features: int, hidden_size: int, *, value: float = 1.0,
        p: float = 0.5, generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Signed-constant input weights ``+/- value`` (min-complexity ESNs)."""
    probabilities = torch.full((n_features, hidden_size), p, dtype=dtype,
                               device=device)
    signs = torch.bernoulli(probabilities, generator=generator)
    return (2.0 * signs - 1.0) * value


def simple_cycle_weights(
        hidden_size: int, forward_weight: float = 0.9, *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Simple Cycle Reservoir: one cycle with weight ``forward_weight``."""
    weights = torch.zeros(hidden_size, hidden_size, dtype=dtype, device=device)
    for i in range(hidden_size):
        weights[i, i - 1] = forward_weight               # subdiagonal + corner
    return weights


def delay_line_weights(
        hidden_size: int, forward_weight: float = 0.9, *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Delay Line Reservoir: subdiagonal only (nilpotent)."""
    weights = torch.zeros(hidden_size, hidden_size, dtype=dtype, device=device)
    for i in range(1, hidden_size):
        weights[i, i - 1] = forward_weight
    return weights


def delay_line_feedback_weights(
        hidden_size: int, forward_weight: float = 0.9,
        feedback_weight: float = 0.1, *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Delay Line Reservoir with feedback: subdiagonal + superdiagonal."""
    weights = torch.zeros(hidden_size, hidden_size, dtype=dtype, device=device)
    for i in range(1, hidden_size):
        weights[i, i - 1] = forward_weight               # forward
        weights[i - 1, i] = feedback_weight              # feedback
    return weights
