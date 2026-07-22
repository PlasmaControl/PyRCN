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

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

from collections.abc import Iterator

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


def _pi_digit_stream() -> Iterator[int]:
    """Yield the decimal digits of pi (3, 1, 4, 1, 5, 9, ...).

    Unbounded Gibbons spigot algorithm in exact integer arithmetic, so the
    digits carry no floating-point error at any length.
    """
    q, r, t, k, n, u = 1, 0, 1, 1, 3, 3
    while True:
        if 4 * q + r - t < n * t:
            yield n
            q, r, t, k, n, u = (
                10 * q, 10 * (r - n * t), t, k,
                (10 * (3 * q + r)) // t - 10 * n, u)
        else:
            q, r, t, k, n, u = (
                q * k, (2 * q + r) * u, t * u, k + 1,
                (q * (7 * k + 2) + r * u) // (t * u), u + 2)


def _pi_fractional_digits(count: int) -> list[int]:
    """First ``count`` digits after the decimal point of pi (d1, d2, ...)."""
    stream = _pi_digit_stream()
    next(stream)                                  # discard integer part d0 = 3
    return [next(stream) for _ in range(count)]


def pi_digit_input_weights(
        n_features: int, hidden_size: int, *, value: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Deterministic signed-constant input weights from the digits of pi [1]_.

    Every weight has the same magnitude ``value``; its sign follows the
    aperiodic decimal expansion of pi thresholded at 4.5 (digit ``<= 4`` gives
    ``-``, digit ``>= 5`` gives ``+``). The fractional digits d1, d2, ... are
    consumed row-major over the ``(n_features, hidden_size)`` matrix, so with
    ``n_features == 1`` the ``k``-th hidden unit takes the sign of the ``k``-th
    digit exactly as in the minimum-complexity ESN papers. Fully deterministic:
    no random state, identical across runs and machines.

    References
    ----------
    .. [1] A. Rodan and P. Tino, "Minimum Complexity Echo State Network", IEEE
       Transactions on Neural Networks, 22(1), 2011.
    """
    out_dtype = dtype if dtype is not None else torch.get_default_dtype()
    digits = torch.tensor(
        _pi_fractional_digits(n_features * hidden_size), device=device)
    signs = (digits >= 5).to(out_dtype) * 2.0 - 1.0  # threshold at 4.5
    return (signs * value).reshape(n_features, hidden_size)


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


def cycle_reservoir_with_jumps_weights(
        hidden_size: int, forward_weight: float = 0.9,
        jump_weight: float = 0.9, jump_size: int | None = None, *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Cycle Reservoir with Jumps (CRJ) [1]_.

    A simple cycle (weight ``forward_weight``) with bidirectional jumps of
    weight ``jump_weight`` connecting units ``jump_size`` (``ell``) apart,
    following the exact construction of Rodan & Tino (2012): there are
    ``hidden_size // ell`` jumps, the first from unit 0 to unit ``ell``. If
    ``ell`` divides ``hidden_size`` the last jump wraps from unit
    ``hidden_size - ell`` back to unit 0 (a closed jump cycle); otherwise the
    jumps form an open chain whose last one ends at unit
    ``ell * (hidden_size // ell)`` with no wrap-around. ``jump_size`` defaults
    to ``round(sqrt(hidden_size))`` and must satisfy ``1 < ell < N // 2`` (the
    paper's range), which also keeps every jump off the cycle's own diagonals.

    References
    ----------
    .. [1] A. Rodan and P. Tino, "Simple Deterministically Constructed Cycle
       Reservoirs with Regular Jumps", Neural Computation, 24(7), 2012.
    """
    if jump_size is None:
        jump_size = round(hidden_size ** 0.5)
    if not 1 < jump_size < hidden_size // 2:
        raise ValueError(
            f"jump_size must satisfy 1 < jump_size < hidden_size // 2, got "
            f"{jump_size} for hidden_size {hidden_size}")
    weights = simple_cycle_weights(
        hidden_size, forward_weight, dtype=dtype, device=device)
    n_jumps = hidden_size // jump_size            # floor(N / ell), paper eq.
    for k in range(n_jumps):
        src = k * jump_size
        dst = (k + 1) * jump_size % hidden_size   # wraps to 0 only if ell | N
        weights[src, dst] = jump_weight           # bidirectional jump
        weights[dst, src] = jump_weight
    return weights


def multi_ring_weights(
        hidden_size: int, n_rings: int = 2, r_min: float = 0.9,
        r_max: float = 0.99, *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None) -> torch.Tensor:
    """Block-diagonal Simple Cycle Reservoirs with log-spaced radii.

    Splits the hidden units into ``n_rings`` blocks (as equal as possible; any
    remainder goes to the earlier blocks) and fills each with a simple cycle
    whose ``forward_weight`` — hence spectral radius — is the ``k``-th of
    ``n_rings`` radii spaced logarithmically over ``[r_min, r_max]``. The full
    recurrent matrix is block-diagonal, so its spectrum is the union of the
    per-ring circles. With ``n_rings == 1`` this reduces to a single cycle at
    radius ``r_min``.
    """
    if n_rings < 1 or n_rings > hidden_size:
        raise ValueError(
            f"n_rings must be in [1, hidden_size], got {n_rings} for "
            f"hidden_size {hidden_size}")
    radii = torch.logspace(
        float(torch.log10(torch.tensor(r_min))),
        float(torch.log10(torch.tensor(r_max))), n_rings)
    base, remainder = divmod(hidden_size, n_rings)
    weights = torch.zeros(hidden_size, hidden_size, dtype=dtype, device=device)
    offset = 0
    for k in range(n_rings):
        size = base + (1 if k < remainder else 0)
        block = simple_cycle_weights(
            size, float(radii[k]), dtype=dtype, device=device)
        weights[offset:offset + size, offset:offset + size] = block
        offset += size
    return weights
