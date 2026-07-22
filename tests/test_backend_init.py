"""Numerical correctness of the torch-native reservoir weight init.

These produce the recurrent matrix for real (non-parity) use, so they are
checked on their defining numerical properties: unit spectral radius for the
random/sparse designs, and the exact structure (and resulting spectral radius)
for the minimum-complexity topologies. Reproducibility uses a torch Generator.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.nn.init import (antisymmetric_recurrent_weights,
                           cycle_reservoir_with_jumps_weights,
                           delay_line_feedback_weights, delay_line_weights,
                           multi_ring_weights, normal_recurrent_weights,
                           simple_cycle_weights)


def _spectral_radius(W: torch.Tensor) -> float:
    return float(torch.linalg.eigvals(W).abs().max())


def test_normal_recurrent_has_unit_spectral_radius() -> None:
    g = torch.Generator().manual_seed(0)
    W = normal_recurrent_weights(50, generator=g, dtype=torch.float64)
    assert W.shape == (50, 50)
    np.testing.assert_allclose(_spectral_radius(W), 1.0, rtol=1e-6)


def test_sparse_recurrent_fan_in_and_radius() -> None:
    g = torch.Generator().manual_seed(0)
    W = normal_recurrent_weights(
        40, fan_in=8, generator=g, dtype=torch.float64)
    nonzeros_per_column = (W != 0).sum(dim=0)
    assert int(nonzeros_per_column.max()) == 8          # exactly fan_in kept
    assert int(nonzeros_per_column.min()) == 8
    np.testing.assert_allclose(_spectral_radius(W), 1.0, rtol=1e-6)


def test_simple_cycle_structure_and_radius() -> None:
    W = simple_cycle_weights(6, forward_weight=0.9, dtype=torch.float64)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    for i in range(6):
        expected[i, i - 1] = 0.9                        # subdiagonal + corner
    assert torch.equal(W, expected)
    np.testing.assert_allclose(_spectral_radius(W), 0.9, rtol=1e-6)


def test_delay_line_structure_is_nilpotent() -> None:
    W = delay_line_weights(6, forward_weight=0.9, dtype=torch.float64)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    for i in range(1, 6):
        expected[i, i - 1] = 0.9                        # subdiagonal only
    assert torch.equal(W, expected)
    assert _spectral_radius(W) < 1e-9                   # nilpotent


def test_delay_line_feedback_structure() -> None:
    W = delay_line_feedback_weights(
        6, forward_weight=0.9, feedback_weight=0.1, dtype=torch.float64)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    for i in range(1, 6):
        expected[i, i - 1] = 0.9                        # forward
        expected[i - 1, i] = 0.1                        # feedback
    assert torch.equal(W, expected)


def test_crj_structure_with_jumps() -> None:
    # simple cycle (subdiagonal + corner) plus bidirectional jumps of size 2
    # placed at 0->2, 2->4, 4->0 (wrap): exact reconstruction of the matrix.
    W = cycle_reservoir_with_jumps_weights(
        6, forward_weight=0.9, jump_weight=0.5, jump_size=2,
        dtype=torch.float64)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    for i in range(6):
        expected[i, i - 1] = 0.9                        # cycle
    for start in (0, 2, 4):
        end = (start + 2) % 6
        expected[start, end] = 0.5                      # bidirectional jump
        expected[end, start] = 0.5
    assert torch.equal(W, expected)


def test_crj_jumps_are_symmetric() -> None:
    # the jump connections (everything beyond the cycle) are symmetric.
    W = cycle_reservoir_with_jumps_weights(
        12, forward_weight=0.9, jump_weight=0.3, jump_size=3,
        dtype=torch.float64)
    cycle = simple_cycle_weights(12, 0.9, dtype=torch.float64)
    jumps = W - cycle                                   # remove the cycle
    assert torch.equal(jumps, jumps.T)                  # symmetric
    assert torch.count_nonzero(jumps) == 2 * (12 // 3)  # N/ell bidir jumps


def test_crj_default_jump_size_is_sqrt_n() -> None:
    # jump_size defaults to round(sqrt(hidden_size)) = 4 for N = 16.
    W = cycle_reservoir_with_jumps_weights(
        16, forward_weight=0.9, jump_weight=0.3, dtype=torch.float64)
    assert W[0, 4] == 0.3 and W[4, 0] == 0.3            # jump at spacing 4
    assert W[0, 15] == 0.9 and W[3, 2] == 0.9           # cycle intact


def test_crj_matches_paper_divisible_case() -> None:
    # Rodan & Tino (2012), Fig. 2(A): N=18, ell=3 divides N -> N/ell = 6 jumps,
    # the last wrapping from unit N-ell back to unit 0 (closed jump cycle).
    W = cycle_reservoir_with_jumps_weights(
        18, forward_weight=0.9, jump_weight=0.4, jump_size=3,
        dtype=torch.float64)
    jumps = W - simple_cycle_weights(18, 0.9, dtype=torch.float64)
    expected = torch.zeros(18, 18, dtype=torch.float64)
    for src, dst in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 0)]:
        expected[src, dst] = 0.4
        expected[dst, src] = 0.4
    assert torch.equal(jumps, expected)
    assert torch.count_nonzero(jumps) == 2 * (18 // 3)


def test_crj_matches_paper_nondivisible_case() -> None:
    # Rodan & Tino (2012), Fig. 2(B): N=18, ell=4, (N mod ell) = 2 != 0 ->
    # floor(N/ell) = 4 jumps forming an OPEN chain whose last jump ends at unit
    # N+1-(N mod ell) = 17 (0-indexed 16); crucially there is NO wrap-around.
    W = cycle_reservoir_with_jumps_weights(
        18, forward_weight=0.9, jump_weight=0.4, jump_size=4,
        dtype=torch.float64)
    jumps = W - simple_cycle_weights(18, 0.9, dtype=torch.float64)
    expected = torch.zeros(18, 18, dtype=torch.float64)
    for src, dst in [(0, 4), (4, 8), (8, 12), (12, 16)]:
        expected[src, dst] = 0.4
        expected[dst, src] = 0.4
    assert torch.equal(jumps, expected)
    assert torch.count_nonzero(jumps) == 2 * (18 // 4)  # 4 jumps, not 5
    assert W[16, 2] == 0.0 and W[2, 16] == 0.0          # no spurious wrap jump


def test_crj_rejects_out_of_range_jump_size() -> None:
    with pytest.raises(ValueError):
        cycle_reservoir_with_jumps_weights(10, jump_size=1)      # not > 1
    with pytest.raises(ValueError):
        cycle_reservoir_with_jumps_weights(10, jump_size=5)      # not < N // 2


def test_multi_ring_is_block_diagonal_with_logspaced_radii() -> None:
    W = multi_ring_weights(
        12, n_rings=3, r_min=0.5, r_max=0.9, dtype=torch.float64)
    # off-block-diagonal entries are exactly zero; each block is a simple
    # cycle whose radius is its own subdiagonal weight.
    for k in range(3):
        lo, hi = 4 * k, 4 * (k + 1)
        block = W[lo:hi, lo:hi]
        radius = float(block[1, 0])
        assert torch.equal(
            block, simple_cycle_weights(4, radius, dtype=torch.float64))
        for j in range(3):
            if j != k:
                assert torch.count_nonzero(
                    W[lo:hi, 4 * j:4 * (j + 1)]) == 0
    # radii increase from r_min to r_max (log-spaced), and the spectral radius
    # of the whole (block-diagonal) matrix is the largest ring radius.
    assert float(W[1, 0]) == pytest.approx(0.5, rel=1e-5)
    assert float(W[9, 8]) == pytest.approx(0.9, rel=1e-5)
    assert float(W[1, 0]) < float(W[5, 4]) < float(W[9, 8])
    np.testing.assert_allclose(_spectral_radius(W), 0.9, rtol=1e-5)


def test_multi_ring_uneven_block_sizes() -> None:
    # 7 units, 2 rings -> sizes [4, 3] (remainder goes to the first block).
    W = multi_ring_weights(7, n_rings=2, dtype=torch.float64)
    assert torch.count_nonzero(W[0:4, 0:4]) == 4        # size-4 cycle
    assert torch.count_nonzero(W[4:7, 4:7]) == 3        # size-3 cycle
    assert torch.count_nonzero(W[0:4, 4:7]) == 0        # block-diagonal
    assert torch.count_nonzero(W[4:7, 0:4]) == 0


def test_multi_ring_single_ring_is_simple_cycle() -> None:
    W = multi_ring_weights(5, n_rings=1, r_min=0.7, r_max=0.7,
                           dtype=torch.float64)
    np.testing.assert_allclose(
        W.numpy(), simple_cycle_weights(5, 0.7, dtype=torch.float64).numpy(),
        rtol=1e-5)


def test_multi_ring_rejects_out_of_range_n_rings() -> None:
    with pytest.raises(ValueError):
        multi_ring_weights(8, n_rings=0)
    with pytest.raises(ValueError):
        multi_ring_weights(8, n_rings=9)                # > hidden_size


def test_antisymmetric_recurrent_is_antisymmetric() -> None:
    g = torch.Generator().manual_seed(0)
    W = antisymmetric_recurrent_weights(10, generator=g, dtype=torch.float64)
    assert W.shape == (10, 10)
    np.testing.assert_allclose(W.numpy(), -W.T.numpy())


def test_antisymmetric_recurrent_sparse_fan_in() -> None:
    # fan_in < hidden_size exercises the per-column sparsification of U
    # before antisymmetrization; the result stays antisymmetric.
    g = torch.Generator().manual_seed(0)
    W = antisymmetric_recurrent_weights(
        12, fan_in=4, generator=g, dtype=torch.float64)
    assert W.shape == (12, 12)
    np.testing.assert_allclose(W.numpy(), -W.T.numpy())
    # sparsifying U to 4 entries per column bounds each row/col of U-U.T.
    assert int((W != 0).sum()) < 12 * 12


def test_reproducible_with_generator() -> None:
    W1 = normal_recurrent_weights(
        20, fan_in=5, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64)
    W2 = normal_recurrent_weights(
        20, fan_in=5, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64)
    assert torch.equal(W1, W2)
