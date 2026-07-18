"""Numerical correctness of the torch-native reservoir weight init.

These produce the recurrent matrix for real (non-parity) use, so they are
checked on their defining numerical properties: unit spectral radius for the
random/sparse designs, and the exact structure (and resulting spectral radius)
for the minimum-complexity topologies. Reproducibility uses a torch Generator.
"""
from __future__ import annotations

import numpy as np
import torch

from pyrcn.backend import (delay_line_feedback_weights, delay_line_weights,
                           normal_recurrent_weights, simple_cycle_weights)


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


def test_reproducible_with_generator() -> None:
    W1 = normal_recurrent_weights(
        20, fan_in=5, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64)
    W2 = normal_recurrent_weights(
        20, fan_in=5, generator=torch.Generator().manual_seed(1),
        dtype=torch.float64)
    assert torch.equal(W1, W2)
