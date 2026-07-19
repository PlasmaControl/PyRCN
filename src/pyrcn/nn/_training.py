"""Gradient-based training utility for :mod:`pyrcn.nn` readouts.

A minimal optimizer loop that trains a readout module (typically
:class:`~pyrcn.nn.LinearReadout`) on precomputed features/states. Optimizer
and loss are selected by name so the choice stays serializable and tunable
via scikit-learn model selection. ``weight_decay`` supplies the L2 (ridge)
regularization, so with a fixed reservoir this converges to the closed-form
:class:`~pyrcn.nn.IncrementalRidge` solution.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

import torch
import torch.nn as nn

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
LOSSES = {"mse": nn.MSELoss}


def train_readout(readout: nn.Module, Z: torch.Tensor, y: torch.Tensor, *,
                  optimizer: str = "adam", learning_rate: float = 1e-3,
                  epochs: int = 100, batch_size: int | None = None,
                  weight_decay: float = 0.0, loss: str = "mse") -> nn.Module:
    """Train ``readout`` on ``(Z, y)`` with a gradient optimizer loop.

    Parameters
    ----------
    readout : nn.Module
        The readout to train in place (e.g. a
        :class:`~pyrcn.nn.LinearReadout`).
    Z : torch.Tensor of shape (n_samples, n_features)
        Readout inputs (reservoir states / input features).
    y : torch.Tensor of shape (n_samples, n_targets)
        Training targets.
    optimizer : {"adam", "sgd"}, default="adam"
    learning_rate : float, default=1e-3
    epochs : int, default=100
    batch_size : int or None, default=None
        Mini-batch size; ``None`` uses full-batch updates.
    weight_decay : float, default=0.0
        L2 penalty (ridge strength).
    loss : {"mse"}, default="mse"

    Returns
    -------
    readout : nn.Module
        The trained readout (same object).
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; supported: "
            f"{sorted(OPTIMIZERS)}")
    if loss not in LOSSES:
        raise ValueError(
            f"unknown loss {loss!r}; supported: {sorted(LOSSES)}")
    opt = OPTIMIZERS[optimizer](
        readout.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = LOSSES[loss]()
    n_samples = Z.shape[0]
    step = n_samples if batch_size is None else int(batch_size)
    readout.train()
    for _ in range(int(epochs)):
        permutation = torch.randperm(n_samples, device=Z.device)
        for start in range(0, n_samples, step):
            index = permutation[start:start + step]
            opt.zero_grad()
            batch_loss = loss_fn(readout(Z[index]), y[index])
            batch_loss.backward()
            opt.step()
    readout.eval()
    return readout
