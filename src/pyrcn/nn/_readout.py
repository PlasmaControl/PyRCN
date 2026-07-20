"""PyTorch readouts for :mod:`pyrcn.nn`.

:class:`IncrementalRidge` is the closed-form (ridge) readout, the torch
companion of ``IncrementalRegression``: it accumulates the normal-equation
statistics ``K = sum(Z^T Z)`` and ``xTy = sum(Z^T y)`` over one or more
batches and solves ``(K + alpha I) W = xTy`` once (mirroring the sequence-fit
flow, where each sequence contributes with ``postpone_inverse=True`` and the
final call triggers the single solve). The statistics are additive, so two
readouts fitted on disjoint data merge with ``+`` (map-reduce).
``fit_intercept`` folds a bias into the weights by appending a column of ones
to ``Z``; its weights carry no gradient.

:class:`LinearReadout` is the trainable counterpart -- a plain ``nn.Linear``
whose weights are optimized by a gradient loop (see
:func:`pyrcn.nn.train_readout`), used when ``solver='gradient'``.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

import math

import torch
import torch.nn as nn


class IncrementalRidge(nn.Module):
    """Accumulate ``Z^T Z`` / ``Z^T y`` and solve ``(K + alpha I) W = xTy``.

    Parameters
    ----------
    alpha : float, default=1e-5
        L2 (ridge) regularization strength.
    fit_intercept : bool, default=True
        Append a column of ones to the features so the last weight row is the
        intercept.
    """

    def __init__(self, *, alpha: float = 1e-5, fit_intercept: bool = True,
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self._device = device
        self._dtype = dtype
        self.register_buffer("_K", None)
        self.register_buffer("_xTy", None)
        self.register_buffer("output_weights", None)

    def _preprocess(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.fit_intercept:
            return Z
        ones = torch.ones(Z.shape[0], 1, dtype=Z.dtype, device=Z.device)
        return torch.cat([Z, ones], dim=1)

    def partial_fit(self, Z: torch.Tensor, y: torch.Tensor, *,
                    reset: bool = False,
                    postpone_inverse: bool = False) -> IncrementalRidge:
        """Accumulate one batch; solve unless ``postpone_inverse``.

        With ``postpone_inverse=True`` only the statistics are updated (used
        for all but the last sequence); the following non-postponed call
        performs the single closed-form solve over the accumulated data.
        """
        if reset:
            self._K = None
            self._xTy = None
            self.output_weights = None
        Zp = self._preprocess(Z)
        gram = Zp.T @ Zp
        rhs = Zp.T @ y
        self._K = gram if self._K is None else self._K + gram
        self._xTy = rhs if self._xTy is None else self._xTy + rhs
        if postpone_inverse and self.output_weights is None:
            return self
        self.solve()
        return self

    def fit(self, Z: torch.Tensor, y: torch.Tensor) -> IncrementalRidge:
        """Fit on a single batch (reset, accumulate, solve)."""
        return self.partial_fit(Z, y, reset=True, postpone_inverse=False)

    def solve(self) -> IncrementalRidge:
        """Solve ``(K + alpha I) W = xTy`` from the accumulated statistics."""
        if self._K is None or self._xTy is None:
            raise RuntimeError("no statistics accumulated; call partial_fit")
        identity = torch.eye(
            self._K.shape[0], dtype=self._K.dtype, device=self._K.device)
        self.output_weights = torch.linalg.solve(
            self._K + self.alpha * identity, self._xTy)
        return self

    def predict(self, Z: torch.Tensor) -> torch.Tensor:
        """Return ``Z @ output_weights`` (intercept folded in)."""
        if self.output_weights is None:
            raise RuntimeError("readout is not fitted; call fit/partial_fit")
        return self._preprocess(Z) @ self.output_weights

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.predict(Z)

    def __add__(self, other: IncrementalRidge) -> IncrementalRidge:
        """Merge accumulated statistics (map-reduce over disjoint data).

        Returns an unsolved readout; call :meth:`solve` before predicting.
        """
        if (self._K is None or other._K is None
                or self._xTy is None or other._xTy is None):
            raise RuntimeError(
                "both readouts must have accumulated statistics")
        merged = IncrementalRidge(
            alpha=self.alpha, fit_intercept=self.fit_intercept,
            device=self._device, dtype=self._dtype)
        merged._K = self._K + other._K
        merged._xTy = self._xTy + other._xTy
        return merged


class LinearReadout(nn.Module):
    """Trainable linear readout (a thin wrapper over ``nn.Linear``).

    The gradient-mode counterpart of :class:`IncrementalRidge`: its weights
    are trainable ``Parameter``s optimized by :func:`pyrcn.nn.train_readout`
    rather than solved in closed form.

    Parameters
    ----------
    in_features, out_features : int
        Readout input and output sizes.
    fit_intercept : bool, default=True
        Whether the underlying ``nn.Linear`` has a bias term.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 fit_intercept: bool = True,
                 generator: torch.Generator | None = None,
                 device: torch.device | str | int | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features, out_features, bias=fit_intercept, device=device,
            dtype=dtype)
        if generator is not None:
            # Reproducible init from a seeded generator (nn.Linear's default
            # init draws from torch's global RNG).
            bound = 1.0 / math.sqrt(in_features)
            with torch.no_grad():
                self.linear.weight.uniform_(-bound, bound, generator=generator)
                if self.linear.bias is not None:
                    self.linear.bias.uniform_(
                        -bound, bound, generator=generator)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.linear(Z)

    def predict(self, Z: torch.Tensor) -> torch.Tensor:
        """Return predictions ``(n_samples, out_features)`` without grad."""
        with torch.no_grad():
            return self.linear(Z)
