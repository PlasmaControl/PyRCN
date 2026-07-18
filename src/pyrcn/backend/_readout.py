"""PyTorch closed-form (ridge) readout, the torch companion of
``IncrementalRegression``.

Accumulates the normal-equation statistics ``K = sum(Z^T Z)`` and
``xTy = sum(Z^T y)`` over one or more batches and solves the ridge system
``(K + alpha I) W = xTy`` once. This mirrors the sequence-fit flow, where
each sequence contributes to ``K``/``xTy`` (``postpone_inverse=True``) and the
final call triggers the single solve. The statistics are additive, so two
readouts fitted on disjoint data can be merged with ``+`` (map-reduce).

``fit_intercept`` folds a bias into the weights by appending a column of ones
to ``Z`` (as the legacy readout does); ``output_weights`` then has one extra
row. All state is on-device; weights carry no gradient in the closed-form path.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

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
