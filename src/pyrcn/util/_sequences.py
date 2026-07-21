"""Normalization of sequence input into a canonical padded batch.

This is the single place where the various accepted public input forms (a 2-D
single sequence, a list / object-array of 2-D sequences, or a 3-D equal-length
batch) are validated and converted to the canonical representation consumed by
the redesigned backend: a zero-padded ``(N, L_max, n_features)`` array plus a
per-sequence ``lengths`` vector. It replaces the ad-hoc
``concatenate_sequences`` plus the ``ndim``-based sequence heuristics.
"""

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Task = Literal["sequence-to-sequence", "sequence-to-value"]


@dataclass
class SequenceBatch:
    """Canonical, validated batch of sequences.

    Attributes
    ----------
    X : np.ndarray of shape (n_sequences, max_length, n_features)
        Zero-padded input sequences.
    lengths : np.ndarray of shape (n_sequences,)
        The true (unpadded) length of each sequence.
    task : {"sequence-to-sequence", "sequence-to-value"} or None
        Resolved task kind. ``None`` when no targets were supplied.
    n_features : int
        Number of input features.
    y : np.ndarray or None
        Normalized targets. For ``"sequence-to-sequence"`` this is
        ``(n_sequences, max_length, n_targets)`` (zero-padded); for
        ``"sequence-to-value"`` this is ``(n_sequences, n_targets)``.
    n_targets : int or None
        Number of target outputs, or ``None`` when no targets were supplied.
    single_sequence : bool
        ``True`` when the input was a single 2-D ``(T, n_features)`` array (so
        the caller should return plain, non-object output).
    """

    X: np.ndarray
    lengths: np.ndarray
    task: Task | None
    n_features: int
    y: np.ndarray | None = None
    n_targets: int | None = None
    single_sequence: bool = False


def _as_2d(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"each sequence must be 2-D (n_samples, n_features), got ndim "
            f"{arr.ndim}")
    return arr


def _as_sequence_list(X: object) -> tuple[list[np.ndarray], bool]:
    """Return X as a list of 2-D arrays plus a single-sequence flag."""
    if isinstance(X, np.ndarray):
        if X.dtype == object:
            if X.ndim != 1:
                raise ValueError(
                    f"object array of sequences must be 1-D, got ndim "
                    f"{X.ndim}")
            return [_as_2d(x) for x in X], False
        if X.ndim == 2:
            return [np.asarray(X, dtype=float)], True
        if X.ndim == 3:
            return [np.asarray(x, dtype=float) for x in X], False
        raise ValueError(
            f"X array must be 2-D, 3-D or a 1-D object array, got ndim "
            f"{X.ndim}")
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            raise ValueError("empty sequence input")
        return [_as_2d(x) for x in X], False
    raise ValueError(f"unsupported X type {type(X)!r}")


def _pad_inputs(
        seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    n_features = seqs[0].shape[1]
    for s in seqs:
        if s.shape[1] != n_features:
            raise ValueError(
                f"inconsistent n_features across sequences: {s.shape[1]} vs "
                f"{n_features}")
    lengths = np.array([s.shape[0] for s in seqs], dtype=int)
    max_length = int(lengths.max())
    X = np.zeros((len(seqs), max_length, n_features), dtype=float)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    return X, lengths, n_features


def _as_target_list(y: object, single_sequence: bool) -> list[np.ndarray]:
    if single_sequence:
        return [np.asarray(y)]
    if isinstance(y, (list, tuple)):
        return [np.asarray(v) for v in y]
    if isinstance(y, np.ndarray):
        return [np.asarray(v) for v in y]
    raise ValueError(f"unsupported y type {type(y)!r}")


def _resolve_task(
        y_list: list[np.ndarray], lengths: np.ndarray,
        task: Literal["auto"] | Task) -> Task:
    per_timestep = [
        yi.ndim >= 1 and yi.shape[0] == length
        for yi, length in zip(y_list, lengths)]
    if task == "sequence-to-sequence":
        if not all(per_timestep):
            raise ValueError(
                "task='sequence-to-sequence' but some targets are not aligned "
                "per timestep with their sequence")
        return task
    if task == "sequence-to-value":
        return task
    if task != "auto":
        raise ValueError(f"unknown task {task!r}")
    if all(per_timestep):
        return "sequence-to-sequence"
    if not any(per_timestep):
        return "sequence-to-value"
    raise ValueError(
        "targets mix sequence-to-sequence and sequence-to-value shapes; pass "
        "task= explicitly")


def _pad_targets(
        y_list: list[np.ndarray],
        lengths: np.ndarray) -> tuple[np.ndarray, int]:
    rows = [yi.reshape(yi.shape[0], -1) for yi in y_list]
    n_targets = rows[0].shape[1]
    for a in rows:
        if a.shape[1] != n_targets:
            raise ValueError("inconsistent n_targets across sequences")
    max_length = int(lengths.max())
    Y = np.zeros((len(rows), max_length, n_targets), dtype=float)
    for i, a in enumerate(rows):
        Y[i, :a.shape[0]] = a
    return Y, n_targets


def _stack_values(y_list: list[np.ndarray]) -> tuple[np.ndarray, int]:
    rows = [np.atleast_1d(yi).reshape(-1) for yi in y_list]
    n_targets = rows[0].shape[0]
    for r in rows:
        if r.shape[0] != n_targets:
            raise ValueError("inconsistent n_targets across sequences")
    return np.stack(rows, axis=0), n_targets


def check_sequences(
        X: object, y: object = None, *,
        task: Literal["auto"] | Task = "auto") -> SequenceBatch:
    """Validate and normalize sequence input to a canonical padded batch.

    Parameters
    ----------
    X : list/tuple of 2-D arrays, 1-D object array of 2-D arrays, a 2-D
        ``(T, n_features)`` array (a single sequence), or a 3-D
        ``(n_sequences, length, n_features)`` array.
    y : optional targets, parallel to ``X``. ``None`` for prediction.
    task : {"auto", "sequence-to-sequence", "sequence-to-value"}
        Default "auto".
        ``"auto"`` infers the task from the target shapes (per-timestep targets
        whose leading dimension equals the sequence length are treated as
        sequence-to-sequence); pass an explicit value to resolve the ambiguous
        case where a per-sequence value has the same length as the sequence.

    Returns
    -------
    batch : SequenceBatch
    """
    seqs, single_sequence = _as_sequence_list(X)
    X_batch, lengths, n_features = _pad_inputs(seqs)

    if y is None:
        resolved: Task | None = None if task == "auto" else task
        return SequenceBatch(
            X=X_batch, lengths=lengths, task=resolved, n_features=n_features,
            single_sequence=single_sequence)

    y_list = _as_target_list(y, single_sequence)
    if len(y_list) != len(seqs):
        raise ValueError(
            f"X and y have different numbers of sequences: {len(seqs)} vs "
            f"{len(y_list)}")

    resolved = _resolve_task(y_list, lengths, task)
    if resolved == "sequence-to-sequence":
        y_batch, n_targets = _pad_targets(y_list, lengths)
    else:
        y_batch, n_targets = _stack_values(y_list)
    return SequenceBatch(
        X=X_batch, lengths=lengths, task=resolved, n_features=n_features,
        y=y_batch, n_targets=n_targets, single_sequence=single_sequence)
