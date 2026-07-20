"""Tests for the sequence normalization used by the redesigned backend."""
from __future__ import annotations

import numpy as np
import pytest

from pyrcn.util import check_sequences


def test_2d_single_sequence() -> None:
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)
    batch = check_sequences(X, y)
    assert batch.X.shape == (1, 10, 3)
    assert batch.lengths.tolist() == [10]
    assert batch.single_sequence is True
    assert batch.task == "sequence-to-sequence"
    assert batch.n_features == 3
    np.testing.assert_array_equal(batch.X[0], X)


def test_list_ragged_seq2seq() -> None:
    X = [np.ones((5, 2)), np.ones((3, 2)) * 2]
    y = [np.ones((5, 1)), np.ones((3, 1))]
    batch = check_sequences(X, y)
    assert batch.X.shape == (2, 5, 2)
    assert batch.lengths.tolist() == [5, 3]
    assert batch.single_sequence is False
    assert batch.task == "sequence-to-sequence"
    assert np.all(batch.X[1, 3:] == 0)              # zero-padded tail
    np.testing.assert_array_equal(batch.X[1, :3], np.ones((3, 2)) * 2)
    assert batch.y.shape == (2, 5, 1)


def test_object_array_seq2value() -> None:
    rng = np.random.RandomState(0)
    X = np.empty(4, dtype=object)
    y = np.empty(4, dtype=object)
    for k in range(4):
        X[k] = rng.rand(8, 8)
        y[k] = np.atleast_1d(k)                      # (1,) per sequence
    batch = check_sequences(X, y)
    assert batch.X.shape == (4, 8, 8)
    assert batch.lengths.tolist() == [8, 8, 8, 8]
    assert batch.task == "sequence-to-value"
    assert batch.y.shape == (4, 1)


def test_3d_equal_length_seq2seq() -> None:
    X = np.ones((4, 6, 2))
    y = [np.ones((6, 1)) for _ in range(4)]
    batch = check_sequences(X, y)
    assert batch.X.shape == (4, 6, 2)
    assert batch.lengths.tolist() == [6, 6, 6, 6]
    assert batch.task == "sequence-to-sequence"


def test_task_override_value() -> None:
    # per-sequence target length equals L (ambiguous) -> force value
    X = [np.ones((3, 2)), np.ones((3, 2))]
    y = [np.arange(3), np.arange(3)]
    batch = check_sequences(X, y, task="sequence-to-value")
    assert batch.task == "sequence-to-value"
    assert batch.y.shape == (2, 3)


def test_predict_no_targets() -> None:
    X = [np.ones((5, 2)), np.ones((3, 2))]
    batch = check_sequences(X)
    assert batch.X.shape == (2, 5, 2)
    assert batch.y is None
    assert batch.task is None


def test_inconsistent_features_raises() -> None:
    with pytest.raises(ValueError):
        check_sequences([np.ones((5, 2)), np.ones((3, 3))])


def test_bad_ndim_raises() -> None:
    with pytest.raises(ValueError):
        check_sequences(np.ones((2, 3, 4, 5)))
