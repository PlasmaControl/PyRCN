"""Testing for pyrcn.utils module"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pyrcn.util import (FeatureExtractor, argument_parser, batched,
                        concatenate_sequences, get_mnist, new_logger,
                        value_to_tuple)
from pyrcn.util._util import seed_everything


def test_new_logger() -> None:
    directory = os.getcwd()
    logger = new_logger(name='test_logger', directory=directory)
    logger.info('Test')
    assert os.path.isfile(os.path.join(directory, 'test_logger.log'))


def test_argument_parser() -> None:
    args = argument_parser.parse_args(['-o', './', 'param0', 'param1'])
    assert os.path.isdir(args.out)
    assert 'param1' in args.params


@pytest.mark.skip(reason="no way of currently testing this")
def test_get_mnist() -> None:
    X, y = get_mnist(os.getcwd())
    assert X.shape[0] == 70000


def test_get_mnist_from_disk(tmp_path) -> None:
    # Covers the cached branch without any network download.
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    np.savez(os.path.join(str(tmp_path), 'MNIST.npz'), X=X, y=y)
    X_out, y_out = get_mnist(str(tmp_path))
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)


def test_batched_yields_batches() -> None:
    result = list(batched('ABCDEFG', 3))
    assert result == [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', )]


def test_batched_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        list(batched('ABC', 0))


def test_value_to_tuple_scalar() -> None:
    assert value_to_tuple(5, 3) == (5, 5, 5)
    assert value_to_tuple(1.5, 2) == (1.5, 1.5)


def test_value_to_tuple_passthrough() -> None:
    assert value_to_tuple((1, 2, 3), 3) == (1, 2, 3)


def test_seed_everything() -> None:
    seed_everything(seed=123)
    assert os.environ['PYTHONHASHSEED'] == '123'
    first = np.random.rand(3)
    seed_everything(seed=123)
    second = np.random.rand(3)
    np.testing.assert_array_equal(first, second)


def test_concatenate_sequences_from_lists() -> None:
    # Equal-length sequences; passing lists exercises the list->array
    # conversion branch of concatenate_sequences.
    X = [np.ones((3, 2)), np.ones((3, 2)) * 2]
    y = [np.zeros((3, 1)), np.ones((3, 1))]
    X_out, y_out, _ = concatenate_sequences(X, y)
    assert X_out.shape == (6, 2)
    assert y_out.shape == (6, 1)


def test_concatenate_sequences_object_array() -> None:
    # Ragged object arrays exercise the ndim==1 sequence-range branch.
    X = np.empty(2, dtype=object)
    y = np.empty(2, dtype=object)
    X[0] = np.ones((3, 2))
    X[1] = np.ones((5, 2)) * 2
    y[0] = np.zeros((3, 1))
    y[1] = np.ones((5, 1))
    X_out, y_out, sequence_ranges = concatenate_sequences(X, y)
    assert X_out.shape == (8, 2)
    assert y_out.shape == (8, 1)
    assert sequence_ranges.tolist() == [[0, 3], [3, 8]]


def test_concatenate_sequences_sequence_to_value() -> None:
    X = np.empty(2, dtype=object)
    y = np.empty(2, dtype=object)
    X[0] = np.ones((3, 2))
    X[1] = np.ones((4, 2))
    y[0] = 1
    y[1] = 2
    X_out, y_out, sequence_ranges = concatenate_sequences(
        X, y, sequence_to_value=True)
    assert X_out.shape == (7, 2)
    assert y_out.shape[0] == 7
    assert sequence_ranges.tolist() == [[0, 3], [3, 7]]


def test_feature_extractor_transform() -> None:
    fe = FeatureExtractor(func=lambda X: X * 2)
    fe.fit(np.ones((4, 2)))
    X_out = fe.transform(np.ones((4, 2)))
    np.testing.assert_array_equal(X_out, np.ones((4, 2)) * 2)


def test_feature_extractor_tuple_output() -> None:
    # A func returning a tuple should have its first element returned.
    fe = FeatureExtractor(func=lambda X: (X * 3, 44100))
    fe.fit(np.ones((2, 2)))
    X_out = fe.transform(np.ones((2, 2)))
    np.testing.assert_array_equal(X_out, np.ones((2, 2)) * 3)


def test_feature_extractor_kw_args() -> None:
    fe = FeatureExtractor(func=lambda X, factor: X * factor,
                          kw_args={'factor': 5})
    fe.fit(np.ones((2, 2)))
    X_out = fe.transform(np.ones((2, 2)))
    np.testing.assert_array_equal(X_out, np.ones((2, 2)) * 5)
