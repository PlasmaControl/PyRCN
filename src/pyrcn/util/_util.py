"""The :mod:`pyrcn.util` has utilities for running, testing and analyzing."""

# Author: Peter Steiner <peter.steiner@tu-dresden.de>,
# Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import sys
from typing import Union, Tuple, Iterable

import random
import os
import torch
import logging
import argparse
import numpy as np
from itertools import islice

from sklearn.utils import check_X_y, check_consistent_length
from sklearn.datasets import fetch_openml


argument_parser = argparse.ArgumentParser(
    description='Standard input parser for HPC on PyRCN.')
argument_parser.add_argument('-o', '--out', metavar='outdir', nargs='?',
                             help='output directory', dest='out', type=str)
argument_parser.add_argument(dest='params', metavar='params', nargs='*',
                             help='optional parameter for scripts')

# noinspection PyArgumentList
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


def batched(iterable: Iterable, n: int) -> Tuple:
    """
    Iterate over batches of size n.

    Parameters
    ----------
    iterable : Iterable
        The object over which to be iterated.
    n : int
        The batch size

    Returns
    -------
    batch : Tuple
        A batch from the iterable.

    Notes
    -----
    Starting from Python 3.12, this is included in `itertools`_.

    .. _itertools:
    https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def value_to_tuple(value: Union[float, int],
                   size: Union[float, int, Tuple[Union[float, int], ...]]) \
        -> Tuple[Union[float, int], ...]:
    """
    Convert a value to a tuple of values.

    Parameters
    ----------
    value : Union[float, int, Tuple[Union[float, int], ...]]
        The value to be inserted in the tuple.
    size : int
        The length of the tuple.

    Returns
    -------
    value : Tuple[Union[float, int], ...]
        Tuple of values.
    """
    if isinstance(value, float) or isinstance(value, int):
        return (value, ) * size
    elif isinstance(value, Tuple):
        return value


def seed_everything(seed: int = 42) -> None:
    """
    Fix all random number generators to reproduce results.

    Parameters
    ----------
    seed : int, default = 42
        The default seed for the random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def new_logger(name: str, directory: str = os.getcwd()) -> logging.Logger:
    """Register a new logger for logfiles."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = logging.FileHandler(
        os.path.join(directory, '{0}.log'.format(name)))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_mnist(directory: str = os.getcwd()) -> Tuple[np.ndarray, np.ndarray]:
    """Load the MNIST dataset from harddisk."""
    npzfilepath = os.path.join(directory, 'MNIST.npz')

    if os.path.isfile(npzfilepath):
        npzfile = np.load(npzfilepath, allow_pickle=True)
        logging.info('Dataset loaded')
        return npzfile['X'], npzfile['y']
    else:
        X, y = fetch_openml(
            data_id=554, return_X_y=True, cache=True, as_frame=False)
        logging.info('Fetched dataset')
        np.savez(npzfilepath, X=X, y=y)
        return X, y


def concatenate_sequences(X: Union[list, np.ndarray],
                          y: Union[list, np.ndarray],
                          sequence_to_value: bool = False)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate multiple sequences to scikit-learn compatible numpy arrays.

    ´Parameters
    -----------
    X : Union[list, np.ndarray] of shape=(n_sequences, )
        All sequences. Note that all elements in ```X```
        must have at least one equal dimension.
    y : Union[list, np.ndarray] of shape=(n_sequences, )
        All sequences. Note that all elements in ```X```
        must have at least one equal dimension.
    sequence_to_value : bool, default=False
        If true, expand each element of y to the sequence length

    Returns
    -------
    X : np.ndarray of shape=(n_samples, n_features)
        Input data where n_samples is the accumulated length of all sequences
    y : np.ndarray of shape=(n_samples, n_features) or shape=(n_samples, )
        Target data where n_samples is the accumulated length of all sequences
    sequence_ranges : Union[None, np.ndarray] of shape=(n_sequences, 2)
        Sequence border indicator matrix
    """
    if isinstance(X, list):
        X = np.asarray(X)
    if isinstance(y, list):
        y = np.asarray(y)
    X = np.array(X)
    y = np.array(y)
    if sequence_to_value:
        for k, _ in enumerate(y):
            y[k] = np.repeat(y[k], X[k].shape[0])

    check_consistent_length(X, y)
    sequence_ranges: np.ndarray = np.ndarray([])
    if X.ndim == 1:
        sequence_ranges = np.zeros((X.shape[0], 2), dtype=int)
        sequence_ranges[:, 1] = np.cumsum(
            [X[k].shape[0] for k, _ in enumerate(X)])
        sequence_ranges[1:, 0] = sequence_ranges[:-1, 1]
        for k, _ in enumerate(X):
            X[k], y[k] = check_X_y(X[k], y[k], multi_output=True)
    return np.concatenate(X), np.concatenate(y), sequence_ranges
