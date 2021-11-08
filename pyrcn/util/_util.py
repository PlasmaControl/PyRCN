"""The :mod:`pyrcn.util` contains utilities for running, testing and analyzing."""

# Author: Peter Steiner <peter.steiner@tu-dresden.de>,
# Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import sys
from typing import Union, Tuple

import os
import logging
import argparse
import numpy as np

from sklearn.utils import check_X_y, check_consistent_length
from sklearn.datasets import fetch_openml


argument_parser = argparse.ArgumentParser(
    description='Standard input parser for HPC on PyRCN.')
argument_parser.add_argument('-o', '--out', metavar='outdir', nargs='?',
                             help='output directory', dest='out', type=str)
argument_parser.add_argument(dest='params', metavar='params', nargs='*',
                             help='optional parameter for scripts')

tud_colors = {
    'darkblue': (0 / 255., 48 / 255., 94 / 255., 1.0),
    'gray': (114 / 255., 120 / 255., 121 / 255., 1.0),
    'lightblue': (0 / 255., 106 / 255., 179 / 255., 1.0),
    'darkgreen': (0 / 255., 125 / 255., 64 / 255., 1.0),
    'lightgreen': (106 / 255., 176 / 255., 35 / 255., 1.0),
    'darkpurple': (84 / 255., 55 / 255., 138 / 255., 1.0),
    'lightpurple': (147 / 255., 16 / 255., 126 / 255., 1.0),
    'orange': (238 / 255., 127 / 255., 0 / 255., 1.0),
    'red': (181 / 255., 28 / 255., 28 / 255., 1.0)
}

# noinspection PyArgumentList
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


def new_logger(name: str, directory: str = os.getcwd()) -> logging.Logger:
    """Register a new logger for logfiles."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = logging.FileHandler(os.path.join(directory, '{0}.log'.format(name)))
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
        X, y = fetch_openml(data_id=554, return_X_y=True, cache=True, as_frame=False)
        logging.info('Fetched dataset')
        np.savez(npzfilepath, X=X, y=y)
        return X, y


def concatenate_sequences(X: Union[list, np.ndarray], y: Union[list, np.ndarray],
                          sequence_to_value: bool = False) \
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
        sequence_ranges[:, 1] = np.cumsum([X[k].shape[0] for k, _ in enumerate(X)])
        sequence_ranges[1:, 0] = sequence_ranges[:-1, 1]
        for k, _ in enumerate(X):
            X[k], y[k] = check_X_y(X[k], y[k], multi_output=True)
    return np.concatenate(X), np.concatenate(y), sequence_ranges
