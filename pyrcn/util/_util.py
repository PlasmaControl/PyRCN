"""
The :mod:`pyrcn.util` contains utilities for runnung, testing and analyzing the reservoir computing modules
"""

# Author: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

import sys
import os
import logging
import argparse
import csv
import numpy as np

from sklearn.utils import check_X_y, check_consistent_length
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.datasets import fetch_openml


argument_parser = argparse.ArgumentParser(description='Standard input parser for HPC on PyRCN.')
argument_parser.add_argument('-o', '--out', metavar='outdir', nargs='?', help='output directory', dest='out', type=str)
argument_parser.add_argument(dest='params', metavar='params', nargs='*', help='optional parameter for scripts')

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


def new_logger(name, directory=os.getcwd()):
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = logging.FileHandler(os.path.join(directory, '{0}.log'.format(name)))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_mnist(directory=os.getcwd()):
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


def export_ragged_time_series(filename, times, values, dtype=float, 
                              delimiter='\t', header=False):
    """
    """
    if header:
        start_row = 1
    else:
        start_row = 0

    with open(filename, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)
        if header:
            writer.writerow(header)
        for t, v in zip(times, values):
            writer.writerow([t]+list(v))


def concatenate_sequences(X, y, sequence_to_value=False):
    if isinstance(X, list):
        X = np.asarray(X)
    if isinstance(y, list):
        y = np.asarray(y)
    if sequence_to_value:
        for k, _ in enumerate(y):
            y[k] = np.repeat(y[k], X[k].shape[0])

    check_consistent_length(X, y)
    sequence_ranges = None
    if X.ndim == 1:
        sequence_ranges = np.zeros((X.shape[0], 2), dtype=int)
        sequence_ranges[:, 1] = np.cumsum([X[k].shape[0] for k, _ in enumerate(X)])
        sequence_ranges[1:, 0] = sequence_ranges[:-1, 1]
        for k, _ in enumerate(X):
            X[k], y[k] = check_X_y(X[k], y[k], multi_output=True)
    return np.concatenate(X), np.concatenate(y), sequence_ranges
