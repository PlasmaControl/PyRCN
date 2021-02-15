#!/bin/python

"""
This file contains several functions testing ELMs in different configurations,
optimize them and save the results in data files and pickles
"""

import sys
import os

import scipy
import numpy as np

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

from pyrcn.util import new_logger, argument_parser, get_mnist

from pyrcn.cluster import KCluster


def elm_basic(directory):
    logger = new_logger('elm_basic', directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))


def main(directory, params):
    if os.path.isdir(directory):
        workdir = directory
    else:
        print('ERROR: directory not found!')

    file_dir = os.path.join(directory, 'mnist-elm')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    logger = new_logger('main', directory=file_dir)
    logger.info('Started main with directory={0} and params={1}'.format(directory, params))

    experiment_names = {
        'elm_basic': elm_basic,
    }

    for param in params:
        if param in experiment_names:
            experiment_names[param](file_dir)


if __name__ == '__main__':
    parsed_args = argument_parser.parse_args(sys.argv[1:])
    if os.path.isdir(parsed_args.out):
        main(parsed_args.out, parsed_args.params)
    else:
        main(parsed_args.params)
    exit(0)
