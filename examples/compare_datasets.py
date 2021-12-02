#!/bin/python

"""
This file contains several functions comparing datasets
"""

import sys
import os
import scipy
import scipy.stats
import numpy as np
import csv

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from pyrcn.util import new_logger, argument_parser


def dataset_imbalance(directory, *args, **kwargs):
    self_name = 'dataset_imbalance'
    logger = new_logger(self_name, directory)
    logger.info('Entering {0}'.format(self_name))

    list_dict_datasets = [{'name': 'abalone19',
                           'id': 41357},
                          {'name': 'abalone',
                           'id': 1557},
                          {'name': 'mnist_784',
                           'id': 554},
                          {'name': 'iris',
                           'id': 61}]

    for dict_dataset in list_dict_datasets:
        filepath = os.path.join(directory, '{0}.npz'
                                .format(dict_dataset['name']))
        if os.path.isfile(filepath):
            logger.info('Loading {0}'.format(filepath))
            npzfile = np.load(filepath, allow_pickle=True)
            X, y = npzfile['X'], npzfile['y']
        else:
            logger.info('Fetching {0}'.format(dict_dataset['name']))
            try:
                frame = fetch_openml(data_id=dict_dataset['id'], as_frame=True)
                X, y = frame['data'], frame['target']
                np.savez(filepath, X=X, y=y)
            except Exception as e:
                logger.warning('Failed to load and save {0}, due to error {1}'
                               .format(dict_dataset['name'], e))
                continue

        label_encoder = LabelEncoder().fit(y)
        labels, label_frequency = np.unique(label_encoder.transform(y),
                                            return_counts=True)
        ir = np.min(label_frequency) / np.max(label_frequency)
        entropy = scipy.stats.entropy(label_frequency, base=2)
        max_possible_entropy = scipy.stats.entropy(
            np.ones(label_frequency.shape), base=2)

        dict_dataset.update({
            'filepath': filepath,
            'labels': label_encoder.classes_,
            'labels_nbr': labels,
            'label_frequency': label_frequency,
            'imbalance_ratio': ir,
            'entropy': entropy,
            'max_possible_entropy': max_possible_entropy,
            'entropy_ratio': entropy / max_possible_entropy,
            'features': labels.size,
        })

    filepath = os.path.join(directory, '{0}.csv'.format(self_name))
    with open(filepath, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, list_dict_datasets[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(list_dict_datasets)
    return


def main(directory, params=()):
    # workdir
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except PermissionError as e:
            print('mkdir failed due to missing privileges: {0}'.format(e))
            exit(1)

    # subfolder for results
    file_dir = os.path.join(directory, 'compare_datasets')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    logger = new_logger('main', directory=file_dir)
    logger.info('Started main with directory={0} and params={1}'
                .format(directory, params))

    # register parameters
    experiment_names = {
        'dataset_imbalance': dataset_imbalance,
    }

    # run specified programs
    for param in params:
        if param in experiment_names:
            experiment_names[param](file_dir)
        else:
            logger.warning('Parameter {0} invalid/not found.'.format(param))


if __name__ == '__main__':
    parsed_args = argument_parser.parse_args(sys.argv[1:])
    if parsed_args.out:
        main(parsed_args.out, parsed_args.params)
    else:
        main(parsed_args.params)
    exit(0)
