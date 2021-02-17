#!/bin/python

"""
This file contains several functions testing ELMs in different configurations,
optimize them and save the results in data files and pickles
"""

import sys
import os

import scipy
import numpy as np

import pickle
import csv

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from pyrcn.util import new_logger, argument_parser, get_mnist
from pyrcn.base import InputToNode
# from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier

from pyrcn.cluster import KCluster


train_size = 70000
hidden_layer_sizes = np.array([300, 450, 1000, 2250])


def preprocessing(X, directory=os.getcwd(), save_file='preprocessor.pickle', overwrite=True):
    self_name = 'preprocessing'
    if not os.path.isdir(directory):
        raise NotADirectoryError('{0} is not a directory.'.format(directory))

    logger = new_logger(self_name, directory=directory)
    logger.info('Preprocessing was called.')

    if os.path.isfile(os.path.join(directory, save_file)) and not overwrite:
        logger.info('Load precalculated preprocessor.')
        try:
            with open(os.path.join(directory, save_file), 'rb') as f:
                prep = pickle.load(f)
        except Exception as e:
            logger.error('Unexpected error: {0}'.format(e))
            exit(1)
    else:
        logger.info('Recalculating preprocessor.')
        prep = Pipeline(steps=[
            ('whiten', PCA(whiten=False, random_state=42, n_components=450)),
            ('scale', StandardScaler()),
        ]).fit(X)
        try:
            with open(os.path.join(directory, save_file), 'wb') as f:
                pickle.dump(prep, f)
        except Exception as e:
            logger.error('Unexpected error: {0}'.format(e))
            exit(1)
    return prep.transform(X)


def elm_hyperparameters(directory):
    self_name = 'elm_hyperparameters'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, random_state=42, shuffle=True)

    param_grid = {
        'input_to_nodes__hidden_layer_size': [500, 1000, 2000, 4000],
        'input_to_nodes__input_scaling': np.linspace(start=.2, stop=5., num=10),
        'input_to_nodes__bias_scaling': np.linspace(start=.2, stop=5., num=10),
        'input_to_nodes__activation': ['tanh', 'relu'],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [.00001, .001, .1],
        'regressor__random_state': [42],
        'random_state': [42]
    }

    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())
    cv = GridSearchCV(estimator, param_grid)
    cv.fit(X_train, y_train, n_jobs=-1)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    cv_results = cv.cv_results_
    del cv_results['params']
    with open(os.path.join(directory, '{0}.csv'.format(self_name)), 'w') as f:
        f.write(','.join(cv_results.keys()) + '\n')
        for row in list(map(list, zip(*cv_results.values()))):
            f.write(','.join(map(str, row)) + '\n')


def elm_basic(directory):
    self_name = 'elm_basic'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, random_state=42, shuffle=True)

    hidden_layer_size_list = []
    best_score = []
    best_time = []
    best_scorer = []

    for hidden_layer_size in hidden_layer_sizes:
        scores = cross_validate(
            ELMClassifier(
                input_to_nodes=[('default', InputToNode(hidden_layer_size=hidden_layer_size, random_state=42))],
                regressor=Ridge(alpha=.001, random_state=42),
                random_state=42
            ),
            X_train, y_train,
            scoring='accuracy',
            cv=10,
            return_estimator=True,
            return_train_score=True,
            n_jobs=-1
        )
        hidden_layer_size_list.append(hidden_layer_size)
        best_score_index = np.argmax(scores['test_score'])
        best_score.append(scores['test_score'][best_score_index])
        best_time.append(scores['score_time'][best_score_index])
        best_scorer.append(scores['estimator'][best_score_index])
        logger.info('run hidden_layer_size = {0}, time = {1}, score = {2}'.format(hidden_layer_size, best_time[-1], best_score[-1]))

    best_overall_score_index = np.argmax(np.array(best_score))
    best_overall_score = best_score[best_overall_score_index]
    best_overall_scorer = best_scorer[best_overall_score_index]

    try:
        # noinspection PyTypeChecker
        np.savetxt(
            fname=os.path.join(directory, '{0}.csv'.format(self_name)),
            X=np.hstack((np.array(hidden_layer_size_list, ndmin=2).T, np.array(best_time, ndmin=2).T, np.array(best_score, ndmin=2).T)),
            fmt='%f,%f,%f',
            header='hidden_layer_size,best_time,best_score',
            comments='best scorer = {0} (score = {1})\n\n'.format(str(best_overall_scorer.get_params()), best_overall_score)
        )
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_preprocessed(directory):
    self_name = 'elm_preprocessed'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)
    X_preprocessed = preprocessing(X, directory=directory, overwrite=False)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, train_size=train_size, random_state=42, shuffle=True)

    hidden_layer_size_list = []
    best_score = []
    best_time = []
    best_scorer = []

    for hidden_layer_size in hidden_layer_sizes:
        scores = cross_validate(
            ELMClassifier(
                input_to_nodes=[('default', InputToNode(hidden_layer_size=hidden_layer_size, random_state=42))],
                regressor=Ridge(alpha=.001, random_state=42),
                random_state=42
            ),
            X_train, y_train,
            scoring='accuracy',
            cv=10,
            return_estimator=True,
            return_train_score=True,
            n_jobs=-1
        )
        hidden_layer_size_list.append(hidden_layer_size)
        best_score_index = np.argmax(scores['test_score'])
        best_score.append(scores['test_score'][best_score_index])
        best_time.append(scores['score_time'][best_score_index])
        best_scorer.append(scores['estimator'][best_score_index])
        logger.info('run hidden_layer_size = {0}, time = {1}, score = {2}'.format(hidden_layer_size, best_time[-1], best_score[-1]))

    best_overall_score_index = np.argmax(np.array(best_score))
    best_overall_score = best_score[best_overall_score_index]
    best_overall_scorer = best_scorer[best_overall_score_index]

    try:
        # noinspection PyTypeChecker
        np.savetxt(
            fname=os.path.join(directory, '{0}.csv'.format(self_name)),
            X=np.hstack((np.array(hidden_layer_size_list, ndmin=2).T, np.array(best_time, ndmin=2).T, np.array(best_score, ndmin=2).T)),
            fmt='%f,%f,%f',
            header='hidden_layer_size,best_time,best_score',
            comments='best scorer = {0} (score = {1})\n\n'.format(str(best_overall_scorer.get_params()), best_overall_score)
        )
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_coates(directory):
    self_name = 'elm_coates'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)
    X_preprocessed = preprocessing(X, directory=directory, overwrite=False)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    cluster = KMeans(n_clusters=20, init='k-means++', n_init=10, random_state=42).fit(X_preprocessed)
    logger.info('cluster.cluster_centers_ = {0}'.format(cluster.cluster_centers_))
    with open(os.path.join(directory, 'clusterer.pickle'), 'wb') as f:
        pickle.dump(cluster, f)

    X_coates = np.dot(X_preprocessed, cluster.cluster_centers_.T)

    X_train, X_test, y_train, y_test = train_test_split(X_coates, y_encoded, train_size=train_size, random_state=42, shuffle=True)

    hidden_layer_size_list = []
    best_score = []
    best_time = []
    best_scorer = []

    for hidden_layer_size in hidden_layer_sizes:
        scores = cross_validate(
            ELMClassifier(
                input_to_nodes=[('default', InputToNode(hidden_layer_size=hidden_layer_size, random_state=42))],
                regressor=Ridge(alpha=.001, random_state=42),
                random_state=42
            ),
            X_train, y_train,
            scoring='accuracy',
            cv=10,
            return_estimator=True,
            return_train_score=True,
            n_jobs=-1
        )
        hidden_layer_size_list.append(hidden_layer_size)
        best_score_index = np.argmax(scores['test_score'])
        best_score.append(scores['test_score'][best_score_index])
        best_time.append(scores['score_time'][best_score_index])
        best_scorer.append(scores['estimator'][best_score_index])
        logger.info('run hidden_layer_size = {0}, time = {1}, score = {2}'.format(hidden_layer_size, best_time[-1], best_score[-1]))

    best_overall_score_index = np.argmax(np.array(best_score))
    best_overall_score = best_score[best_overall_score_index]
    best_overall_scorer = best_scorer[best_overall_score_index]

    try:
        # noinspection PyTypeChecker
        np.savetxt(
            fname=os.path.join(directory, '{0}.csv'.format(self_name)),
            X=np.hstack((np.array(hidden_layer_size_list, ndmin=2).T, np.array(best_time, ndmin=2).T, np.array(best_score, ndmin=2).T)),
            fmt='%f,%f,%f',
            header='hidden_layer_size,best_time,best_score',
            comments='best scorer = {0} (score = {1})\n\n'.format(str(best_overall_scorer.get_params()), best_overall_score)
        )
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def main(directory, params):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except PermissionError as e:
            print('mkdir failed due to missing privileges: {0}'.format(e))
            exit(1)

    workdir = directory

    file_dir = os.path.join(directory, 'mnist-elm')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    logger = new_logger('main', directory=file_dir)
    logger.info('Started main with directory={0} and params={1}'.format(directory, params))

    # register parameters
    experiment_names = {
        'elm_hyperparameters': elm_hyperparameters,
        'elm_basic': elm_basic,
        'elm_preprocessed': elm_preprocessed,
        'elm_coates': elm_coates
    }

    for param in params:
        if param in experiment_names:
            experiment_names[param](file_dir)
        else:
            logger.warning('Parameter {0} invalid/not found.'.format(param))


if __name__ == '__main__':
    parsed_args = argument_parser.parse_args(sys.argv[1:])
    if os.path.isdir(parsed_args.out):
        main(parsed_args.out, parsed_args.params)
    else:
        main(parsed_args.params)
    exit(0)
