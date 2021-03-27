#!/bin/python

"""
This file contains several functions testing ELMs in different configurations,
optimize them and save the results in data files and pickles
"""

import sys
import os, glob

import scipy
import numpy as np

import pickle
import csv
import copy

import time

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedShuffleSplit

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import Ridge

from pyrcn.util import new_logger, argument_parser, get_mnist
from pyrcn.base import InputToNode, ACTIVATIONS, BatchIntrinsicPlasticity, PredefinedWeightsInputToNode
from pyrcn.cluster import KCluster
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier

train_size = 60000


def train_kmeans(directory):
    self_name = 'train_kmeans'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    # scale X, so $X \in [0, 1]$
    X /= 255.

    list_n_components = [50]  # [50, 100]
    list_n_clusters = [200]  # [20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]

    for n_components in list_n_components:
        pca = PCA(n_components=n_components, random_state=42).fit(X)
        X_pca = pca.transform(X)
        logger.info('pca{0}: explained variance ratio = {1}'.format(n_components, np.sum(pca.explained_variance_ratio_)))

        for n_clusters in list_n_clusters:
            # minibatch kmeans
            kmeans_basename = 'minibatch-pca{0}+kmeans{1}'.format(n_components, n_clusters)

            # only if file does not exist
            if not os.path.isfile(os.path.join(directory, '{0}_matrix.npy'.format(kmeans_basename))):
                clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=42, batch_size=5000, n_init=5).fit(X_pca)
                np.save(os.path.join(directory, '{0}_matrix.npy'.format(kmeans_basename)), np.dot(pca.components_.T, clusterer.cluster_centers_.T))

                # assemble pipeline
                p = make_pipeline(pca, clusterer)
                with open(os.path.join(directory, '{0}_pipeline.pickle'.format(kmeans_basename)), 'wb') as f:
                    pickle.dump(p, f)

                logger.info('successfuly trained MiniBatchKMeans and saved to npy/pickle {0}'.format(kmeans_basename))

            # original kmeans
            kmeans_basename = 'original-pca{0}+kmeans{1}'.format(n_components, n_clusters)

            if n_clusters < 2000 and not os.path.isfile(os.path.join(directory, '{0}_matrix.npy'.format(kmeans_basename))):
                clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=5).fit(X_pca)
                np.save(os.path.join(directory, '{0}_matrix.npy'.format(kmeans_basename)), np.dot(pca.components_.T, clusterer.cluster_centers_.T))

                # assemble pipeline
                p = make_pipeline(pca, clusterer)
                with open(os.path.join(directory, '{0}_pipeline.pickle'.format(kmeans_basename)), 'wb') as f:
                    pickle.dump(p, f)

                logger.info('successfuly trained KMeans and saved to npy/pickle {0}'.format(kmeans_basename))


def elm_hyperparameters(directory):
    self_name = 'elm_hyperparameters'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    X = X/255.

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[:train_size], y_encoded[train_size:]

    param_grid = {
        'input_to_nodes__hidden_layer_size': [2000],
        'input_to_nodes__input_scaling': np.logspace(start=-2, stop=2, base=10, num=7),
        'input_to_nodes__bias_scaling': np.logspace(start=-2, stop=2, base=10, num=7),
        'input_to_nodes__activation': ['tanh'],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [1e-5],
        'regressor__random_state': [42],
        'random_state': [42]
    }

    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())
    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    cv.fit(X_train, y_train)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    cv_results = cv.cv_results_
    del cv_results['params']
    with open(os.path.join(directory, '{0}_scaling.csv'.format(self_name)), 'w') as f:
        f.write(','.join(cv_results.keys()) + '\n')
        for row in list(map(list, zip(*cv_results.values()))):
            f.write(','.join(map(str, row)) + '\n')

    param_grid = {
        'input_to_nodes__hidden_layer_size': [500, 1000, 2000, 4000],
        'input_to_nodes__input_scaling': [cv.best_params_['input_to_nodes__input_scaling']],
        'input_to_nodes__bias_scaling': [cv.best_params_['input_to_nodes__bias_scaling']],
        'input_to_nodes__activation': ['tanh', 'relu', 'bounded_relu', 'logistic', 'identity'],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [1e-5],
        'regressor__random_state': [42],
        'random_state': [42]
    }

    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    cv.fit(X_train, y_train)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    cv_results = cv.cv_results_
    del cv_results['params']
    with open(os.path.join(directory, '{0}_size.csv'.format(self_name)), 'w') as f:
        f.write(','.join(cv_results.keys()) + '\n')
        for row in list(map(list, zip(*cv_results.values()))):
            f.write(','.join(map(str, row)) + '\n')

    param_grid = {
        'input_to_nodes__hidden_layer_size': [cv.best_params_['input_to_nodes__hidden_layer_size']],
        'input_to_nodes__input_scaling': [cv.best_params_['input_to_nodes__input_scaling']],
        'input_to_nodes__bias_scaling': [cv.best_params_['input_to_nodes__bias_scaling']],
        'input_to_nodes__activation': [cv.best_params_['input_to_nodes__activation']],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [.00001, .001, .1],
        'regressor__random_state': [42],
        'random_state': [42]
    }

    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=1, scoring='accuracy')
    cv.fit(X_train, y_train)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    cv_results = cv.cv_results_
    del cv_results['params']
    with open(os.path.join(directory, '{0}_alpha.csv'.format(self_name)), 'w') as f:
        f.write(','.join(cv_results.keys()) + '\n')
        for row in list(map(list, zip(*cv_results.values()))):
            f.write(','.join(map(str, row)) + '\n')


def elm_basic(directory):
    self_name = 'elm_basic'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    # scaling
    X /= 255.

    # encode labels
    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, random_state=42)

    param_grid = [{
            'input_to_nodes__hidden_layer_size': [500, 2000],
            'input_to_nodes__input_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__bias_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        },
        {
            'input_to_nodes__hidden_layer_size': [2000],
            'input_to_nodes__input_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__bias_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__activation': ['tanh'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        }
    ]

    # prepare grid search
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        refit=False,
        cv=StratifiedShuffleSplit(n_splits=1, test_size=1/7, random_state=42))

    # run!
    cv.fit(X_train, y_train)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']

    # save results
    try:
        with open(os.path.join(directory, 'elm_basic.csv'), 'w') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_pca(directory):
    self_name = 'elm_pca'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    # encode y
    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # scale X
    X /= 255.

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X[:train_size], y[:train_size], train_size=50000, random_state=42)

    # prepare parameter grids
    param_grid_basic = {
            'input_to_nodes__hidden_layer_size': 2000,
            'input_to_nodes__input_scaling': 1.,
            'input_to_nodes__bias_scaling': 0.,
            'input_to_nodes__activation': 'relu',
            'input_to_nodes__random_state': 42,
            'regressor__alpha': 1e-5,
            'regressor__random_state': 42,
            'random_state': 42
    }

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())

    # initialize filepath
    filepath = os.path.join(directory, '{0}_basic.csv'.format(self_name))

    # initialize param dict
    param_dict_job = estimator.get_params().copy()
    param_dict_job.update(param_grid_basic)

    # initialize results dict
    results_dict_job = param_dict_job.copy()
    # add dummy results
    results_dict_job.update({'time_fit': 0, 'time_pred': 0, 'score': 0, 'pca_n_components': 0})

    # preprocessing pca
    try:
        # write header
        with open(filepath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
            writer.writeheader()

        for pca_n_components in [10, 20, 50, 100, 200, 500, 784]:
            results_dict_job.update({'pca_n_components': pca_n_components})
            estimator.set_params(**param_dict_job)

            # preprocessing
            pca = PCA(n_components=pca_n_components).fit(X_train)
            X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)

            # run!
            time_start = time.time()
            estimator.fit(X_train_pca, y_train)
            time_fit = time.time()
            y_pred = estimator.predict(X_test_pca)
            time_pred = time.time()
            # run end!

            results_dict_job.update({
                'time_fit': time_fit - time_start,
                'time_pred': time_pred - time_fit,
                'score': accuracy_score(y_test, y_pred)
            })

            logger.info('pca.n_components_: {0}, score: {1}'.format(pca_n_components, results_dict_job['score']))

            with open(filepath, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
                writer.writerow(results_dict_job)
    except MemoryError as e:
        logger.error('Memory error: {0}'.format(e))
    except PermissionError as e:
        logger.error('Missing privileges: {0}'.format(e))
    except Exception as e:
        logger.error('Unexpected exception: {0}'.format(e))


def elm_preprocessed(directory):
    self_name = 'elm_preprocessed'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # preprocessing
    X /= 255.
    pca = PCA(n_components=50).fit(X)
    X_preprocessed = pca.transform(X)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, train_size=train_size, random_state=42)

    # prepare parameter grid
    param_grid = [{
            'input_to_nodes__hidden_layer_size': [500, 2000],
            'input_to_nodes__input_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__bias_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        },
        {
            'input_to_nodes__hidden_layer_size': [2000],
            'input_to_nodes__input_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__bias_scaling': np.logspace(start=-3, stop=1, base=10, num=6),
            'input_to_nodes__activation': ['tanh'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        }
    ]

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())

    # setup grid search
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        refit=False,
        cv=StratifiedShuffleSplit(n_splits=1, test_size=1/7, random_state=42))

    # run!
    cv.fit(X_train, y_train)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']

    # save results
    try:
        with open(os.path.join(directory, 'elm_preprocessed.csv'), 'w') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_random_state(directory):
    self_name = 'elm_random_state'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    X /= 255.

    pca = PCA(n_components=50).fit(X)
    X_preprocessed = pca.transform(X)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    # prepare parameter grid
    param_grid = [{
            'input_to_nodes__hidden_layer_size': [2000],
            'input_to_nodes__input_scaling': [1.],
            'input_to_nodes__bias_scaling': [0.],
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': np.random.randint(low=1, high=2**16-1, size=10),
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        },
    ]

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())

    # setup grid search
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        refit=False,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    # run!
    cv.fit(X_preprocessed, y_encoded)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']

    # save results
    try:
        with open(os.path.join(directory, '{0}.csv'.format(self_name)), 'w') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_bip(directory):
    self_name = 'elm_bip'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # preprocessing
    X /= 255.
    pca = PCA(n_components=50).fit(X)
    X_preprocessed = pca.transform(X)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    # prepare parameter grid
    param_grid = [
        {
            'input_to_nodes__hidden_layer_size': [500, 1000, 2000, 4000],
            'input_to_nodes__activation': ['tanh'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
        }
    ]

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=BatchIntrinsicPlasticity(), regressor=Ridge())

    # setup grid search
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        refit=False,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    # run!
    cv.fit(X, y_encoded)
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']

    # save results
    try:
        with open(os.path.join(directory, '{0}.csv'.format(self_name)), 'w') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def elm_hidden_layer_size(directory):
    self_name = 'elm_hidden_layer_size'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    # encode y
    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # scale X
    X /= 255.

    # split train test
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[:train_size], y_encoded[train_size:]

    # fan-out from paper
    fan_out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    # prepare parameter grids
    param_grid_basic = {
            'input_to_nodes__hidden_layer_size': 0,
            'input_to_nodes__input_scaling': 1.,
            'input_to_nodes__bias_scaling': 0.,
            'input_to_nodes__activation': 'relu',
            'input_to_nodes__random_state': 42,
            'chunk_size': 1000,
            'regressor__alpha': 1e-5,
            'random_state': 42
    }

    param_grid_pca = {
            'input_to_nodes__hidden_layer_size': 0,
            'input_to_nodes__input_scaling': 1.,
            'input_to_nodes__bias_scaling': 0.,
            'input_to_nodes__activation': 'relu',
            'input_to_nodes__random_state': 42,
            'chunk_size': 1000,
            'regressor__alpha': 1e-5,
            'random_state': 42
    }

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=IncrementalRegression())

    # basic
    try:
        # initialize filepath
        csv_filepath = os.path.join(directory, '{0}_basic.csv'.format(self_name))

        # initialize param dict
        param_dict_job = estimator.get_params().copy()
        param_dict_job.update(param_grid_basic)

        # initialize results dict
        results_dict_job = param_dict_job.copy()
        # add dummy results
        results_dict_job.update({'time_fit': 0, 'time_pred': 0, 'score': 0})

        # write header
        with open(csv_filepath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
            writer.writeheader()

        for hls in 784 * np.array(fan_out):
            param_dict_job.update({'input_to_nodes__hidden_layer_size': hls})
            estimator.set_params(**param_dict_job)

            # run!
            time_start = time.time()
            estimator.fit(X_train, y_train)
            time_fit = time.time()
            y_pred = estimator.predict(X_test)
            time_pred = time.time()
            # run end!

            results_dict_job.update(estimator.get_params())

            results_dict_job.update({
                'time_fit': time_fit - time_start,
                'time_pred': time_pred - time_fit,
                'score': accuracy_score(y_test, y_pred)
            })

            logger.info('hidden_layer_size: {0}, score: {1}'.format(hls, results_dict_job['score']))

            with open(csv_filepath, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
                writer.writerow(results_dict_job)

            del estimator.input_to_nodes._hidden_layer_state

            with open(os.path.join(directory, 'elmc_hls{0}_basic.pickle'.format(hls)), 'wb') as f:
                pickle.dump(estimator, f)
    except MemoryError as e:
        logger.error('Memory error: {0}'.format(e))
        pass
    except PermissionError as e:
        logger.error('Missing privileges: {0}'.format(e))
        pass

    # preprocessing pca
    try:
        # initialize filepath
        csv_filepath = os.path.join(directory, '{0}_pca.csv'.format(self_name))

        # preprocessing
        pca50 = PCA(n_components=50).fit(X_train)
        X_train_pca50, X_test_pca50 = pca50.transform(X_train), pca50.transform(X_test)

        pca100 = PCA(n_components=100).fit(X_train)
        X_train_pca100, X_test_pca100 = pca100.transform(X_train), pca100.transform(X_test)

        list_dict_pca = [{
            'n_components': 50,
            'X_train': X_train_pca50,
            'X_test': X_test_pca50
        }, {
            'n_components': 100,
            'X_train': X_train_pca100,
            'X_test': X_test_pca100
        }]
        logger.info('Preprocessing successful!')

        # initialize param dict
        param_dict_job = estimator.get_params().copy()
        param_dict_job.update(param_grid_pca)

        # initialize results dict
        results_dict_job = param_dict_job.copy()
        # add dummy results
        results_dict_job.update({'time_fit': 0, 'time_pred': 0, 'score': 0, 'pca_n_components': 0})

        # write header
        with open(csv_filepath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
            writer.writeheader()

        for dict_pca in list_dict_pca:
            results_dict_job.update({'pca_n_components': dict_pca['n_components']})
            for hls in np.concatenate((100 * np.array(fan_out), 784 * np.array(fan_out)), axis=0):
                param_dict_job.update({'input_to_nodes__hidden_layer_size': hls})
                estimator.set_params(**param_dict_job)

                # run!
                time_start = time.time()
                estimator.fit(dict_pca['X_train'], y_train)
                time_fit = time.time()
                y_pred = estimator.predict(dict_pca['X_test'])
                time_pred = time.time()
                # run end!

                results_dict_job.update(estimator.get_params())

                results_dict_job.update({
                    'time_fit': time_fit - time_start,
                    'time_pred': time_pred - time_fit,
                    'score': accuracy_score(y_test, y_pred)
                })

                logger.info('n_components: {2}, hidden_layer_size: {0}, score: {1}'.format(hls, results_dict_job['score'], results_dict_job['pca_n_components']))

                with open(csv_filepath, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=results_dict_job.keys())
                    writer.writerow(results_dict_job)

                with open(os.path.join(directory, 'elmc_hls{0}_pca{1}.pickle'.format(hls, results_dict_job['pca_n_components'])), 'wb') as f:
                    pickle.dump(estimator, f)
    except MemoryError as e:
        logger.error('Memory error: {0}'.format(e))
        pass
    except PermissionError as e:
        logger.error('Missing privileges: {0}'.format(e))
        pass


def elm_coates(directory):
    self_name = 'elm_coates'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    filepath_label_encoder = os.path.join(directory, 'label_encoder_{0}.pickle'.format(self_name))

    # save label_encoder
    try:
        with open(filepath_label_encoder, 'wb') as f:
            pickle.dump(label_encoder, f)
    except Exception as e:
        logger.error('Unexpected error: {0}'.format(e))
        exit(1)

    # scale X so X in [0, 1]
    X /= 255.

    X_train, X_test, y_train, y_test = X[:train_size, ...], X[train_size:], y_encoded[:train_size], y_encoded[train_size:]

    csv_filepath = os.path.join(directory, '{0}.csv'.format(self_name))

    # read input matrices from files
    list_filepaths = []
    for filepath in glob.glob(os.path.join(directory, '*pca*+kmeans*_matrix.npy')):
        logger.info('matrix file found: {0}'.format(filepath))
        list_filepaths.append(filepath)
        filename = os.path.splitext(os.path.basename(filepath))[0]

        est_filepath = os.path.join(directory, 'est_coates-{0}.pickle'.format(filename))
        pred_filpath = os.path.join(directory, 'est_coates-{0}-predicted.npz'.format(filename))

        # only if files do not exist yet
        if not os.path.isfile(csv_filepath) or not os.path.isfile(est_filepath) or not os.path.isfile(pred_filpath):
            # setup estimator
            estimator = ELMClassifier(
                input_to_nodes=PredefinedWeightsInputToNode(
                    predefined_input_weights=np.load(filepath),
                    input_scaling=1.0,
                    bias_scaling=0.0,
                    activation='relu',
                    random_state=42,
                ),
                regressor=IncrementalRegression(alpha=1e-5),
                chunk_size=1000,
                random_state=42,
            )
            logger.info('Estimator params: {0}'.format(estimator.get_params().keys()))

            # !run
            time_start = time.time()
            estimator.fit(X_train, y_train)
            time_fitted = time.time()
            y_pred = estimator.predict(X_test)
            time_predicted = time.time()
            # !run

            # results
            dict_results = estimator.get_params()
            dict_results.update({
                'filename': filename,
                'fit_time': time_fitted - time_start,
                'score_time': time_predicted - time_fitted,
                'score': accuracy_score(y_test, y_pred)
            })

            # drop data
            dict_results.pop('input_to_nodes__predefined_input_weights')
            dict_results.pop('input_to_nodes')
            dict_results.pop('regressor')

            logger.info('fitted time {1}, score on test set: {0}'.format(dict_results['score'], dict_results['fit_time']))

            # save estimator
            try:
                with open(est_filepath, 'wb') as f:
                    pickle.dump(estimator, f)
            except Exception as e:
                logger.error('Unexpected error: {0}'.format(e))
                exit(1)

            # save results
            try:
                if not os.path.isfile(csv_filepath):
                    with open(csv_filepath, 'a') as f:
                        f.write(','.join(dict_results.keys()) + '\n')
                        f.write(','.join([str(item) for item in dict_results.values()]) + '\n')
                else:
                    with open(csv_filepath, 'a') as f:
                        f.write(','.join([str(item) for item in dict_results.values()]) + '\n')
            except PermissionError as e:
                print('Missing privileges: {0}'.format(e))

            # save prediction
            np.savez_compressed(
                pred_filpath,
                X_test=X_test,
                y_test=label_encoder.inverse_transform(y_test),
                y_pred=label_encoder.inverse_transform(y_pred),
            )

    if not list_filepaths:
        logger.warning('no input weights matrices found')
        return


def elm_coates_stacked(directory):
    self_name = 'elm_coates_stacked'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # scale X so X in [0, 1]
    X /= 255.

    # setup parameter grid
    param_grid = {
        'chunk_size': [10000],
        'input_to_nodes__input_scaling': np.logspace(start=-3, stop=1, base=10, num=3),
        'input_to_nodes__bias_scaling': [0.],  # np.logspace(start=-3, stop=1, base=10, num=6),
        'input_to_nodes__activation': ['relu'],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [1e-5],
        'random_state': [42]
    }

    # read input matrices from files
    list_filepaths = []
    predefined_input_weights = np.empty((784, 0))
    for filepath in glob.glob(os.path.join(directory, '*kmeans1*matrix.npy')):
        logger.info('matrix file found: {0}'.format(filepath))
        list_filepaths.append(filepath)
        predefined_input_weights = np.append(predefined_input_weights, np.load(filepath), axis=1)

    # setup estimator
    estimator = ELMClassifier(
        PredefinedWeightsInputToNode(predefined_input_weights=predefined_input_weights),
        IncrementalRegression())
    # logger.info('[pass] Estimator params: {0}'.format(estimator.get_params()))
    logger.info('Estimator params: {0}'.format(estimator.get_params().keys()))
    # return

    # setup grid search
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        verbose=1,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    # run!
    cv.fit(X, y_encoded)
    cv_best_params = cv.best_params_
    del cv_best_params['input_to_nodes__predefined_input_weights']

    # refine best params
    logger.info('best parameters: {0} (score: {1})'.format(cv_best_params, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']
    del cv_results['param_input_to_nodes__predefined_input_weights']

    # save results
    try:
        with open(os.path.join(directory, '{0}.csv'.format(self_name)), 'w') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))

    if not list_filepaths:
        logger.warning('no input weights matrices found')
        return


def significance(directory):
    self_name = 'significance'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    # setup modified input to node
    class PCAKMeansInputToNode(InputToNode):
        def __init__(self, hidden_layer_size=500, activation='relu', input_scaling=1., bias_scaling=0., random_state=None, pca_components=np.array([])):
            super().__init__(sparsity=1., hidden_layer_size=hidden_layer_size, activation=activation, input_scaling=input_scaling, bias_scaling=bias_scaling, random_state=random_state)
            self.clusterer = MiniBatchKMeans(init='k-means++', batch_size=5000, n_init=5)
            self.pca_components = pca_components

        def fit(self, X, y=None):
            # no validation!
            if self.random_state is None:
                self.random_state = np.random.RandomState()
            elif isinstance(self.random_state, (int, np.integer)):
                self.random_state = np.random.RandomState(self.random_state)
            elif isinstance(self.random_state, np.random.RandomState):
                pass
            else:
                raise ValueError('random_state is not valid, got {0}.'.format(self.random_state))

            if self.pca_components.size != 0:
                X_preprocessed = np.matmul(X - np.mean(X, axis=0), self.pca_components.T)
                # X_preprocessed = np.matmul(X, self.pca_components.T)
            else:
                X_preprocessed = X

            dict_params = {'n_clusters': self.hidden_layer_size, 'random_state': self.random_state}
            self.clusterer.set_params(**dict_params)
            self.clusterer.fit(X_preprocessed)
            # self.clusterer.fit(np.divide(X.T, np.linalg.norm(X, axis=1)).T)
            #self._input_weights = (self.clusterer.cluster_centers_ / np.linalg.norm(self.clusterer.cluster_centers_, axis=0)).T
            # self._input_weights = np.load(os.path.join(directory, 'original-pca50+kmeans200_matrix.npy'), allow_pickle=True)
            self._input_weights = np.matmul(self.pca_components.T, self.clusterer.cluster_centers_.T)
            self._bias = self._uniform_random_bias(
                hidden_layer_size=self.hidden_layer_size,
                random_state=self.random_state)
            return self

        """
        def transform(self, X):
            # self._hidden_layer_state = - self.clusterer.transform(X) * self.input_scaling + np.ones((X.shape[0], self.hidden_layer_size)) * self.bias_scaling
            # ACTIVATIONS[self.activation](self._hidden_layer_state)
            # return self._hidden_layer_state
            return super().transform(np.divide(X.T, np.linalg.norm(X, axis=1)).T)
        """

    # preprocessing
    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    X /= 255.
    pca = PCA(n_components=50, random_state=42).fit(X)
    # X_preprocessed = pca.transform(X)
    # X_preprocessed = np.dot(X - np.mean(X, axis=0), pca.components_.T)
    # logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    # number of initializations
    n_inits = 100
    random_state = np.random.RandomState(43)
    random_state_inits = random_state.choice(int(2**16-1), size=n_inits)

    # prepare parameter grid
    param_grid = [{
        'input_to_nodes': [InputToNode()],
        'input_to_nodes__hidden_layer_size': [200],
        'input_to_nodes__input_scaling': [1.],
        'input_to_nodes__bias_scaling': [0.],
        'input_to_nodes__activation': ['relu'],
        'input_to_nodes__random_state': [42],  # random_state_inits,
        'regressor__alpha': [1e-5],
        'random_state': [42],
        'chunk_size': [1000],
    }, {
        'input_to_nodes': [PCAKMeansInputToNode()],  # [BatchIntrinsicPlasticity(activation='tanh', hidden_layer_size=200, random_state=42, distribution='normal')],
        'input_to_nodes__hidden_layer_size': [200],
        'input_to_nodes__input_scaling': [1.],
        'input_to_nodes__bias_scaling': [0.],
        'input_to_nodes__activation': ['relu'],
        'input_to_nodes__pca_components': [pca.components_],
        'input_to_nodes__random_state': random_state_inits,
        'regressor__alpha': [1e-5],
        'random_state': [42],
        'chunk_size': [1000],
    }]

    # setup estimator
    estimator = ELMClassifier(regressor=IncrementalRegression())
    print(ELMClassifier(input_to_nodes=PCAKMeansInputToNode()).get_params().keys())

    # setup grid search
    cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=1,
        # pre_dispatch=2,
        verbose=2,
        refit=False,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    # run!
    cv.fit(X, y_encoded)

    # refine best params
    logger.info('best parameters: {0} (score: {1})'.format(cv.best_params_, cv.best_score_))

    # refine results
    cv_results = cv.cv_results_
    del cv_results['params']
    del cv_results['param_input_to_nodes__pca_components']
    cv_results.update({'param_input_to_nodes': [type(p).__name__ for p in cv_results['param_input_to_nodes']]})
    cv_results.update({'mean_test_error_rate': [1 - v for v in cv_results['mean_test_score']]})

    # save results
    try:
        with open(os.path.join(directory, '{0}_pca{1}_kmeans200_cosine_prepmatrix.csv'.format(self_name, pca.n_components_)), 'a') as f:
            f.write(','.join(cv_results.keys()) + '\n')
            for row in list(map(list, zip(*cv_results.values()))):
                f.write(','.join(map(str, row)) + '\n')
    except PermissionError as e:
        print('Missing privileges: {0}'.format(e))


def silhouette_n_clusters(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_n_clusters', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    scaler = StandardScaler().fit(X)
    X /= 255.

    pca = PCA(n_components=50, whiten=False, random_state=42).fit(X)
    min_var = 3088.6875

    # reduce train size
    # X = X[:10000, ...]
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=10000, random_state=42)

    # variance threshold
    X_var_threshold = X_train[..., scaler.var_ > min_var]

    # pca
    X_pca = pca.transform(X_train)

    # n_clusters
    k = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000, 4000]

    # n_init
    n_init = 10

    dict_results = {
        'n_clusters': [],
        'n_init': [],
        'variance_threshold': [],
        'pca_n_components': [],
        'pca_explained_variance': [],
        'pca_explained_variance_ratio': [],
        'silhouette_original': [],
        'silhouette_variance_threshold': [],
        'silhouette_pca': [],
        'fittime_original': [],
        'fittime_variance_threshold': [],
        'fittime_pca': [],
        'inertia_original': [],
        'inertia_variance_threshold': [],
        'inertia_pca': [],
        'n_iter_original': [],
        'n_iter_variance_threshold': [],
        'n_iter_pca': []
    }

    for n_clusters in k:
        dict_results['n_clusters'].append(n_clusters)
        dict_results['n_init'].append(n_init)
        dict_results['variance_threshold'].append(min_var)
        dict_results['pca_n_components'].append(pca.n_components_)
        dict_results['pca_explained_variance'].append(np.sum(pca.explained_variance_))
        dict_results['pca_explained_variance_ratio'].append(np.sum(pca.explained_variance_ratio_))

        clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, random_state=42)

        # original
        t = time.time()
        clusterer.fit(X_train)
        dict_results['fittime_original'].append(time.time() - t)
        dict_results['inertia_original'].append(clusterer.inertia_)
        dict_results['n_iter_original'].append(clusterer.n_iter_)
        dict_results['silhouette_original'].append(
            silhouette_score(X_train, clusterer.predict(X_train), metric='euclidean', random_state=42))

        np.save('./cluster_critical.npy', clusterer.cluster_centers_)

        # var threshold
        t = time.time()
        clusterer.fit(X_var_threshold)
        dict_results['fittime_variance_threshold'].append(time.time() - t)
        dict_results['inertia_variance_threshold'].append(clusterer.inertia_)
        dict_results['n_iter_variance_threshold'].append(clusterer.n_iter_)
        dict_results['silhouette_variance_threshold'].append(
            silhouette_score(X_train, clusterer.predict(X_var_threshold), metric='euclidean', random_state=42))

        # pca
        t = time.time()
        clusterer.fit(X_pca)
        dict_results['fittime_pca'].append(time.time() - t)
        dict_results['inertia_pca'].append(clusterer.inertia_)
        dict_results['n_iter_pca'].append(clusterer.n_iter_)
        dict_results['silhouette_pca'].append(
            silhouette_score(X_train, clusterer.predict(X_pca), metric='euclidean', random_state=42))

        logger.info('n_clusters = {0}, pca kmeans score: {1}'.format(n_clusters, dict_results['silhouette_pca'][-1]))
        logger.info('n_clusters = {0}'.format(n_clusters))

    # save results to csv
    with open(os.path.join(directory, 'silhouette_n_clusters.csv'), 'w') as f:
        f.write(','.join(dict_results.keys()) + '\n')
        for row in list(map(list, zip(*dict_results.values()))):
            f.write(','.join(map(str, row)) + '\n')
    return


def silhouette_kcluster(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_kcluster', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    X /= 255.

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=50, whiten=False, random_state=42).fit(X)

    # PCA preprocessed
    X_pca = pca.transform(X)

    k = [10, 15, 20, 25, 30, 35, 50, 100, 200]

    dict_results = {
        'n_clusters': [],
        'pca_n_components': [],
        'pca_expl_var': [],
        'pca_expl_var_ratio': [],
        'silhouette_kcosine': [],
        'silhouette_kmeans': [],
        'fittime_kcosine': [],
        'fittime_kmeans': []
    }

    for n_clusters in k:
        dict_results['n_clusters'].append(n_clusters)
        dict_results['pca_n_components'].append(pca.n_components_)
        dict_results['pca_expl_var'].append(np.sum(pca.explained_variance_))
        dict_results['pca_expl_var_ratio'].append(np.sum(pca.explained_variance_ratio_))

        # kmeans
        clusterer_euclid = KMeans(n_clusters=n_clusters, random_state=42)
        t = time.time()
        clusterer_euclid.fit(X_pca)
        dict_results['fittime_kmeans'].append(time.time() - t)
        dict_results['silhouette_kmeans'].append(
            silhouette_score(X, clusterer_euclid.predict(X_pca), metric='euclidean', random_state=42))

        # kcosine
        clusterer_cosine = KCluster(n_clusters=n_clusters, metric='cosine', random_state=42)
        t = time.time()
        clusterer_cosine.fit(X_pca)
        dict_results['fittime_kcosine'].append(time.time() - t)
        dict_results['silhouette_kcosine'].append(
            silhouette_score(X, clusterer_cosine.predict(X_pca), metric='cosine', random_state=42))

    # save results to csv
    with open(os.path.join(directory, 'silhouette_kcluster.csv'), 'w') as f:
        f.write(','.join(dict_results.keys()) + '\n')
        for row in list(map(list, zip(*dict_results.values()))):
            f.write(','.join(map(str, row)) + '\n')
    return


def silhouette_subset(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_subset', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    X /= 255.

    pca = PCA(n_components=50, whiten=False, random_state=42)

    # preprocessing
    X_pca = pca.fit_transform(X)

    # define subset sizes
    subset_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 60000]

    # number of centroids
    k_list = [20]

    dict_results = {
        'subset_size': [],
        'k': [],
        'n_init': [],
        'silhouette_raninit': [],
        'silhouette_preinit': [],
        'fittime_raninit': [],
        'fittime_preinit': [],
        'scoretime_raninit': [],
        'scoretime_preinit': []
    }

    for k in k_list:
        # preinit
        # initial training set
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, random_state=42, train_size=subset_sizes[0], shuffle=True, stratify=y)
        clusterer_init = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10).fit(X_train)

        # random inits
        clusterer = KMeans(n_clusters=k, n_init=10, random_state=42)

        for subset_size in subset_sizes:
            # split on subset size
            dict_results['subset_size'].append(subset_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, random_state=42, train_size=subset_size, shuffle=True, stratify=y)

            # train preinit
            t = time.time()
            clusterer_init = KMeans(n_clusters=k, random_state=42, n_init=1, init=clusterer_init.cluster_centers_)
            clusterer_init.fit_predict(X_train)
            dict_results['fittime_preinit'].append(time.time() - t)

            # score preinit
            t = time.time()
            dict_results['silhouette_preinit'].append(
                silhouette_score(X_train, clusterer_init.predict(X_train), metric='euclidean', random_state=42))
            dict_results['scoretime_preinit'].append(time.time() - t)

            # train randinit
            t = time.time()
            clusterer.fit(X_train)
            dict_results['fittime_raninit'].append(time.time() - t)

            # score raninit
            t = time.time()
            dict_results['silhouette_raninit'].append(
                silhouette_score(X_train, clusterer.predict(X_train), metric='euclidean', random_state=42))
            dict_results['scoretime_raninit'].append(time.time() - t)

            # store results
            dict_results['k'].append(k)
            dict_results['n_init'].append(clusterer.n_init)

            logger.info('silhouette (preinit) at subset size {1}: {0}'.format(dict_results['silhouette_preinit'][-1], dict_results['subset_size'][-1]))

    # save results to csv
    with open(os.path.join(directory, 'silhouette_kmeans_subset_size.csv'), 'w') as f:
        f.write(','.join(dict_results.keys()) + '\n')
        for row in list(map(list, zip(*dict_results.values()))):
            f.write(','.join(map(str, row)) + '\n')
    return


def silhouette_features(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_features', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    X /= 255.

    X = X[:10000, ...]

    scaler = StandardScaler().fit(X)
    pca = PCA(whiten=False, random_state=42).fit(X)

    X_pca = pca.transform(X)

    # sort scaler variances
    variance_indices = np.argsort(scaler.var_)[::-1]

    n_features_list = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 784]

    rs = np.random.RandomState(42)

    k = 20

    dict_results = {
        'nfeatures': [],
        'fittime_random': [],
        'fittime_maxvar': [],
        'fittime_pca': [],
        'silhouette_random': [],
        'silhouette_maxvar': [],
        'silhouette_pca': [],
        'explainvar_random': [],
        'explainvar_maxvar': [],
        'explainvar_pca': [],
        'explvarrat_random': [],
        'explvarrat_maxvar': [],
        'explvarrat_pca': [],
        'n_clusters': [],
    }

    for n_features in n_features_list:
        clusterer = KMeans(n_clusters=k, random_state=42)
        dict_results['nfeatures'].append(n_features)
        dict_results['n_clusters'].append(clusterer.n_clusters)

        indices = rs.choice(X.shape[1], size=n_features)
        t = time.time()
        pred = clusterer.fit_predict(X[:, indices])
        dict_results['fittime_random'].append(time.time() - t)
        dict_results['silhouette_random'].append(silhouette_score(X, pred, metric='euclidean', random_state=42))
        dict_results['explainvar_random'].append(np.sum(scaler.var_[indices]))
        dict_results['explvarrat_random'].append(np.sum(scaler.var_[indices]) / np.sum(scaler.var_))

        t = time.time()
        indices = variance_indices[:n_features]
        pred = clusterer.fit_predict(X[:, indices])
        dict_results['fittime_maxvar'].append(time.time() - t)
        dict_results['silhouette_maxvar'].append(silhouette_score(X, pred, metric='euclidean', random_state=42))
        dict_results['explainvar_maxvar'].append(np.sum(scaler.var_[indices]))
        dict_results['explvarrat_maxvar'].append(np.sum(scaler.var_[indices]) / np.sum(scaler.var_))

        t = time.time()
        pred = clusterer.fit_predict(X_pca[:, :n_features])
        dict_results['fittime_pca'].append(time.time() - t)
        dict_results['silhouette_pca'].append(silhouette_score(X, pred, metric='euclidean', random_state=42))
        dict_results['explainvar_pca'].append(np.sum(pca.explained_variance_[:n_features]))
        dict_results['explvarrat_pca'].append(np.sum(pca.explained_variance_ratio_[:n_features]))

        logger.info('pca silhouette at n_features={1:.0f}: {0}'.format(dict_results['silhouette_pca'][-1], n_features))

    # save results to csv
    with open(os.path.join(directory, 'silhouette_kmeans{0:.0f}_features.csv'.format(k)), 'w') as f:
        f.write(','.join(dict_results.keys()) + '\n')
        for row in list(map(list, zip(*dict_results.values()))):
            f.write(','.join(map(str, row)) + '\n')
    return


def main(directory, params):
    # workdir
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except PermissionError as e:
            print('mkdir failed due to missing privileges: {0}'.format(e))
            exit(1)

    workdir = directory

    # subfolder for results
    file_dir = os.path.join(directory, 'mnist-elm')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    logger = new_logger('main', directory=file_dir)
    logger.info('Started main with directory={0} and params={1}'.format(directory, params))

    # register parameters
    experiment_names = {
        'train_kmeans': train_kmeans,
        'elm_hyperparameters': elm_hyperparameters,
        'elm_basic': elm_basic,
        'elm_pca': elm_pca,
        'elm_preprocessed': elm_preprocessed,
        'elm_random_state': elm_random_state,
        'elm_hidden_layer_size': elm_hidden_layer_size,
        'elm_coates': elm_coates,
        'elm_coates_stacked': elm_coates_stacked,
        'significance': significance,
        'silhouette_n_clusters': silhouette_n_clusters,
        'silhouette_subset': silhouette_subset,
        'silhouette_kcluster': silhouette_kcluster,
        'silhouette_features': silhouette_features
    }

    # run specified programs
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
