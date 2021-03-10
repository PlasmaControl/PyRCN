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
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier

train_size = 60000


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
    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
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

    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
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

    cv = GridSearchCV(estimator, param_grid, cv=5, n_jobs=1)
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
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[:train_size], y_encoded[train_size:]

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


def elm_random_state(directory):
    self_name = 'elm_preprocessed_relu_rs'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    pca = PCA(n_components=200).fit(X)
    X_preprocessed = pca.transform(X) / 255.
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    # prepare parameter grid
    param_grid = [{
            'input_to_nodes__hidden_layer_size': [2000],
            'input_to_nodes__input_scaling': [1/400],
            'input_to_nodes__bias_scaling': [40/400],
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': np.random.randint(low=1, high=2**8-1, size=10),
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

    # setup parameter grid
    param_grid = {
        'input_to_nodes__input_scaling': [1.],  # np.logspace(start=-3, stop=1, base=10, num=6),
        'input_to_nodes__bias_scaling': [0.],  # np.logspace(start=-3, stop=1, base=10, num=6),
        'input_to_nodes__activation': ['relu'],
        'input_to_nodes__random_state': [42],
        'regressor__alpha': [1e-5],
        'chunk_size': [1000],
        'random_state': [42]
    }

    # read input matrices from files
    list_filepaths = []
    for filepath in glob.glob(os.path.join(directory, '*pca*+kmeans*_matrix.npy')):
        logger.info('matrix file found: {0}'.format(filepath))
        list_filepaths.append(filepath)
        filename = os.path.splitext(os.path.basename(filepath))[0]

        est_filepath = os.path.join(directory, 'est_coates-{0}.pickle'.format(filename))
        csv_filepath = os.path.join(directory, '{0}.csv'.format(filename))
        pred_filpath = os.path.join(directory, 'est_coates-{0}-predicted.npz'.format(filename))

        # only if files do not exist yet
        if not os.path.isfile(csv_filepath) or not os.path.isfile(est_filepath):
            # set input weights
            # param_grid.update({'input_to_nodes__predefined_input_weights': [np.load(filepath)]})

            # setup estimator
            estimator = ELMClassifier(
                PredefinedWeightsInputToNode(
                    predefined_input_weights=np.load(filepath)),
                IncrementalRegression(alpha=1e-5))
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
            # del cv_best_params['input_to_nodes__predefined_input_weights']

            # refine best params
            logger.info('file {2}, best parameters: {0} (score: {1})'.format(cv_best_params, cv.best_score_, filepath))

            # refine results
            cv_results = cv.cv_results_
            # del cv_results['params']
            # del cv_results['param_input_to_nodes__predefined_input_weights']

            # save estimator
            try:
                with open(est_filepath, 'wb') as f:
                    pickle.dump(cv.best_estimator_, f)
            except Exception as e:
                logger.error('Unexpected error: {0}'.format(e))
                exit(1)

            # save results
            try:
                with open(os.path.join(directory, '{0}.csv'.format(os.path.splitext(os.path.basename(filepath))[0])), 'w') as f:
                    f.write(','.join(cv_results.keys()) + '\n')
                    for row in list(map(list, zip(*cv_results.values()))):
                        f.write(','.join(map(str, row)) + '\n')
            except PermissionError as e:
                print('Missing privileges: {0}'.format(e))

            # save prediction
            np.savez_compressed(pred_filpath, X_test=X[train_size:, ...], y_test=y_encoded[train_size:], y_pred=cv.best_estimator_.predict(X[train_size:, ...]))

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
