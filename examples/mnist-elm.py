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

import time

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedShuffleSplit

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import Ridge

from pyrcn.util import new_logger, argument_parser, get_mnist
from pyrcn.base import InputToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier

# import importlib
# prep = importlib.import_module(os.path.abspath('../preprocessing.py'))

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

    # scale X -> $X \in [0, 1]$
    X /= 255.

    list_n_components = [50, 100]
    list_n_clusters = [20, 50, 100, 200, 500, 1000]

    for n_components in list_n_components:
        pca = PCA(n_components=n_components, random_state=42).fit(X)
        X_pca = pca.transform(X)
        np.save('pca{0}_components.npy', pca.components_)
        logger.info('pca{0}: explained variance ratio = {1}'.format(n_components, np.sum(pca.explained_variance_ratio_)))

        for n_clusters in list_n_clusters:
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=42, batch_size=5000, n_init=5).fit(X_pca)
            np.save('kmeans{0}-pca{1}_centroids.npy'.format(n_clusters, n_components), clusterer.cluster_centers_)
            logger.info('successfuly trained MiniBatchKMeans and saved to kmeans{0}-pca{1}_centroids.npy'.format(n_clusters, n_components))


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
        n_jobs=2,
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
    pca = PCA(n_components=100).fit(X)
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
        n_jobs=2,
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


def elm_random_state(directory):
    self_name = 'elm_preprocessed_relu_rs'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    pca = PCA(n_components=200).fit(X)

    # X_preprocessed = preprocessing(X, directory=directory, overwrite=False)
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
    X_preprocessed = preprocessing(X, directory=directory, overwrite=False)
    logger.info('{0} features remaining after preprocessing.'.format(X_preprocessed.shape[1]))

    cluster = KMeans(n_clusters=20, init='k-means++', n_init=10, random_state=42).fit(X_preprocessed)
    logger.info('cluster.cluster_centers_ = {0}'.format(cluster.cluster_centers_))
    with open(os.path.join(directory, 'clusterer.pickle'), 'wb') as f:
        pickle.dump(cluster, f)

    X_coates = np.dot(X_preprocessed, cluster.cluster_centers_.T)

    # X_train, X_test, y_train, y_test = train_test_split(X_coates, y_encoded, train_size=train_size, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[:train_size], y_encoded[train_size:]

    hidden_layer_sizes = np.array([300, 450, 1000, 2250])

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
