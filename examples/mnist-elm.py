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

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedShuffleSplit

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import Ridge

from pyrcn.util import new_logger, argument_parser, get_mnist
from pyrcn.base import InputToNode
# from pyrcn.linear_model import IncrementalRegression
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

    # 20 centroids
    pca20 = PCA(n_components=20).fit(X)
    kmeans20 = MiniBatchKMeans(n_clusters=20, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(pca20.transform(X))
    with open(os.path.join(directory, 'kmeans-mnist-pca-20.pickle'), 'wb') as f:
        pickle.dump(kmeans20, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-pca-20.pickle')

    # 50 centroids
    pca50 = PCA(n_components=50).fit(X)
    kmeans50 = MiniBatchKMeans(n_clusters=50, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(pca50.transform(X))
    with open(os.path.join(directory, 'kmeans-mnist-pca-50.pickle'), 'wb') as f:
        pickle.dump(kmeans50, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-pca-50.pickle')

    # 100 centroids
    pca100 = PCA(n_components=100).fit(X)
    kmeans100 = MiniBatchKMeans(n_clusters=100, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(pca100.transform(X))
    with open(os.path.join(directory, 'kmeans-mnist-pca-100.pickle'), 'wb') as f:
        pickle.dump(kmeans100, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-pca-100.pickle')

    # 200 centroids
    pca200 = PCA(n_components=200).fit(X)
    kmeans200 = MiniBatchKMeans(n_clusters=200, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(pca200.transform(X))
    with open(os.path.join(directory, 'kmeans-mnist-pca-200.pickle'), 'wb') as f:
        pickle.dump(kmeans200, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-pca-200.pickle')

    # 450 centroids
    pca450 = PCA(n_components=450).fit(X)
    kmeans450 = MiniBatchKMeans(n_clusters=450, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(pca450.transform(X))
    with open(os.path.join(directory, 'kmeans-mnist-pca-450.pickle'), 'wb') as f:
        pickle.dump(kmeans450, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-pca-450.pickle')

    # 900 centroids, no preprocessing
    kmeans900 = MiniBatchKMeans(n_clusters=900, init='k-means++', n_init=10, batch_size=1000, random_state=42).fit(X)
    with open(os.path.join(directory, 'kmeans-mnist-900.pickle'), 'wb') as f:
        pickle.dump(kmeans900, f)
    logger.info('successfulyy trained and saved to kmeans-mnist-900.pickle')


def elm_hyperparameters(directory):
    self_name = 'elm_hyperparameters'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    X = X/255.

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[train_size:], y_encoded[:train_size]

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


def elm_preprocessed(directory):
    self_name = 'elm_pca'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # preprocessing
    pca = PCA(n_components=200).fit(X / 255.)
    X_preprocessed = pca.transform(X / 255.)
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

    # preprocessing
    pca450 = PCA(n_components=450).fit(X)
    logger.info('Preprocessing successful!')

    # fan-out from paper
    fan_out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    # prepare parameter grids
    param_grid_basic = [{
            'input_to_nodes__hidden_layer_size': 784 * np.array(fan_out),
            'input_to_nodes__input_scaling': [1.],
            'input_to_nodes__bias_scaling': [1.],
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
    }]

    param_grid_pca = [{
            'input_to_nodes__hidden_layer_size': np.concatenate((450 * np.array(fan_out), 784 * np.array(fan_out)), axis=0),
            'input_to_nodes__input_scaling': [1./40],
            'input_to_nodes__bias_scaling': [1.],
            'input_to_nodes__activation': ['relu'],
            'input_to_nodes__random_state': [42],
            'regressor__alpha': [1e-5],
            'regressor__random_state': [42],
            'random_state': [42]
    }]

    # setup estimator
    estimator = ELMClassifier(input_to_nodes=InputToNode(), regressor=Ridge())

    # setup grid search
    cv_basic = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid_basic,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    cv_pca = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid_pca,
        scoring='accuracy',
        n_jobs=1,
        verbose=2,
        cv=[(np.arange(0, train_size), np.arange(train_size, 70000))])  # split train test (dataset size = 70k)

    # run!
    try:
        cv_basic.fit(X, y_encoded)
        logger.info('best parameters: {0} (score: {1})'.format(cv_basic.best_params_, cv_basic.best_score_))
    except MemoryError as e:
        logger.error('Memory error: {0}'.format(e))
    except Exception as e:
        logger.error('Unexpected exception: {0}'.format(e))
    finally:
        # refine results
        cv_basic_results = cv_basic.cv_results_
        del cv_basic_results['params']

        # save results
        try:
            with open(os.path.join(directory, '{0}_basic.csv'.format(self_name)), 'w') as f:
                f.write(','.join(cv_basic_results.keys()) + '\n')
                for row in list(map(list, zip(*cv_basic_results.values()))):
                    f.write(','.join(map(str, row)) + '\n')
        except PermissionError as e:
            print('Missing privileges: {0}'.format(e))

    try:
        cv_pca.fit(pca450.transform(X), y_encoded)
        logger.info('best parameters: {0} (score: {1})'.format(cv_pca.best_params_, cv_pca.best_score_))
    except MemoryError as e:
        logger.error('Memory error: {0}'.format(e))
    except Exception as e:
        logger.error('Unexpected exception: {0}'.format(e))
    finally:
        # refine results
        cv_pca_results = cv_pca.cv_results_
        del cv_pca_results['params']

        # save results
        try:
            with open(os.path.join(directory, '{0}_pca.csv'.format(self_name)), 'w') as f:
                f.write(','.join(cv_pca_results.keys()) + '\n')
                for row in list(map(list, zip(*cv_pca_results.values()))):
                    f.write(','.join(map(str, row)) + '\n')
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

    # X_train, X_test, y_train, y_test = train_test_split(X_coates, y_encoded, train_size=train_size, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test = X[:train_size, :], X[train_size:, :], y_encoded[train_size:], y_encoded[:train_size]

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
