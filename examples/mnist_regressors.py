# MNIST classification using Extreme Learning Machines

import numpy as np
import time
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.utils.fixes import loguniform
from sklearn.metrics import accuracy_score

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.extreme_learning_machine import ELMClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode


# Load the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Provide standard split in training and test. Normalize to a range between [-1, 1].
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X=X)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000].astype(int), y[60000:].astype(int)

# Prepare sequential hyperparameter tuning
initially_fixed_params = {'input_to_node__hidden_layer_size': 500,
                          'input_to_node__activation': 'tanh',
                          'input_to_node__k_in': 10,
                          'input_to_node__random_state': 42,
                          'input_to_node__bias_scaling': 0.0,
                          'regressor__alpha': 1e-5,
                          'random_state': 42}

step1_params = {'input_to_node__input_scaling': loguniform(1e-5, 1e1)}
kwargs1 = {'random_state': 42,
           'verbose': 1,
           'n_jobs': -1,
           'n_iter': 50,
           'scoring': 'accuracy'}
step2_params = {'input_to_node__bias_scaling': np.linspace(0.0, 1.6, 16)}
kwargs2 = {'verbose': 1,
           'n_jobs': -1,
           'scoring': 'accuracy'}

elm = ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()).set_params(**initially_fixed_params)

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_params, kwargs1),
            ('step2', GridSearchCV, step2_params, kwargs2)]  # Note that we pass functors, not instances (no '()')!


sequential_search = SequentialSearchCV(elm, searches=searches).fit(X_train, y_train)

sequential_search = load("sequential_search_mnist_elm.joblib")
final_fixed_params = initially_fixed_params
final_fixed_params.update(sequential_search.all_best_params_["step1"])
final_fixed_params.update(sequential_search.all_best_params_["step2"])

base_elm_ridge = ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()).set_params(**final_fixed_params)
base_elm_inc = ELMClassifier(input_to_node=InputToNode(), regressor=IncrementalRegression()).set_params(**final_fixed_params)
base_elm_inc_chunk = clone(base_elm_inc).set_params(chunk_size=6000)

param_grid = {'input_to_node__hidden_layer_size': [32000]}

print("Estimator\tFit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in ParameterGrid(param_grid):
    t1 = time.time()
    elm_ridge = clone(base_elm_ridge).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    mem_size = elm_ridge.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, elm_ridge.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(elm_ridge, t_fit, t_inference, acc_score, mem_size))
    t1 = time.time()
    elm_inc = clone(base_elm_inc).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    mem_size = elm_inc.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, elm_inc.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(elm_inc, t_fit, t_inference, acc_score, mem_size))
    t1 = time.time()
    elm_inc_chunk = clone(base_elm_inc_chunk).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    mem_size = elm_inc_chunk.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, elm_inc_chunk.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(elm_inc_chunk, t_fit, t_inference, acc_score, mem_size))
