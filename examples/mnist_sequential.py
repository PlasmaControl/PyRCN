# MNIST classification using Echo State Networks

import numpy as np
from random import randint, seed
import time
from joblib import Parallel, delayed, dump, load
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import ESNClassifier, SeqToLabelESNClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.metrics import accuracy_score


# Load the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# Provide standard split in training and test. Normalize to a range between [-1, 1].
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X=X)
y = y.astype(int)
np.random.seed(42)
n_corrupt = 0
idx_corrupt = np.random.randint(low=0, high=60000, size=n_corrupt)
X_train = np.empty(shape=(60000 + n_corrupt,), dtype=object)
X_test = np.empty(shape=(10000,), dtype=object)
y_train = np.empty(shape=(60000 + n_corrupt,), dtype=object)
y_test = np.empty(shape=(10000,), dtype=object)
for k, (seq, label) in enumerate(zip(X[:60000], y[:60000])):
    X_train[k] = seq.reshape(28, 28).T
    y_train[k] = np.repeat(label, repeats=28, axis=0)

seed(42)
for k, (seq, label) in enumerate(zip(X[idx_corrupt], y[idx_corrupt])):
    n_remove = randint(1, 3)
    idx_keep = np.random.randint(low=0, high=28, size=28-n_remove)
    X_train[60000+k] = seq.reshape(28,28).T[np.sort(idx_keep), :]
    y_train[60000+k] = np.repeat(label, repeats=28-n_remove, axis=0)
for k, (seq, label) in enumerate(zip(X[60000:], y[60000:])):
    X_test[k] = seq.reshape(28, 28).T
    y_test[k] = np.repeat(label, repeats=28, axis=0)
# Prepare sequential hyperparameter tuning
initially_fixed_params = {'input_to_node__hidden_layer_size': 500,
                          'input_to_node__activation': 'identity',
                          'input_to_node__k_in': 10,
                          'input_to_node__random_state': 42,
                          'input_to_node__bias_scaling': 0.0,
                          'node_to_node__hidden_layer_size': 500,
                          'node_to_node__activation': 'tanh',
                          'node_to_node__leakage': 1.0,
                          'node_to_node__bias_scaling': 0.0,
                          'node_to_node__bi_directional': False,
                          'node_to_node__k_rec': 10,
                          'node_to_node__wash_out': 0,
                          'node_to_node__continuation': False,
                          'node_to_node__random_state': 42,
                          'regressor__alpha': 1e-5,
                          'random_state': 42}
"""
step1_esn_params = {'input_to_node__input_scaling': np.linspace(0.1, 1.0, 10),
                    'node_to_node__spectral_radius': np.linspace(0.0, 1.5, 16)}

step2_esn_params = {'node_to_node__leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'node_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}
kwargs = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs)]

base_esn = SequenceToSequenceClassifier().set_params(**initially_fixed_params)

sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)

dump(sequential_search, "sequential_search_esn_mnist_no_noise.joblib")
"""
sequential_search = load("sequential_search_esn_mnist_no_noise.joblib")

final_fixed_params = initially_fixed_params
final_fixed_params.update(sequential_search.all_best_params_["step1"])
final_fixed_params.update(sequential_search.all_best_params_["step2"])
final_fixed_params.update(sequential_search.all_best_params_["step3"])

base_esn = SequenceToSequenceClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression()).set_params(**final_fixed_params)

param_grid = {'input_to_node__hidden_layer_size': [500, 1000, 2000, 4000, 8000, 16000, 32000]}

print("CV results\tFit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in ParameterGrid(param_grid):
    params["node_to_node__hidden_layer_size"] = params["input_to_node__hidden_layer_size"]
    esn_cv = cross_validate(clone(base_esn).set_params(**params), X=X_train, y=y_train)
    t1 = time.time()
    esn = clone(base_esn).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    mem_size = esn.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, esn.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(esn_cv, t_fit, t_inference, acc_score, mem_size))
