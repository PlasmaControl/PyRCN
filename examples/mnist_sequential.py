# MNIST classification using Echo State Networks

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.util import SequenceToSequenceClassifier


def optimize_esn(base_esn, params, X_train, y_train):
    clf = SequenceToSequenceClassifier(clone(base_esn), estimator_params=params)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_true = [np.argmax(np.bincount(y)) for y in y_train]
    y_pred = [np.argmax(np.bincount(y)) for y in y_train_pred]
    return accuracy_score(y_true, y_pred)


# Load the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Provide standard split in training and test. Normalize to a range between [-1, 1].
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X=X)
y = y.astype(int)
X_train = np.empty(shape=(60000,), dtype=object)
X_test = np.empty(shape=(60000,), dtype=object)
y_train = np.empty(shape=(60000,), dtype=object)
y_test = np.empty(shape=(60000,), dtype=object)
for k, (seq, label) in enumerate(zip(X[:60000], y[:60000])):
    X_train[k] = seq.reshape(28, 28).T
    y_train[k] = np.repeat(label, repeats=28, axis=0)
for k, (seq, label) in enumerate(zip(X[60000:], y[60000:])):
    X_test[k] = seq.reshape(28, 28).T
    y_test[k] = np.repeat(label, repeats=28, axis=0)

# Prepare sequential hyperparameter tuning
initially_fixed_params = {'input_to_node__hidden_layer_size': 500,
                          'input_to_node__activation': 'identity',
                          'input_to_node__k_in': 10,
                          'input_to_node__random_state': 42,
                          'input_to_node__bias_scaling': 0.0,
                          'input_to_node__k_in': 1,
                          'input_to_node__random_state': 42,
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
final_fixed_params = initially_fixed_params

step1_esn_params = {'input_to_node__input_scaling': np.linspace(0.1, 1.5, 15),
                    'node_to_node__spectral_radius': np.linspace(0.0, 1.5, 16)}
step2_esn_params = {'node_to_node__leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'node_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}


base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression()).set_params(**initially_fixed_params)

acc_scores = Parallel(n_jobs=2, verbose=10)(delayed(optimize_esn)(base_esn, params, X_train, y_train) for params in ParameterGrid(step1_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])

acc_scores = Parallel(n_jobs=2, verbose=10)(delayed(optimize_esn)(base_esn, params, X_train, y_train) for params in ParameterGrid(step2_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step2_esn_params)[np.argmax(acc_scores)])

acc_scores = Parallel(n_jobs=2, verbose=10)(delayed(optimize_esn)(base_esn, params, X_train, y_train) for params in ParameterGrid(step3_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step3_esn_params)[np.argmax(acc_scores)])

base_elm_ridge = ESNClassifier(input_to_node=InputToNode(), regressor=Ridge()).set_params(**final_fixed_params)
base_elm_inc = ESNClassifier(input_to_node=InputToNode(), regressor=IncrementalRegression()).set_params(**final_fixed_params)
base_elm_inc_chunk = clone(base_elm_inc).set_params(chunk_size=6000)

param_grid = {'input_to_node__hidden_layer_size': [16000]}

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
