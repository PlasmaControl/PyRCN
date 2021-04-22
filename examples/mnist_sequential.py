# MNIST classification using Echo State Networks

import numpy as np
from random import randint, seed
import time
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

from tqdm import tqdm


def optimize_esn(clf, X_train, y_train):
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
# np.random.seed(42)
n_corrupt = 0
# idx_corrupt = np.random.randint(low=0, high=60000, size=n_corrupt)
X_train = np.empty(shape=(60000 + n_corrupt,), dtype=object)
X_test = np.empty(shape=(10000,), dtype=object)
y_train = np.empty(shape=(60000 + n_corrupt,), dtype=object)
y_test = np.empty(shape=(10000,), dtype=object)
for k, (seq, label) in enumerate(zip(X[:60000], y[:60000])):
    X_train[k] = seq.reshape(28, 28).T
    y_train[k] = np.repeat(label, repeats=28, axis=0)

# seed(42)
# for k, (seq, label) in enumerate(zip(X[idx_corrupt], y[idx_corrupt])):
#     n_remove = randint(1, 3)
#     idx_keep = np.random.randint(low=0, high=28, size=28-n_remove)
#     X_train[60000+k] = seq.reshape(28,28).T[np.sort(idx_keep), :]
#     y_train[60000+k] = np.repeat(label, repeats=28-n_remove, axis=0)
# for k, (seq, label) in enumerate(zip(X[60000:], y[60000:])):
#     X_test[k] = seq.reshape(28, 28).T
#     y_test[k] = np.repeat(label, repeats=28, axis=0)


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
final_fixed_params = initially_fixed_params

step1_esn_params = {'input_to_node__input_scaling': np.linspace(0.1, 1.5, 15),
                    'node_to_node__spectral_radius': np.linspace(0.0, 1.5, 16)}
step2_esn_params = {'node_to_node__leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'node_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}


base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression()).set_params(**initially_fixed_params)
"""
acc_scores = Parallel(n_jobs=-1, verbose=10)(delayed(optimize_esn)
                                             (SequenceToSequenceClassifier(clone(base_esn), estimator_params=params), X_train, y_train) 
                                             for params in ParameterGrid(step1_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
print(final_fixed_params)
acc_scores = Parallel(n_jobs=-1, verbose=10)(delayed(optimize_esn)
                                             (SequenceToSequenceClassifier(clone(base_esn), estimator_params=params), X_train, y_train) 
                                             for params in ParameterGrid(step2_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step2_esn_params)[np.argmax(acc_scores)])
print(final_fixed_params)
acc_scores = Parallel(n_jobs=-1, verbose=10)(delayed(optimize_esn)
                                             (SequenceToSequenceClassifier(clone(base_esn), estimator_params=params), X_train, y_train) 
                                             for params in ParameterGrid(step3_esn_params))
base_esn.set_params(**ParameterGrid(step1_esn_params)[np.argmax(acc_scores)])
final_fixed_params.update(ParameterGrid(step3_esn_params)[np.argmax(acc_scores)])
print(final_fixed_params)
"""
final_fixed_params = {'input_to_node__hidden_layer_size': 500, 
                      'input_to_node__activation': 'identity', 
                      'input_to_node__k_in': 10, 
                      'input_to_node__random_state': 42, 
                      'input_to_node__bias_scaling': 0.0, 
                      'input_to_node__input_scaling': 0.2, 
                      'node_to_node__hidden_layer_size': 500, 
                      'node_to_node__activation': 'tanh', 
                      'node_to_node__leakage': 0.1, 
                      'node_to_node__spectral_radius': 1.1, 
                      'node_to_node__bias_scaling': 0.2, 
                      'node_to_node__bi_directional': False, 
                      'node_to_node__k_rec': 10, 
                      'node_to_node__wash_out': 0, 
                      'node_to_node__continuation': False, 
                      'node_to_node__random_state': 42, 
                      'regressor__alpha': 1e-05, 
                      'random_state': 42}
base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression()).set_params(**final_fixed_params)

param_grid = {'input_to_node__hidden_layer_size': [16000, 32000, 64000]}

with tqdm(total=len(ParameterGrid(param_grid))) as pb:
    print("Fit time\tInference time\tAccuracy score\tSize[Bytes]")
    for params in ParameterGrid(param_grid):
        params['node_to_node__hidden_layer_size'] = params['input_to_node__hidden_layer_size']
        t1 = time.time()
        clf = SequenceToSequenceClassifier(clone(base_esn), estimator_params=params).fit(X_train, y_train)
        t_fit = time.time() - t1
        mem_size = clf.estimator.__sizeof__()
        t1 = time.time()
        y_test_pred = clf.predict(X_test)
        y_true = [np.argmax(np.bincount(y)) for y in y_test]
        y_pred = [np.argmax(np.bincount(y)) for y in y_test_pred]
        acc_score = accuracy_score(y_true, y_pred)
        t_inference = time.time() - t1
        print("{0}\t{1}\t{2}\t{3}".format(t_fit, t_inference, acc_score, mem_size))
        pbar.update(1)