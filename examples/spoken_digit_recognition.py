import os
import glob
import numpy as np
from tqdm import tqdm
import time

from sklearn.pipeline import FeatureUnion
from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.linear_model import IncrementalRegression, FastIncrementalRegression
from pyrcn.base import InputToNode, NodeToNode

import librosa


def optimize_esn(base_input_to_node, base_node_to_node, base_reg, params, X_train, X_val, y_train, y_val):
    input_to_node = clone(base_input_to_node)
    node_to_node = clone(base_node_to_node)
    reg = clone(base_reg)
    if "input_scaling" in params:
        input_to_node = input_to_node.set_params(**{'input_scaling': params['input_scaling']})
        del params["input_scaling"]
    if "k_in" in params:
        input_to_node = input_to_node.set_params(**{'k_in': params['k_in']})
        del params["k_in"]
    if "alpha" in params:
        base_reg.set_params(**{'alpha': params['alpha']})
        del params["alpha"]
    node_to_node.set_params(**params)
    esn = ESNClassifier(input_to_nodes=[('default', input_to_node)],
                        nodes_to_nodes=[('default', node_to_node)],
                        regressor=reg)
    for X, y in zip(X_train[:-1], y_train[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), update_output_weights=False)

    X = X_train[-1]
    y = np.repeat(y_train[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, update_output_weights=True)

    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = esn.predict_proba(X=X)
        Y_true_train.append(y)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    Y_true_val = []
    Y_pred_val = []
    for X, y in zip(X_val, y_val):
        y_pred = esn.predict_proba(X=X)
        Y_true_val.append(y)
        Y_pred_val.append(np.argmax(y_pred.sum(axis=0)))
    return [accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_val, Y_pred_val)]


def optimize_esn_kmeans(base_input_to_node, base_node_to_node, base_reg, params, w_in, X_train, X_val, y_train, y_val):
    input_to_node = clone(base_input_to_node)
    node_to_node = clone(base_node_to_node)
    reg = clone(base_reg)
    if "input_scaling" in params:
        input_to_node = input_to_node.set_params(**{'input_scaling': params['input_scaling']})
        del params["input_scaling"]
    if "k_in" in params:
        input_to_node = input_to_node.set_params(**{'k_in': params['k_in']})
        del params["k_in"]
    if "alpha" in params:
        base_reg.set_params(**{'alpha': params['alpha']})
        del params["alpha"]
    node_to_node.set_params(**params)
    esn = ESNClassifier(input_to_nodes=[('default', input_to_node)],
                        nodes_to_nodes=[('default', node_to_node)],
                        regressor=reg)
    esn._input_to_node = FeatureUnion(transformer_list=[('default', input_to_node)], n_jobs=None,
                                      transformer_weights=None).fit(X_train[0])
    esn._input_to_node.transformer_list[0][1]._input_weights = w_in.T
    for X, y in zip(X_train[:-1], y_train[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), update_output_weights=False)

    X = X_train[-1]
    y = np.repeat(y_train[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, update_output_weights=True)

    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = esn.predict_proba(X=X)
        Y_true_train.append(y)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    Y_true_val = []
    Y_pred_val = []
    for X, y in zip(X_val, y_val):
        y_pred = esn.predict_proba(X=X)
        Y_true_val.append(y)
        Y_pred_val.append(np.argmax(y_pred.sum(axis=0)))
    return [accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_val, Y_pred_val)]


all_files = glob.glob(r"E:\free-spoken-digit-dataset\recordings\*.wav")
print(len(all_files))

X_train = []
X_test = []
y_train = []
y_test = []
print("extracting features...")
with tqdm(total=len(all_files)) as pbar:
    for k, f in enumerate(all_files):
        basename = os.path.basename(f).split('.')[0]
        # Get label (0-9) of recording.
        label = int(basename.split('_')[0])
        idx = int(basename.split('_')[2])
        # Load the audio signal and normalize it.
        x, sr = librosa.core.load(f, sr=None, mono=False)
        # x /= np.max(np.abs(x))
        mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(0.01*sr), n_fft=256, htk=True, n_mels=100, n_mfcc=13)
        if idx <= 4:
            X_test.append(mfcc.T)
            y_test.append(label)
        else:
            X_train.append(mfcc.T)
            y_train.append(label)
        pbar.update(1)
print("done!")


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

# Validate training and test sizes
print(len(X_train), len(y_train), X_train[0].shape, y_train[0])
print(len(X_val), len(y_val), X_val[0].shape, y_val[0])
print(len(X_test), len(y_test), X_test[0].shape, y_test[0])


# Normalize all features using the StandardScaler from scikit-learn.
scaler = StandardScaler().fit(X=np.vstack(X_train))
X_train_scaled = [scaler.transform(X) for X in X_train]
X_val_scaled = [scaler.transform(X) for X in X_val]
X_test_scaled = [scaler.transform(X) for X in X_test]

grid = {'bias_scaling': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }

kmeans = MiniBatchKMeans(n_clusters=500, n_init=20, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=1, random_state=0)
kmeans.fit(X=np.concatenate(X_train_scaled))
# w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
w_in = np.pad(np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]), ((0, 1500), (0, 0)), mode='constant', constant_values=0)

base_input_to_nodes = InputToNode(hidden_layer_size=2000, activation='identity', k_in=10, input_scaling=0.3, bias_scaling=0.0, random_state=10)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=2000, spectral_radius=0.5, leakage=0.1, bias_scaling=0.0, k_rec=10, random_state=10)
base_reg = FastIncrementalRegression(alpha=1e-3)

esn = ESNClassifier(input_to_nodes=[('default', base_input_to_nodes)],
                    nodes_to_nodes=[('default', base_nodes_to_nodes)],
                    regressor=base_reg)

t1 = time.time()

losses = Parallel(n_jobs=6, verbose=50)(
    delayed(optimize_esn_kmeans)(base_input_to_nodes, base_nodes_to_nodes, base_reg, params, w_in,
                                 X_train_scaled, X_val_scaled, y_train, y_val)
    for params in ParameterGrid(grid))
"""
losses = Parallel(n_jobs=-1, verbose=50)(
    delayed(optimize_esn)(base_input_to_nodes, base_nodes_to_nodes, base_reg, params,
                          X_train_scaled, X_val_scaled, y_train, y_val)
    for params in ParameterGrid(grid))
"""
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.max(losses[:, 0]), ParameterGrid(grid)[np.argmax(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.max(losses[:, 1]), ParameterGrid(grid)[np.argmax(losses[:, 1])]))
fname = open("losses_08_500_2000.csv", "wa")
np.savetxt(f, X=losses, delimiter=",")
fname.close()
