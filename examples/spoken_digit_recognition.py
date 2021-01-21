import os
import glob
import numpy as np
from tqdm import tqdm
import time

from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode
import matplotlib
from matplotlib import pyplot as plt
#Options
params = {'image.cmap' : 'RdBu',
          'text.usetex' : True,
          'font.size' : 11,
          'axes.titlesize' : 24,
          'axes.labelsize' : 20,
          'lines.linewidth' : 3,
          'lines.markersize' : 10,
          'xtick.labelsize' : 16,
          'ytick.labelsize' : 16,
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

import librosa


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
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
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


# One-Hot encoding of labels
enc = OneHotEncoder(sparse=False).fit(X=np.asarray(y_train).reshape(-1, 1))


# Clone the base_esn and fit it on the training data
def opt_function(base_input_to_node, base_node_to_node, encoder, params, X_train, y_train, X_test, y_test, w_in=None):
    input_to_nodes = clone(base_input_to_node)      # .set_params(**{'input_scaling': params['input_scaling']})
    input_to_nodes.fit(X=X_train_scaled[0])
    if w_in is not None:
        input_to_nodes._input_weights = w_in
    # nodes_to_nodes = clone(base_node_to_node).set_params(**{'spectral_radius': params['spectral_radius']})
    nodes_to_nodes = clone(base_node_to_node).set_params(**params)

    reg = ESNRegressor(input_to_nodes=[('default', input_to_nodes)],
                       nodes_to_nodes=[('default', nodes_to_nodes)],
                       regressor=IncrementalRegression(alpha=5e-3), random_state=10)
    for X, y in zip(X_train, y_train):
        y = encoder.transform(np.asarray(y).reshape(1, -1))
        y = np.repeat(np.atleast_2d(y), repeats=X.shape[0], axis=0)
        reg.partial_fit(X=X, y=y)
    err_train = []
    Y_pred_train = []
    Y_true_train = []
    for X, y in zip(X_train, y_train):
        Y_true_train.append(y)
        y_true = encoder.transform(np.asarray(y).reshape(1, -1))
        y_true = np.repeat(np.atleast_2d(y_true), repeats=X.shape[0], axis=0)
        y_pred = reg.predict(X=X)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        err_train.append(mean_squared_error(y_true, y_pred))
    err_test = []
    Y_pred_test = []
    Y_true_test = []
    for X, y in zip(X_test, y_test):
        Y_true_test.append(y)
        y_true = encoder.transform(np.asarray(y).reshape(1, -1))
        y_true = np.repeat(np.atleast_2d(y_true), repeats=X.shape[0], axis=0)
        y_pred = reg.predict(X=X)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        err_train.append(mean_squared_error(y_true, y_pred))
    return [1 - accuracy_score(Y_true_train, Y_pred_train), 1 - accuracy_score(Y_true_test, Y_pred_test)]


kmeans = MiniBatchKMeans(n_clusters=300, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=1, random_state=0)
kmeans.fit(X=np.vstack(X_train_scaled))

w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
"""
base_input_to_nodes = InputToNode(hidden_layer_size=300, activation='identity', k_in=10, input_scaling=0.1, bias_scaling=0.0, random_state=1)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=300, activation='tanh', spectral_radius=0.0, leakage=0.1, bias_scaling=0.0, bi_directional=False, k_rec=10, random_state=1)

esn = ESNRegressor(input_to_nodes=[('default', base_input_to_nodes)],
                   nodes_to_nodes=[('default', base_nodes_to_nodes)],
                   regressor=IncrementalRegression(alpha=1e-3), random_state=1)


grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        'spectral_radius': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }

t1 = time.time()
# opt_function(, , encoder, params, X_train, y_train, X_test, y_test, w_in=None)
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_input_to_nodes, base_nodes_to_nodes, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))


base_input_to_nodes = InputToNode(hidden_layer_size=300, activation='identity', k_in=10, input_scaling=0.4, bias_scaling=0.0, random_state=1)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=300, activation='tanh', spectral_radius=0.6, leakage=0.1, bias_scaling=0.0, bi_directional=False, k_rec=10, random_state=1)

esn = ESNRegressor(input_to_nodes=[('default', base_input_to_nodes)],
                   nodes_to_nodes=[('default', base_nodes_to_nodes)],
                   regressor=IncrementalRegression(alpha=1e-3), random_state=1)

grid = {'bias_scaling': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_input_to_nodes, base_nodes_to_nodes, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))
"""
base_input_to_nodes = InputToNode(hidden_layer_size=300, activation='identity', k_in=10, input_scaling=0.4, bias_scaling=0.0, random_state=1)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=300, activation='tanh', spectral_radius=0.6, leakage=0.1, bias_scaling=0.7, bi_directional=False, k_rec=10, random_state=1)

esn = ESNRegressor(input_to_nodes=[('default', base_input_to_nodes)],
                   nodes_to_nodes=[('default', base_nodes_to_nodes)],
                   regressor=IncrementalRegression(alpha=1e-3), random_state=1)

grid = {'k_rec': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_input_to_nodes, base_nodes_to_nodes, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))
"""

"""

