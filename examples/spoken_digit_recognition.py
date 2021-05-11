#!/usr/bin/env python
# coding: utf-8

# Spoken digit recognition using the Free Spoken Digit Dataset (FSDD)
# 
# At first, import packages to be used for the experiments

# In[1]:


import os, sys
cwd = os.getcwd()
module_path = os.path.dirname(cwd)  # target working directory

sys.path = [item for item in sys.path if item != module_path]  # remove module_path from sys.path
sys.path.append(module_path)  # add module_path to sys.path

import glob
import os
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.utils import shuffle
from joblib import Parallel, delayed, dump, load
from pyrcn.echo_state_network import SeqToLabelESNClassifier
from pyrcn.base import PredefinedWeightsInputToNode
from pyrcn.metrics import accuracy_score, classification_report, confusion_matrix
from pyrcn.model_selection import SequentialSearchCV
import matplotlib
from matplotlib import pyplot as plt
#Options
import scipy.stats
plt.rc('image', cmap='RdBu')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import librosa
import librosa.display


# Print number of files that are included in the dataset
all_files = glob.glob(r"E:\free-spoken-digit-dataset\recordings\*.wav")
print(len(all_files))





# Extract features and labels from all signals

X_train = []
X_test = []
y_train = []
y_test = []
print("extracting features...")
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
print("done!")


# ## Normalize all features using the StandardScaler from scikit-learn."
scaler = StandardScaler().fit(X=np.vstack(X_train))
X_train_scaled = np.empty(shape=(len(X_train),), dtype=object)
X_test_scaled = np.empty(shape=(len(X_test),), dtype=object)
y_train = np.array(y_train, dtype=object)
X_train, X_train_scaled, y_train = shuffle(X_train, X_train_scaled, y_train)
y_test = np.array(y_test, dtype=object)
for k in range(len(X_train)):
    X_train_scaled[k] = scaler.transform(X_train[k])
    y_train[k] = np.atleast_1d(y_train[k]).astype(int)
for k in range(len(X_test)):
    X_test_scaled[k] = scaler.transform(X_test[k])
    y_test[k] = np.atleast_1d(y_test[k]).astype(int)

# Validate training and test sizes
print(len(X_train), len(y_train), X_train[0].shape, y_train[0])
print(len(X_test), len(y_test), X_test[0].shape, y_test[0])


initially_fixed_params = {'hidden_layer_size': 100,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 1.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bi_directional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 10}

step1_esn_params = {'input_scaling': np.linspace(0.1, 1.0, 10),
                    'spectral_radius': np.linspace(0.0, 1.0, 11)}

step2_esn_params = {'leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]}

kwargs = {'verbose': 1, 'n_jobs': 1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs),
            ('step4', GridSearchCV, step4_esn_params, kwargs)]

base_esn = SeqToLabelESNClassifier(**initially_fixed_params)

try:
    sequential_search = load("sequential_search.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train_scaled, y_train)
    dump(sequential_search, "sequential_search.joblib")

# Initialize an Echo State Network

# In[ ]:


base_input_to_node = InputToNode(hidden_layer_size=100, activation='identity', k_in=10, input_scaling=0.4, bias_scaling=0.0, random_state=10)
base_node_to_node = NodeToNode(hidden_layer_size=100, spectral_radius=1.0, leakage=0.1, bias_scaling=0.0, k_rec=10, random_state=10)

base_esn = ESNClassifier(input_to_node=base_input_to_node,
                         node_to_node=base_node_to_node,
                         regressor=FastIncrementalRegression(alpha=1e-3),
                         random_state=1)


# Clone the base_esn and fit it on the training data

# In[ ]:


esn = clone(base_esn)
print("Train the ESN model...")
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled[:-1], y_train + y_val[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), postpone_inverse=True)
        pbar.update(1)
    X = X_val[-1]
    y = np.repeat(y_val[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, postpone_inverse=False)
    pbar.update(1)
print("... done!")


# Test the model on the training and test set

# In[ ]:


Y_true_train = []
Y_pred_train = []
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled, y_train + y_val):
        Y_true_train.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_train, Y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train, digits=10)))
plt.show()

Y_true_test = []
Y_pred_test = []
with tqdm(total=len(X_test_scaled)) as pbar:
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_test, Y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test, digits=10)))
plt.show()


# Visualization of time signals from the training set

# In[ ]:


fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax1 = plt.subplot(111)
im = ax1.imshow(X_train_scaled[361].T,vmin=np.min(X_train_scaled[361]), vmax=np.max(X_train_scaled[361]))

plt.xlim([0,X_train_scaled[361].shape[0]])
plt.ylim([0, X_train_scaled[361].shape[1] - 1])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{u}[n]$')
plt.grid()

divider = make_axes_locatable(ax1)
ax2 = divider.append_axes("top", size="100%", pad=0.7)
cax = divider.append_axes("right", size="3%", pad=0.2)
cb = plt.colorbar( im, ax=ax1, cax=cax )

t = np.arange(len(time_signals_train[361])) / sr
#ax2 = plt.subplot( gs[-1,:] )  # , sharex=ax1
ax2.plot(t, time_signals_train[361], 'dimgrey')
ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(-0.4, 0.4)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$y(t)$')
ax2.grid(True)

# fig.tight_layout()
width = 3.487
height =3 * width / 1.618
fig.set_size_inches(width, height)
plt.savefig('time_signal_and_features_train.pdf', bbox_inches = 'tight', pad_inches = 0)
# np.savetxt(X=np.vstack((t, time_signals_train[1])).T, fname="time_signal_train.txt", delimiter="\t")


# Visualization of features from the training set

# In[ ]:


fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
im = plt.imshow(X_train_scaled[361].T,vmin=np.min(X_train_scaled[361]), vmax=np.max(X_train_scaled[361]))
plt.xlim([0,X_train_scaled[361].shape[0]])
plt.ylim([0, X_train_scaled[361].shape[1] - 1])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{u}[n]$')
# plt.colorbar(im)
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)

plt.colorbar(im, cax=cax)
fig.set_size_inches(width, height)
plt.savefig('features_train.pdf', bbox_inches = 'tight', pad_inches = 0)


# Visualizations of features from the test set

# In[ ]:


fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
im = plt.imshow(X_test_scaled[0].T,vmin=np.min(X_test_scaled[0]), vmax=np.max(X_test_scaled[0]))
plt.xlim([0, X_test[0].shape[0]])
plt.ylim([0, X_test[0].shape[1] - 1])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{u}[n]$')
# plt.colorbar(im)
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)

plt.colorbar(im, cax=cax)
fig.set_size_inches(width, height)
plt.savefig('features_test.pdf', bbox_inches = 'tight', pad_inches = 0)


# Visualization of a reservoir state from the training set

# In[ ]:


_ = esn.predict(X=X_train_scaled[361])
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
rng = np.random.default_rng(12345)
idx = rng.choice(100, 50)
im = plt.imshow(esn.nodes_to_nodes[0][1]._hidden_layer_state[:, idx].T,vmin=-1, vmax=1)
plt.xlim([0, esn.nodes_to_nodes[0][1]._hidden_layer_state.shape[0]])
plt.ylim([0, 50])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
tick_locator = ticker.MaxNLocator(nbins=5)
cb = plt.colorbar(im, cax=cax)
cb.locator = tick_locator
cb.update_ticks()
fig.set_size_inches(0.5*width, height)
plt.savefig('input_scaling_rand_train.pdf', bbox_inches = 'tight', pad_inches = 0)

print(f_name_train[361])


# Visualization of a reservoir state from the test set

# In[ ]:


_ = esn.predict(X=X_test_scaled[0])
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
rng = np.random.default_rng(12345)
idx = rng.choice(100, 50)
im = plt.imshow(esn.nodes_to_nodes[0][1]._hidden_layer_state[:, idx].T,vmin=-1, vmax=1)
plt.xlim([0, esn.nodes_to_nodes[0][1]._hidden_layer_state.shape[0]])
plt.ylim([0, 50])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cb = plt.colorbar(im, cax=cax)
cb.locator = tick_locator
cb.update_ticks()
fig.set_size_inches(0.5*width, height)
plt.savefig('spectral_radius_rand_test.pdf', bbox_inches = 'tight', pad_inches = 0)


# Random experiments

# In[ ]:


for rs in range(20):
    kmeans = MiniBatchKMeans(n_clusters=100, n_init=20, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=0, random_state=rs)
    kmeans.fit(X=np.concatenate(X_train_scaled+X_val_scaled))
    w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
    print("Train the ESN model...")
    base_input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T, activation='identity', input_scaling=0.8)
    base_node_to_node = NodeToNode(hidden_layer_size=100, spectral_radius=0.4, leakage=0.1, bias_scaling=0.0, k_rec=10, random_state=10)
    base_reg = FastIncrementalRegression(alpha=1e-3)
    
    esn = ESNClassifier(input_to_node=base_input_to_node,
                        node_to_node=base_node_to_node,
                        regressor=base_reg)
    
    for X, y in zip(X_train_scaled + X_val_scaled[:-1], y_train + y_val[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), postpone_inverse=True)
    X = X_val[-1]
    y = np.repeat(y_val[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, postpone_inverse=False)
    print("... done!")
    Y_true_train = []
    Y_pred_train = []
    mse_train = []
    mse_test = []
    for X, y in zip(X_train_scaled + X_val_scaled, y_train + y_val):
        Y_true_train.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        # mse_train.append(mean_squared_error(y, y_pred))
    print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train, digits=10)))
    # print("MSE training: %f\n" % (np.mean(mse_train)))
    
    Y_true_test = []
    Y_pred_test = []
    mse_test = []
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        # mse_test.append(mean_squared_error(y, y_pred))
    print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test, digits=10)))
    # print("MSE test: %f\n" % (np.mean(mse_test)))


# K-Means Clustering

# In[ ]:


kmeans = MiniBatchKMeans(n_clusters=100, n_init=20, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=2, random_state=0)
kmeans.fit(X=np.concatenate(X_train_scaled+X_val_scaled))


# 

# In[ ]:


w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
base_input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T, activation='identity', input_scaling=0.8)
base_node_to_node = NodeToNode(hidden_layer_size=100, spectral_radius=0.4, leakage=0.1, bias_scaling=0.0, k_rec=10, random_state=10)
base_reg = FastIncrementalRegression(alpha=1e-3)

esn = ESNClassifier(input_to_node=base_input_to_node,
                    node_to_node=base_node_to_node,
                    regressor=base_reg)


# In[ ]:


print("Train the ESN model...")
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled[:-1], y_train + y_val[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), postpone_inverse=True)
        pbar.update(1)
    X = X_val[-1]
    y = np.repeat(y_val[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, postpone_inverse=False)
    pbar.update(1)
print("... done!")


# Test

# In[ ]:


Y_true_train = []
Y_pred_train = []
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled, y_train + y_val):
        Y_true_train.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_train, Y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train, digits=10)))
plt.show()

Y_true_test = []
Y_pred_test = []
with tqdm(total=len(X_test_scaled)) as pbar:
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_test, Y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test, digits=10)))
plt.show()


# Visualization of a reservoir state from the training set

# In[ ]:


_ = esn.predict(X=X_train_scaled[361])
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
rng = np.random.default_rng(12345)
idx = rng.choice(100, 50)
im = plt.imshow(esn.nodes_to_nodes[0][1]._hidden_layer_state[:, idx].T,vmin=-1, vmax=1)
plt.xlim([0, esn.nodes_to_nodes[0][1]._hidden_layer_state.shape[0]])
plt.ylim([0, len(idx)])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cb = plt.colorbar(im, cax=cax)
cb.locator = tick_locator
cb.update_ticks()
fig.set_size_inches(.5*width, height)
plt.savefig('input_scaling_kmeans_train.pdf', bbox_inches = 'tight', pad_inches = 0)


# Visualization of a reservoir state from the test set

# In[ ]:


_ = esn.predict(X=X_test_scaled[0])
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = plt.gca()
rng = np.random.default_rng(12345)
idx = rng.choice(100, 50)
im = plt.imshow(esn.nodes_to_nodes[0][1]._hidden_layer_state[:, idx].T,vmin=-1, vmax=1)
plt.xlim([0, esn.nodes_to_nodes[0][1]._hidden_layer_state.shape[0]])
plt.ylim([0, len(idx)])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.grid()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cb = plt.colorbar(im, cax=cax)
cb.locator = tick_locator
cb.update_ticks()
fig.set_size_inches(.5*width, height)
plt.savefig('leakage_kmeans_test.pdf', bbox_inches = 'tight', pad_inches = 0)


# In[ ]:


plt.hist(np.var(esn.nodes_to_nodes[0][1]._hidden_layer_state, axis=0), bins=10, orientation='horizontal')
kmeans_states_test = esn.nodes_to_nodes[0][1]._hidden_layer_state
plt.hist(np.var(kmeans_states_test, axis=0), orientation='horizontal')
np.min(np.var(esn.nodes_to_nodes[0][1]._hidden_layer_state, axis=0)), np.max(np.var(esn.nodes_to_nodes[0][1]._hidden_layer_state, axis=0))


# In[ ]:


plt.hist(np.var(rand_states_test, axis=0), alpha=0.4, bins=10, color="black", label='R-ESN')
plt.hist(np.var(kmeans_states_test, axis=0), bins=10, color="black", label='KM-ESN')
plt.legend(loc='upper right')
fig.set_size_inches(width, height)
plt.show()
np.var(rand_states_training, axis=0).shape


# Increase the reservoir size

# In[ ]:


kmeans = MiniBatchKMeans(n_clusters=500, n_init=20, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=1, random_state=0)
kmeans.fit(X=np.concatenate(X_train_scaled))
# w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
w_in = np.pad(np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]), ((0, 4500), (0, 0)), mode='constant', constant_values=0)

base_input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T, activation='identity', input_scaling=0.2)
base_node_to_node = NodeToNode(hidden_layer_size=5000, spectral_radius=0.6, leakage=0.1, bias_scaling=0.0, k_rec=10, random_state=10)
base_reg = FastIncrementalRegression(alpha=1e-3)

esn = ESNClassifier(input_to_node=base_input_to_node,
                    node_to_node=base_node_to_node,
                    regressor=base_reg)


# In[ ]:


print("Train the ESN model...")
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled[:-1], y_train + y_val[:-1]):
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, classes=np.arange(10), postpone_inverse=True)
        pbar.update(1)
    X = X_val[-1]
    y = np.repeat(y_val[-1], repeats=X.shape[0], axis=0)
    esn.partial_fit(X=X, y=y, postpone_inverse=True)
    pbar.update(1)
print("... done!")


# Test

# In[ ]:


Y_true_train = []
Y_pred_train = []
with tqdm(total=len(X_train_scaled) + len(X_val_scaled)) as pbar:
    for X, y in zip(X_train_scaled + X_val_scaled, y_train + y_val):
        Y_true_train.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_train, Y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train, digits=10)))
plt.show()

Y_true_test = []
Y_pred_test = []
with tqdm(total=len(X_test_scaled)) as pbar:
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict_proba(X=X)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        y = np.repeat(y, repeats=X.shape[0], axis=0)
        pbar.update(1)
cm = confusion_matrix(Y_true_test, Y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test, digits=10)))
plt.show()


# Visualization of a reservoir state from the training set

# In[ ]:


_ = esn.predict(X=X_train_scaled[0], keep_reservoir_state=True)
np.random.seed(0)
index = np.random.choice(esn.reservoir_state.shape[1], 50, replace=False)
plt.figure(figsize=(4, 3))
im = plt.imshow(esn.reservoir_state[:, index].T,vmin=-1, vmax=1)
plt.xlim([0, esn.reservoir_state.shape[0]])
plt.ylim([0, 50])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.colorbar(im)
plt.grid()
plt.tight_layout()
plt.savefig('reservoir_size_kmeans_train.pdf')


# Visualization of a reservoir state from the test set

# In[ ]:


_ = esn.predict(X=X_test_scaled[0], keep_reservoir_state=True)
np.random.seed(0)
index = np.random.choice(esn.reservoir_state.shape[1], 50, replace=False)
plt.figure(figsize=(4, 3))
im = plt.imshow(esn.reservoir_state[:, index].T,vmin=-1, vmax=1)
plt.xlim([0, esn.reservoir_state.shape[0]])
plt.ylim([0, 50])
plt.xlabel(r'$n$')
plt.ylabel(r'$\mathbf{r}[n]$')
plt.colorbar(im)
plt.grid()
plt.tight_layout()
plt.savefig('reservoir_size_kmeans_test.pdf')


# Functions to fit KMeans and KMedoids for different settings (K, minibatch)

# In[ ]:


def fit_k_means(k, mini_batch=False, X=np.ndarray):
    print(k)
    if mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=k, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=0)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=0)
    kmeans.fit(X=np.vstack(X_train_scaled))
    return kmeans.inertia_

def fit_k_medoids(k):
    kmedoids = KMedoids(n_clusters=k, metric='euclidean',init='k-medoids++', max_iter=300, random_state=0)
    kmedoids.fit(X=np.vstack(X_train_scaled))
    return kmedoids.inertia_


# Sweep along various $K$ and compare $K$-means, Mini-batch $K$-means and $K$-medoids

# In[ ]:


inertias_k_means = Parallel(n_jobs=-1, verbose=50)(delayed(fit_k_means)(k, True) for k in [1460])
# silhouette_scores_k_means = [None] * len(range(2, 1001))
# for k in range(2, 10):
#     print(k)
#     silhouette_scores_k_means[k-2] = fit_k_means(k, True, X_train_scaled)
# inertias_k_medoids = Parallel(n_jobs=-1, verbose=50)(delayed(fit_k_medoids)(k) for k in range(10, 501, 10))


# In[ ]:


np.savetxt(X=inertias_k_means, fname="inertias_minibatch_k_means.csv")


# Visualization

# In[ ]:


plt.figure(figsize=(6, 3))
# plt.plot(range(10, 1001, 10), inertias_k_means, label=r"$K$-Means")
plt.plot(range(10, 100, 10), silhouette_scores_k_means, label=r"Mini-batch-$K$-Means")
# plt.plot(range(10, 501, 10), inertias_k_medoids, label="$K-\text{Medoids}$")
plt.xlabel(r'$K$')
plt.ylabel(r'Silhouette score')
plt.xlim([10, 100])
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("silhouette_score.pdf")


# In[ ]:




