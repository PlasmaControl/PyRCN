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
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from pyrcn.echo_state_network import ESNRegressor
import matplotlib.pyplot as plt

import librosa


all_files = glob.glob(r"E:\free-spoken-digit-dataset\recordings\*.wav")
print(len(all_files))

X = [None] * len(all_files)
y = [None] * len(all_files)
print("extracting features...")
with tqdm(total=len(all_files)) as pbar:
    for k, f in enumerate(all_files):
        basename = os.path.basename(f).split('.')[0]
        # Get label (0-9) of recording.
        label = int(basename.split('_')[0])
        # Load the audio signal and normalize it.
        x, sr = librosa.core.load(f, sr=None, mono=False)
        # x /= np.max(np.abs(x))
        mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(0.01*sr), n_fft=256, htk=True, n_mels=100, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        X[k] = mfcc.T
        y[k] = label
        pbar.update(1)
print("done!")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1, stratify=y_test)

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
def opt_function(base_reg, encoder, params, X_train, y_train, X_test, y_test, w_in=None):
    reg = clone(base_reg).set_params(**params)
    if w_in is not None:
        reg.initialize_from_outside(y=enc.transform(np.asarray(y_train[0]).reshape(-1, 1)),
                                    n_features=X_train[0].shape[0], input_weights=w_in, reservoir_weights=None,
                                    bias_weights=None)
    for X, y in zip(X_train, y_train):
        y = encoder.transform(np.asarray(y).reshape(1, -1))
        y = np.repeat(np.atleast_2d(y), repeats=X.shape[0], axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    err_train = []
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        Y_true_train.append(y)
        y_true = encoder.transform(np.asarray(y).reshape(1, -1))
        y_true = np.repeat(np.atleast_2d(y_true), repeats=X.shape[0], axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_train.append(mean_squared_error(y_true, y_pred))
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    err_test = []
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        Y_true_test.append(y)
        y_true = encoder.transform(np.asarray(y).reshape(1, -1))
        y_true = np.repeat(np.atleast_2d(y_true), repeats=X.shape[0], axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_test.append(mean_squared_error(y_true, y_pred))
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
    # return [np.mean(err_train), np.mean(err_test)]
    return [1 - accuracy_score(Y_true_train, Y_pred_train), 1 - accuracy_score(Y_true_test, Y_pred_test)]


"""
base_esn = ESNRegressor(k_in=-1, input_scaling=0.1, spectral_radius=0.0, bias=0.0, leakage=1.0, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=-1, input_scaling=0.3, spectral_radius=0.0, bias=0.0, leakage=1.0, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'leakage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=-1, input_scaling=0.3, spectral_radius=0.0, bias=0.0, leakage=0.1, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'spectral_radius': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=-1, input_scaling=0.8, spectral_radius=0.3, bias=0.0, leakage=0.1, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'bias': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=-1, input_scaling=0.8, spectral_radius=0.3, bias=0.6, leakage=0.1, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'k_res': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=-1, input_scaling=0.8, spectral_radius=0.3, bias=0.6, leakage=0.1, reservoir_size=1000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

esn = clone(base_esn)
print("Train the ESN model...")
with tqdm(total=len(X_train_scaled)) as pbar:
    for X, y in zip(X_train_scaled, y_train):
        y = enc.transform(np.asarray(y).reshape(1, -1))
        y = np.repeat(np.atleast_2d(y), repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, update_output_weights=False)
        pbar.update(1)
esn.finalize()
print("... done!")

Y_true_train = []
Y_pred_train = []
print("Test the ESN model on the training data...")
with tqdm(total=len(X_train_scaled)) as pbar:
    for X, y in zip(X_train_scaled, y_train):
        Y_true_train.append(y)
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        pbar.update(1)
cm = confusion_matrix(Y_true_train, Y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train)))
plt.show()

Y_true_test = []
Y_pred_test = []
print("Test the ESN model on the test data...")
with tqdm(total=len(X_test_scaled)) as pbar:
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        pbar.update(1)
cm = confusion_matrix(Y_true_test, Y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test)))
plt.show()
"""

kmeans = MiniBatchKMeans(n_clusters=500, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=1, random_state=0)
kmeans.fit(X=np.vstack(X_train_scaled))

new_input_weights = kmeans.cluster_centers_
# To compute the norm of the cluster centers, use the following line:
w_in = np.pad(np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]), ((0, 3500), (0, 0)), mode='constant', constant_values=0)
# w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])

base_esn = ESNRegressor(k_in=10, input_scaling=0.1, spectral_radius=0.0, bias=0.0, leakage=1.0, reservoir_size=4000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

"""
grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_val_scaled, y_val, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=10, input_scaling=0.4, spectral_radius=0.0, bias=0.0, leakage=1.0, reservoir_size=3000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'leakage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))
"""
base_esn = ESNRegressor(k_in=10, input_scaling=0.4, spectral_radius=0.0, bias=0.0, leakage=0.1, reservoir_size=4000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'spectral_radius': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=10, input_scaling=1.2, spectral_radius=0.6, bias=0.0, leakage=0.1, reservoir_size=4000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'bias': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=10, input_scaling=1.2, spectral_radius=0.6, bias=0.7, leakage=0.1, reservoir_size=4000,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

grid = {'k_res': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
       }

t1 = time.time()
losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, enc, params, X_train_scaled, y_train, X_test_scaled, y_test, w_in) for params in ParameterGrid(grid))
losses = np.asarray(losses)
print("Finished in {0} seconds!".format(time.time() - t1))

print("The lowest training DER: {0}; parameter combination: {1}".format(np.min(losses[:, 0]), ParameterGrid(grid)[np.argmin(losses[:, 0])]))
print("The lowest validation DER: {0}; parameter combination: {1}".format(np.min(losses[:, 1]), ParameterGrid(grid)[np.argmin(losses[:, 1])]))

base_esn = ESNRegressor(k_in=10, input_scaling=1.2, spectral_radius=0.6, bias=0.7, leakage=0.1, reservoir_size=4000,
                        k_res=5, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-3, random_state=1)

esn = clone(base_esn)
esn.initialize_from_outside(y=enc.transform(np.asarray(y_train[0]).reshape(-1, 1)), n_features=X_train[0].shape[0],
                            input_weights=w_in, reservoir_weights=None, bias_weights=None)
print("Train the ESN model...")
with tqdm(total=len(X_train_scaled)) as pbar:
    for X, y in zip(X_train_scaled, y_train):
        y = enc.transform(np.asarray(y).reshape(1, -1))
        y = np.repeat(np.atleast_2d(y), repeats=X.shape[0], axis=0)
        esn.partial_fit(X=X, y=y, update_output_weights=False)
        pbar.update(1)
esn.finalize()
print("... done!")

Y_true_train = []
Y_pred_train = []
print("Test the ESN model on the training data...")
with tqdm(total=len(X_train_scaled)) as pbar:
    for X, y in zip(X_train_scaled, y_train):
        Y_true_train.append(y)
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
        pbar.update(1)
cm = confusion_matrix(Y_true_train, Y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_train, Y_pred_train)))
plt.show()

Y_true_test = []
Y_pred_test = []
print("Test the ESN model on the test data...")
with tqdm(total=len(X_test_scaled)) as pbar:
    for X, y in zip(X_test_scaled, y_test):
        Y_true_test.append(y)
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
        pbar.update(1)
cm = confusion_matrix(Y_true_test, Y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(Y_true_test, Y_pred_test)))
plt.show()



