import numpy as np
import librosa
import librosa.display
import glob
from pyrcn.echo_state_network import ESNRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import ParameterGrid
from scipy.signal import find_peaks
import time
from joblib import Parallel, delayed, dump, load

from matplotlib import pyplot as plt


def opt_function(base_esn, params, X_train, y_train, X_test, y_test):
    esn = clone(base_esn)
    esn.set_params(**params)
    for X, y in zip(X_train, y_train):
        esn.partial_fit(X=X, y=y, update_output_weights=False)
    esn.finalize()
    err_train = []
    for X, y in zip(X_train, y_train):
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        err_train.append(mean_squared_error(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y_pred = esn.predict(X=X, keep_reservoir_state=False)
        err_test.append(mean_squared_error(y, y_pred))
    return [np.mean(err_train), np.mean(err_test)]


def extract_features(filename: str, sr: float = 4000., frame_length: int = 81, target_widening: bool = True):
    s, sr = librosa.load(filename, sr=sr, mono=False)
    X = librosa.util.frame(s[0, :], frame_length=frame_length, hop_length=1).T
    y = librosa.util.frame(binarize_signal(s[1, :], 0.04), frame_length=frame_length, hop_length=1).T
    if target_widening:
        return X, np.convolve(y[:, int(frame_length / 2)], [0.5, 1.0, 0.5], 'same')
    else:
        return X, y[:, int(frame_length / 2)]


def binarize_signal(y, thr=0.04):
    y_diff = np.maximum(np.diff(y, prepend=0), thr)
    peaks, _ = find_peaks(y_diff)
    y_bin = np.zeros_like(y_diff, dtype=int)
    y_bin[peaks] = 1
    return y_bin


all_wavs_m = glob.glob(r"C:\Temp\SpLxDataLondonStudents2008\M\*.wav")
print(len(all_wavs_m))
all_wavs_n = glob.glob(r"C:\Temp\SpLxDataLondonStudents2008\N\*.wav")
print(len(all_wavs_n))

# Extract features for training and hyperparameter optimization
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []
print("Training files:")
for file in all_wavs_m[:8]:
    print(file)
    X, y = extract_features(file, sr = 4000., frame_length = 21)
    X_train.append(X)
    y_train.append(y)
    print(X_train[-1].shape)
    print(y_train[-1].shape)
print("Test files:")
for file in all_wavs_m[8:13]:
    print(file)
    X, y = extract_features(file, sr = 4000., frame_length = 21)
    X_test.append(X)
    y_test.append(y)
    print(X_test[-1].shape)
    print(y_test[-1].shape)
print("Validation files:")
for file in all_wavs_m[13:]:
    print(file)
    X, y = extract_features(file, sr = 4000., frame_length = 21)
    X_val.append(X)
    y_val.append(y)
    print(X_val[-1].shape)
    print(y_val[-1].shape)

base_esn = ESNRegressor(k_in=10, input_scaling=9.0, spectral_radius=0.0, bias=0.0, leakage=1.0, reservoir_size=500,
                        k_res=10, reservoir_activation='tanh', teacher_scaling=1.0, teacher_shift=0.0,
                        bi_directional=False, solver='ridge', beta=1e-5, random_state=1)

grid = {'input_scaling': [9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
       }

losses = Parallel(n_jobs=-1, verbose=50)(delayed(opt_function)(base_esn, params, X_train, y_train, X_test, y_test) for params in ParameterGrid(grid))
print(losses)
exit(0)
