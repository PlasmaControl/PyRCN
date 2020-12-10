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


def extract_features(filename: str, sr: float = 4000., frame_length: int = 81, target_widening: bool = True):
    s, sr = librosa.load(filename, sr=sr, mono=False)
    plt.figure()
    plt.plot(s[0, :5000])
    plt.show()
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


esn = load("tmp_models/esn_500u_81.joblib")
X, y = extract_features(filename=all_wavs_m[0], sr=4000., frame_length=81, target_widening=True)
y_pred = esn.predict(X=X)
t = np.arange(len(y_pred)) / 4000
plt.figure(figsize=(6, 1.8))
plt.plot(t, y, color='gray', label='Reference Tx')
plt.plot(t, y_pred, color='blue', label='Tx probabilities')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.xlim([t[13100], t[13400]])
plt.tick_params(direction='in')
plt.title("Tx")
plt.legend()
plt.tight_layout()
plt.show()
exit(0)
