from pathlib import Path
import os
import glob
import numpy as np
import time
import librosa
import pandas as pd
from tqdm import tqdm

from joblib import dump, load

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy import sparse
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import make_scorer
from pyrcn.metrics import mean_squared_error, accuracy_score
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.util import FeatureExtractor
from pyrcn.datasets import fetch_ptdb_tug_dataset
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.base.blocks import InputToNode, PredefinedWeightsInputToNode, NodeToNode, PredefinedWeightsNodeToNode


training_sentences = list(Path("/scratch/ws/1/s2575425-CSTR_VCTK_Corpus/TIMIT/train").rglob("*.wav"))
test_sentences = list(Path("/scratch/ws/1/s2575425-CSTR_VCTK_Corpus/TIMIT/test").rglob("*.wav"))


def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states
    M = np.zeros(shape=(n,n))
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M


def phn_label(phn, frame, hop_length, num_of_frame):
    label = np.empty(num_of_frame, dtype='U5')
    label_number = 0
    idx = int(phn[0][0])
    for i in range(num_of_frame):
        if label_number >= len(phn):
            label[i] = phn[-1][2]
        elif int(phn[label_number][0]) <= idx < int(phn[label_number][1]):
            label[i] = phn[label_number][2]
        else:
            if idx - int(phn[label_number][1]) <= frame / 2:
                label[i] = phn[label_number][2]
                label_number += 1
            else:
                label_number += 1
                label[i] = phn[label_number][2]
        idx += hop_length
    return label


def set_label_number(label):
    phone_39set = {"iy": 0, "ih": 1, "ix": 1, "eh": 2, "ae": 3, "ah": 4, "ax": 4, "ax-h": 4, "uw": 5, "ux": 5, "uh": 6,
                   "aa": 7, "ao": 7, "ey": 8, "ay": 9, "oy": 10, "aw": 11, "ow": 12, "er": 13, "axr": 13,
                   "l": 14, "el": 14, "r": 15, "w": 16, "y": 17, "m": 18, "em": 18, "n": 19, "en": 19, "nx": 19,
                   "ng": 20, "eng": 20, "dx": 21, "jh": 22, "ch": 23, "z": 24, "s": 25, "sh": 26, "zh": 26,
                   "hh": 27, "hv": 27, "v": 28, "f": 29, "dh": 30, "th": 31, "b": 32, "p": 33, "d": 34, "t": 35,
                   "g": 36, "k": 37, "bcl": 38, "pcl": 38, "dcl": 38, "tcl": 38, "gcl": 38, "kcl": 38, "epi": 38,
                   "pau": 38, "h": 38, "q": 38}
    label_idx = np.zeros(len(label))
    for i in range(len(label)):
        label_idx[i] = phone_39set[label[i]]

    return label_idx


scaler = StandardScaler()
X_train = []
y_train = []
X_test = []
y_test = []
for k, f in tqdm(enumerate(shuffle(training_sentences, random_state=42))):
    if not "sa" in str(f):
        y, sr = librosa.core.load(str(f), sr=None, mono=False)
        y = librosa.util.normalize(y)
        y = librosa.effects.preemphasis(y)
        mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=400, hop_length=160, 
                                             n_mels=40, center=False, window="hamming")
        mel = np.log(mel + 1e-5)
        mel_delta = librosa.feature.delta(mel, width=5, order=1)
        mel_deltadelta = librosa.feature.delta(mel, width=5, order=2)
        X_train.append(np.vstack((mel, mel_delta, mel_deltadelta)).T)
        scaler.partial_fit(X_train[-1])
        phn = np.loadtxt(str(f).replace(".wav", ".phn"), dtype=str)
        label = phn_label(phn=phn, frame=400, hop_length=160, num_of_frame=X_train[-1].shape[0])
        label_idx = set_label_number(label)
        y_train.append(label_idx)

for k, f in tqdm(enumerate(shuffle(test_sentences, random_state=42))):
    if not "sa" in str(f):
        y, sr = librosa.core.load(str(f), sr=None, mono=False)
        y = librosa.util.normalize(y)
        y = librosa.effects.preemphasis(y)
        mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=400, hop_length=160, 
                                             n_mels=40, center=False, window="hamming")
        mel = np.log(mel + 1e-5)
        mel_delta = librosa.feature.delta(mel, width=5, order=1)
        mel_deltadelta = librosa.feature.delta(mel, width=5, order=2)
        X_test.append(np.vstack((mel, mel_delta, mel_deltadelta)).T)
        phn = np.loadtxt(str(f).replace(".wav", ".phn"), dtype=str)
        label = phn_label(phn=phn, frame=400, hop_length=160, num_of_frame=X_test[-1].shape[0])
        label_idx = set_label_number(label)
        y_test.append(label_idx)

X_train = np.asarray(X_train, dtype=object)
X_test = np.asarray(X_test, dtype=object)
y_train = np.asarray(y_train, dtype=object)
y_test = np.asarray(y_test, dtype=object)
for k, X in enumerate(X_train):
    X_train[k] = scaler.transform(X)
for k, X in enumerate(X_test):
    X_test[k] = scaler.transform(X)

for k in [50, 100, 200, 400, 500, 800, 1000, 1600, 2000, 3200, 4000, 6400, 8000, 16000]:
    try:
        kmeans = load("../kmeans_" + str(k) + ".joblib")
    except FileNotFoundError:
        kmeans = kmeans = MiniBatchKMeans(n_clusters=k, n_init=200, reassignment_ratio=0, 
                                          max_no_improvement=50, init='k-means++', 
                                          verbose=2, random_state=42)
        kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
        dump(kmeans, "../kmeans_" + str(k) + ".joblib")

initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 1.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bidirectional': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': True}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}
step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}
scoring = {"MSE": make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True), 
           "Acc": make_scorer(accuracy_score)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': scoring, 'refit': 'MSE'}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': scoring, 'refit': 'MSE'}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': scoring, 'refit': 'MSE'}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': scoring, 'refit': 'MSE'}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

try:
    kmeans = load("../kmeans_50.joblib")
except FileNotFoundError:
    kmeans = kmeans = MiniBatchKMeans(n_clusters=50, n_init=200, reassignment_ratio=0, 
                                        max_no_improvement=50, init='k-means++', 
                                        verbose=2, random_state=42)
    kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
    dump(kmeans, "../kmeans_50.joblib")

w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
w_rec = transition_matrix(kmeans.labels_)
input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T)
node_to_node = PredefinedWeightsNodeToNode(predefined_recurrent_weights=w_rec / np.max(np.abs(np.linalg.eigvals(w_rec))))
base_esn = ESNClassifier(input_to_node=input_to_node, node_to_node=node_to_node).set_params(**initially_fixed_params)

try:
    sequential_search = load("../sequential_search_speech_timit_kmeans_rec.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search, "../sequential_search_speech_timit_kmeans_rec.joblib")
print(sequential_search.all_best_params_, sequential_search.all_best_score_)
