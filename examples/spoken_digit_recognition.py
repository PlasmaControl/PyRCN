import glob
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from joblib import dump, load
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.base.blocks import PredefinedWeightsInputToNode
from pyrcn.metrics import accuracy_score
from pyrcn.model_selection import SequentialSearchCV
import librosa
import librosa.display


all_files = glob.glob(r"E:\free-spoken-digit-dataset\recordings\*.wav")
print(len(all_files))

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
    mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(0.01 * sr),
                                n_fft=256, htk=True, n_mels=100, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    if idx <= 4:
        X_test.append(mfcc.T)
        y_test.append(label)
    else:
        X_train.append(mfcc.T)
        y_train.append(label)
print("done!")

scaler = StandardScaler().fit(X=np.vstack(X_train + X_test))
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

print(len(X_train), len(y_train), X_train[0].shape, y_train[0])
print(len(X_test), len(y_test), X_test[0].shape, y_test[0])

kmeans = MiniBatchKMeans(n_clusters=200, n_init=200, reassignment_ratio=0,
                         max_no_improvement=50, init='k-means++', verbose=2,
                         random_state=0)
kmeans.fit(X=np.concatenate(np.concatenate((X_train_scaled, X_test_scaled))))
w_in = np.divide(kmeans.cluster_centers_,
                 np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
w_in = np.pad(w_in, ((0, 800 - 200), (0, 0)), mode='constant',
              constant_values=0)

initially_fixed_params = {
    'hidden_layer_size': 800,
    'k_in': 10,
    'input_scaling': 0.4,
    'input_activation': 'identity',
    'bias_scaling': 0.0,
    'spectral_radius': 0.0,
    'leakage': 0.1,
    'k_rec': 10,
    'reservoir_activation': 'tanh',
    'bidirectional': False,
    'wash_out': 0,
    'continuation': False,
    'alpha': 1e-3,
    'random_state': 42
}

step1_esn_params = {
    'input_scaling': uniform(loc=1e-2, scale=1),
    'spectral_radius': uniform(loc=0, scale=2)
}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {
    'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': 1,
    'scoring': make_scorer(accuracy_score)
}
kwargs_step2 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(accuracy_score)
}
kwargs_step3 = {
    'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)
}
kwargs_step4 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(accuracy_score)
}

searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_km_esn = ESNClassifier(
    input_to_node=PredefinedWeightsInputToNode(
        predefined_input_weights=w_in.T), **initially_fixed_params)

try:
    sequential_search = load("../sequential_search_fsdd_km_sparse_200.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_km_esn, searches=searches).fit(
        X_train_scaled, y_train)
    dump(sequential_search, "../sequential_search_fsdd_km_sparse_200.joblib")
