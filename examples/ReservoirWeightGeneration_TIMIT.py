from pathlib import Path
import numpy as np
from scipy.stats import uniform
import librosa
from tqdm import tqdm

from joblib import dump, load

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from sklearn.base import clone
from sklearn.model_selection import (RandomizedSearchCV, ParameterGrid,
                                     GridSearchCV)
from sklearn.metrics import make_scorer
from pyrcn.metrics import accuracy_score
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.base.blocks import (PredefinedWeightsInputToNode,
                               PredefinedWeightsNodeToNode,
                               AttentionWeightsNodeToNode)
from pyrcn.util import FeatureExtractor


training_sentences = list(
    Path("/scratch/ws/1/s2575425-CSTR_VCTK_Corpus/TIMIT/train").rglob("*.wav"))
test_sentences = list(
    Path("/scratch/ws/1/s2575425-CSTR_VCTK_Corpus/TIMIT/test").rglob("*.wav"))


def add_constant(x, constant=1e-5):
    return np.add(x, constant)


def create_feature_extraction_pipeline():
    step1 = FeatureExtractor(func=librosa.load,
                             kw_args={'sr': 16000, 'mono': True})
    step2 = FeatureExtractor(func=librosa.util.normalize)
    step3 = FeatureExtractor(func=librosa.effects.preemphasis)
    step4 = FeatureExtractor(librosa.feature.melspectrogram,
                             kw_args={
                                 'sr': 16000, 'n_fft': 400, 'hop_length': 160,
                                 'n_mels': 40, 'center': False,
                                 'window': "hamming"
                             })
    step5 = FunctionTransformer(add_constant, kw_args={'constant': 1e-5})
    step6 = FunctionTransformer(np.log)
    mel_spectrogram = Pipeline([("load_audio", step1), ("normalize", step2),
                                ("preemphasis", step3),
                                ("mel_spectrogram", step4),
                                ("add", step5), ("log", step6)])
    delta_spectrogram = Pipeline(
        [("mel_spectrogram", mel_spectrogram),
         ("delta", FeatureExtractor(func=librosa.feature.delta,
                                    kw_args={'width': 5, 'order': 1}))])
    delta_delta_spectrogram = Pipeline(
        [("mel_spectrogram", mel_spectrogram),
         ("delta", FeatureExtractor(func=librosa.feature.delta,
                                    kw_args={'width': 5, 'order': 2}))])
    return mel_spectrogram, delta_spectrogram, delta_delta_spectrogram


def transition_matrix(transitions):
    n = 1 + max(transitions)  # number of states
    M = np.zeros(shape=(n, n))
    for (i, j) in zip(transitions, transitions[1:]):
        M[i][j] += 1
    # now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
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
    phone_39set = {
        "iy": 0, "ih": 1, "ix": 1, "eh": 2, "ae": 3, "ah": 4, "ax": 4,
        "ax-h": 4,
        "uw": 5, "ux": 5, "uh": 6, "aa": 7, "ao": 7, "ey": 8, "ay": 9,
        "oy": 10,
        "aw": 11, "ow": 12, "er": 13, "axr": 13, "l": 14, "el": 14, "r": 15,
        "w": 16,
        "y": 17, "m": 18, "em": 18, "n": 19, "en": 19, "nx": 19, "ng": 20,
        "eng": 20,
        "dx": 21, "jh": 22, "ch": 23, "z": 24, "s": 25, "sh": 26, "zh": 26,
        "hh": 27,
        "hv": 27, "v": 28, "f": 29, "dh": 30, "th": 31, "b": 32, "p": 33,
        "d": 34,
        "t": 35, "g": 36, "k": 37, "bcl": 38, "pcl": 38, "dcl": 38, "tcl": 38,
        "gcl": 38, "kcl": 38, "epi": 38, "pau": 38, "h": 38, "q": 38
    }
    label_idx = np.zeros(len(label))
    for i in range(len(label)):
        label_idx[i] = phone_39set[label[i]]

    return label_idx


mel_spectrogram, delta_spectrogram, delta_delta_spectrogram \
    = create_feature_extraction_pipeline()
scaler = StandardScaler()
X_train = []
y_train = []
for k, f in tqdm(enumerate(shuffle(training_sentences, random_state=42))):
    if "sa" not in str(f):
        mel = mel_spectrogram.transform(X=str(f))
        mel_delta = delta_spectrogram.transform(X=str(f))
        mel_delta_delta = delta_delta_spectrogram.transform(X=str(f))
        X_train.append(np.vstack((mel, mel_delta, mel_delta_delta)).T)
        scaler.partial_fit(X_train[-1])
        phn = np.loadtxt(str(f).replace(".wav", ".phn"), dtype=str)
        label = phn_label(phn=phn, frame=400, hop_length=160,
                          num_of_frame=X_train[-1].shape[0])
        label_idx = set_label_number(label)
        y_train.append(label_idx)

X_test = []
y_test = []
for k, f in tqdm(enumerate(shuffle(test_sentences, random_state=42))):
    if "sa" not in str(f):
        mel = mel_spectrogram.transform(X=str(f))
        mel_delta = delta_spectrogram.transform(X=str(f))
        mel_delta_delta = delta_delta_spectrogram.transform(X=str(f))
        X_test.append(np.vstack((mel, mel_delta, mel_delta_delta)).T)
        phn = np.loadtxt(str(f).replace(".wav", ".phn"), dtype=str)
        label = phn_label(phn=phn, frame=400, hop_length=160,
                          num_of_frame=X_test[-1].shape[0])
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

kmeans = load("../kmeans_50.joblib")
w_in = np.divide(kmeans.cluster_centers_,
                 np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
input_to_node = PredefinedWeightsInputToNode(
    predefined_input_weights=w_in.T,
)
w_rec = 2 * transition_matrix(kmeans.labels_) - 1
node_to_node = PredefinedWeightsNodeToNode(predefined_recurrent_weights=w_rec)

initially_fixed_params = {
    'hidden_layer_size': 50,
    'k_in': 10,
    'input_scaling': 0.4,
    'input_activation': 'identity',
    'bias_scaling': 0.2,
    'spectral_radius': 0.0,
    'leakage': 1.0,
    'k_rec': 10,
    'reservoir_activation': 'tanh',
    'bidirectional': False,
    'alpha': 1e-5,
    'random_state': 42,
    'requires_sequence': True
}

step1_esn_params = {
    'input_scaling': uniform(loc=1e-2, scale=1),
    'spectral_radius': uniform(loc=0, scale=2)
}
step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': uniform(loc=0, scale=3)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}
scoring = make_scorer(accuracy_score)

kwargs_step1 = {
    'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': scoring
}
kwargs_step2 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': scoring
}
kwargs_step3 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': scoring
}
kwargs_step4 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': scoring
}

searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', RandomizedSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = ESNClassifier(input_to_node=input_to_node,
                         node_to_node=node_to_node
                         ).set_params(**initially_fixed_params)

try:
    sequential_search = load(
        "../sequential_search_speech_timit_km_esn_rec_-1_1"
        ".joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn,
                                           searches=searches).fit(X_train,
                                                                  y_train)
    dump(sequential_search,
         "../sequential_search_speech_timit_km_esn_rec_-1_1"
         ".joblib")
print(sequential_search.all_best_params_, sequential_search.all_best_score_)

param_grid = {
    'hidden_layer_size': [50, 100, 200, 400, 500, 800, 1000,
                          1600, 2000, 3200, 4000, 6400, 8000, 16000],
}
for params in ParameterGrid(param_grid):
    estimator = clone(sequential_search.best_estimator_).set_params(**params)
    kmeans = load("../kmeans_" + str(params["hidden_layer_size"])
                  + ".joblib")
    w_in = np.divide(kmeans.cluster_centers_,
                     np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
    w_rec = 2 * transition_matrix(kmeans.labels_) - 1
    estimator.input_to_node.predefined_input_weights = w_in.T
    try:
        cv = load("../speech_timit_km_esn_rec_-1_1_"
                  + str(params["hidden_layer_size"]) + ".joblib")
    except FileNotFoundError:
        cv = GridSearchCV(estimator=estimator, param_grid={}, scoring=scoring,
                          n_jobs=5, verbose=10).fit(X=X_train, y=y_train)
        dump(cv, "../speech_timit_km_esn_rec_-1_1_" +
             str(params["hidden_layer_size"]) + ".joblib")
    print(cv.cv_results_)
