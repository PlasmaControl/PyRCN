import numpy as np
import librosa

from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (ParameterGrid, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.metrics import make_scorer, zero_one_loss
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.util import FeatureExtractor
from pyrcn.datasets import fetch_ptdb_tug_dataset
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.base.blocks import PredefinedWeightsInputToNode


def create_feature_extraction_pipeline(sr=16000):
    audio_loading = Pipeline([("load_audio",
                               FeatureExtractor(func=librosa.load,
                                                kw_args={"sr": sr,
                                                         "mono": True})),
                              ("normal",
                               FeatureExtractor(func=librosa.util.normalize,
                                                kw_args={"norm": np.inf}))])

    feature_extractor = Pipeline([("mel_spectrogram",
                                   FeatureExtractor(
                                       func=librosa.feature.melspectrogram,
                                       kw_args={"sr": sr, "n_fft": 1024,
                                                "hop_length": 160,
                                                "window": 'hann',
                                                "center": False, "power": 2.0,
                                                "n_mels": 80, "fmin": 40,
                                                "fmax": 4000, "htk": True})),
                                  ("power_to_db",
                                   FeatureExtractor(func=librosa.power_to_db,
                                                    kw_args={"ref": 1}))])

    feature_extraction_pipeline = Pipeline(
        [("audio_loading", audio_loading),
         ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


# Load and preprocess the dataset
feature_extraction_pipeline = create_feature_extraction_pipeline()

X_train, X_test, y_train, y_test = fetch_ptdb_tug_dataset(
    data_origin="/scratch/ws/1/s2575425-CSTR_VCTK_Corpus/SPEECH_DATA",
    data_home="/scratch/ws/1/s2575425-pyrcn/f0_estimation/dataset/5",
    preprocessor=feature_extraction_pipeline, force_preprocessing=False,
    augment=5)
X_train, y_train = shuffle(X_train, y_train, random_state=0)

scaler = StandardScaler().fit(np.concatenate(X_train))
for k, X in enumerate(X_train):
    X_train[k] = scaler.transform(X=X)
for k, X in enumerate(X_test):
    X_test[k] = scaler.transform(X=X)


# Define several error functions for $f_{0}$ extraction
def gpe(y_true, y_pred):
    """
    Gross pitch error
    -----------------

    All frames that are considered voiced by both pitch tracker and ground
    truth, for which the relative pitch error is higher than a certain
    threshold (20 percent).
    """
    idx = np.nonzero(y_true*y_pred)[0]
    if idx.size == 0:
        return np.inf
    else:
        return np.sum(np.abs(y_true[idx] - y_pred[idx]) > 0.2 * y_true[idx])\
               / len(np.nonzero(y_true)[0])


def vde(y_true, y_pred):
    """
    Voicing Decision Error
    ----------------------

    Proportion of frames for which an incorrect voiced/unvoiced decision is
    made.
    """
    return zero_one_loss(y_true, y_pred)


def fpe(y_true, y_pred):
    """
    Fine Pitch Error
    ----------------

    Standard deviation of the distribution of relative error values (in cents)
    from the frames that do not have gross pitch errors.
    """
    idx_voiced = np.nonzero(y_true * y_pred)[0]
    idx_correct = np.argwhere(np.abs(y_true - y_pred) <= 0.2 * y_true).ravel()
    idx = np.intersect1d(idx_voiced, idx_correct)
    if idx.size == 0:
        return 0
    else:
        return 100 * np.std(np.log2(y_pred[idx] / y_true[idx]))


def ffe(y_true, y_pred):
    """
    $f_{0}$ Frame Error
    -------------------

    Proportion of frames for which an error (either according to the GPE or the
    VDE criterion) is made.

    FFE can be seen as a single measure for assessing the overall performance
    of a pitch tracker.
    """
    idx_correct = np.argwhere(np.abs(y_true - y_pred) <= 0.2 * y_true).ravel()
    return 1 - len(idx_correct) / len(y_true)


def custom_scorer(y_true, y_pred):
    gross_pitch_error = np.zeros(shape=(len(y_true),))
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        gross_pitch_error[k] = gpe(y_true=y_t[:, 0]*y_t[:, 1],
                                   y_pred=y_p[:, 0]*(y_p[:, 1] >= .5))
    return np.mean(gross_pitch_error)


gpe_scorer = make_scorer(custom_scorer, greater_is_better=False)

kmeans = load("../f0/kmeans_500.joblib")
w_in = np.divide(kmeans.cluster_centers_,
                 np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])

input_to_node = PredefinedWeightsInputToNode(
    predefined_input_weights=w_in.T,
)

# Set up a ESN
# To develop an ESN model for f0 estimation, we need to tune several hyper-
# parameters, e.g., input_scaling, spectral_radius, bias_scaling and leaky
# integration.
# We follow the way proposed in the paper for multipitch tracking and for
# acoustic modeling of piano music to optimize hyper-parameters sequentially.
# We define the search spaces for each step together with the type of search
# (a grid search in this context).
# At last, we initialize an ESNRegressor with the desired output strategy and
# with the initially fixed parameters.

initially_fixed_params = {'hidden_layer_size': 500,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 1.0,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bidirectional': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': uniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': uniform(loc=0, scale=3)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': gpe_scorer}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': gpe_scorer}
kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': gpe_scorer}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                'scoring': gpe_scorer}

searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', RandomizedSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = ESNRegressor(input_to_node=input_to_node).set_params(
    **initially_fixed_params)

try:
    sequential_search = load(
        "../f0/sequential_search_f0_mel_km_500.joblib")
except FileNotFoundError:
    print(FileNotFoundError)
    sequential_search = SequentialSearchCV(
        base_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search,
         "../f0/sequential_search_f0_mel_km_500.joblib")

print(sequential_search)

param_grid = {
    'hidden_layer_size': [50, 100, 200, 400, 500, 800, 1000,
                          1600, 2000, 3200, 4000, 6400, 8000, 16000],
}
for params in ParameterGrid(param_grid):
    estimator = clone(sequential_search.best_estimator_).set_params(**params)
    kmeans = load("../f0/kmeans_" + str(params["hidden_layer_size"])
                  + ".joblib")
    w_in = np.divide(kmeans.cluster_centers_,
                     np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
    estimator.input_to_node.predefined_input_weights = w_in.T

    try:
        cv = load("../f0/speech_ptdb_tug_kmeans_esn_"
                  + str(params["hidden_layer_size"]) + "_0_5.joblib")
    except FileNotFoundError:
        cv = GridSearchCV(estimator=estimator, param_grid={},
                          scoring=gpe_scorer, n_jobs=5, verbose=10).fit(
            X=X_train, y=y_train)
        dump(cv, "../f0/speech_ptdb_tug_kmeans_esn_"
             + str(params["hidden_layer_size"]) + "_0_5.joblib")
    print(cv.cv_results_)
