import numpy as np
import os
import csv
from sklearn.utils.fixes import loguniform
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from joblib import dump, load
import time

# import librosa
from madmom.audio.signal import normalize
from madmom.io.audio import load_wave_file
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor

from pyrcn.util import FeatureExtractor
from pyrcn.echo_state_network import SeqToSeqESNClassifier
from pyrcn.datasets import fetch_maps_piano_dataset
from pyrcn.metrics import accuracy_score, mean_squared_error
from pyrcn.model_selection import SequentialSearchCV


# Create a feature extraction pipeline. This is basically a transformer that takes an audio file name
# and returns a feature vector sequence.

def create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100.):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))])

    sig = SignalProcessor(num_channels=1, sample_rate=sr)
    multi = ParallelProcessor([])
    for frame_size in frame_sizes:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps_hz)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
                                            norm_filters=True, unique_filters=True)
        spec = LogarithmicSpectrogramProcessor(log=np.log10, mul=5, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor([frames, stft, filt, spec, diff]))
    feature_extractor = FeatureExtractor(SequentialProcessor([sig, multi, np.hstack]))

    feature_extraction_pipeline = Pipeline([("audio_loading", audio_loading),
                                            ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


# Load and preprocess the dataset
feature_extraction_pipeline = create_feature_extraction_pipeline(sr=44100, frame_sizes=[2048], fps_hz=100)

X_train, X_test, y_train, y_test = fetch_maps_piano_dataset(data_origin="/projects/p_transcriber/MAPS", 
                                                            data_home=None, preprocessor=feature_extraction_pipeline,
                                                            force_preprocessing=True, label_type="pitch")

# ESN preparation
initially_fixed_params = {'hidden_layer_size': 500,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'k_in': 10,
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'spectral_radius': 0.1,
                          'leakage': 0.1,
                          'bi_directional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'leakage': loguniform(1e-5, 1e0)}
kwargs_1 = {'random_state': 42,
           'verbose': 2,
           'n_jobs': -1,
           'n_iter': 14,
           'scoring': make_scorer(accuracy_score)}
step2_esn_params = {'input_scaling': np.linspace(0.1, 1.0, 10),
                    'spectral_radius': np.linspace(0.0, 1.5, 16)}

step3_esn_params = {'bias_scaling': np.linspace(0.0, 3.0, 31)}

kwargs_2_3 = {'verbose': 2, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_1),
            ('step2', GridSearchCV, step2_esn_params, kwargs_2_3),
            ('step3', GridSearchCV, step3_esn_params, kwargs_2_3)]

base_esn = SeqToSeqESNClassifier(**initially_fixed_params)

try:
    sequential_search = load("sequential_search.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search, "sequential_search.joblib")

# Use the ESN with final hyper-parameters
base_esn = sequential_search.best_estimator_

# Test the ESN

param_grid = {'hidden_layer_size': [500, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 32000],
              'bi_directional': [False, True]}

print("Fit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in ParameterGrid(param_grid):
    esn_cv = cross_validate(clone(base_esn).set_params(**params), X=X_train, y=y_train, scoring=make_scorer(accuracy_score), n_jobs=-1)
    t1 = time.time()
    esn = clone(base_esn).set_params(**params).fit(X_train, y_train, n_jobs=-1)
    t_fit = time.time() - t1
    dump(esn, "esn_" + str(params["hidden_layer_size"]) + "_" + str(params["bi_directional"]) + ".joblib")
    mem_size = esn.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, esn.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(esn_cv, t_fit, t_inference, acc_score, mem_size))
