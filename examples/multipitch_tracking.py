import numpy as np
import os
import csv
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

import librosa
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor

from pyrcn.util import FeatureExtractor
from pyrcn.echo_state_network import SeqToSeqESNClassifier
from pyrcn.datasets import fetch_maps_piano_dataset
from pyrcn.metrics import accuracy_score
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
                          'input_activation': 'identity',
                          'k_in': 5,
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': 1.0,
                          'bi_directional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42}

step1_esn_params = {'input_scaling': np.linspace(0.1, 1.0, 10),
                    'spectral_radius': np.linspace(0.0, 1.5, 16)}

step2_esn_params = {'leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}

kwargs = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs)]

base_esn = SeqToSeqESNClassifier(**initially_fixed_params)

sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)

dump(sequential_search, "sequential_search.joblib")

# ## Train a model
# 
# This might require a huge amount of time and memory. We have deactivated it for now, because we already offer pre-trained models.
# 
# The pipeline is clear: 
# 
# - Extract features and labels
# - Pass features and labels through ESN
# - Compute output weights and clear no more required helper variables.
# 
# Notice that you can pass the option "update_output_weights=False" to the call to partial_fit. If this option is passed, no output weights are computed after passing the sequence through the network. This is computationally more efficient.
# 
# After passing the entire sequence through the ESN, we need to call finalize() in order to compute the update weights.
# 
# By default, "update_output_weights=False" and it is not necessary to call finalize().

# In[5]:


should_train = False

if should_train:
    train_ids, test_ids = get_dataset()
    for fid in train_ids:
        X, y_true = extract_features(os.path.join(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\MusicNet", 'train_data'), file_name=fid)
        esn.partial_fit(X=X, y=y_true, update_output_weights=False)
    esn.finalize()


# ## Validate the ESN model
# 
# This might require a huge amount of time if a lot of audio files need to be analyzed. 
# 
# The pipeline is similar as before: 
# 
# - Extract features and labels
# - Pass features through ESN and compute outputs
# - Binarize the outputs by simply thresholding
# - Save labels, outputs, binarized outputs for further processing, e.g. visualization or evaluation.

# In[6]:


train_ids, test_ids = get_dataset()
X_train = []
Y_pred_train = []
Y_pred_bin_train = []
Y_true_train = []
for fid in train_ids:
    X, y_true = extract_features(os.path.join(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\MusicNet", 'train_data'), file_name=fid)
    X_train.append(X)
    y_pred = esn.predict(X=X)
    Y_true_train.append(y_true)
    Y_pred_train.append(y_pred)
    Y_pred_bin_train.append(np.asarray(y_pred > 0.3, dtype=int))

X_test = []
Y_pred_test = []
Y_true_test = []
for fid in test_ids:
    X, y_true = extract_features(os.path.join(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\MusicNet", 'test_data'), file_name=fid)
    X_test.append(X)
    y_pred = esn.predict(X=X)
    Y_true_test.append(y_true)
    Y_pred_test.append(np.asarray(y_pred > 0.3, dtype=int))


# ## Visualization
# 
# We visualize inputs, labels, raw outputs and thresholded outputs in four plots, respectively. 

# In[7]:


fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0, 0].imshow(Y_true_train[0].T, origin='lower', vmin=0, vmax=1, cmap='gray', aspect='auto')
axs[0, 0].set_xlabel('n')
axs[0, 0].set_ylabel('Y_true[n]')
axs[0, 0].grid()
axs[0, 0].set_ylim([21, 108])

axs[0, 1].imshow(X_train[0].T, origin='lower', vmin=0, vmax=1, aspect='auto')
axs[0, 1].set_xlabel('n')
axs[0, 1].set_ylabel('X[n]')
axs[0, 1].grid()

im = axs[1, 0].imshow(Y_pred_train[0].T, origin='lower', aspect='auto')
axs[1, 0].set_xlabel('n')
axs[1, 0].set_ylabel('Y_pred[n]')
axs[1, 0].grid()
axs[1, 0].set_ylim([21, 108])
# fig.colorbar(im, ax=axs[1, 0])

axs[1, 1].imshow(Y_pred_bin_train[0].T, origin='lower', vmin=0, vmax=1, cmap='gray', aspect='auto')
axs[1, 1].set_xlabel('n')
axs[1, 1].set_ylabel('Y_pred_bin[n]')
axs[1, 1].grid()
axs[1, 1].set_ylim([21, 108])

plt.tight_layout()

