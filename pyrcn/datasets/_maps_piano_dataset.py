"""MAPS piano dataset.
The original database is available at

    https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/

The license is restricted, and one needs to register and download the dataset.

"""
from pathlib import Path
from os.path import dirname, exists, join
from os import makedirs, remove

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import get_data_home
from sklearn.datasets._base import RemoteFileMetadata, _pkl_filepath
from sklearn.utils import _deprecate_positional_args
from madmom.utils import quantize_notes


@_deprecate_positional_args
def fetch_maps_piano_dataset(*, data_origin=None, data_home=None, preprocessor=None, 
                             force_preprocessing=False, label_type="pitch"):
    """
    Load the MAPS piano dataset from Telecom Paris (classification)

    =================   =====================
    Classes                              TODO
    Samples total                        TODO
    Dimensionality                       TODO
    Features                             TODO
    =================   =====================

    Parameters
    ----------
    data_origin : str, default=None
        Specify where the original dataset can be found. By default,
        all pyrcn data is stored in '~/pyrcn_data' and all scikit-learn data in
       '~/scikit_learn_data' subfolders.

    data_home : str, default=None
        Specify another download and cache folder fo the datasets. By default,
        all pyrcn data is stored in '~/pyrcn_data' and all scikit-learn data in
       '~/scikit_learn_data' subfolders.

    preprocessor : default=None,
        Estimator for preprocessing the dataset (create features and targets from 
        audio and label files).

    label_type : str, default="pitch",
        Type of labels to return. Possible are pitch labels or onset and offset labels
        for each pitch.

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = _pkl_filepath(data_home, 'maps.pkz')
    if not exists(filepath) or force_preprocessing:

        print('preprocessing MAPS dataset from %s to %s'
              % (data_origin, data_home))
        train_files = np.loadtxt(join(data_origin, Path("mapsSplits/sigtia-conf3-splits/train")), dtype=object)
        test_files = np.loadtxt(join(data_origin, Path("mapsSplits/sigtia-conf3-splits/test")), dtype=object)

        X_train = np.empty(shape=(len(train_files),), dtype=object)
        X_test = np.empty(shape=(len(test_files),), dtype=object)
        y_train = np.empty(shape=(len(train_files),), dtype=object)
        y_test = np.empty(shape=(len(test_files),), dtype=object)

        for k, f in enumerate(train_files):
            X_train[k] = preprocessor.transform(join(data_origin, Path(f + ".wav")))
            y_train[k] = pd.read_csv(join(data_origin, Path(f + ".txt")), sep="\t")

        for k, f in enumerate(test_files):
            X_test[k] = preprocessor.transform(join(data_origin, Path(f + ".wav")))
            y_test[k] = pd.read_csv(join(data_origin, Path(f + ".txt")), sep="\t")

        joblib.dump([X_train, X_test, y_train, y_test], filepath, compress=6)
    else:
        X_train, X_test, y_train, y_test = joblib.load(filepath)

    x_shape_zero = np.unique([X.shape[0] for X in X_train] + [X.shape[0] for X in X_test])
    x_shape_one = np.unique([X.shape[1] for X in X_train] + [X.shape[1] for X in X_test])
    if len(x_shape_zero) == 1 and len(x_shape_one) > 1:
        for k in range(len(X_train)):
            X_train[k] = X_train[k].T
        for k in range(len(X_test)):
            X_test[k] = X_test[k].T 
    elif len(x_shape_zero) > 1 and len(x_shape_one) == 1:
        pass
    else:
        raise TypeError("Invalid dataformat. Expected at least one equal dimension of all sequences.")

    for k in range(len(X_train)):
        if label_type == "pitch":
            y_train[k] = _get_pitch_labels(X_train[k], y_train[k])
        elif label_type == "onset":
            y_train[k] = _get_onset_labels(X_train[k], y_train[k])
        elif label_type == "onset":
            y_train[k] = _get_offset_labels(X_train[k], y_train[k])
        else:
            raise TypeError("Invalid label type.")

    for k in range(len(X_test)):
        if label_type == "pitch":
            y_test[k] = _get_pitch_labels(X_test[k], y_test[k])
        elif label_type == "onset":
            y_test[k] = _get_onset_labels(X_test[k], y_test[k])
        elif label_type == "onset":
            y_test[k] = _get_offset_labels(X_test[k], y_test[k])
        else:
            raise TypeError("Invalid label type.")

    return X_train, X_test, y_train, y_test


def _get_pitch_labels(X, df_label):
    df_label["Duration"] = df_label["OffsetTime"] - df_label["OnsetTime"]
    notes = df_label[["OnsetTime", "MidiPitch", "Duration"]].to_numpy()
    pitch_labels = quantize_notes(notes, fps=100., num_pitches=128, length=X.shape[0])
    y = np.zeros(shape=(pitch_labels.shape[0], pitch_labels.shape[1] + 1))
    y[:, 1:] = pitch_labels
    y[np.argwhere(pitch_labels.sum(axis=1) == 0), 0] = 1
    return y


def get_note_events(self, utterance):
    """get_onset_events(utterance)
    Given a file name of a specific utterance, e.g.
        ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
    returns the note events with start and stop in seconds

    Returns:
    start, note, duration
    """
    notes = []
    with open(utterance, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for label in reader:
            start_time = float(label['OnsetTime'])
            end_time = float(label['OffsetTime'])
            note = int(label['MidiPitch'])
            notes.append([start_time, note, end_time - start_time])
    return np.array(notes)

def get_onset_events(self, utterance):
    """get_onset_events(utterance)
    Given a file name of a specific utterance, e.g.
        ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
    returns the instrument events with start and stop in seconds

    If fs is None, returns instrument events with start and stop in samples

    Returns:
    start, note, duration
"""
    onset_labels = []
    with open(utterance, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for label in reader:
            start_time = float(label['OnsetTime'])
            note = int(label['MidiPitch'])
            onset_labels.append([start_time, note])
    return madmom.utils.combine_events(list(dict.fromkeys(onset_labels)), 0.03, combine='mean')

def get_offset_events(self, utterance):
    """get_offset_events(utterance)
    Given a file name of a specific utterance, e.g.
        ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
    returns the instrument events with start and stop in seconds

    If fs is None, returns instrument events with start and stop in samples

    Returns:
    start, note, duration
    """
    offset_labels = []
    with open(utterance, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for label in reader:
            start_time = float(label['OnsetTime'])
            note = int(label['MidiPitch'])
            offset_labels.append([start_time, note])
    return madmom.utils.combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')
