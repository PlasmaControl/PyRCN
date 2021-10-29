"""MAPS piano dataset.
The original database is available at

    https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/

The license is restricted, and one needs to register and download the dataset.

"""
import sys
if sys.version_info >= (3, 8):
    from typing import Union, Literal
else:
    from typing_extensions import Literal
    from typing import Union

from pathlib import Path
from os.path import dirname, exists, join
from os import makedirs, remove
import numpy as np
import pandas as pd
import joblib

from sklearn.base import TransformerMixin
from sklearn.datasets import get_data_home
from sklearn.datasets._base import RemoteFileMetadata, _pkl_filepath
from sklearn.utils.validation import _deprecate_positional_args


def _combine_events(events: Union[list, np.ndarray], 
                    delta : float, 
                    combine: Literal['mean', 'left', 'right']) -> np.ndarray:
    """
    Combine all events within a certain range.
    
    Parameters
    ----------
    events : Union[list, ndarray]
        Events to be combined.
    delta : float
        Combination delta. All events within this `delta` are combined.
    combine : {'mean', 'left', 'right'}
        How to combine two adjacent events:
            - 'mean': replace by the mean of the two events
            - 'left': replace by the left of the two events
            - 'right': replace by the right of the two events
    
    Returns
    -------
    events : ndarray
        Combined events.
    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # convert to numpy array or create a copy if needed
    events = np.array(events, dtype=np.float)
    # can handle only 1D events
    if events.ndim > 1:
        raise ValueError('only 1-dimensional events supported.')
    # set start position
    idx = 0
    # get first event
    left = events[idx]
    # iterate over all remaining events
    for right in events[1:]:
        if right - left <= delta:
            # combine the two events
            if combine == 'mean':
                left = events[idx] = 0.5 * (right + left)
            elif combine == 'left':
                left = events[idx] = left
            elif combine == 'right':
                left = events[idx] = right
            else:
                raise ValueError("don't know how to combine two events with "
                                 "%s" % combine)
        else:
            # move forward
            idx += 1
            left = events[idx] = right
    # return the combined events
    return events[:idx + 1]


def _quantize_notes(notes: np.ndarray, fps: float, 
                    length: Union[int, np.integer] = None, 
                    num_pitches: Union[int, np.integer] = None, 
                    velocity: float = None) -> np.ndarray:
    """
    Quantize the notes with the given resolution.
    Create a sparse 2D array with rows corresponding to points in time
    (according to `fps` and `length`), and columns to note pitches (according
    to `num_pitches`). The values of the array correspond to the velocity of a
    sounding note at a given point in time (based on the note pitch, onset,
    duration and velocity). If no values for `length` and `num_pitches` are
    given, they are inferred from `notes`.
    Parameters
    ----------
    notes : np.ndarray
        Notes to be quantized. Expected columns:
        'note_time' 'note_number' ['duration' ['velocity']]
        If `notes` contains no 'duration' column, only the frame of the
        onset will be set. If `notes` has no velocity column, a velocity
        of 1 is assumed.
    fps : float
        Quantize with `fps` frames per second.
    length : Union[int, np.integer]
        Length of the returned array. If 'None', the length will be set
        according to the latest sounding note.
    num_pitches : Union[int, np.integer]
        Number of pitches of the returned array. If 'None', the number of
        pitches will be based on the highest pitch in the `notes` array.
    velocity : float
        Use this velocity for all quantized notes. If set, the last column of
        `notes` (if present) will be ignored.

    Returns
    -------
    np.ndarray
        Quantized notes.
    """
    # convert to numpy array or create a copy if needed
    notes = np.array(np.array(notes).T, dtype=np.float, ndmin=2).T
    # check supported dims and shapes
    if notes.ndim != 2:
        raise ValueError('only 2-dimensional notes supported.')
    if notes.shape[1] < 2:
        raise ValueError('notes must have at least 2 columns.')
    # split the notes into columns
    note_onsets = notes[:, 0]
    note_numbers = notes[:, 1].astype(np.int)
    note_offsets = np.copy(note_onsets)
    if notes.shape[1] > 2:
        note_offsets += notes[:, 2]
    if notes.shape[1] > 3 and velocity is None:
        note_velocities = notes[:, 3]
    else:
        velocity = velocity or 1
        note_velocities = np.ones(len(notes)) * velocity
    # determine length and width of quantized array
    if length is None:
        # set the length to be long enough to cover all notes
        length = int(round(np.max(note_offsets) * float(fps))) + 1
    if num_pitches is None:
        num_pitches = int(np.max(note_numbers)) + 1
    # init array
    quantized = np.zeros((length, num_pitches))
    # quantize onsets and offsets
    note_onsets = np.round((note_onsets * fps)).astype(np.int)
    note_offsets = np.round((note_offsets * fps)).astype(np.int) + 1
    # iterate over all notes
    for n, note in enumerate(notes):
        # use only the notes which fit in the array and note number >= 0
        if num_pitches > note_numbers[n] >= 0:
            quantized[note_onsets[n]:note_offsets[n], note_numbers[n]] = \
                note_velocities[n]
    # return quantized array
    return quantized


@_deprecate_positional_args
def fetch_maps_piano_dataset(*, data_origin: str = None, data_home: str = None, 
                             preprocessor: TransformerMixin = None, 
                             force_preprocessing: bool = False, 
                             label_type: Literal["pitch", "onset", "offset"]) -> (np.ndarray, 
                                                                                  np.ndarray, 
                                                                                  np.ndarray, 
                                                                                  np.ndarray):
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
    preprocessor : sklearn.TransformerMixin, default=None,
        Estimator for preprocessing the dataset (create features and targets from 
        audio and label files).
    force_preprocessing: bool, default=False
        Force preprocessing (label computation and feature extraction)
    label_type : Literal["pitch", "onset", "offset"], default="pitch",
        Type of labels to return. Possible are pitch labels or onset and offset labels
        for each pitch.

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
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


def _get_pitch_labels(X: np.ndarray, df_label: pd.DataFrame) -> np.ndarray:
    """
    Get the pitch labels of a recording

    Parameters
    ----------
    X: np.ndarray
        Feature matrix to know the shape of the input data.
    df_label, pandas.DataFrame
        Pandas dataframe that contains the annotations.
    Returns
    -------
    y : np.ndarray
    """
    df_label["Duration"] = df_label["OffsetTime"] - df_label["OnsetTime"]
    notes = df_label[["OnsetTime", "MidiPitch", "Duration"]].to_numpy()
    pitch_labels = _quantize_notes(notes, fps=100., num_pitches=128, length=X.shape[0])
    y = np.zeros(shape=(pitch_labels.shape[0], pitch_labels.shape[1] + 1))
    y[:, 1:] = pitch_labels
    y[np.argwhere(pitch_labels.sum(axis=1) == 0), 0] = 1
    return y


def get_note_events(utterance: str):
    """
    Obtain note events from a stored file.

    Parameters
    ----------
    utterance : str
        file name of an utterance

    Returns
    -------
    notes : ndarray 
        MIDI pitches, start and stop in seconds
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

def get_onset_events(utterance: str) -> np.ndarray:
    """
    Obtain onset events from a stored file.

    Parameters
    ----------
    utterance : str
        file name of an utterance

    Returns
    -------
    onset_events : ndarray 
        onset times in seconds
    """
    onset_labels = []
    with open(utterance, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for label in reader:
            start_time = float(label['OnsetTime'])
            note = int(label['MidiPitch'])
            onset_labels.append([start_time, note])
    return _combine_events(list(dict.fromkeys(onset_labels)), 0.03, combine='mean')

def get_offset_events(utterance: str) -> np.ndarray:
    """
    Obtain onset events from a stored file.

    Parameters
    ----------
    utterance : str
        file name of an utterance

    Returns
    -------
    offset_events : ndarray 
        offset times in seconds
    """
    offset_labels = []
    with open(utterance, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for label in reader:
            start_time = float(label['OnsetTime'])
            note = int(label['MidiPitch'])
            offset_labels.append([start_time, note])
    return _combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')
