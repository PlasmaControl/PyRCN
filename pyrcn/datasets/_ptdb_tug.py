"""PTDB-TUG: Pitch Tracking Database from Graz University of Technology
The original database is available at

    https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html


"""
import sys
if sys.version_info >= (3, 8):
    from typing import Union, Literal
else:
    from typing_extensions import Literal
    from typing import Union

from pathlib import Path
from os.path import dirname, exists, join
from os import makedirs, remove, walk
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import get_data_home
from sklearn.datasets._base import RemoteFileMetadata, _pkl_filepath
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.base import TransformerMixin


@_deprecate_positional_args
def fetch_ptdb_tug_dataset(*, data_origin: str = None, 
                           data_home: str = None, 
                           preprocessor: TransformerMixin = None, 
                           augment : Union[int, np.integer] = 0, 
                           force_preprocessing: bool = False) -> (np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray):
    """
    Load the PTDB-TUG: Pitch Tracking Database from Graz University of Technology
    (classification and regression)

    =================   =====================
    Outputs                                 2
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
    augment : Union[int, np.integer], default = 0
        Semitone range used for data augmentation
    force_preprocessing: bool, default=False
        Force preprocessing (label computation and feature extraction)

    Returns
    -------
    (X, y) : tuple
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = _pkl_filepath(data_home, 'ptdb_tug.pkz')
    if not exists(filepath) or force_preprocessing:
        print('preprocessing PTDB-TUG database from %s to %s'
              % (data_origin, data_home))
        all_training_files = []
        all_test_files = []
        for root, dirs, files in walk(data_origin):
            for f in files:
                if f.endswith(".wav") and f.startswith("mic") and not re.search('\_[0-9]\.wav$', f) and not  re.search('\_\-[0-9]\.wav$', f):
                    if "F09" in f or "F10" in f or "M09" in f or "M10" in f:
                        all_test_files.append(join(root, f))
                    else:
                        all_training_files.append(join(root, f))

        if augment > 0:
            augment = list(range(-augment, augment + 1))
            augment.remove(0)
        else:
            augment = [0]
        if len(augment) == 1:
            X_train = np.empty(shape=(len(all_training_files),), dtype=object)
            y_train = np.empty(shape=(len(all_training_files),), dtype=object)
        else:
            X_train = np.empty(shape=((1 + len(augment)) * len(all_training_files),), dtype=object)
            y_train = np.empty(shape=((1 + len(augment)) * len(all_training_files),), dtype=object)
        X_test = np.empty(shape=(len(all_test_files),), dtype=object)
        y_test = np.empty(shape=(len(all_test_files),), dtype=object)

        if len(augment) > 1:
            for k, f in enumerate(all_training_files):
                X_train[k] = preprocessor.transform(f)
                y_train[k] = pd.read_csv(f.replace("MIC", "REF").replace("mic", "ref").replace(".wav", ".f0"), sep=" ", header=None)
            for m, st in enumerate(augment):
                for k, f in enumerate(all_training_files):
                    X_train[k + int((m+1) * len(all_training_files))] = preprocessor.transform(f.replace(".wav", "_" + str(st) + ".wav"))
                    df = pd.read_csv(f.replace("MIC", "REF").replace("mic", "ref").replace(".wav", ".f0"), sep=" ", header=None)
                    df[[0]] = df[[0]] * 2**(st/12)
                    y_train[k + int((m+1) * len(all_training_files))] = df
        else:
            for k, f in enumerate(all_training_files):
                X_train[k] = preprocessor.transform(f)
                y_train[k] = pd.read_csv(f.replace("MIC", "REF").replace("mic", "ref").replace(".wav", ".f0"), sep=" ", header=None)
        for k, f in enumerate(all_test_files):
            X_test[k] = preprocessor.transform(f)
            y_test[k] = pd.read_csv(f.replace("MIC", "REF").replace("mic", "ref").replace(".wav", ".f0"), sep=" ", header=None)
        joblib.dump([X_train, X_test, y_train, y_test], filepath, compress=6)
    else:
        X_train, X_test, y_train, y_test = joblib.load(filepath)

    x_shape_zero = np.unique([x.shape[0] for x in X_train] + [x.shape[0] for x in X_test])
    x_shape_one = np.unique([x.shape[1] for x in X_train] + [x.shape[1] for x in X_test])
    if len(x_shape_zero) == 1 and len(x_shape_one) > 1:
        for k in range(len(X_train)):
            X_train[k] = X_train[k].T
            y_train[k] = _get_labels(X_train[k], y_train[k])
        for k in range(len(X_test)):
            X_test[k] = X_test[k].T
            y_test[k] = _get_labels(X_test[k], y_test[k])
    elif len(x_shape_zero) > 1 and len(x_shape_one) == 1:
        for k in range(len(X_train)):
            y_train[k] = _get_labels(X_train[k], y_train[k])
        for k in range(len(X_test)):
            y_test[k] = _get_labels(X_test[k], y_test[k])
    else:
        raise TypeError("Invalid dataformat. Expected at least one equal dimension of all sequences.")

    return X_train, X_test, y_train, y_test


def _get_labels(X: np.ndarray, df_label: pd.DataFrame) -> np.ndarray:
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
    labels = df_label[[0, 1]].to_numpy()
    y = np.zeros(shape=(X.shape[0], 2))
    if X.shape[0] == labels.shape[0]:
        y[:, :] = labels
        return y
    elif X.shape[0] == labels.shape[0] + 2 or X.shape[0] == labels.shape[0] + 1:
        y[1:1+len(labels), :] = labels
        return y
    elif X.shape[0] == 2*labels.shape[0]:
        y[:, 0] = np.interp(np.arange(len(labels), step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 0])
        y[:, 1] = np.interp(np.arange(len(labels), step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 1])
        return y
    elif X.shape[0] == 2*labels.shape[0] - 1:
        y[1:1+2*len(labels)-1, 0] = np.interp(np.arange(len(labels) - 1, step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 0])
        y[1:1+2*len(labels)-1, 1] = np.interp(np.arange(len(labels) - 1, step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 1])
        return y
    elif X.shape[0] == 2*labels.shape[0] + 1:
        y[1:1+2*len(labels)+1, 0] = np.interp(np.arange(len(labels), step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 0])
        y[1:1+2*len(labels)+1, 1] = np.interp(np.arange(len(labels), step=0.5), xp=np.arange(len(labels), step=1), fp=labels[:, 1])
        return y
    else:
        print("Test")
