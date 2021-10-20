import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


def winner_takes_all(X, weights=None):
    return np.argmax(np.sum(X, axis=0))


def median(X):
    return np.argmax(np.median(X, axis=0))


class SequenceToLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    TODO: DOCSTRING
    """
    def __init__(self, output_strategy="winner_takes_all"):
        self._output_strategy = output_strategy

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        if self._output_strategy == "winner_takes_all":
            X = np.sum(X, axis=0)
        elif self._output_strategy == "median":
            X = np.median(X, axis=0)
        elif self._output_strategy == "last_value":
            X = X[-1, :]
        return np.atleast_1d(np.argmax(X))

    def predict_proba(self, X, y=None):
        if self._output_strategy == "winner_takes_all":
            X = np.sum(X, axis=0)
        elif self._output_strategy == "median":
            X = np.median(X, axis=0)
        elif self._output_strategy == "last_value":
            X = X[-1, :]
        return X