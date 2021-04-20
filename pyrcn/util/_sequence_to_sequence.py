"""
Sequence-to-sequence model
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class SequenceToSequenceRegressor(BaseEstimator, RegressorMixin, MultiOutputMixin):
    """
    A trainer to simplify sequence-to-sequence regression.
    """
    def __init__(self, estimator, estimator_params=None, n_jobs=None):
        self.estimator = estimator.set_params(**estimator_params)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        for X_train, y_train in zip(X[:-1], y[:-1]):
            self.estimator.partial_fit(X_train, y_train, postpone_inverse=True)
        self.estimator.partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def partial_fit(self, X, y):
        for X_train, y_train in zip(X[:-1], y[:-1]):
            self.estimator.partial_fit(X_train, y_train, postpone_inverse=True)
        self.estimator.partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = self.estimator.predict(X_test)
        return y_pred


class SequenceToSequenceClassifier(SequenceToSequenceRegressor, ClassifierMixin):
    """
    A trainer to simplify sequence-to-sequence classification.
    """
    def __init__(self, estimator, estimator_params=None, n_jobs=None):
        super().__init__(estimator=estimator, estimator_params=estimator_params, n_jobs=n_jobs)
        self._encoder = None
    def fit(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        self._encoder = LabelBinarizer().fit(np.hstack(y))
        return self.partial_fit(X=X, y=y, classes=self._encoder.classes_)

    def partial_fit(self, X, y, classes=None):
        if self._encoder is None:
            self._encoder = LabelBinarizer().fit(classes)
        for X_train, y_train in zip(X[:-1], y[:-1]):
            self.estimator.partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
        self.estimator.partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = self.estimator.predict(X_test)
        return y_pred

    def predict_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = self.estimator.predict_proba(X_test)
        return y_pred

    def predict_log_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = self.estimator.predict_log_proba(X_test)
        return y_pred
