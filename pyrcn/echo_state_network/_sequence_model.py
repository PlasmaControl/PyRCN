"""
Sequence-to-sequence model
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from sklearn.preprocessing import LabelBinarizer
from pyrcn.base import InputToNode, NodeToNode, FeedbackNodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNRegressor, ESNClassifier, FeedbackESNRegressor
import numpy as np
from tqdm import tqdm


class SequenceToSequenceRegressor(ESNRegressor):
    """
    A trainer to simplify sequence-to-sequence regression.
    """
    def __init__(self,
                 input_to_node=InputToNode(),
                 node_to_node=NodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 random_state=None,
                 n_jobs=None):
        super().__init__(input_to_node=input_to_node, 
                         node_to_node=node_to_node,
                         regressor=regressor,
                         chunk_size=chunk_size,
                         random_state=random_state)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        for X_train, y_train in tqdm(zip(X[:-1], y[:-1])):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        super().partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def partial_fit(self, X, y):
        for X_train, y_train in tqdm(zip(X, y)):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in tqdm(enumerate(X)):
            y_pred[k] = super().predict(X_test)
        return y_pred


class SequenceToSequenceClassifier(ESNClassifier):
    """
    A trainer to simplify sequence-to-sequence classification.
    """
    def __init__(self,
                 input_to_node=InputToNode(),
                 node_to_node=NodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 random_state=None,
                 n_jobs=None):
        super().__init__(input_to_node=input_to_node, 
                         node_to_node=node_to_node,
                         regressor=regressor,
                         chunk_size=chunk_size,
                         random_state=random_state)
        self.n_jobs = n_jobs

    def partial_fit(self, X, y, classes=None, n_jobs=None, transformer_weights=None, postpone_inverse=False):
        for X_train, y_train in tqdm(zip(X, y)):
            super().partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
        return self

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        classes = LabelBinarizer().fit(np.concatenate(y)).classes_
        for X_train, y_train in tqdm(zip(X[:-1], y[:-1])):
            super().partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
        super().partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in tqdm(enumerate(X)):
            y_pred[k] = super().predict(X_test)
        return y_pred

    def predict_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in tqdm(enumerate(X)):
            y_pred[k] = super().predict_proba(X_test)
        return y_pred

    def predict_log_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in tqdm(enumerate(X)):
            y_pred[k] = super().predict_log_proba(X_test)
        return y_pred


class SequenceToSequenceFeedbackRegressor(FeedbackESNRegressor):
    """
    A trainer to simplify sequence-to-sequence regression with feedback.
    """
    def __init__(self,
                 input_to_node=InputToNode(),
                 node_to_node=FeedbackNodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 random_state=None,
                 n_jobs=None):
        super().__init__(input_to_node=input_to_node, 
                         node_to_node=node_to_node,
                         regressor=regressor,
                         chunk_size=chunk_size,
                         random_state=random_state)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        for X_train, y_train in tqdm(zip(X[:-1], y[:-1])):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        super().partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def partial_fit(self, X, y):
        for X_train, y_train in tqdm(zip(X, y)):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in tqdm(enumerate(X)):
            y_pred[k] = super().predict(X_test)
        return y_pred
