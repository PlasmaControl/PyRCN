"""
Sequence-to-sequence model
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _deprecate_positional_args

from pyrcn.base import InputToNode, NodeToNode, FeedbackNodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNRegressor, ESNClassifier, FeedbackESNRegressor
import numpy as np
from joblib import Parallel, delayed


class SeqToSeqESNRegressor(ESNRegressor):
    """
    A trainer to simplify sequence-to-sequence regression.
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node=InputToNode(),
                 node_to_node=NodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 **kwargs):
        super().__init__(input_to_node=input_to_node, 
                         node_to_node=node_to_node,
                         regressor=regressor,
                         chunk_size=chunk_size,
                         **kwargs)

    def fit(self, X, y, n_jobs=None):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        if n_jobs is None:
            for X_train, y_train in zip(X[:-1], y[:-1]):
                super().partial_fit(X_train, y_train, postpone_inverse=True)
            super().partial_fit(X[-1], y[-1], postpone_inverse=False)
        else:
            regs = Parallel(n_jobs=n_jobs, verbose=2)(delayed(self._parallel_fit)(X=X_train, y=y_train)
                                                      for X_train, y_train in zip(np.array_split(X[:-1], n_jobs), np.array_split(y[:-1], n_jobs)))
            final_reg = sum(regs).partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
            self.input_to_node = final_reg.input_to_node
            self.node_to_node = final_reg.node_to_node
            self.regressor = final_reg.regressor
        self._base_estimator = clone(self._base_estimator)
        return self

    def partial_fit(self, X, y):
        if n_jobs is None:
            for X_train, y_train in zip(X, y):
                super().partial_fit(X_train, y_train, postpone_inverse=True)
        else:
            regs = Parallel(n_jobs=n_jobs, verbose=2)(delayed(self._parallel_fit)(X=X_train, y=y_train)
                                                      for X_train, y_train in zip(X, y))
            final_reg = sum(regs)
            regs = None
            self.input_to_node = final_reg.input_to_node
            self.node_to_node = final_reg.node_to_node
            self.regressor = final_reg.regressor
        self._base_estimator = clone(self._base_estimator)
        return self

    def _parallel_fit(self, X, y):
        clf = clone(self._base_estimator)
        for X_train, y_train in zip(X, y):
            clf.partial_fit(X_train, y_train, postpone_inverse=True)
        return clf

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = super().predict(X_test)
        return y_pred


class SeqToSeqESNClassifier(ESNClassifier):
    """
    A trainer to simplify sequence-to-sequence classification.
    """
    @_deprecate_positional_args
    def __init__(self, *,
                 input_to_node=InputToNode(),
                 node_to_node=NodeToNode(),
                 regressor=IncrementalRegression(alpha=.0001),
                 chunk_size=None,
                 **kwargs):
        super().__init__(input_to_node=input_to_node, 
                         node_to_node=node_to_node,
                         regressor=regressor,
                         chunk_size=chunk_size,
                         **kwargs)

    def partial_fit(self, X, y, classes=None, n_jobs=None, transformer_weights=None, postpone_inverse=False):
        if n_jobs is None:
            for X_train, y_train in zip(X, y):
                super().partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
        else:
            clfs = Parallel(n_jobs=n_jobs, verbose=2)(delayed(self._parallel_fit)(X=X_train, y=y_train, classes=classes)
                                                      for X_train, y_train in zip(X, y))
            final_clf = sum(clfs)
            self.input_to_node = final_clf.input_to_node
            self.node_to_node = final_clf.node_to_node
            self.regressor = final_clf.regressor
            self._encoder = final_clf._encoder
            self._base_estimator = clone(self._base_estimator)
        return self

    def fit(self, X, y, n_jobs=None, transformer_weights=None):
        lab = LabelBinarizer().fit(np.concatenate(y))
        if lab.y_type_.startswith('multilabel'):
            classes = np.zeros(shape=(1, y[0].shape[1]))
        else:
            classes = lab.classes_
        if n_jobs is None:
            for X_train, y_train in zip(X[:-1], y[:-1]):
                super().partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
            super().partial_fit(X[-1], np.repeat(y[-1], X[-1].shape[0]), classes=classes, postpone_inverse=False)
        else:
            clfs = Parallel(n_jobs=n_jobs, verbose=2)(delayed(self._parallel_fit)(X=X_train, y=y_train, classes=classes)
                                                      for X_train, y_train in zip(np.array_split(X[:-1], n_jobs), np.array_split(y[:-1], n_jobs)))
            final_clf = sum(clfs).partial_fit(X=X[-1], y=np.repeat(y[-1], X[-1].shape[0]), postpone_inverse=False)
            self.input_to_node = final_clf.input_to_node
            self.node_to_node = final_clf.node_to_node
            self.regressor = final_clf.regressor
            self._encoder = final_clf._encoder
            self._base_estimator = clone(self._base_estimator)
        return self

    def _parallel_fit(self, X, y, classes):
        clf = clone(self._base_estimator)
        for X_train, y_train in zip(X, y):
            clf.partial_fit(X_train, y_train, classes=classes, postpone_inverse=True)
        return clf

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = super().predict(X_test)
        return y_pred

    def predict_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = super().predict_proba(X_test)
        return y_pred

    def predict_log_proba(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = super().predict_log_proba(X_test)
        return y_pred


class SeqToSeqESNFeedbackRegressor(FeedbackESNRegressor):
    """
    A trainer to simplify sequence-to-sequence regression with feedback.
    """
    @_deprecate_positional_args
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
        for X_train, y_train in zip(X[:-1], y[:-1]):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        super().partial_fit(X=X[-1], y=y[-1], postpone_inverse=False)
        return self

    def partial_fit(self, X, y):
        for X_train, y_train in zip(X, y):
            super().partial_fit(X_train, y_train, postpone_inverse=True)
        return self

    def predict(self, X):
        y_pred = np.empty(shape=(X.shape[0], ), dtype=object)
        for k, X_test in enumerate(X):
            y_pred[k] = super().predict(X_test)
        return y_pred
