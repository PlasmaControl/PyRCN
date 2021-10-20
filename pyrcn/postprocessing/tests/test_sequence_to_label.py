"""
Testing for Echo State Network module (pyrcn.echo_state_network)
"""
import numpy as np
from pyrcn.postprocessing import SequenceToLabelClassifier

import pytest


def test_sequence_to_label_classifier():
    print('\ntest_sequence_to_label_classifier():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = SequenceToLabelClassifier().fit(X, y=None).predict(X)
    np.testing.assert_equal(y, 0)


def test_sequence_to_label_classifier_median():
    print('\ntest_sequence_to_label_classifier_median():')
    clf = SequenceToLabelClassifier(output_strategy="median")
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = clf.fit(X, y=None).predict(X)
    np.testing.assert_equal(y, np.argmax(np.median(X, axis=0)))


def test_sequence_to_label_classifier_lv():
    print('\ntest_sequence_to_label_classifier_lv():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    clf = SequenceToLabelClassifier(output_strategy="last_value").fit(X, y=None)
    y = clf.predict(X)
    np.testing.assert_equal(y, np.argmax(X[-1, :]))


def test_sequence_to_label_classifier_predict_proba():
    print('\ntest_sequence_to_label_classifier_predict_proba():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    clf = SequenceToLabelClassifier(output_strategy="last_value").fit(X, y=None)
    y = clf.predict_proba(X)
    np.testing.assert_equal(y, X[-1, :])


if __name__ == "__main__":
    test_sequence_to_label_classifier()
    test_sequence_to_label_classifier_median()
    test_sequence_to_label_classifier_lv()
    test_sequence_to_label_classifier_predict_proba()
