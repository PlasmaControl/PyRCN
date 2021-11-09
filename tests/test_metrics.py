"""Testing for Metrics module."""

import pyrcn.metrics
import sklearn.metrics
from sklearn.datasets import make_classification, make_multilabel_classification
import numpy as np
import pytest


rng_true = np.random.RandomState(42)
rng_pred = np.random.RandomState(1234)
y_true_bin = np.empty(shape=(10,), dtype=object)
y_pred_bin = np.empty(shape=(10,), dtype=object)
sample_weight = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_bin[k] = make_classification(n_samples=10 * (k + 1), n_features=20,
                                           random_state=rng_true)
    _, y_pred_bin[k] = make_classification(n_samples=10 * (k + 1), n_features=20,
                                           random_state=rng_pred)
    sample_weight[k] = np.ones_like(y_true_bin[k])

y_true_mlb = np.empty(shape=(10,), dtype=object)
y_pred_mlb = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_mlb[k] = make_multilabel_classification(n_samples=10 * (k + 1),
                                                      n_features=20,
                                                      random_state=rng_true)
    _, y_pred_mlb[k] = make_multilabel_classification(n_samples=10 * (k + 1),
                                                      n_features=20,
                                                      random_state=rng_pred)


def test_accuracy_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.accuracy_score(y_true=y_true_bin, y_pred=y_pred_bin), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_bin, y_pred=y_pred_bin,
                                     sample_weight=sample_weight),
        sklearn.metrics.accuracy_score(y_true=np.concatenate(y_true_bin),
                                       y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_mlb, y_pred=y_true_mlb), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.accuracy_score(y_true=y_true_mlb, y_pred=y_pred_mlb), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.accuracy_score(y_true=np.concatenate(y_true_mlb),
                                       y_pred=np.concatenate(y_pred_mlb)))
    with pytest.raises(TypeError):
        pyrcn.metrics.accuracy_score(y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_confusion_matrix() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.confusion_matrix(y_true=y_true_bin, y_pred=y_true_bin).shape,
        (2, 2))
    np.testing.assert_equal(
        np.where(pyrcn.metrics.confusion_matrix(y_true=y_true_bin, y_pred=y_true_bin,
                                                sample_weight=sample_weight)),
        np.where(np.eye(2)))
    np.testing.assert_equal(
        pyrcn.metrics.confusion_matrix(y_true=y_true_bin, y_pred=y_pred_bin),
        sklearn.metrics.confusion_matrix(y_true=np.concatenate(y_true_bin),
                                         y_pred=np.concatenate(y_pred_bin)))

    with pytest.raises(TypeError):
        pyrcn.metrics.confusion_matrix(y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_multilabel_confusion_matrix() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.multilabel_confusion_matrix(y_true=y_true_mlb,
                                                  y_pred=y_true_mlb).shape,
        (5, 2, 2))
    np.testing.assert_equal(
        np.where(
            pyrcn.metrics.multilabel_confusion_matrix(y_true=y_true_mlb,
                                                      y_pred=y_true_mlb,
                                                      sample_weight=sample_weight)),
        np.where(
            pyrcn.metrics.multilabel_confusion_matrix(y_true=y_true_mlb,
                                                      y_pred=y_true_mlb)))
    np.testing.assert_equal(
        pyrcn.metrics.multilabel_confusion_matrix(y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.multilabel_confusion_matrix(y_true=np.concatenate(y_true_mlb),
                                                    y_pred=np.concatenate(y_pred_mlb)))


if __name__ == "__main__":
    test_multilabel_confusion_matrix()
