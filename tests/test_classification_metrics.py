"""Testing for Metrics module."""

import pyrcn.metrics
import sklearn.metrics
from sklearn.datasets import (make_classification,
                              make_multilabel_classification)
import numpy as np
import pytest


rng_true = np.random.RandomState(42)
rng_pred = np.random.RandomState(1234)
y_true_bin = np.empty(shape=(10,), dtype=object)
y_pred_bin = np.empty(shape=(10,), dtype=object)
sample_weight = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_bin[k] = make_classification(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_true)
    _, y_pred_bin[k] = make_classification(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_pred)
    sample_weight[k] = np.ones_like(y_true_bin[k])

y_true_mlb = np.empty(shape=(10,), dtype=object)
y_pred_mlb = np.empty(shape=(10,), dtype=object)
for k in range(10):
    _, y_true_mlb[k] = make_multilabel_classification(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_true)
    _, y_pred_mlb[k] = make_multilabel_classification(
        n_samples=10 * (k + 1), n_features=20, random_state=rng_pred)


def test_accuracy_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.accuracy_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.accuracy_score(y_true=np.concatenate(y_true_bin),
                                       y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_mlb, y_pred=y_true_mlb), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.accuracy_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.accuracy_score(y_true=np.concatenate(y_true_mlb),
                                       y_pred=np.concatenate(y_pred_mlb)))
    with pytest.raises(TypeError):
        pyrcn.metrics.accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_confusion_matrix() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.confusion_matrix(
            y_true=y_true_bin, y_pred=y_true_bin).shape, (2, 2))
    np.testing.assert_equal(
        np.where(pyrcn.metrics.confusion_matrix(
            y_true=y_true_bin, y_pred=y_true_bin,
            sample_weight=sample_weight)), np.where(np.eye(2)))
    np.testing.assert_equal(
        pyrcn.metrics.confusion_matrix(y_true=y_true_bin, y_pred=y_pred_bin),
        sklearn.metrics.confusion_matrix(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin)))

    with pytest.raises(TypeError):
        pyrcn.metrics.confusion_matrix(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_multilabel_confusion_matrix() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.multilabel_confusion_matrix(
            y_true=y_true_mlb, y_pred=y_true_mlb).shape, (5, 2, 2))
    np.testing.assert_equal(
        np.where(
            pyrcn.metrics.multilabel_confusion_matrix(
                y_true=y_true_mlb, y_pred=y_true_mlb,
                sample_weight=sample_weight)),
        np.where(
            pyrcn.metrics.multilabel_confusion_matrix(
                y_true=y_true_mlb, y_pred=y_true_mlb)))
    np.testing.assert_equal(
        pyrcn.metrics.multilabel_confusion_matrix(
            y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.multilabel_confusion_matrix(
            y_true=np.concatenate(y_true_mlb),
            y_pred=np.concatenate(y_pred_mlb)))


def test_cohen_kappa_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.cohen_kappa_score(
            y1=y_true_bin, y2=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.cohen_kappa_score(
            y1=y_true_bin, y2=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.cohen_kappa_score(
            y1=y_true_bin, y2=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.cohen_kappa_score(
            y1=np.concatenate(y_true_bin), y2=np.concatenate(y_pred_bin)))
    with pytest.raises(TypeError):
        pyrcn.metrics.cohen_kappa_score(y1=y_true_bin[0], y2=y_pred_bin[0])


def test_jaccard_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.jaccard_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.jaccard_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.jaccard_score(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.jaccard_score(y_true=np.concatenate(y_true_bin),
                                      y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.jaccard_score(
            y_true=y_true_mlb, y_pred=y_true_mlb, average=None), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.jaccard_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="micro"), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.jaccard_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="macro"),
        sklearn.metrics.jaccard_score(y_true=np.concatenate(y_true_mlb),
                                      y_pred=np.concatenate(y_pred_mlb),
                                      average="macro"))
    with pytest.raises(TypeError):
        pyrcn.metrics.jaccard_score(y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_matthews_corrcoef() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.matthews_corrcoef(
            y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(
            pyrcn.metrics.matthews_corrcoef(
                y_true=y_true_bin, y_pred=y_pred_bin), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.matthews_corrcoef(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.matthews_corrcoef(y_true=np.concatenate(y_true_bin),
                                          y_pred=np.concatenate(y_pred_bin)))
    with pytest.raises(TypeError):
        pyrcn.metrics.matthews_corrcoef(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_zero_one_loss() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.zero_one_loss(y_true=y_true_bin, y_pred=y_true_bin), 0)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.zero_one_loss(
            y_true=y_true_bin, y_pred=y_pred_bin), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.zero_one_loss(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.zero_one_loss(y_true=np.concatenate(y_true_bin),
                                      y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.zero_one_loss(y_true=y_true_mlb, y_pred=y_true_mlb), 0)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.zero_one_loss(
            y_true=y_true_mlb, y_pred=y_pred_mlb), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.zero_one_loss(y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.zero_one_loss(y_true=np.concatenate(y_true_mlb),
                                      y_pred=np.concatenate(y_pred_mlb)))
    with pytest.raises(TypeError):
        pyrcn.metrics.zero_one_loss(y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_f1_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.f1_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.f1_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.f1_score(y_true=y_true_bin, y_pred=y_pred_bin,
                               sample_weight=sample_weight),
        sklearn.metrics.f1_score(y_true=np.concatenate(y_true_bin),
                                 y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.f1_score(
            y_true=y_true_mlb, y_pred=y_true_mlb, average=None), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.f1_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="micro"), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.f1_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="weighted"),
        sklearn.metrics.f1_score(
            y_true=np.concatenate(y_true_mlb),
            y_pred=np.concatenate(y_pred_mlb), average="weighted"))
    with pytest.raises(TypeError):
        pyrcn.metrics.f1_score(y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_fbeta_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.fbeta_score(
            y_true=y_true_bin, y_pred=y_true_bin, beta=1), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.fbeta_score(
            y_true=y_true_bin, y_pred=y_pred_bin, beta=0), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.fbeta_score(y_true=y_true_bin, y_pred=y_pred_bin,
                                  sample_weight=sample_weight, beta=0.5),
        sklearn.metrics.fbeta_score(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin), beta=0.5))

    np.testing.assert_equal(
        pyrcn.metrics.fbeta_score(
            y_true=y_true_mlb, y_pred=y_true_mlb, average=None, beta=0), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.fbeta_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb,
            average="micro", beta=1), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.fbeta_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb,
            average="weighted", beta=0.5),
        sklearn.metrics.fbeta_score(
            y_true=np.concatenate(y_true_mlb),
            y_pred=np.concatenate(y_pred_mlb), average="weighted", beta=0.5))
    with pytest.raises(TypeError):
        pyrcn.metrics.fbeta_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0], beta=0)


def test_precision_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.precision_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.precision_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.precision_score(y_true=y_true_bin, y_pred=y_pred_bin,
                                      sample_weight=sample_weight),
        sklearn.metrics.precision_score(y_true=np.concatenate(y_true_bin),
                                        y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.precision_score(
            y_true=y_true_mlb, y_pred=y_true_mlb, average=None), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.precision_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="micro"), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.precision_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="weighted"),
        sklearn.metrics.precision_score(y_true=np.concatenate(y_true_mlb),
                                        y_pred=np.concatenate(y_pred_mlb),
                                        average="weighted"))
    with pytest.raises(TypeError):
        pyrcn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_recall_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.recall_score(y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.recall_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.recall_score(y_true=y_true_bin, y_pred=y_pred_bin,
                                   sample_weight=sample_weight),
        sklearn.metrics.recall_score(y_true=np.concatenate(y_true_bin),
                                     y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.recall_score(
            y_true=y_true_mlb, y_pred=y_true_mlb, average=None), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.recall_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="micro"), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.recall_score(
            y_true=y_true_mlb, y_pred=y_pred_mlb, average="weighted"),
        sklearn.metrics.recall_score(y_true=np.concatenate(y_true_mlb),
                                     y_pred=np.concatenate(y_pred_mlb),
                                     average="weighted"))
    with pytest.raises(TypeError):
        pyrcn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_balanced_accuracy_score() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.balanced_accuracy_score(
            y_true=y_true_bin, y_pred=y_true_bin), 1)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.balanced_accuracy_score(
            y_true=y_true_bin, y_pred=y_pred_bin), 1), True)
    np.testing.assert_equal(
        pyrcn.metrics.balanced_accuracy_score(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight,
            adjusted=True),
        sklearn.metrics.balanced_accuracy_score(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin), adjusted=True))
    with pytest.raises(TypeError):
        pyrcn.metrics.balanced_accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0])


def test_hamming_loss() -> None:
    np.testing.assert_equal(
        pyrcn.metrics.hamming_loss(y_true=y_true_bin, y_pred=y_true_bin), 0)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.hamming_loss(
            y_true=y_true_bin, y_pred=y_pred_bin), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.hamming_loss(
            y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight),
        sklearn.metrics.hamming_loss(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin)))

    np.testing.assert_equal(
        pyrcn.metrics.hamming_loss(y_true=y_true_mlb, y_pred=y_true_mlb), 0)
    np.testing.assert_equal(
        np.less(pyrcn.metrics.hamming_loss(
            y_true=y_true_mlb, y_pred=y_pred_mlb), 1),
        True)
    np.testing.assert_equal(
        pyrcn.metrics.hamming_loss(y_true=y_true_mlb, y_pred=y_pred_mlb),
        sklearn.metrics.hamming_loss(
            y_true=np.concatenate(y_true_mlb),
            y_pred=np.concatenate(y_pred_mlb)))
    with pytest.raises(TypeError):
        pyrcn.metrics.hamming_loss(y_true=y_true_bin[0], y_pred=y_pred_bin[0])
