"""Testing for Metrics module."""

from __future__ import annotations

import numpy as np
import sklearn.metrics
from sklearn.datasets import (make_classification,
                              make_multilabel_classification)

import pyrcn.metrics

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

rng_score = np.random.RandomState(7)
y_prob_bin = np.empty(shape=(10,), dtype=object)
pred_decision = np.empty(shape=(10,), dtype=object)
for k in range(10):
    y_prob_bin[k] = rng_score.uniform(size=10 * (k + 1))
    pred_decision[k] = rng_score.uniform(-1, 1, size=10 * (k + 1))


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
    # Plain (non-sequence) arrays are now accepted and match scikit-learn.
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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

    np.testing.assert_equal(
        pyrcn.metrics.confusion_matrix(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.confusion_matrix(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.cohen_kappa_score(y1=y_true_bin[0], y2=y_pred_bin[0]),
        sklearn.metrics.cohen_kappa_score(y1=y_true_bin[0], y2=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.jaccard_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.jaccard_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.matthews_corrcoef(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.matthews_corrcoef(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.zero_one_loss(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.zero_one_loss(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.f1_score(y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.f1_score(y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.fbeta_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0], beta=0),
        sklearn.metrics.fbeta_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0], beta=0))


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
    np.testing.assert_equal(
        pyrcn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.precision_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.balanced_accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.balanced_accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


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
    np.testing.assert_equal(
        pyrcn.metrics.hamming_loss(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.hamming_loss(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


def test_classification_report() -> None:
    report = pyrcn.metrics.classification_report(
        y_true=y_true_bin, y_pred=y_true_bin)
    assert isinstance(report, str)
    report_dict = pyrcn.metrics.classification_report(
        y_true=y_true_bin, y_pred=y_pred_bin, sample_weight=sample_weight,
        output_dict=True)
    assert isinstance(report_dict, dict)
    np.testing.assert_equal(
        pyrcn.metrics.classification_report(
            y_true=y_true_bin, y_pred=y_pred_bin, output_dict=True),
        sklearn.metrics.classification_report(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin), output_dict=True))
    np.testing.assert_equal(
        pyrcn.metrics.classification_report(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.classification_report(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


def test_hinge_loss() -> None:
    np.testing.assert_equal(
        np.greater_equal(pyrcn.metrics.hinge_loss(
            y_true=y_true_bin, pred_decision=pred_decision), 0), True)
    np.testing.assert_almost_equal(
        pyrcn.metrics.hinge_loss(
            y_true=y_true_bin, pred_decision=pred_decision,
            sample_weight=sample_weight),
        sklearn.metrics.hinge_loss(
            y_true=np.concatenate(y_true_bin),
            pred_decision=np.concatenate(pred_decision)))
    np.testing.assert_almost_equal(
        pyrcn.metrics.hinge_loss(
            y_true=y_true_bin[0], pred_decision=pred_decision[0]),
        sklearn.metrics.hinge_loss(
            y_true=y_true_bin[0], pred_decision=pred_decision[0]))


def test_log_loss() -> None:
    # The generic wrapper mirrors whatever signature the installed
    # scikit-learn exposes (the probability arg was renamed y_pred->y_proba
    # across versions), so call positionally to stay version-agnostic; the
    # pyrcn metric matches scikit-learn on both sequence and plain input.
    np.testing.assert_almost_equal(
        pyrcn.metrics.log_loss(y_true_bin, y_prob_bin,
                               sample_weight=sample_weight),
        sklearn.metrics.log_loss(np.concatenate(y_true_bin),
                                 np.concatenate(y_prob_bin),
                                 sample_weight=np.concatenate(sample_weight)))
    np.testing.assert_almost_equal(
        pyrcn.metrics.log_loss(y_true_bin, y_prob_bin),
        sklearn.metrics.log_loss(np.concatenate(y_true_bin),
                                 np.concatenate(y_prob_bin)))
    # Plain (non-sequence) arrays are now accepted.
    np.testing.assert_almost_equal(
        pyrcn.metrics.log_loss(y_true_bin[0], y_prob_bin[0]),
        sklearn.metrics.log_loss(y_true_bin[0], y_prob_bin[0]))


def test_brier_score_loss() -> None:
    # Call positionally to stay agnostic to the y_prob->y_proba rename across
    # scikit-learn versions; the pyrcn metric matches scikit-learn on both
    # sequence and plain input.
    np.testing.assert_almost_equal(
        pyrcn.metrics.brier_score_loss(y_true_bin, y_prob_bin,
                                       sample_weight=sample_weight),
        sklearn.metrics.brier_score_loss(
            np.concatenate(y_true_bin), np.concatenate(y_prob_bin),
            sample_weight=np.concatenate(sample_weight)))
    np.testing.assert_almost_equal(
        pyrcn.metrics.brier_score_loss(y_true_bin[0], y_prob_bin[0]),
        sklearn.metrics.brier_score_loss(y_true_bin[0], y_prob_bin[0]))


def test_plain_array_flexibility() -> None:
    # A plain (non-object) array is passed straight through to scikit-learn.
    plain = y_true_bin[0]
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=plain, y_pred=plain), 1)
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]),
        sklearn.metrics.accuracy_score(
            y_true=y_true_bin[0], y_pred=y_pred_bin[0]))


def test_sequence_equals_concatenation() -> None:
    # A sequence and its concatenation yield the same result.
    np.testing.assert_equal(
        pyrcn.metrics.accuracy_score(y_true=y_true_bin, y_pred=y_pred_bin),
        pyrcn.metrics.accuracy_score(
            y_true=np.concatenate(y_true_bin),
            y_pred=np.concatenate(y_pred_bin)))
