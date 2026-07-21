"""Metrics to assess performance on classification task given class prediction.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.

Each public metric is generated from the corresponding scikit-learn metric by
:func:`pyrcn.metrics._wrappers._wrap`, which adds PyRCN sequence handling
(concatenate per-sequence arrays) while forwarding every configuration
argument unchanged. This keeps the signatures in sync with scikit-learn.
"""

from __future__ import annotations

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import sklearn.metrics as sklearn_metrics

from ._wrappers import _wrap

accuracy_score = _wrap(sklearn_metrics.accuracy_score)
confusion_matrix = _wrap(sklearn_metrics.confusion_matrix)
multilabel_confusion_matrix = _wrap(
    sklearn_metrics.multilabel_confusion_matrix)
cohen_kappa_score = _wrap(sklearn_metrics.cohen_kappa_score)
jaccard_score = _wrap(sklearn_metrics.jaccard_score)
matthews_corrcoef = _wrap(sklearn_metrics.matthews_corrcoef)
zero_one_loss = _wrap(sklearn_metrics.zero_one_loss)
f1_score = _wrap(sklearn_metrics.f1_score)
fbeta_score = _wrap(sklearn_metrics.fbeta_score)
precision_recall_fscore_support = _wrap(
    sklearn_metrics.precision_recall_fscore_support)
precision_score = _wrap(sklearn_metrics.precision_score)
recall_score = _wrap(sklearn_metrics.recall_score)
balanced_accuracy_score = _wrap(sklearn_metrics.balanced_accuracy_score)
classification_report = _wrap(sklearn_metrics.classification_report)
hamming_loss = _wrap(sklearn_metrics.hamming_loss)
log_loss = _wrap(sklearn_metrics.log_loss)
hinge_loss = _wrap(sklearn_metrics.hinge_loss)
brier_score_loss = _wrap(sklearn_metrics.brier_score_loss)
