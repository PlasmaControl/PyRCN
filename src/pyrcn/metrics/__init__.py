"""
The :mod:`pyrcn.metrics` module includes score functions, performance metrics.

Also, pairwise metrics and distance computations for sequence-to-sequence
results.
"""

from __future__ import annotations

# Author: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause
from ..metrics._regression import (explained_variance_score, max_error,
                                   mean_absolute_error,
                                   mean_absolute_percentage_error,
                                   mean_gamma_deviance, mean_poisson_deviance,
                                   mean_squared_error, mean_squared_log_error,
                                   mean_tweedie_deviance,
                                   median_absolute_error, r2_score)
from ._classification import (accuracy_score, balanced_accuracy_score,
                              brier_score_loss, classification_report,
                              cohen_kappa_score, confusion_matrix, f1_score,
                              fbeta_score, hamming_loss, hinge_loss,
                              jaccard_score, log_loss, matthews_corrcoef,
                              multilabel_confusion_matrix,
                              precision_recall_fscore_support, precision_score,
                              recall_score, zero_one_loss)

__all__ = ('accuracy_score',
           'balanced_accuracy_score',
           'classification_report',
           'cohen_kappa_score',
           'confusion_matrix',
           'f1_score',
           'fbeta_score',
           'hamming_loss',
           'hinge_loss',
           'jaccard_score',
           'log_loss',
           'matthews_corrcoef',
           'precision_recall_fscore_support',
           'precision_score',
           'recall_score',
           'zero_one_loss',
           'brier_score_loss',
           'multilabel_confusion_matrix',
           'explained_variance_score',
           'max_error',
           'mean_absolute_error',
           'mean_squared_error',
           'mean_squared_log_error',
           'median_absolute_error',
           'mean_absolute_percentage_error',
           'r2_score',
           'mean_tweedie_deviance',
           'mean_poisson_deviance',
           'mean_gamma_deviance',
           )
