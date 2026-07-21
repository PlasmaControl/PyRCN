"""Metrics to assess performance on regression task.

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

# Authors: Peter Steiner <peter.steiner@princeton.edu>
# License: BSD 3 clause

import sklearn.metrics as sklearn_metrics

from ._wrappers import _wrap

mean_absolute_error = _wrap(sklearn_metrics.mean_absolute_error)
mean_absolute_percentage_error = _wrap(
    sklearn_metrics.mean_absolute_percentage_error)
mean_squared_error = _wrap(sklearn_metrics.mean_squared_error)
mean_squared_log_error = _wrap(sklearn_metrics.mean_squared_log_error)
median_absolute_error = _wrap(sklearn_metrics.median_absolute_error)
explained_variance_score = _wrap(sklearn_metrics.explained_variance_score)
r2_score = _wrap(sklearn_metrics.r2_score)
max_error = _wrap(sklearn_metrics.max_error)
mean_tweedie_deviance = _wrap(sklearn_metrics.mean_tweedie_deviance)
mean_poisson_deviance = _wrap(sklearn_metrics.mean_poisson_deviance)
mean_gamma_deviance = _wrap(sklearn_metrics.mean_gamma_deviance)
