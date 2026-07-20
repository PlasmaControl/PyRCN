"""Metrics to assess performance on regression task.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from typing import Any, Literal

import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length


def _check_reg_targets(y_true: np.ndarray, y_pred: np.ndarray,
                       sample_weight: np.ndarray | None = None,
                       multioutput: (np.ndarray | Literal[
                           "raw_values", "uniform_average",
                           "variance_weighted"] | None) = None,
                       dtype: str = "numeric")\
        -> tuple[
            Any, np.ndarray, np.ndarray, np.ndarray | None,
            np.ndarray
            | Literal["raw_values", "uniform_average", "variance_weighted"]
            | None]:
    """
    Check that y_true and y_pred belong to the same regression task.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
    y_pred : np.ndarray, dtype=object
    sample_weight: Optional[np.ndarray], default=None
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    dtype : str, default="numeric"

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    sample_weight : np.ndarray
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.
    """
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
        [check_consistent_length(y_t, y_p, s_w)
         for y_t, y_p, s_w in zip(y_true, y_pred, sample_weight)]
        sample_weight = np.concatenate(sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
        [check_consistent_length(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_type = type_of_target(y_true)
    return y_type, y_true, y_pred, sample_weight, multioutput


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                        sample_weight: np.ndarray | None = None,
                        multioutput: (np.ndarray | Literal[
                            "raw_values", "uniform_average",
                            "variance_weighted"] | None) = "uniform_average")\
        -> float:
    """Mean absolute error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output values, one of
        {'raw_values', 'uniform_average'} or an array-like of shape
        (n_outputs,). An array-like value defines weights used to average
        errors. With 'raw_values', a full set of errors is returned in case
        of multioutput input. With 'uniform_average', errors of all outputs
        are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately. If multioutput is 'uniform_average' or
        an ndarray of weights, then the weighted average of all output
        errors is returned. MAE output is non-negative floating point. The
        best value is 0.0.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_absolute_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray,
                                   sample_weight: np.ndarray | None = None,
                                   multioutput: (np.ndarray | Literal[
                                       "raw_values", "uniform_average",
                                       "variance_weighted"] | None) =
                                   "uniform_average")\
        -> float:
    """Mean absolute percentage error regression loss.

    Note that the output is not represented as a percentage in range
    [0, 100]. Instead, it is represented in range [0, 1/eps].

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output values, one of
        {'raw_values', 'uniform_average'} or an array-like of shape
        (n_outputs,). An array-like value defines weights used to average
        errors. With 'raw_values', a full set of errors is returned in case
        of multioutput input. With 'uniform_average', errors of all outputs
        are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1/eps]
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately. If multioutput is
        'uniform_average' or an ndarray of weights, then the weighted average
        of all output errors is returned. MAPE output is non-negative
        floating point. The best value is 0.0. Note that bad predictions can
        lead to arbitrarily large MAPE values, especially if some y_true
        values are very close to zero. A large value is returned instead of
        ``inf`` when y_true is zero.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_absolute_percentage_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                       sample_weight: np.ndarray | None = None,
                       multioutput: (np.ndarray | Literal[
                           "raw_values", "uniform_average",
                           "variance_weighted"] | None) = "uniform_average",
                       squared: bool = True) -> float:
    """Mean squared error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output values, one of
        {'raw_values', 'uniform_average'} or an array-like of shape
        (n_outputs,). An array-like value defines weights used to average
        errors. With 'raw_values', a full set of errors is returned in case
        of multioutput input. With 'uniform_average', errors of all outputs
        are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    if squared:
        return sklearn_metrics.mean_squared_error(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
            multioutput=multioutput)
    return sklearn_metrics.root_mean_squared_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                           sample_weight: np.ndarray | None = None,
                           multioutput: (np.ndarray | Literal[
                               "raw_values", "uniform_average",
                               "variance_weighted"] | None) =
                           "uniform_average") -> float:
    """Mean squared logarithmic error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output values, one of
        {'raw_values', 'uniform_average'} or an array-like of shape
        (n_outputs,). An array-like value defines weights used to average
        errors. With 'raw_values', a full set of errors is returned in case
        of multioutput input. With 'uniform_average', errors of all outputs
        are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_squared_log_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                          multioutput: (np.ndarray | Literal[
                              "raw_values", "uniform_average",
                              "variance_weighted"] | None) = "uniform_average",
                          sample_weight: np.ndarray | None = None) -> float:
    """Median absolute error regression loss.

    Median absolute error output is non-negative floating point. The best
    value is 0.0.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output values, one of
        {'raw_values', 'uniform_average'} or an array-like of shape
        (n_outputs,). An array-like value defines weights used to average
        errors. With 'raw_values', a full set of errors is returned in case
        of multioutput input. With 'uniform_average', errors of all outputs
        are averaged with uniform weight.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately. If multioutput is 'uniform_average' or
        an ndarray of weights, then the weighted average of all output
        errors is returned.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.median_absolute_error(
        y_true=y_true, y_pred=y_pred, multioutput=multioutput,
        sample_weight=sample_weight)


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                             sample_weight: np.ndarray | None = None,
                             multioutput: (np.ndarray | Literal[
                                 "raw_values", "uniform_average",
                                 "variance_weighted"] | None) =
                             "uniform_average") -> float:
    """Explained variance regression score function.

    Best possible score is 1.0, lower values are worse.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str or array-like, default='uniform_average'
        Defines aggregating of multiple output scores, one of
        {'raw_values', 'uniform_average', 'variance_weighted'} or an
        array-like of shape (n_outputs,). An array-like value defines
        weights used to average scores. With 'raw_values', a full set of
        scores is returned in case of multioutput input. With
        'uniform_average', scores of all outputs are averaged with uniform
        weight. With 'variance_weighted', scores of all outputs are
        averaged, weighted by the variances of each individual output.

    Returns
    -------
    score : float or ndarray of floats
        The explained variance or ndarray if 'multioutput' is 'raw_values'.

    Notes
    -----
    This is not a symmetric function.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.explained_variance_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, *,
             sample_weight: np.ndarray | None = None,
             multioutput: (np.ndarray | Literal[
                 "raw_values", "uniform_average", "variance_weighted"] |
                                None) = "uniform_average") -> float:
    """R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : str, array-like or None, default='uniform_average'
        Defines aggregating of multiple output scores, one of
        {'raw_values', 'uniform_average', 'variance_weighted'}, an
        array-like of shape (n_outputs,) or None. An array-like value
        defines weights used to average scores. With 'raw_values', a full
        set of scores is returned in case of multioutput input. With
        'uniform_average', scores of all outputs are averaged with uniform
        weight. With 'variance_weighted', scores of all outputs are
        averaged, weighted by the variances of each individual output.

    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function. Unlike most other scores, the R^2
    score may be negative (it need not actually be the square of a quantity
    R). This metric is not well-defined for single samples and will return a
    NaN value if n_samples is less than two.
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput=multioutput)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.r2_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the maximum residual error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    max_error : float
        A positive floating point value (the best value is 0.0).
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, None)
    if sample_weight is not None:  # pragma: no cover
        # sample_weight is always None here (passed as None above), so this
        # branch is unreachable; kept for parity with the other metrics.
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.max_error(y_true=y_true, y_pred=y_pred)


def mean_tweedie_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                          sample_weight: np.ndarray | None = None,
                          power: float = 0) -> float:
    """Mean Tweedie deviance regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1. The higher
        the power, the less weight is given to extreme deviations between
        true and predicted targets. The distribution depends on power:
        power < 0 is the extreme stable distribution (requires y_pred > 0);
        power = 0 is the normal distribution (output corresponds to
        mean_squared_error, y_true and y_pred can be any real numbers);
        power = 1 is the Poisson distribution (requires y_true >= 0 and
        y_pred > 0); 1 < power < 2 is the compound Poisson distribution
        (requires y_true >= 0 and y_pred > 0); power = 2 is the Gamma
        distribution (requires y_true > 0 and y_pred > 0); power = 3 is the
        inverse Gaussian distribution (requires y_true > 0 and y_pred > 0);
        otherwise it is the positive stable distribution (requires
        y_true > 0 and y_pred > 0).

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    """
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_tweedie_deviance(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, power=power)


def mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                          sample_weight: np.ndarray | None = None) -> float:
    """Mean Poisson deviance regression loss.

    Poisson deviance is equivalent to the Tweedie deviance with the power
    parameter power=1.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Requires y_true >= 0.
    y_pred : array-like of shape (n_samples,)
        Estimated target values. Requires y_pred > 0.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    """
    return mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=1)


def mean_gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                        sample_weight: np.ndarray | None = None) -> float:
    """Mean Gamma deviance regression loss.

    Gamma deviance is equivalent to the Tweedie deviance with the power
    parameter power=2. It is invariant to scaling of the target variable,
    and measures relative errors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Requires y_true > 0.
    y_pred : array-like of shape (n_samples,)
        Estimated target values. Requires y_pred > 0.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    """
    return mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=2)
