"""Metrics to assess performance on regression task.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Karan Desai <karandesai281196@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
#          Konstantin Shmelkov <konstantin.shmelkov@polytechnique.edu>
#          Christian Lorentzen <lorentzen.ch@googlemail.com>
#          Ashutosh Hathidara <ashutoshhathidara98@gmail.com>
# License: BSD 3 clause
from typing import no_type_check
import warnings
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import (_deprecate_positional_args,
                                      check_consistent_length, check_array,
                                      _num_samples)
from sklearn._loss.glm_distribution import TweedieDistribution
from sklearn.utils.stats import _weighted_percentile


@no_type_check
def _check_reg_targets(y_true: np.ndarray, y_pred: np.ndarray, multioutput, dtype):
    """
    Check that y_true and y_pred belong to the same regression task.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.
    """
    check_consistent_length(y_true, y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different number of sequences "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    y_true_out = np.empty(shape=y_true.shape, dtype=object)
    y_pred_out = np.empty(shape=y_pred.shape, dtype=object)

    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        y_t = check_array(y_t, ensure_2d=False, dtype=dtype)
        y_p = check_array(y_p, ensure_2d=False, dtype=dtype)
        if y_t.ndim == 1:
            y_t = y_t.reshape(-1, 1)
        if y_p.ndim == 1:
            y_p = y_p.reshape(-1, 1)
        if y_t.shape[1] != y_p.shape[1]:
            raise ValueError("y_true and y_pred have different number of outputs "
                             "({0}!={1})".format(y_t.shape[1], y_p.shape[1]))
        y_true_out[k] = y_t
        y_pred_out[k] = y_p

    n_outputs = np.unique(y.shape[1] for y in y_true)
    if len(n_outputs) > 1:
        raise ValueError("y_true contains sequences with differing numbers of outputs "
                         "({0})".format(n_outputs))
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


@no_type_check
@_deprecate_positional_args
def mean_absolute_error(y_true, y_pred, *, sample_weight=None,
                        multioutput='uniform_average'):
    """Mean absolute error regression loss.

    Read more in the :ref:`User Guide <mean_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape
    (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        MAE output is non-negative floating point. The best value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_error(y_true, y_pred)
    0.75
    >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = np.empty(shape=y_true.shape, dtype=object)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        output_errors[k] = np.average(np.abs(y_p - y_t),
                                      weights=sample_weight, axis=0)
        if isinstance(multioutput, str):
            if multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                output_errors[k] = np.average(output_errors[k], weights=None)
    return np.average(output_errors, weights=None)


@no_type_check
def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None,
                                   multioutput='uniform_average'):
    """
    Mean absolute percentage error regression loss.

    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
    :ref:`User Guide <mean_absolute_percentage_error>`.
    .. versionadded:: 0.24

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1/eps]
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3273...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5515...
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.6198...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred,
                                                             multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    output_errors = np.empty(shape=y_true.shape, dtype=object)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        mape = np.abs(y_p - y_t) / np.maximum(np.abs(y_t), epsilon)
        output_errors[k] = np.average(mape, weights=sample_weight, axis=0)
        if isinstance(multioutput, str):
            if multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                output_errors[k] = np.average(output_errors[k], weights=None)

    return np.average(output_errors, weights=None)


@no_type_check
@_deprecate_positional_args
def mean_squared_error(y_true, y_pred, *, sample_weight=None,
                       multioutput='uniform_average', squared=True):
    """Mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
    (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.822...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred,
                                                             multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = np.empty(shape=y_true.shape, dtype=object)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        output_errors[k] = np.average((y_t - y_p) ** 2, axis=0, weights=sample_weight)
        if not squared:
            output_errors[k] = np.sqrt(output_errors[k])
        if isinstance(multioutput, str):
            if multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                output_errors[k] = np.average(output_errors[k], weights=None)
    return np.average(output_errors, weights=None)


@no_type_check
@_deprecate_positional_args
def mean_squared_log_error(y_true, y_pred, *, sample_weight=None,
                           multioutput='uniform_average'):
    """Mean squared logarithmic error regression loss.

    Read more in the :ref:`User Guide <mean_squared_log_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
    (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y_true, y_pred)
    0.039...
    >>> y_true = [[0.5, 1], [1, 2], [7, 6]]
    >>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
    >>> mean_squared_log_error(y_true, y_pred)
    0.044...
    >>> mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
    array([0.00462428, 0.08377444])
    >>> mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.060...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred,
                                                             multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    if (np.concatenate(y_true) < 0).any() or (np.concatenate(y_pred) < 0).any():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
    y_true_log = np.empty(shape=y_true.shape, dtype=object)
    y_pred_log = np.empty(shape=y_pred.shape, dtype=object)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        y_true_log[k] = np.log1p(y_t)
        y_pred_log[k] = np.log1p(y_p)
    return mean_squared_error(y_true_log, y_pred_log, sample_weight=sample_weight,
                              multioutput=multioutput)


@no_type_check
@_deprecate_positional_args
def median_absolute_error(y_true, y_pred, *, multioutput='uniform_average',
                          sample_weight=None):
    """Median absolute error regression loss.

    Median absolute error output is non-negative floating point. The best value
    is 0.0. Read more in the :ref:`User Guide <median_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
    (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines
        weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        .. versionadded:: 0.24

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

    Examples
    --------
    >>> from sklearn.metrics import median_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> median_absolute_error(y_true, y_pred)
    0.75
    >>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred,
                                                             multioutput)
    output_errors = np.empty(shape=y_true.shape, dtype=object)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        if sample_weight is None:
            output_errors[k] = np.median(np.abs(y_t - y_p), axis=0)
        else:
            output_errors[k] = _weighted_percentile(np.abs(y_pred - y_true),
                                                    sample_weight=sample_weight)
        if isinstance(multioutput, str):
            if multioutput == 'uniform_average':
                # pass None as weights to np.average: uniform mean
                output_errors[k] = np.average(output_errors[k], weights=None)

    return np.average(output_errors, weights=None)


@no_type_check
@_deprecate_positional_args
def explained_variance_score(y_true, y_pred, *, sample_weight=None,
                             multioutput='uniform_average'):
    """Explained variance regression score function.

    Best possible score is 1.0, lower values are worse.
    Read more in the :ref:`User Guide <explained_variance_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'} or
    array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

    Returns
    -------
    score : float or ndarray of floats
        The explained variance or ndarray if 'multioutput' is 'raw_values'.

    Notes
    -----
    This is not a symmetric function.

    Examples
    --------
    >>> from sklearn.metrics import explained_variance_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y_true, y_pred)
    0.957...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    0.983...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    y_diff_avg = np.empty(shape=y_true.shape, dtype=object)
    numerator = np.empty(shape=y_true.shape, dtype=object)
    y_true_avg = np.empty(shape=y_true.shape, dtype=object)
    denominator = np.empty(shape=y_true.shape, dtype=object)
    nonzero_numerator = np.empty(shape=y_true.shape, dtype=object)
    nonzero_denominator = np.empty(shape=y_true.shape, dtype=object)
    valid_score = np.empty(shape=y_true.shape, dtype=object)
    output_scores = np.empty(shape=y_true.shape, dtype=object)

    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        y_diff_avg[k] = np.average(y_t - y_p, weights=sample_weight, axis=0)
        numerator[k] = np.average((y_t - y_p - y_diff_avg[k]) ** 2,
                                  weights=sample_weight, axis=0)
        y_true_avg[k] = np.average(y_t, weights=sample_weight, axis=0)
        denominator[k] = np.average((y_t - y_true_avg[k]) ** 2,
                                    weights=sample_weight, axis=0)
        nonzero_numerator[k] = numerator[k] != 0
        nonzero_denominator[k] = denominator[k] != 0
        valid_score[k] = nonzero_numerator[k] & nonzero_denominator[k]
        output_scores[k] = np.ones(y_t.shape[1])

        output_scores[k][valid_score[k]] = 1 - (numerator[k][valid_score[k]]
                                                / denominator[k][valid_score[k]])
        output_scores[k][nonzero_numerator[k] & ~nonzero_denominator[k]] = 0.
        if isinstance(multioutput, str):
            if multioutput == 'uniform_average':
                # passing to np.average() None as weights results is uniform mean
                avg_weights = None
            elif multioutput == 'variance_weighted':
                avg_weights = denominator
        else:
            avg_weights = multioutput
        output_scores[k] = np.average(output_scores[k], weights=avg_weights)
    if multioutput == 'raw_values':
        # return scores individually
        return np.mean(output_scores)
    else:
        return np.mean([np.average(scores, weights=avg_weights)
                        for scores in output_scores])


@no_type_check
@_deprecate_positional_args
def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'},
    array-like of shape (n_outputs,) or None, default='uniform_average'
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.

    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted')
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = np.empty(shape=y_true.shape, dtype=object)
    denominator = np.empty(shape=y_true.shape, dtype=object)
    nonzero_numerator = np.empty(shape=y_true.shape, dtype=object)
    nonzero_denominator = np.empty(shape=y_true.shape, dtype=object)
    valid_score = np.empty(shape=y_true.shape, dtype=object)
    output_scores = np.empty(shape=y_true.shape, dtype=object)

    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        numerator[k] = (weight * (y_t - y_p) ** 2).sum(axis=0, dtype=np.float64)
        denominator[k] = (weight * (y_t - np.average(y_t, axis=0,
                                                     weights=sample_weight)) ** 2).sum(
                                                         axis=0, dtype=np.float64)
        nonzero_numerator[k] = numerator[k] != 0
        nonzero_denominator[k] = denominator[k] != 0
        valid_score[k] = nonzero_denominator[k] & nonzero_numerator[k]
        output_scores[k] = np.ones([y_t.shape[1]])

        output_scores[k][valid_score[k]] = 1 - (numerator[k][valid_score[k]]
                                                / denominator[k][valid_score[k]])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[k][nonzero_numerator[k] & ~nonzero_denominator[k]] = 0.
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                # return scores individually
                output_scores[k] = output_scores[k]
            elif multioutput == 'uniform_average':
                # passing None as weights results is uniform mean
                avg_weights = None
                output_scores[k] = np.average(output_scores[k], weights=avg_weights)
            elif multioutput == 'variance_weighted':
                avg_weights = denominator[k]
                # avoid fail on constant y or one-element arrays
                if not np.any(nonzero_denominator[k]):
                    if not np.any(nonzero_numerator[k]):
                        output_scores[k] = 1.0
                    else:
                        output_scores[k] = 0.0
                else:
                    output_scores[k] = np.average(output_scores[k], weights=avg_weights)
        else:
            avg_weights = multioutput
            output_scores[k] = np.average(output_scores[k], weights=avg_weights)
    return np.average(output_scores, weights=None)


@no_type_check
def max_error(y_true, y_pred):
    """
    max_error metric calculates the maximum residual error.

    Read more in the :ref:`User Guide <max_error>`.

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

    Examples
    --------
    >>> from sklearn.metrics import max_error
    >>> y_true = [3, 2, 7, 1]
    >>> y_pred = [4, 2, 7, 1]
    >>> max_error(y_true, y_pred)
    1
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)
    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in max_error")
    return np.max(np.abs(y_true - y_pred))


@no_type_check
@_deprecate_positional_args
def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.
        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.
        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to
          mean_squared_error. y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_tweedie_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_tweedie_deviance(y_true, y_pred, power=1)
    1.4260...
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[np.float64, np.float32])
    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")
    check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]

    dist = TweedieDistribution(power=power)
    dev = dist.unit_deviance(y_true, y_pred, check_input=True)

    return np.average(dev, weights=sample_weight)


@no_type_check
@_deprecate_positional_args
def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None):
    """Mean Poisson deviance regression loss.

    Poisson deviance is equivalent to the Tweedie deviance with
    the power parameter `power=1`.
    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

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

    Examples
    --------
    >>> from sklearn.metrics import mean_poisson_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_poisson_deviance(y_true, y_pred)
    1.4260...
    """
    return mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=1
    )


@no_type_check
@_deprecate_positional_args
def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None):
    """
    Mean Gamma deviance regression loss.

    Gamma deviance is equivalent to the Tweedie deviance with
    the power parameter `power=2`. It is invariant to scaling of
    the target variable, and measures relative errors.
    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

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

    Examples
    --------
    >>> from sklearn.metrics import mean_gamma_deviance
    >>> y_true = [2, 0.5, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_gamma_deviance(y_true, y_pred)
    1.0568...
    """
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2)
