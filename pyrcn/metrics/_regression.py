"""Metrics to assess performance on regression task.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause
import sys
if sys.version_info >= (3, 8):
    from typing import Any, Tuple, Union, Optional, Literal
else:
    from typing_extensions import Literal
    from typing import Any, Tuple, Union, Optional
import numpy as np

from sklearn.utils.validation import check_consistent_length, _deprecate_positional_args
import sklearn.metrics as sklearn_metrics
from sklearn.metrics._regression import _check_reg_targets as sklearn_check_reg_targets


def _check_reg_targets(y_true: np.ndarray, y_pred: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None,
                       multioutput: Union[np.ndarray, Literal["raw_values",
                                                              "uniform_average",
                                                              "variance_weighted"],
                                          None] = None,
                       dtype: str = "numeric") \
                           -> Tuple[Any,
                                    np.ndarray, np.ndarray, Optional[np.ndarray],
                                    Union[np.ndarray, Literal["raw_values",
                                                              "uniform_average",
                                                              "variance_weighted"],
                                          None]]:
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
    y_type, y_true, y_pred, multioutput = sklearn_check_reg_targets(y_true, y_pred,
                                                                    multioutput, dtype)
    return y_type, y_true, y_pred, sample_weight, multioutput


@_deprecate_positional_args
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                        sample_weight: Optional[np.ndarray] = None,
                        multioutput: Union[np.ndarray, Literal["raw_values",
                                                               "uniform_average",
                                                               "variance_weighted"],
                                           None] = "uniform_average") -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred,
                                               sample_weight=sample_weight,
                                               multioutput=multioutput)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray,
                                   sample_weight: Optional[np.ndarray] = None,
                                   multioutput: Union[np.ndarray,
                                                      Literal["raw_values",
                                                              "uniform_average",
                                                              "variance_weighted"],
                                                      None] = "uniform_average")\
        -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_absolute_percentage_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


@_deprecate_positional_args
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                       sample_weight: Optional[np.ndarray] = None,
                       multioutput: Union[np.ndarray,
                                          Literal["raw_values", "uniform_average",
                                                  "variance_weighted"],
                                          None] = "uniform_average",
                       squared: bool = True) -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_squared_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput, squared=squared)


@_deprecate_positional_args
def mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                           sample_weight: Optional[np.ndarray] = None,
                           multioutput: Union[np.ndarray,
                                              Literal["raw_values", "uniform_average",
                                                      "variance_weighted"],
                                              None] = "uniform_average") -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_squared_log_error(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


@_deprecate_positional_args
def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, *,
                          multioutput: Union[np.ndarray,
                                             Literal["raw_values", "uniform_average",
                                                     "variance_weighted"],
                                             None] = "uniform_average",
                          sample_weight: Optional[np.ndarray] = None) -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.median_absolute_error(
        y_true=y_true, y_pred=y_pred, multioutput=multioutput,
        sample_weight=sample_weight)


@_deprecate_positional_args
def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                             sample_weight: Optional[np.ndarray] = None,
                             multioutput: Union[np.ndarray,
                                                Literal["raw_values", "uniform_average",
                                                        "variance_weighted"],
                                                None] = "uniform_average") -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.explained_variance_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


@_deprecate_positional_args
def r2_score(y_true: np.ndarray, y_pred: np.ndarray, *,
             sample_weight: Optional[np.ndarray] = None,
             multioutput: Union[np.ndarray, Literal["raw_values", "uniform_average",
                                                    "variance_weighted"],
                                None] = "uniform_average") -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.r2_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        multioutput=multioutput)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, None)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.max_error(y_true=y_true, y_pred=y_pred)


@_deprecate_positional_args
def mean_tweedie_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                          sample_weight: Optional[np.ndarray] = None,
                          power: float = 0) -> float:
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
    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.mean_tweedie_deviance(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, power=power)


@_deprecate_positional_args
def mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                          sample_weight: Optional[np.ndarray] = None) -> float:
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
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1)


@_deprecate_positional_args
def mean_gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray, *,
                        sample_weight: Optional[np.ndarray] = None) -> float:
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
