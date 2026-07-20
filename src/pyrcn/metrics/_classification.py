"""Metrics to assess performance on classification task given class prediction.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
import sklearn.metrics as sklearn_metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length


def _check_targets(y_true: np.ndarray, y_pred: np.ndarray,
                   sample_weight: np.ndarray | None = None) \
                       -> tuple[Literal["multilabel-indicator",
                                        "multiclass", "binary"],
                                np.ndarray | csr_matrix,
                                np.ndarray | csr_matrix,
                                np.ndarray | csr_matrix | None]:
    """
    Check that y_true and y_pred belong to the same classification task.

    This converts sequential types to  a common shape that can be handled by
    scikit-learn. It raises a ValueError if the conversion fails, e.g. due to
    different sequence lengths or a a mix of sequence-to-sequence and
    sequence-to-label tasks.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
    y_pred : np.ndarray, dtype=object
    sample_weight: Optional[np.ndarray], default=None

    Returns
    -------
    type_true : Literal["multilabel-indicator", "multiclass", "binary"]
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``.
    y_true : np.ndarray
    y_pred : np.ndarray
    sample_weight : np.ndarray
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
    return y_type, y_true, y_pred, sample_weight


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                   normalize: bool = True,
                   sample_weight: np.ndarray | None = None) -> float:
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See Also
    --------
    jaccard_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_score`` function.
    """
    # Compute accuracy for each possible representation
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.accuracy_score(y_true, y_pred, normalize=normalize,
                                          sample_weight=sample_weight)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                     labels: np.ndarray | None = None,
                     sample_weight: np.ndarray | None = None,
                     normalize: Literal["true", "predicted"] | None = None)\
                         -> np.ndarray:
    """Compute confusion matrix to evaluate classification accuracy.

    By definition a confusion matrix C is such that C[i, j] is equal to the
    number of observations known to be in group i and predicted to be in
    group j. Thus in binary classification, the count of true negatives is
    C[0, 0], false negatives is C[1, 0], true positives is C[1, 1] and false
    positives is C[0, 1].

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) target values.
    y_pred : np.ndarray, dtype=object
        Estimated targets as returned by a classifier.
    labels : Optional[np.ndarray] of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight :  Optional[np.ndarray], default=None
        Sample weights.
    normalize :  Optional[Literal["true", "predicted"]], default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted
        label being j-th class.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=labels,
        sample_weight=sample_weight, normalize=normalize)


def multilabel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                                sample_weight: np.ndarray | None = None,
                                labels: np.ndarray | None = None,
                                samplewise: bool = False) -> np.ndarray:
    """Compute a confusion matrix for each class or sample.

    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate classification accuracy, and output
    confusion matrices for each class or sample. In the multilabel confusion
    matrix MCM, the count of true negatives is MCM[:, 0, 0], false negatives
    is MCM[:, 1, 0], true positives is MCM[:, 1, 1] and false positives is
    MCM[:, 0, 1]. Multiclass data are treated as if binarized under a
    one-vs-rest transformation. Returned confusion matrices are in the order
    of sorted unique labels in the union of (y_true, y_pred).

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) target values.
    y_pred : np.ndarray, dtype=object
        Estimated targets as returned by a classifier.
    sample_weight :  Optional[np.ndarray], default=None
        Sample weights.
    labels : Optional[np.ndarray] of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.

    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results are returned in the order specified in ``labels``,
        otherwise the results are returned in sorted order by default.

    Notes
    -----
    The multilabel_confusion_matrix calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while confusion_matrix calculates
    one confusion matrix for confusion between every two classes.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.multilabel_confusion_matrix(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        labels=labels, samplewise=samplewise)


def cohen_kappa_score(y1: np.ndarray, y2: np.ndarray, *,
                      labels: np.ndarray | None = None,
                      weights: Literal["linear", "quadratic"] | None = None,
                      sample_weight: np.ndarray | None = None) -> float:
    """Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as kappa = (p_o - p_e) / (1 - p_e), where p_o is the empirical
    probability of agreement on the label assigned to any sample (the
    observed agreement ratio), and p_e is the expected agreement when both
    annotators assign labels randomly. p_e is estimated using a
    per-annotator empirical prior over the class labels.

    Parameters
    ----------
    y1 : np.ndarray, dtype=object
        Labels assigned by the first annotator.
    y2 : np.ndarray, dtype=object
        Labels assigned by the second annotator. The kappa statistic is
        symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.
    labels : Optional[np.ndarray] of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to select a
        subset of labels. If None, all labels that appear at least once in
        ``y1`` or ``y2`` are used.
    weights : Optional[Literal["linear", "quadratic"]], default=None
        Weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.
    sample_weight :  Optional[np.ndarray], default=None
        Sample weights.

    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    """
    y_type, y1, y2, sample_weight = _check_targets(y1, y2, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y1, y2, sample_weight)
    else:
        check_consistent_length(y1, y2)
    return sklearn_metrics.cohen_kappa_score(y1=y1, y2=y2,
                                             labels=labels, weights=weights,
                                             sample_weight=sample_weight)


def jaccard_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                  labels: np.ndarray | None = None,
                  pos_label: str | int = 1,
                  average: None | (Literal['micro', 'macro', 'samples',
                                   'weighted', 'binary']) = 'binary',
                  sample_weight: np.ndarray | None = None,
                  zero_division: Literal["warn", 0, 1] = "warn")\
        -> float | np.ndarray:
    """Jaccard similarity coefficient score.

    The Jaccard index, or Jaccard similarity coefficient, defined as the size
    of the intersection divided by the size of the union of two label sets,
    is used to compare the set of predicted labels for a sample to the
    corresponding set of labels in ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    labels : Optional[np.ndarray], default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : pos_label Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default='binary'
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data. With
        'binary', only results for the class specified by ``pos_label`` are
        reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which accounts for label
        imbalance. With 'samples', metrics are calculated for each instance
        and averaged (only meaningful for multilabel classification).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : {"warn", 0, 1}, default="warn"
        Sets the value to return when there is a zero division, i.e. when
        there are no negative values in predictions and labels. If set to
        "warn", this acts like 0, but a warning is also raised.

    Returns
    -------
    score : float or ndarray of floats
        A float if average is not None, otherwise an array of shape
        (n_unique_labels,).

    Notes
    -----
    The Jaccard score may be a poor metric if there are no positives for
    some samples or classes. Jaccard is undefined if there are no true or
    predicted labels, and this implementation returns a score of 0 with a
    warning.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.jaccard_score(
        y_true=y_true, y_pred=y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight,
        zero_division=zero_division)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray, *,
                      sample_weight: np.ndarray | None = None,) -> float:
    """Compute the Matthews correlation coefficient (MCC).

    The Matthews correlation coefficient is used as a measure of the quality
    of binary and multiclass classifications. It takes into account true and
    false positives and negatives and is generally regarded as a balanced
    measure which can be used even if the classes are of very different
    sizes. The MCC is in essence a correlation coefficient value between -1
    and +1. A coefficient of +1 represents a perfect prediction, 0 an
    average random prediction and -1 an inverse prediction. The statistic is
    also known as the phi coefficient. Binary and multiclass labels are
    supported. Only in the binary case does this relate to information about
    true and false positives and negatives.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Returns
    -------
    mcc : float
        The Matthews correlation coefficient (+1 represents a perfect
        prediction, 0 an average random prediction and -1 an inverse
        prediction).
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred,
                                             sample_weight=sample_weight)


def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray, *,
                  normalize: bool = True,
                  sample_weight: np.ndarray | None = None) -> float:
    """Zero-one classification loss.

    If normalize is ``True``, return the fraction of misclassifications
    (float), else return the number of misclassifications (int). The best
    performance is 0.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float or int
        If ``normalize == True``, return the fraction of misclassifications
        (float), else return the number of misclassifications (int).

    Notes
    -----
    In multilabel classification, the zero_one_loss function corresponds to
    the subset zero-one loss: for each sample, the entire set of labels must
    be correctly predicted, otherwise the loss for that sample is equal to
    one.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.zero_one_loss(y_true, y_pred, normalize=normalize,
                                         sample_weight=sample_weight)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *,
             labels: np.ndarray | None = None,
             pos_label: str | int = 1,
             average: None | (Literal['micro', 'macro', 'samples', 'weighted',
                              'binary']) = 'binary',
             sample_weight: np.ndarray | None = None,
             zero_division: Literal["warn", 0, 1] = "warn")\
        -> float | np.ndarray:
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a weighted average of precision and
    recall, where an F1 score reaches its best value at 1 and worst score at
    0. The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is
    F1 = 2 * (precision * recall) / (precision + recall). In the multi-class
    and multi-label case, this is the average of the F1 score of each class
    with weighting depending on the ``average`` parameter.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    labels : Optional[np.ndarray], default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default='binary'
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data.
        With 'binary', only results for the class specified by ``pos_label``
        are reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which can result in an
        F-score that is not between precision and recall. With 'samples',
        metrics are calculated for each instance and averaged (only
        meaningful for multilabel classification).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : Literal["warn", 0, 1], default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    f1_score : float or ndarray of floats
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task. An
        array of shape (n_unique_labels,) is returned if average is None.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric is set to 0, as is f-score, and
    ``UndefinedMetricWarning`` is raised. This behavior can be modified with
    ``zero_division``.
    """
    return fbeta_score(
        y_true, y_pred, beta=1, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight,
        zero_division=zero_division)


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float, *,
                labels: np.ndarray | None = None,
                pos_label: str | int = 1,
                average: None | (Literal['micro', 'macro', 'samples',
                                 'weighted', 'binary']) = 'binary',
                sample_weight: np.ndarray | None = None,
                zero_division: Literal["warn", 0, 1] = "warn")\
        -> float | np.ndarray:
    """Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0. The ``beta``
    parameter determines the weight of recall in the combined score.
    ``beta < 1`` lends more weight to precision, while ``beta > 1`` favors
    recall (``beta -> 0`` considers only precision, ``beta -> +inf`` only
    recall).

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    beta : float
        Determines the weight of recall in the combined score.
    labels : Optional[np.ndarray], default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default='binary'
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data.
        With 'binary', only results for the class specified by ``pos_label``
        are reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which can result in an
        F-score that is not between precision and recall. With 'samples',
        metrics are calculated for each instance and averaged (only
        meaningful for multilabel classification).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : Literal["warn", 0, 1], default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    fbeta_score : float or ndarray of floats
        F-beta score of the positive class in binary classification or
        weighted average of the F-beta score of each class for the
        multiclass task. An array of shape (n_unique_labels,) is returned if
        average is None.

    Notes
    -----
    When ``true positive + false positive == 0`` or
    ``true positive + false negative == 0``, f-score returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label,
        average=average, warn_for=('f-score',), sample_weight=sample_weight,
        zero_division=zero_division)
    return f


def precision_recall_fscore_support(y_true: np.ndarray, y_pred: np.ndarray, *,
                                    beta: float = 1.0,
                                    labels: np.ndarray | None = None,
                                    pos_label: str | int = 1,
                                    average: None | (Literal[
                                        'micro', 'macro', 'samples',
                                        'weighted', 'binary']) = 'binary',
                                    warn_for: tuple = ('precision', 'recall',
                                                       'f-score'),
                                    sample_weight: np.ndarray | None = None,
                                    zero_division: Literal["warn",
                                                           0, 1] = "warn")\
        -> tuple[float | np.ndarray, float | np.ndarray,
                 float | np.ndarray, np.ndarray | None]:
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number
    of true positives and ``fp`` the number of false positives. The
    precision is intuitively the ability of the classifier not to label as
    positive a sample that is negative. The recall is the ratio
    ``tp / (tp + fn)`` where ``tp`` is the number of true positives and
    ``fn`` the number of false negatives. The recall is intuitively the
    ability of the classifier to find all the positive samples. The F-beta
    score can be interpreted as a weighted harmonic mean of the precision
    and recall, where an F-beta score reaches its best value at 1 and worst
    score at 0. The F-beta score weights recall more than precision by a
    factor of ``beta``. ``beta == 1.0`` means recall and precision are
    equally important. The support is the number of occurrences of each
    class in ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object
        Ground truth (correct) labels.
    y_pred : np.ndarray, dtype=object
        Predicted labels, as returned by a classifier.
    beta : float, default=1.0
        The strength of recall versus precision in the F-score.
    labels : np.ndarray, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default=None
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data.
        With 'binary', only results for the class specified by ``pos_label``
        are reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which can result in an
        F-score that is not between precision and recall. With 'samples',
        metrics are calculated for each instance and averaged (only
        meaningful for multilabel classification).
    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : {"warn", 0, 1}, default="warn"
        Sets the value to return when there is a zero division: for recall
        when there are no positive labels, for precision when there are no
        positive predictions, and for f-score for both cases. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float or ndarray of floats
        A float if average is not None, otherwise an array of shape
        (n_unique_labels,).
    recall : float or ndarray of floats
        A float if average is not None, otherwise an array of shape
        (n_unique_labels,).
    fbeta_score : float or ndarray of floats
        A float if average is not None, otherwise an array of shape
        (n_unique_labels,).
    support : None or ndarray of ints
        The number of occurrences of each label in ``y_true``. None if
        average is not None, otherwise an array of shape (n_unique_labels,).

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric is set to 0, as is f-score, and
    ``UndefinedMetricWarning`` is raised. This behavior can be modified with
    ``zero_division``.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, beta=beta, labels=labels,
        pos_label=pos_label, average=average, warn_for=warn_for,
        sample_weight=sample_weight, zero_division=zero_division)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                    labels: np.ndarray | None = None,
                    pos_label: str | int = 1,
                    average: None | (Literal[
                        'micro', 'macro', 'samples', 'weighted', 'binary'])
                    = 'binary',
                    sample_weight: np.ndarray | None = None,
                    zero_division: Literal["warn", 0, 1] = "warn")\
        -> float | np.ndarray:
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number
    of true positives and ``fp`` the number of false positives. The
    precision is intuitively the ability of the classifier not to label as
    positive a sample that is negative. The best value is 1 and the worst
    value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default='binary'
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data.
        With 'binary', only results for the class specified by ``pos_label``
        are reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which can result in an
        F-score that is not between precision and recall. With 'samples',
        metrics are calculated for each instance and averaged (only
        meaningful for multilabel classification).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float or ndarray of floats
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task. An
        array of shape (n_unique_labels,) is returned if average is None.

    Notes
    -----
    When ``true positive + false positive == 0``, precision returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    p, _, _, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels, pos_label=pos_label,
        average=average, warn_for=('precision',), sample_weight=sample_weight,
        zero_division=zero_division)
    return p


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                 labels: np.ndarray | None = None,
                 pos_label: str | int = 1,
                 average: None | (Literal['micro', 'macro', 'samples',
                                  'weighted', 'binary']) = 'binary',
                 sample_weight: np.ndarray | None = None,
                 zero_division: Literal["warn", 0, 1] = "warn")\
        -> float | np.ndarray:
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive
    samples. The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : str or None, default='binary'
        One of {'micro', 'macro', 'samples', 'weighted', 'binary'} or None.
        This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data.
        With 'binary', only results for the class specified by ``pos_label``
        are reported (applicable only if targets are binary). With 'micro',
        metrics are calculated globally by counting the total true positives,
        false negatives and false positives. With 'macro', metrics are
        calculated for each label and their unweighted mean is taken, which
        does not take label imbalance into account. With 'weighted', metrics
        are calculated for each label and averaged, weighted by support (the
        number of true instances for each label), which can result in an
        F-score that is not between precision and recall. With 'samples',
        metrics are calculated for each instance and averaged (only
        meaningful for multilabel classification).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall : float or ndarray of floats
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task. An
        array of shape (n_unique_labels,) is returned if average is None.

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    _, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
        warn_for=('recall',), sample_weight=sample_weight,
        zero_division=zero_division)
    return r


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                            sample_weight: np.ndarray | None = None,
                            adjusted: bool = False) -> float:
    """Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems
    deals with imbalanced datasets. It is defined as the average of recall
    obtained on each class. The best value is 1 and the worst value is 0
    when ``adjusted=False``.

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.

    Returns
    -------
    balanced_accuracy : float
        The balanced accuracy score.

    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy.
    This definition is equivalent to accuracy_score with class-balanced
    sample weights, and shares desirable properties with the binary case.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.balanced_accuracy_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        adjusted=adjusted)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, *,
                          labels: np.ndarray | None = None,
                          target_names: np.ndarray | None = None,
                          sample_weight: np.ndarray | None = None,
                          digits: int = 2, output_dict: bool = False,
                          zero_division: Literal["warn", 0, 1] = "warn") \
                              -> str | dict:
    """Build a text report showing the main classification metrics.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.
    target_names : list of str of shape (n_labels,), default=None
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    output_dict : bool, default=False
        If True, return output as dict.
    zero_division : {"warn", 0, 1}, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    report : str or dict
        Text summary of the precision, recall, F1 score for each class. A
        dictionary is returned if output_dict is True, keyed by label with a
        nested dict of 'precision', 'recall', 'f1-score' and 'support' for
        each class. The reported averages include macro average (averaging
        the unweighted mean per label), weighted average (averaging the
        support-weighted mean per label), and sample average (only for
        multilabel classification). Micro average (averaging the total true
        positives, false negatives and false positives) is only shown for
        multi-label or multi-class with a subset of classes, because it
        corresponds to accuracy otherwise and would be the same for all
        metrics. Note that in binary classification, recall of the positive
        class is also known as "sensitivity"; recall of the negative class
        is "specificity".
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.classification_report(
        y_true=y_true, y_pred=y_pred, labels=labels, target_names=target_names,
        sample_weight=sample_weight, digits=digits, output_dict=output_dict,
        zero_division=zero_division)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray, *,
                 sample_weight: np.ndarray | None = None) -> float:
    """Compute the average Hamming loss.

    The Hamming loss is the fraction of labels that are incorrectly
    predicted.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float or int
        The average Hamming loss between elements of ``y_true`` and
        ``y_pred``.

    Notes
    -----
    In multiclass classification, the Hamming loss corresponds to the
    Hamming distance between ``y_true`` and ``y_pred`` which is equivalent
    to the subset ``zero_one_loss`` function, when the normalize parameter
    is set to True. In multilabel classification, the Hamming loss is
    different from the subset zero-one loss. The zero-one loss considers the
    entire set of labels for a given sample incorrect if it does not
    entirely match the true set of labels. Hamming loss is more forgiving in
    that it penalizes only the individual labels. The Hamming loss is
    upper-bounded by the subset zero-one loss, when the normalize parameter
    is set to True. It is always between 0 and 1, lower being better.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.hamming_loss(
        y_true, y_pred, sample_weight=sample_weight)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-15,
             normalize: bool = True,
             sample_weight: np.ndarray | None = None,
             labels: np.ndarray | None = None) -> float:
    """Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression and
    extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``. The log loss is only defined for two
    or more labels. For a single sample with true label y in {0, 1} and a
    probability estimate p = Pr(y = 1), the log loss is
    -(y * log(p) + (1 - y) * log(1 - p)).

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of floats
        Predicted probabilities of shape (n_samples, n_classes) or
        (n_samples,), as returned by a classifier's predict_proba method. If
        ``y_pred.shape = (n_samples,)`` the probabilities provided are
        assumed to be that of the positive class. The labels in ``y_pred``
        are assumed to be ordered alphabetically.
    eps : float, default=1e-15
        Log loss is undefined for p=0 or p=1, so probabilities are clipped to
        max(eps, min(1 - eps, p)).
    normalize : bool, default=True
        If true, return the mean loss per sample. Otherwise, return the sum
        of the per-sample losses.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    labels : array-like, default=None
        If not provided, labels are inferred from y_true. If ``labels`` is
        ``None`` and ``y_pred`` has shape (n_samples,) the labels are assumed
        to be binary and are inferred from ``y_true``.

    Returns
    -------
    loss : float
        The log loss.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.log_loss(
        y_true=y_true, y_pred=y_pred, eps=eps, normalize=normalize,
        sample_weight=sample_weight, labels=labels)


def hinge_loss(y_true: np.ndarray, pred_decision: np.ndarray, *,
               labels: np.ndarray | None = None,
               sample_weight: np.ndarray | None = None) -> float:
    """Average hinge loss (non-regularized).

    In the binary case, assuming labels in y_true are encoded with +1 and
    -1, when a prediction mistake is made, ``margin = y_true *
    pred_decision`` is always negative (since the signs disagree), implying
    ``1 - margin`` is always greater than 1. The cumulated hinge loss is
    therefore an upper bound of the number of mistakes made by the
    classifier. In the multiclass case, the function expects that either all
    the labels are included in y_true or an optional labels argument is
    provided which contains all the labels. The multilabel margin is
    calculated according to Crammer-Singer's method. As in the binary case,
    the cumulated hinge loss is an upper bound of the number of mistakes made
    by the classifier.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.
    pred_decision : array of shape (n_samples,) or (n_samples, n_classes)
        Predicted decisions, as output by decision_function (floats).
    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass hinge loss.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The average hinge loss.
    """
    y_type, y_true, pred_decision, sample_weight = _check_targets(
        y_true, pred_decision, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, pred_decision, sample_weight)
    else:
        check_consistent_length(y_true, pred_decision)
    return sklearn_metrics.hinge_loss(
        y_true=y_true, pred_decision=pred_decision, labels=labels,
        sample_weight=sample_weight)


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray, *,
                     sample_weight: np.ndarray | None = None,
                     pos_label: int | None = None) -> float:
    """Compute the Brier score loss.

    The smaller the Brier score loss, the better, hence the naming with
    "loss". The Brier score measures the mean squared difference between the
    predicted probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest possible
    difference between a predicted probability (which must be between zero
    and one) and the actual outcome (which can take on values of only 0 and
    1). It can be decomposed as the sum of refinement loss and calibration
    loss. The Brier score is appropriate for binary and categorical outcomes
    that can be structured as true or false, but is inappropriate for
    ordinal variables which can take on three or more values (this is because
    the Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter ``pos_label``, which defaults to
    the greater label unless ``y_true`` is all 0 or all -1, in which case
    ``pos_label`` defaults to 1.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets.
    y_prob : array of shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    pos_label : int or str, default=None
        Label of the positive class. ``pos_label`` is inferred as follows:
        if ``y_true`` is in {-1, 1} or {0, 1}, ``pos_label`` defaults to 1;
        else if ``y_true`` contains a string, an error is raised and
        ``pos_label`` should be explicitly specified; otherwise
        ``pos_label`` defaults to the greater label, i.e.
        ``np.unique(y_true)[-1]``.

    Returns
    -------
    score : float
        Brier score loss.
    """
    y_type, y_true, y_prob, sample_weight = _check_targets(
        y_true, y_prob, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_prob, sample_weight)
    else:
        check_consistent_length(y_true, y_prob)
    return sklearn_metrics.brier_score_loss(
        y_true=y_true, y_prob=y_prob, sample_weight=sample_weight,
        pos_label=pos_label)
