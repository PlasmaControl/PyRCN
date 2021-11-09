"""Metrics to assess performance on classification task given class prediction.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause
import sys
if sys.version_info >= (3, 8):
    from typing import Tuple, Union, Optional, Dict, Literal
else:
    from typing_extensions import Literal
    from typing import Tuple, Union, Optional, Dict
import numpy as np

from sklearn.utils.validation import check_consistent_length, _deprecate_positional_args
import sklearn.metrics as sklearn_metrics
from sklearn.metrics._classification import _check_targets as sklearn_check_targets


def _check_targets(y_true: np.ndarray, y_pred: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None) \
                       -> Tuple[Literal["multilabel-indicator", "multiclass", "binary"],
                                np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Check that y_true and y_pred belong to the same classification task.

    This converts sequential types to  a common shape that can be handled by
    scikit-learn. It raises a ValueError if the conversion fails, e.g. due to
    different sequence lengths or a a mix of sequence-to-sequence and sequence-to-label
    tasks.

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
    y_type, y_true, y_pred = sklearn_check_targets(y_true, y_pred)
    return y_type, y_true, y_pred, sample_weight


@_deprecate_positional_args
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, *, normalize: bool = True,
                   sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.

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


@_deprecate_positional_args
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                     labels: Optional[np.ndarray] = None,
                     sample_weight: Optional[np.ndarray] = None,
                     normalize: Optional[Literal["true", "predicted"]] = None)\
                         -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.
    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    Read more in the :ref:`User Guide <confusion_matrix>`.

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
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.

    See Also
    --------
    plot_confusion_matrix : Plot Confusion Matrix.
    ConfusionMatrixDisplay : Confusion Matrix visualization.
    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
           (Wikipedia and other references may use a different
           convention for axes).
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels,
                                            sample_weight=sample_weight,
                                            normalize=normalize)


@_deprecate_positional_args
def multilabel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                                sample_weight: Optional[np.ndarray] = None,
                                labels: Optional[np.ndarray] = None,
                                samplewise: bool = False) -> np.ndarray:
    """
    Compute a confusion matrix for each class or sample.

    .. versionadded:: 0.21
    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.
    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.
    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).
    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

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
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default

    See Also
    --------
    confusion_matrix

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
    return sklearn_metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred,
                                                       sample_weight=sample_weight,
                                                       labels=labels,
                                                       samplewise=samplewise)


@_deprecate_positional_args
def cohen_kappa_score(y1: np.ndarray, y2: np.ndarray, *,
                      labels: Optional[np.ndarray] = None,
                      weights: Optional[Literal["linear", "quadratic"]] = None,
                      sample_weight: Optional[np.ndarray] = None) -> float:
    r"""
    Cohen' kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as
    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)
    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.
    Read more in the :ref:`User Guide <cohen_kappa>`.

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

    References
    ----------
    .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
           Educational and Psychological Measurement 20(1):37-46.
           doi:10.1177/001316446002000104.
    .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
           computational linguistics". Computational Linguistics 34(4):555-596
           <https://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-034-R2>`_.
    .. [3] `Wikipedia entry for the Cohen's kappa
            <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_.
    """
    y_type, y1, y2, sample_weight = _check_targets(y1, y2, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y1, y2, sample_weight)
    else:
        check_consistent_length(y1, y2)
    return sklearn_metrics.cohen_kappa_score(y1=y1, y2=y2,
                                             labels=labels, weights=weights,
                                             sample_weight=sample_weight)


@_deprecate_positional_args
def jaccard_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                  labels: Optional[np.ndarray] = None, pos_label: Union[str, int] = 1,
                  average: Optional[Literal['micro', 'macro', 'samples', 'weighted',
                                            'binary']] = 'binary',
                  sample_weight: Optional[np.ndarray] = None,
                  zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """
    Jaccard similarity coefficient score.

    The Jaccard index [1], or Jaccard similarity coefficient, defined as
    the size of the intersection divided by the size of the union of two label
    sets, is used to compare set of predicted labels for a sample to the
    corresponding set of labels in ``y_true``.
    Read more in the :ref:`User Guide <jaccard_similarity_score>`.

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
    average : Optional[Literal['micro', 'macro', 'samples', 'weighted', 'binary']],
    default='binary'
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : Literal["warn", 0.0, 1.0], default="warn"
        Sets the value to return when there is a zero division, i.e. when there
        there are no negative values in predictions and labels. If set to
        "warn", this acts like 0, but a warning is also raised.

    Returns
    -------
    score : float (if average is not None) or array of floats, shape =
    [n_unique_labels]

    See Also
    --------
    accuracy_score, f_score, multilabel_confusion_matrix

    Notes
    -----
    :func:`jaccard_score` may be a poor metric if there are no
    positives for some samples or classes. Jaccard is undefined if there are
    no true or predicted labels, and our implementation will return a score
    of 0 with a warning.

    References
    ----------
    .. [1] `Wikipedia entry for the Jaccard index
           <https://en.wikipedia.org/wiki/Jaccard_index>`_.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.jaccard_score(y_true=y_true, y_pred=y_pred, labels=labels,
                                         pos_label=pos_label, average=average,
                                         sample_weight=sample_weight,
                                         zero_division=zero_division)


@_deprecate_positional_args
def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray, *,
                      sample_weight: Optional[np.ndarray] = None,) -> float:
    """
    Compute the Matthews correlation coefficient (MCC).

    The Matthews correlation coefficient is used in machine learning as a
    measure of the quality of binary and multiclass classifications. It takes
    into account true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even if the classes are of
    very different sizes. The MCC is in essence a correlation coefficient value
    between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
    an average random prediction and -1 an inverse prediction.  The statistic
    is also known as the phi coefficient. [source: Wikipedia]
    Binary and multiclass labels are supported.  Only in the binary case does
    this relate to information about true and false positives and negatives.
    See references below.
    Read more in the :ref:`User Guide <matthews_corrcoef>`.

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
        prediction, 0 an average random prediction and -1 and inverse
        prediction).

    References
    ----------
    .. [1] `Baldi, Brunak, Chauvin, Andersen and Nielsen, (2000). Assessing the
       accuracy of prediction algorithms for classification: an overview
       <https://doi.org/10.1093/bioinformatics/16.5.412>`_.
    .. [2] `Wikipedia entry for the Matthews Correlation Coefficient
       <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_.
    .. [3] `Gorodkin, (2004). Comparing two K-category assignments by a
        K-category correlation coefficient
        <https://www.sciencedirect.com/science/article/pii/S1476927104000799>`_.
    .. [4] `Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
        Error Measures in MultiClass Prediction
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882>`_.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred,
                                             sample_weight=sample_weight)


@_deprecate_positional_args
def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray, *, normalize: bool = True,
                  sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Zero-one classification loss.

    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.
    Read more in the :ref:`User Guide <zero_one_loss>`.

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
    loss : float or int,
        If ``normalize == True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).

    Notes
    -----
    In multilabel classification, the zero_one_loss function corresponds to
    the subset zero-one loss: for each sample, the entire set of labels must be
    correctly predicted, otherwise the loss for that sample is equal to one.

    See Also
    --------
    accuracy_score, hamming_loss, jaccard_score
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.zero_one_loss(y_true, y_pred, normalize=normalize,
                                         sample_weight=sample_weight)


@_deprecate_positional_args
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *,
             labels: Optional[np.ndarray] = None, pos_label: Union[str, int] = 1,
             average: Optional[Literal['micro', 'macro', 'samples', 'weighted',
                                       'binary']] = 'binary',
             sample_weight: Optional[np.ndarray] = None,
             zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """
    Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

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
        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.
    pos_label : Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : Optional[Literal['micro', 'macro', 'samples','weighted', 'binary']],
    default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : Literal["warn", 0, 1], default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.

    See Also
    --------
    fbeta_score, precision_recall_fscore_support, jaccard_score,
    multilabel_confusion_matrix

    References
    ----------
    .. [1] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    return fbeta_score(y_true, y_pred, beta=1, labels=labels, pos_label=pos_label,
                       average=average, sample_weight=sample_weight,
                       zero_division=zero_division)


@_deprecate_positional_args
def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float, *,
                labels: Optional[np.ndarray] = None, pos_label: Union[str, int] = 1,
                average: Optional[Literal['micro', 'macro', 'samples',
                                          'weighted', 'binary']] = 'binary',
                sample_weight: Optional[np.ndarray] = None,
                zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """
    Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.
    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

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
        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.
    pos_label : Union[str, int], default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : Optional[Literal['micro', 'macro', 'samples','weighted', 'binary']],
    default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : Literal["warn", 0, 1], default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    fbeta_score : float (if average is not None) or array of float, shape =
    [n_unique_labels]
        F-beta score of the positive class in binary classification or weighted
        average of the F-beta score of each class for the multiclass task.

    See Also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix
    Notes
    -----
    When ``true positive + false positive == 0`` or
    ``true positive + false negative == 0``, f-score returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.
    References
    ----------
    .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.
    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.
    """
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta,
                                                 labels=labels, pos_label=pos_label,
                                                 average=average, warn_for=('f-score',),
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division)
    return f


@_deprecate_positional_args
def precision_recall_fscore_support(y_true: np.ndarray, y_pred: np.ndarray, *,
                                    beta: float = 1.0,
                                    labels: Optional[np.ndarray] = None,
                                    pos_label: Union[str, int] = 1,
                                    average: Optional[Literal['micro', 'macro',
                                                              'samples', 'weighted',
                                                              'binary']] = 'binary',
                                    warn_for: Tuple = ('precision', 'recall',
                                                       'f-score'),
                                    sample_weight: Optional[np.ndarray] = None,
                                    zero_division: Literal["warn", 0, 1] = "warn")\
                                        -> Tuple[float, float, float, None]:
    """
    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.
    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.
    The support is the number of occurrences of each class in ``y_true``.
    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

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
    average : Optional[Literal['micro', 'macro', 'samples', 'weighted', 'binary']],
    default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both
        If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =
    [n_unique_labels]
    recall : float (if average is not None) or array of float, , shape =
    [n_unique_labels]
    fbeta_score : float (if average is not None) or array of float, shape =
    [n_unique_labels]
    support : None (if average is not None) or array of int, shape =
    [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_.
    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.
    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                           beta=beta, labels=labels,
                                                           pos_label=pos_label,
                                                           average=average,
                                                           warn_for=warn_for,
                                                           sample_weight=sample_weight,
                                                           zero_division=zero_division)


@_deprecate_positional_args
def precision_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                    labels: Optional[np.ndarray] = None,
                    pos_label: Union[str, int] = 1,
                    average: Optional[Literal['micro', 'macro', 'samples', 'weighted',
                                              'binary']] = 'binary',
                    sample_weight: Optional[np.ndarray] = None,
                    zero_division: Literal["warn", 0, 1] = "warn") \
                        -> Union[float, np.ndarray]:
    """
    Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The best value is 1 and the worst value is 0.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

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
        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'}
    default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float of shape
        (n_unique_labels,)
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.

    See Also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix

    Notes
    -----
    When ``true positive + false positive == 0``, precision returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.
    """
    p, _, _, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                 labels=labels, pos_label=pos_label,
                                                 average=average,
                                                 warn_for=('precision',),
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division)
    return p


@_deprecate_positional_args
def recall_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                 labels: Optional[np.ndarray] = None, pos_label: Union[str, int] = 1,
                 average: Optional[Literal['micro', 'macro', 'samples', 'weighted',
                                           'binary']] = 'binary',
                 sample_weight: Optional[np.ndarray] = None,
                 zero_division: Literal["warn", 0, 1] = "warn") \
                     -> Union[float, np.ndarray]:
    """
    Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

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
        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'}, default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall : float (if average is not None) or array of float of shape
        (n_unique_labels,)
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.
    See Also
    --------
    precision_recall_fscore_support, balanced_accuracy_score,
    multilabel_confusion_matrix

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 pos_label=pos_label, average=average,
                                                 warn_for=('recall',),
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division)
    return r


@_deprecate_positional_args
def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                            sample_weight: Optional[np.ndarray] = None,
                            adjusted: bool = False) -> float:
    """
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    .. versionadded:: 0.20

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

    See Also
    --------
    recall_score, roc_auc_score

    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.

    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred,
                                                   sample_weight=sample_weight,
                                                   adjusted=adjusted)


@_deprecate_positional_args
def classification_report(y_true: np.ndarray, y_pred: np.ndarray, *,
                          labels: Optional[np.ndarray] = None,
                          target_names: Optional[np.ndarray] = None,
                          sample_weight: Optional[np.ndarray] = None,
                          digits: int = 2, output_dict: bool = False,
                          zero_division: Literal["warn", 0, 1] = "warn") \
                              -> Union[str, Dict]:
    """
    Build a text report showing the main classification metrics.

    Read more in the :ref:`User Guide <classification_report>`.

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
        .. versionadded:: 0.20
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy
        otherwise and would be the same for all metrics.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".

    See Also
    --------
    precision_recall_fscore_support, confusion_matrix,
    multilabel_confusion_matrix
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.classification_report(y_true=y_true, y_pred=y_pred,
                                                 labels=labels,
                                                 target_names=target_names,
                                                 sample_weight=sample_weight,
                                                 digits=digits, output_dict=output_dict,
                                                 zero_division=zero_division)


@_deprecate_positional_args
def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray, *,
                 sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Compute the average Hamming loss.

    The Hamming loss is the fraction of labels that are incorrectly predicted.
    Read more in the :ref:`User Guide <hamming_loss>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        .. versionadded:: 0.18

    Returns
    -------
    loss : float or int
        Return the average Hamming loss between element of ``y_true`` and
        ``y_pred``.

    See Also
    --------
    accuracy_score, jaccard_score, zero_one_loss

    Notes
    -----
    In multiclass classification, the Hamming loss corresponds to the Hamming
    distance between ``y_true`` and ``y_pred`` which is equivalent to the
    subset ``zero_one_loss`` function, when `normalize` parameter is set to
    True.
    In multilabel classification, the Hamming loss is different from the
    subset zero-one loss. The zero-one loss considers the entire set of labels
    for a given sample incorrect if it does not entirely match the true set of
    labels. Hamming loss is more forgiving in that it penalizes only the
    individual labels.
    The Hamming loss is upperbounded by the subset zero-one loss, when
    `normalize` parameter is set to True. It is always between 0 and 1,
    lower being better.

    References
    ----------
    .. [1] Grigorios Tsoumakas, Ioannis Katakis. Multi-Label Classification:
           An Overview. International Journal of Data Warehousing & Mining,
           3(3), 1-13, July-September 2007.
    .. [2] `Wikipedia entry on the Hamming distance
           <https://en.wikipedia.org/wiki/Hamming_distance>`_.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.hamming_loss(y_true, y_pred, sample_weight=sample_weight)


@_deprecate_positional_args
def log_loss(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-15,
             normalize: bool = True, sample_weight: Optional[np.ndarray] = None,
             labels: Optional[np.ndarray] = None) -> float:
    r"""
    Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    and a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:
    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))
    Read more in the :ref:`User Guide <log_loss>`.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.
    eps : float, default=1e-15
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).
    normalize : bool, default=True
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.
        .. versionadded:: 0.18

    Returns
    -------
    loss : float

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    """
    y_type, y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_pred, sample_weight)
    else:
        check_consistent_length(y_true, y_pred)
    return sklearn_metrics.log_loss(y_true=y_true, y_pred=y_pred, eps=eps,
                                    normalize=normalize, sample_weight=sample_weight,
                                    labels=labels)


@_deprecate_positional_args
def hinge_loss(y_true: np.ndarray, pred_decision: np.ndarray, *,
               labels: Optional[np.ndarray] = None,
               sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Average hinge loss (non-regularized).

    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * pred_decision`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.  The cumulated hinge loss is therefore an upper
    bound of the number of mistakes made by the classifier.
    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method. As in the binary case, the cumulated hinge loss
    is an upper bound of the number of mistakes made by the classifier.
    Read more in the :ref:`User Guide <hinge_loss>`.

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

    References
    ----------
    .. [1] `Wikipedia entry on the Hinge loss
           <https://en.wikipedia.org/wiki/Hinge_loss>`_.
    .. [2] Koby Crammer, Yoram Singer. On the Algorithmic
           Implementation of Multiclass Kernel-based Vector
           Machines. Journal of Machine Learning Research 2,
           (2001), 265-292.
    .. [3] `L1 AND L2 Regularization for Multiclass Hinge Loss Models
           by Robert C. Moore, John DeNero
           <http://www.ttic.edu/sigml/symposium2011/papers/
           Moore+DeNero_Regularization.pdf>`_.
    """
    y_type, y_true, pred_decision, sample_weight = _check_targets(
        y_true, pred_decision, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, pred_decision, sample_weight)
    else:
        check_consistent_length(y_true, pred_decision)
    return sklearn_metrics.hinge_loss(y_true=y_true, pred_decision=pred_decision,
                                      labels=labels, sample_weight=sample_weight)


@_deprecate_positional_args
def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray, *,
                     sample_weight: Optional[np.ndarray] = None,
                     pos_label: Optional[int] = None) -> float:
    """
    Compute the Brier score loss.

    The smaller the Brier score loss, the better, hence the naming with "loss".
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed is the sum of refinement loss and
    calibration loss.
    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.
    Read more in the :ref:`User Guide <brier_score_loss>`.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets.
    y_prob : array of shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    pos_label : int or str, default=None
        Label of the positive class. `pos_label` will be infered in the
        following manner:
        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitely specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

    Returns
    -------
    score : float
        Brier score loss.

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.
    """
    y_type, y_true, y_prob, sample_weight = _check_targets(
        y_true, y_prob, sample_weight)
    if sample_weight is not None:
        check_consistent_length(y_true, y_prob, sample_weight)
    else:
        check_consistent_length(y_true, y_prob)
    return sklearn_metrics.brier_score_loss(y_true=y_true, y_prob=y_prob,
                                            sample_weight=sample_weight,
                                            pos_label=pos_label)
