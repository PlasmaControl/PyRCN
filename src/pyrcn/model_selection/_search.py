"""SearchCV pipeline for incremental hyper-parameter search."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de> and
# Simon Stone <simon.stone@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import indexable, _check_method_params
from sklearn.model_selection._split import check_cv

import numpy as np
import time
from scipy import optimize
from collections.abc import Iterable

from typing import Union, Optional, Callable, Dict, Any, List, Tuple


class SequentialSearchCV(BaseSearchCV):
    """
    A series of searches on hyper-parameters.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Any object derived from ```sklearn.base.BaseEstimator```
        to be sequentially optimized.
    searches : Iterable
        Any ```Iterable``` that contains tuples of search steps.
    scoring : Union[str, Callable, list, tuple, dict, None], default=None
        Strategy to evaluate the performance of the cross-validated model
        on the test set.

        If ```scoring``` represents a single score, one can use:
        -a single string (see The scoring parameter:
        defining model evaluation rules);
        - a callable (see Defining your scoring strategy from metric functions)
        that returns a single value.

        If ```scoring``` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric names
        and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
    refit : bool, default = False
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a ```str``` denoting
        the scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in choosing a
        best estimator, refit can be set to a function which returns the
        selected ```best_index_``` given ```cv_results_```. In that case, the
        ```best_estimator_``` and ```best_params_``` will be set according to
        the returned ```best_index_``` while the ```best_score_``` attribute
        will not be available.

        The refitted estimator is made available at the ```best_estimator_```
        attribute and permits using predict directly on this ```GridSearchCV```
        instance.

        Also for multiple metric evaluation, the attributes ```best_index_```,
        ```best_score_``` and ```best_params_``` will only be available if
        refit is set and all of them will be determined w.r.t this specific
        scorer.

        See ```scoring``` parameter to know more about multiple metric
        evaluation.
    cv : Union[int, np.integer, Iterable, None], default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for ```cv``` are:
        - ```None```, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a ```(Stratified)KFold```,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ```y```
        is either binary or multiclass,
        ```sklearn.model_selection.StratifiedKFold``` is used.
        In all other cases,  ```sklearn.model_selection.KFold``` is used.
        These splitters are instantiated with ```shuffle=False``` so the splits
        will be the same across calls.
    verbose : Union[int, np.integer]
        Controls the verbosity: the higher, the more messages.
        - >1 : the computation time for each fold and parameter candidate is
        displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
        together with the starting time of the computation.
    pre_dispatch: Union[int, np.integer, str], default = '2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion
        of memory consumption when more jobs get dispatched than CPUs can
        process. This parameter can be:
        - None, in which case all the jobs are immediately created and spawned.
        Use this for lightweight and fast-running jobs, to avoid delays due to
        on-demand spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
    error_score: Union[int, float, np.integer, np.float],
    default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.  If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    """

    def __init__(self, estimator: BaseEstimator,
                 searches: list,
                 scoring: Union[str, Callable, list, tuple, dict, None] = None,
                 n_jobs: Union[int, np.integer, None] = None,
                 refit: bool = True,
                 cv: Union[int, np.integer, Iterable, None] = None,
                 verbose: Union[int, np.integer] = 0,
                 pre_dispatch: Union[int, np.integer, str] = '2*n_jobs',
                 error_score: Union[int, float] = np.nan) -> None:
        """Construct the SequentialSearchCV."""
        self.estimator: Optional[BaseEstimator] = None
        super().__init__(
            estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv,
            verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=True)
        self.searches = searches

    def _run_search(self, evaluate_candidates: Callable) -> None:
        """
        Run all the searches.

        Parameters
        ----------
        evaluate_candidates: Callable
        """
        evaluate_candidates(self.searches)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray], *,
            groups: Optional[np.ndarray] = None, **fit_params: Any)\
            -> SequentialSearchCV:
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_features) or (n_sequences)
            Training input.
        y : np.ndarray, shape=(n_samples, n_features) or shape=(n_samples, )
        or (n_sequences)
            Training target.
        groups : Optional[ndarray], shape=(n_samples, ), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            Only used in conjunction with a "Group" cv instance.
        **fit_params : Any
            Parameters passed to the ```fit``` method of the estimator.
        """
        def evaluate_candidates(searches: list) -> None:
            self.all_cv_results_: Dict[str, dict] = {}
            self.all_best_estimator_: Dict[str, BaseEstimator] = {}
            self.all_best_score_: Dict[str, Any] = {}
            self.all_best_params_: Dict[str, dict] = {}
            self.all_best_index_: Dict[str, int] = {}
            self.all_scorer_: Dict[str, Any] = {}
            self.all_n_splits_: Dict[str, int] = {}
            self.all_refit_time_: Dict[str, float] = {}
            self.all_multimetric_: Dict[str, bool] = {}
            for name, search, params, *kwargs in searches:
                if len(kwargs) == 1 and 'refit' in kwargs[0].keys():
                    result = search(
                        self.estimator, params, **kwargs[0]).fit(X, y)
                elif len(kwargs) == 1 and 'refit' not in kwargs[0].keys():
                    result = search(
                        self.estimator, params, refit=True, **kwargs[0])\
                        .fit(X, y)
                else:
                    result = search(
                        self.estimator, params, refit=True).fit(X, y)
                # Save the attributes of the intermediate search results
                self.all_cv_results_[name] = result.cv_results_
                self.all_best_estimator_[name] = result.best_estimator_
                self.all_best_score_[name] = result.best_score_
                self.all_best_params_[name] = result.best_params_
                self.all_best_index_[name] = result.best_index_
                self.all_scorer_[name] = result.scorer_
                self.all_n_splits_[name] = result.n_splits_
                self.all_refit_time_[name] = result.refit_time_
                self.all_multimetric_[name] = result.multimetric_
                self.estimator = result.best_estimator_
        self._run_search(evaluate_candidates)
        return self

    @property
    def cv_results_(self) -> dict:
        """
        A dict with keys as column headers and values as columns.

        It can be imported into a pandas DataFrame. For instance the below
        given table will be represented by a cv_results_ dict of:

        {
        'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                      mask = False),
        'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
        'split0_test_score'  : [0.80, 0.84, 0.70],
        'split1_test_score'  : [0.82, 0.50, 0.70],
        'mean_test_score'    : [0.81, 0.67, 0.70],
        'std_test_score'     : [0.01, 0.24, 0.00],
        'rank_test_score'    : [1, 3, 2],
        'split0_train_score' : [0.80, 0.92, 0.70],
        'split1_train_score' : [0.82, 0.55, 0.70],
        'mean_train_score'   : [0.81, 0.74, 0.70],
        'std_train_score'    : [0.01, 0.19, 0.00],
        'mean_fit_time'      : [0.73, 0.63, 0.43],
        'std_fit_time'       : [0.01, 0.02, 0.01],
        'mean_score_time'    : [0.01, 0.06, 0.04],
        'std_score_time'     : [0.00, 0.00, 0.00],
        'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
        }

        Note
        ----
        The key 'params' is used to store a list of parameter settings dicts
        for all the parameter candidates.

        The mean_fit_time, std_fit_time, mean_score_time and std_score_time
        are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the cv_results_ dict at the keys ending with that scorer’s
        name ('_<scorer_name>') instead of '_score' shown above.
        (‘split0_test_precision’, ‘mean_train_precision’ etc.)

        Returns
        -------
        dict
        """
        return self.all_cv_results_[self.searches[-1][0]]

    @property
    def best_estimator_(self) -> Any:
        """
        Estimator that was chosen by the search.

        I.e. estimator which gave highest score
        (or smallest loss if specified) on the left out data.
        Not available if refit=False.

        See refit parameter for more information on allowed values.

        Returns
        -------
        Estimator
        """
        if self.refit:
            return self.all_best_estimator_[self.searches[-1][0]]
        return None

    @property
    def best_score_(self) -> float:
        """
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is present only if refit is
        specified.

        This attribute is not available if refit is a function.

        Returns
        -------
        float
        """
        if self.refit:
            return self.all_best_score_[self.searches[-1][0]]
        return np.nan

    @property
    def best_params_(self) -> dict:
        """
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if refit is
        specified.

        Returns
        -------
        dict
        """
        if self.refit:
            return self.all_best_params_[self.searches[-1][0]]
        return {}

    @property
    def best_index_(self) -> Union[int, np.integer]:
        """
        The index (of the cv_results_ arrays) which corresponds to the best
        candidate.

        The dict at search.cv_results_['params'][search.best_index_] gives the
        parameter setting for the best model, that gives the highest mean score
        (search.best_score_).

        For multi-metric evaluation, this is present only if refit is
        specified.

        Returns
        -------
        Union[int, np.integer]
        """
        if self.refit:
            return self.all_best_index_[self.searches[-1][0]]
        return 0

    @property
    def scorer_(self) -> Dict:
        """
        Scorer function used on the held out data.

        To choose the best parameters for the model.

        For multi-metric evaluation, this attribute holds the validated scoring
        dict which maps the scorer key to the scorer callable.

        Returns
        -------
        function or a dict
        """
        return self.all_scorer_[self.searches[-1][0]]

    @property
    def n_splits_(self) -> Union[int, np.integer]:
        """
        The number of cross-validation splits (folds/iterations).

        Returns
        -------
        Union[int, np.integer]
        """
        return self.all_n_splits_[self.searches[-1][0]]

    @property
    def refit_time_(self) -> float:
        """
        Second used for refitting the best model on the whole dataset.

        Returns
        -------
        Union[int, np.integer]
        """
        return self.all_refit_time_[self.searches[-1][0]]

    @property
    def multimetric(self) -> bool:
        """
        Whether or not the scorers compute several metrics.

        Returns
        -------
        bool
        """
        return self.all_multimetric_[self.searches[-1][0]]


class SHGOSearchCV(BaseSearchCV):
    """
    Simplicial homology global optimization (SHGO) for hyper-parameters.

    SHGOSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are
    optimized by a guided cross-validated minimum search over the parameter
    settings.

    Parameters
    ----------
    estimator : estimator object
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    func : Callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.
    params : Dict
        Dictionary with parameters names (`str`) as keys and tuples ``(min,
        max)``, defining the lower and upper bounds for the optimizing
        argument of ``func``.
    args : Tuple, default=()
        Any additional fixed parameters needed to completely specify the
        objective function.
    constraints : Dict, List[Dict], None, default=None
        Constraints definitions, where each definition is a dictionary with
        fields:
            type : str
                Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
            fun : Callable
                The function defining the constraint.
            jac : Optional[Callable]
                The Jacobian of ``fun`` (only for SLSQP).
            args : List, Tuple
                Extra arguments to be passed to the function and Jacobian.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``SHGOSearchCV`` instance.
        evaluation.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.
        See ``refit`` parameter for more information on allowed values.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

    See Also
    --------
    GridSearchCV : Does exhaustive search over a grid of parameters.
    """

    def __init__(self, estimator: BaseEstimator, func: Callable, params: Dict,
                 *, args: Tuple = (),
                 constraints: Union[Dict, List, None] = None,
                 refit: bool = True, cv: Optional[int] = None,
                 return_train_score: bool = False) -> None:
        super().__init__(estimator=estimator, refit=refit, cv=cv,
                         return_train_score=return_train_score)
        self.func = func
        self.params = params
        self.args = args
        self.constraints = constraints

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, *,
            groups: Optional[np.ndarray] = None,
            **fit_params: dict) -> SHGOSearchCV:
        """
        Run the optimization based on the parameters defined before.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
             Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : np.ndarray of shape(n_samples, n_output) or (n_samples, ),
        default = None
            Target relative to X for classification or regression; None for
            unsupervised learning.
        groups : np.ndarray of shape(n_samples, ), default = None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        func = self.func

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_method_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)
        param_names = sorted(self.params)
        bounds = [self.params[name] for name in param_names]
        constraints = self.constraints
        train = [None] * n_splits
        test = [None] * n_splits
        for k, (tr, te) in enumerate(self.cv.split(X, y, groups)):
            train[k] = tr
            test[k] = te

        res = optimize.shgo(func=func, bounds=bounds, constraints=constraints,
                            args=(param_names, clone(base_estimator),
                                  X, y, train, test))

        result = {}
        for param_name, param_value in zip(param_names, res.x):
            result[param_name] = param_value
        self.best_params_ = result

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
        self.n_splits_ = n_splits

        return self
