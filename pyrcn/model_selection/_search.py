"""SearchCV pipeline for incremental hyper-parameter search."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# Simon Stone <simon.stone@tu-dresden.de>
# License: BSD 3 clause
import sys
if sys.version_info >= (3, 8):
    from typing import Union, Optional, Callable, Dict, Any, Literal
else:
    from typing import Union, Optional, Callable, Dict, Any
    from typing_extensions import Literal
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from collections.abc import Iterable


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
        -a single string (see The scoring parameter: defining model evaluation rules);
        - a callable (see Defining your scoring strategy from metric functions)
        that returns a single value.

        If ```scoring``` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric names and
        the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
    refit : bool, default = False
        Refit an estimator using the best found parameters on the whole dataset.

        For multiple metric evaluation, this needs to be a ```str``` denoting the scorer
        that would be used to find the best parameters for refitting the estimator
        at the end.

        Where there are considerations other than maximum score in choosing a best
        estimator, refit can be set to a function which returns the selected
        ```best_index_``` given ```cv_results_```. In that case, the
        ```best_estimator_``` and ```best_params_``` will be set according to the
        returned ```best_index_``` while the ```best_score_``` attribute will
        not be available.

        The refitted estimator is made available at the ```best_estimator_``` attribute
        and permits using predict directly on this ```GridSearchCV``` instance.

        Also for multiple metric evaluation, the attributes ```best_index_```,
        ```best_score_``` and ```best_params_``` will only be available if refit is set
        and all of them will be determined w.r.t this specific scorer.

        See ```scoring``` parameter to know more about multiple metric evaluation.
    cv : Union[int, np.integer, Iterable, None], default=None
        Determines the cross-validation splitting strategy. Possible inputs for ```cv```
        are:
        - ```None```, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a ```(Stratified)KFold```,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ```y``` is either
        binary or multiclass, ```sklearn.model_selection.StratifiedKFold``` is used.
        In all other cases,  ```sklearn.model_selection.KFold``` is used.
        These splitters are instantiated with ```shuffle=False``` so the splits will be
        the same across calls.
    verbose : Union[int, np.integer]
        Controls the verbosity: the higher, the more messages.
        - >1 : the computation time for each fold and parameter candidate is displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed together with
        the starting time of the computation.
    pre_dispatch: Union[int, np.integer, str], default = '2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        Reducing this number can be useful to avoid an explosion of memory consumption
        when more jobs get dispatched than CPUs can process. This parameter can be:
        - None, in which case all the jobs are immediately created and spawned.
        Use this for lightweight and fast-running jobs, to avoid delays due to on-demand
        spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
    error_score: Union[Literal['raise'], int, float, np.integer, np.float],
    default=np.nan
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.  If a numeric value is given, FitFailedWarning
        is raised. This parameter does not affect the refit step,
        which will always raise the error.
    """

    def __init__(self, estimator: BaseEstimator,
                 searches: list,
                 scoring: Union[str, Callable, list, tuple, dict, None] = None,
                 n_jobs: Union[int, np.integer, None] = None,
                 refit: bool = True,
                 cv: Union[int, np.integer, Iterable, None] = None,
                 verbose: Union[int, np.integer] = 0,
                 pre_dispatch: Union[int, np.integer, str] = '2*n_jobs',
                 error_score: Union[Literal['raise'], int,
                                    float, np.integer] = np.nan) -> None:
        """Construct the SequentialSearchCV."""
        self.estimator: Optional[BaseEstimator] = None
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv,
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

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None,
            **fit_params: Any) -> BaseSearchCV:
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
                    result = search(self.estimator, params, **kwargs[0]).fit(X, y)
                elif len(kwargs) == 1 and 'refit' not in kwargs[0].keys():
                    result = search(self.estimator, params, refit=True,
                                    **kwargs[0]).fit(X, y)
                else:
                    result = search(self.estimator, params, refit=True).fit(X, y)
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

        It can be imported into a pandas DataFrame. For instance the below given table
        will be represented by a cv_results_ dict of:

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

        For multi-metric evaluation, the scores for all the scorers are available in the
        cv_results_ dict at the keys ending with that scorer’s name ('_<scorer_name>')
        instead of '_score' shown above.
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

        For multi-metric evaluation, this is present only if refit is specified.

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

        For multi-metric evaluation, this is present only if refit is specified.

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
        The index (of the cv_results_ arrays) which corresponds to the best candidate.

        The dict at search.cv_results_['params'][search.best_index_] gives the parameter
        setting for the best model, that gives the highest mean score
        (search.best_score_).

        For multi-metric evaluation, this is present only if refit is specified.

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

        For multi-metric evaluation, this attribute holds the validated scoring dict
        which maps the scorer key to the scorer callable.

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
