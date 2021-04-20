"""
SearchCV pipeline for incremental hyper-parameter search
"""

# Authors: Simon Stone <simon.stone@tu-dresden.de>, Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
import numpy as np


class GridEvaluationCV(BaseEstimator):
    """
    A train_test evaluation on a series of hyper-parameters
    """
    def __init__(self, estimator, params, scoring=None, n_jobs=None, refit=True, cv=None, verbose=None):
        super().__init__(self)
        self.estimator = estimator
        self.params = params



class SequentialSearchCV(BaseSearchCV):
    """
    A series of searches on hyper-parameters.
    """
    def __init__(self, estimator, searches, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)
        self.searches = searches
        # self.best_estimator_ = None

    def _run_search(self, evaluate_candidates):
        """ Run all the searches """
        evaluate_candidates(self.searches)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """
        TODO
        :param groups:
        :param X:
        :param y:
        :return:
        """
        def evaluate_candidates(searches):
            self.all_cv_results_ = {}
            self.all_best_estimator_ = {}
            self.all_best_score_ = {}
            self.all_best_params_ = {}
            self.all_best_index_ = {}
            self.all_scorer_ = {}
            self.all_n_splits_ = {}
            self.all_refit_time_ = {}
            self.all_multimetric_ = {}
            for name, search, params, *kwargs in searches:
                if len(kwargs) == 1:
                    result = search(self.estimator, params, refit=True, **kwargs[0]).fit(X, y)
                else:
                    result = search(self.estimator, params, refit=True).fit(X, y)
                # Save the attributes of the intermediate search results
                # TODO: Should we add a flag to just keep the results of the final optimization step?
                # This would make the object smaller but we cannot check plausibility of previous optimization steps.
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
    def cv_results_(self):
        """
        A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.

        For instance the below given table


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

        NOTE

        The key 'params' is used to store a list of parameter settings dicts for all the parameter candidates.

        The mean_fit_time, std_fit_time, mean_score_time and std_score_time are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are available in the cv_results_ dict at the keys ending with that scorer’s name ('_<scorer_name>') instead of '_score' shown above. (‘split0_test_precision’, ‘mean_train_precision’ etc.)

                Returns
                -------
                dict
        """
        return self.all_cv_results_[self.searches[-1][0]]

    @property
    def best_estimator_(self):
        """
        Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data. Not available if refit=False.

        See refit parameter for more information on allowed values.

        Returns
        -------
        Estimator
        """
        if self.refit:
            return self.all_best_estimator_[self.searches[-1][0]]

    @property
    def best_score_(self):
        """
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if refit is specified.

        This attribute is not available if refit is a function.

        Returns
        -------
        float
        """
        if self.refit:
            return self.all_best_score_[self.searches[-1][0]]

    @property
    def best_params_(self):
        """
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if refit is specified.

        Returns
        -------
        dict
        """
        if self.refit:
            return self.all_best_params_[self.searches[-1][0]]

    @property
    def best_index_(self):
        """
        The index (of the cv_results_ arrays) which corresponds to the best candidate parameter setting.

        The dict at search.cv_results_['params'][search.best_index_] gives the parameter setting for the best model, that gives the highest mean score (search.best_score_).

        For multi-metric evaluation, this is present only if refit is specified.

        Returns
        -------
        int
        """
        if self.refit:
            return self.all_best_index_[self.searches[-1][0]]

    @property
    def scorer_(self):
        """
        Scorer function used on the held out data to choose the best parameters for the model.

        For multi-metric evaluation, this attribute holds the validated scoring dict which maps the scorer key to the scorer callable.

        Returns
        -------
        function or a dict
        """
        return self.all_scorer_[self.searches[-1][0]]

    @property
    def n_splits_(self):
        """
        The number of cross-validation splits (folds/iterations).

        Returns
        -------
        int
        """
        return self.all_n_splits_[self.searches[-1][0]]

    @property
    def refit_time_(self):
        """
        Seconds used for refitting the best model on the whole dataset.

        Returns
        -------
        float
        """
        return self.all_refit_time_[self.searches[-1][0]]

    @property
    def multimetric(self):
        """
        Whether or not the scorers compute several metrics.

        Returns
        -------
        bool
        """
        return self.all_multimetric_[self.searches[-1][0]]
