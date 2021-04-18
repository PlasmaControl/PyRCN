"""
SearchCV pipeline for incremental hyper-parameter search
"""

# Authors: Simon Stone <simon.stone@tu-dresden.de>
# License: BSD 3 clause


from sklearn.model_selection._search import BaseSearchCV
import numpy as np


class SequentialSearchCV(BaseSearchCV):
    """
    A series of searches on hyper-parameters.
    """
    def __init__(self, estimator, searches, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)
        self.searches = searches
        self.best_estimator_ = None

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
            self.cv_results_ = {}
            self.best_estimator_ = {}
            self.best_score_ = {}
            self.best_params_ = {}
            self.best_index_ = {}
            self.scorer_ = {}
            self.n_splits_ = {}
            self.refit_time_ = {}
            self.multimetric_ = {}
            for name, search, params, *kwargs in searches:
                if len(kwargs) == 1:
                    result = search(self.estimator, params, refit=True, **kwargs[0]).fit(X, y)
                else:
                    result = search(self.estimator, params, refit=True).fit(X, y)
                # Save the attributes of the intermediate search results
                # TODO: Is it possible to make a call to, e.g., self.cv_results_ return the final cv_results_?
                # If not, do we maybe need something like self.intermediate_cv_results_ to avoid confusion?
                self.cv_results_[name] = result.cv_results_
                self.best_estimator_[name] = result.best_estimator_
                self.best_score_[name] = result.best_score_
                self.best_params_[name] = result.best_params_
                self.best_index_[name] = result.best_index_
                self.scorer_[name] = result.scorer_
                self.n_splits_[name] = result.n_splits_
                self.refit_time_[name] = result.refit_time_
                self.multimetric_[name] = result.multimetric_

                self.estimator = result.best_estimator_
        self._run_search(evaluate_candidates)
        return self

