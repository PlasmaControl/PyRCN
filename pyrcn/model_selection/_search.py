"""
SearchCV pipeline for incremental hyper-parameter search
"""

# Authors: Simon Stone <simon.stone@tu-dresden.de>
# License: BSD 3 clause


from sklearn.model_selection._search import BaseSearchCV
import numpy as np


class SequentialSearch(BaseSearchCV):
    """
    A series of searches on hyper-parameters.
    """
    def __init__(self, estimator, searches, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)
        self.searches = searches
        self.best_estimator_ = None

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """
        TODO
        :param groups:
        :param X:
        :param y:
        :return:
        """
        for name, search, params, *kwargs in self.searches:
            if len(kwargs) == 1:
                result = search(self.estimator, params, refit=True, **kwargs[0]).fit(X, y)
            else:
                result = search(self.estimator, params, refit=True).fit(X, y)
                # TODO
                """
                Add all attributes of the search result object as a dictionary to a results dictionary:
                self.results = { 'step1': results_dict_from_step1, 'step2': results_dict_from_step2, ...}       
                """

            self.estimator = result.best_estimator_
        self.best_estimator_ = self.estimator

        return self

