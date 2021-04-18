"""
SearchCV pipeline for incremental hyper-parameter search
"""

# Authors: Simon Stone <simon.stone@tu-dresden.de>, Peter Steiner <peter.steiner@tu-dresden.de>
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
        self.results_ = {}
        self.best_estimators_ = {}
        self.best_scores_ = {}
        self.best_parameters_ = {}
        self.best_indices_ = {}
        self.scorers_ = {}
        self.numbers_of_splits_ = {}
        self.refit_times_ = {}
        self.multimetrics_ = {}

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        for step, (name, search, params, *kwargs) in enumerate(self.searches):
            if len(kwargs) == 1:
                result = search(self.estimator, params, refit=True, **kwargs[0]).fit(X, y, groups, **fit_params)
            else:
                result = search(self.estimator, params, refit=True).fit(X, y)
            # TODO
            """
            Add all attributes of the search result object as a dictionary to a results dictionary:
            self.results_ = { 'step1': results_dict_from_step1, 'step2': results_dict_from_step2, ...}       
            """
            self.results_['step'+str(step)] = result.cv_results_
            self.best_estimators_['step'+str(step)] = result.best_estimator_
            self.best_scores_['step'+str(step)] = result.best_score_
            self.best_parameters_['step'+str(step)] = result.best_params_
            self.best_indices_['step'+str(step)] = result.best_index_
            self.scorers_['step'+str(step)] = result.scorer_
            self.numbers_of_splits_['step'+str(step)] = result.n_splits_
            self.refit_times_['step'+str(step)] = result.refit_time_
            self.multimetrics_['step'+str(step)] = result.multimetric_

            self.estimator = result.best_estimator_
        if self.refit:
            self.best_estimator_ = self.estimator

        return self

