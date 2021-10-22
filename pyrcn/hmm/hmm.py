from sklearn.base import BaseEstimator
from sklearn.utils import check_array

import numpy as np


class _BaseHMM(BaseEstimator):
    """
    Base class for Hidden Markov Models (HMMs).


    """
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None, n_iter=10, tol=1e-2, **kwargs):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol

    def _init_model_parameter(X,y=None):
        self.startprob_ = np.full(shape=(self.n_components, ), 
                                  fill_value=1/self.n_components)
        self.transmat_ = np.full(shape=(self.n_components, self.n_components), 
                                 fill_value=1/self.n_components)


    def fit(self, X, y=None):
        X = check_array(X)
        self._init_model_parameters(X, y=y)

        for k in range(self.n_iter):
            costs = {"state_cost": np.zeros(self.n_components),
                     "transition_cost": np.zeros((self.n_components, self.n_components))}
