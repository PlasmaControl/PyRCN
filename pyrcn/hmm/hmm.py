"""The :mod:`hmm` contains a HMM model."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

import sys
if sys.version_info >= (3, 8):
    from typing import Union, Optional, Any, Literal
else:
    from typing import Union, Optional, Any
    from typing_extensions import Literal
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args

import numpy as np


class _BaseHMM(BaseEstimator):
    """Base class for Hidden Markov Models (HMMs)."""

    @_deprecate_positional_args
    def __init__(self, *, n_components: int = 1,
                 startprob_prior: float = 1.0,
                 transmat_prior: float = 1.0,
                 algorithm: Literal["viterbi"] = "viterbi",
                 random_state: Union[int, np.random.RandomState, None] = None,
                 n_iter: int = 10, tol: float = 1e-2,
                 **kwargs: Any) -> None:
        """Construct the BaseHMM."""
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol

    def _init_model_parameter(self, X: np.ndarray,
                              y: Optional[np.ndarray] = None) -> None:
        self.startprob_ = np.full(shape=(self.n_components, ),
                                  fill_value=1/self.n_components)
        self.transmat_ = np.full(shape=(self.n_components, self.n_components),
                                 fill_value=1/self.n_components)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        X = check_array(X)
        self._init_model_parameters(X, y=y)
