"""The :mod:`viterbi_decoder` contains a Viterbi decoder."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args

import numpy as np

from typing import Optional


class ViterbiDecoder(BaseEstimator):
    """Class for a Viterbi Decoder."""

    @_deprecate_positional_args
    def __init__(self) -> None:
        """Construct the BaseHMM."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        X = check_array(X)
        self._init_model_parameters(X, y=y)
