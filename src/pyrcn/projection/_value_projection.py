"""The :mod:`value_projection` contains the MatrixToValueProjection."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

from __future__ import annotations
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info >= (3, 8):
    from typing import Literal, cast
else:
    from typing_extensions import Literal
    from typing import cast


class MatrixToValueProjection(BaseEstimator, TransformerMixin):
    """
    Projection of a matrix to any kind of indices, e.g. of the maximum value.

    Parameters
    ----------
    output_strategy : Literal["winner_takes_all", "median", "last_value"],
    default=winner_takes_all"
        Strategy utilized to compute the index
    needs_proba : bool, default=False
        Whether to return a probability estimate or the index.
    """

    def __init__(self, output_strategy: Literal[
        "winner_takes_all", "median", "last_value"] = "winner_takes_all",
                 needs_proba: bool = False):
        """Construct the MatrixToValueProjection."""
        self._output_strategy = output_strategy
        self._needs_proba = needs_proba

    def fit(self, X: np.ndarray, y: None = None) -> MatrixToValueProjection:
        """
        Fit the MatrixToValueProjection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or of shape (n_samples, )
        y : None
            Ignored.

        Returns
        -------
        self : Returns a trained MatrixToValueProjection model.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform matrix to a value as defined.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or of shape (n_samples, )

        Returns
        -------
        y : np.ndarray
        """
        if self._output_strategy == "winner_takes_all":
            X = cast(np.ndarray, np.sum(X, axis=0))
        elif self._output_strategy == "median":
            X = cast(np.ndarray, np.median(X, axis=0))
        elif self._output_strategy == "last_value":
            X = X[-1, :]
        if self._needs_proba:
            return X
        else:
            return np.atleast_1d(np.argmax(X))
