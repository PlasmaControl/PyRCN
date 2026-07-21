"""Generic adapter that turns a scikit-learn metric into a PyRCN metric.

PyRCN passes targets and predictions as sequences: a list/tuple or an
object-dtype ``ndarray`` whose elements are the per-sequence arrays. The
adapter concatenates those per-sequence arrays into the flat arrays that
scikit-learn expects, then forwards everything else unchanged. Plain
(non-object) numeric arrays are passed through as-is, so the public metrics
also work on ordinary single arrays.
"""

from __future__ import annotations

# Authors: Peter Steiner <peter.steiner@pyrcn.net>
# License: BSD 3 clause

import functools
from typing import Any, Callable

import numpy as np

# Keyword arguments that carry label/prediction/weight arrays and therefore
# need the same sequence-to-flat treatment as the first two positionals.
_ARRAY_KWARGS = frozenset({
    "y_true", "y_pred", "y1", "y2", "y_prob", "y_proba", "y_score",
    "pred_decision", "sample_weight",
})


def _flatten(a: Any) -> Any:
    """Concatenate a sequence of per-sequence arrays into a flat array.

    ``None`` is returned unchanged. A plain (non-object dtype) ``ndarray`` is
    returned as-is. Anything else -- an object-dtype ``ndarray`` or a
    list/tuple of per-sequence arrays -- is concatenated, reproducing the
    historical PyRCN behaviour.
    """
    if a is None:
        return None
    if isinstance(a, np.ndarray) and a.dtype != object:
        return a
    return np.concatenate([np.asarray(s) for s in a])


def _wrap(sklearn_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a scikit-learn metric to accept PyRCN sequence input.

    The returned callable flattens the first two positional arguments (always
    the label/prediction arrays in scikit-learn metrics) and any array-valued
    keyword argument, then forwards all remaining arguments unchanged to
    ``sklearn_fn``. ``functools.wraps`` makes the public metric mirror
    scikit-learn's current name, docstring and signature, so it never drifts.
    """
    @functools.wraps(sklearn_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args:
            new_args = [_flatten(a) for a in args[:2]] + list(args[2:])
            args = tuple(new_args)
        for key in _ARRAY_KWARGS:
            if key in kwargs:
                kwargs[key] = _flatten(kwargs[key])
        return sklearn_fn(*args, **kwargs)
    return wrapper
