"""
The :mod:`pyrcn.linear_model` module implements a functionality to compute regression piecewise.
"""

# See https://github.com/TUD-STKS/PyRCN and for documentation.

from pyrcn.linear_model._incremental_regression import IncrementalRegression
from pyrcn.linear_model._fast_incremental_regression import FastIncrementalRegression

__all__ = ['IncrementalRegression', 'FastIncrementalRegression']
