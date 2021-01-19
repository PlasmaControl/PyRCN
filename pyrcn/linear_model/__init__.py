"""
The :mod:`pyrcn.linear_model` module implements a functionality to compute regression piecewise.
"""

# See https://github.com/TUD-STKS/PyRCN and for documentation.

from pyrcn.linear_model._incremental_regression import IncrementalRegression

__all__ = ['IncrementalRegression']
