"""
The :mod:`pyrcn.elm` module includes Extreme Learning Machine algorithms.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

from pyrcn.extreme_learning_machine._elm import ELMClassifier, ELMRegressor

__all__ = ['ELMClassifier',
           'ELMRegressor'
           ]
