"""
The :mod:`pyrcn.extreme_learning_machine` contains  a simple object-oriented implementation 
of Extreme Learning Machines [#]_.

Separate implementations of Classifiers and Regressors as specified by scikit-learn.

References
----------
    .. [#] Guang-Bin Huang et al., ‘Extreme learning machine: Theory and applications’, 
           p. 489-501, 2006, doi: 10.1016/j.neucom.2005.12.126.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

from pyrcn.extreme_learning_machine._elm import ELMClassifier, ELMRegressor

__all__ = ['ELMClassifier',
           'ELMRegressor'
           ]
