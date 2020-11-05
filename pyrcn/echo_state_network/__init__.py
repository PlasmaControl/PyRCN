"""
The :mod:`pyrcn.echo_state_network` module includes Echo State Network algorithms.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Author: Peter Steiner <peter.steiner@tu-dresden.de> with help from
#         the pyrcn community. ESNs are copyright of their respective owners.
# License: BSD 3-Clause (C) TU Dresden 2020

from pyrcn.echo_state_network._echo_state_network import ESNClassifier, ESNRegressor

__all__ = ['ESNClassifier',
           'ESNRegressor'
           ]
