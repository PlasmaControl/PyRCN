"""
The :mod:`pyrcn.esn` module includes Echo State Network algorithms.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

from pyrcn.echo_state_network._esn import ESNClassifier, ESNRegressor
from pyrcn.echo_state_network._esn_fb import ESNFeedbackRegressor

__all__ = ['ESNClassifier',
           'ESNRegressor',
           'ESNFeedbackRegressor'
           ]
