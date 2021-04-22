"""
The :mod:`pyrcn.esn` module includes Echo State Network algorithms.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

from pyrcn.echo_state_network._esn import ESNClassifier, ESNRegressor
from pyrcn.echo_state_network._feedback_esn import FeedbackESNRegressor
from pyrcn.echo_state_network._sequence_model import SequenceToSequenceRegressor, SequenceToSequenceClassifier

__all__ = ['ESNClassifier',
           'ESNRegressor',
           'FeedbackESNRegressor',
           'SequenceToSequenceRegressor',
           'SequenceToSequenceClassifier',
           ]
