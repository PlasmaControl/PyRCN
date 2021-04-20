"""
The :mod:`pyrcn.util` contains utilities for runnung, testing and analyzing the reservoir computing modules
"""

# Author: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

from pyrcn.util._util import new_logger, tud_colors, get_mnist, argument_parser
from pyrcn.util._sequence_to_sequence import SequenceToSequenceRegressor, SequenceToSequenceClassifier

__all__ = ['new_logger',
           'tud_colors',
           'get_mnist',
           'argument_parser',
           'SequenceToSequenceRegressor',
           'SequenceToSequenceClassifier',
           ]
