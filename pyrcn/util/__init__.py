"""The :mod:`pyrcn.util` contains utilities for running, testing and analyzing."""

# Author: Peter Steiner <peter.steiner@tu-dresden.de>,
# Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

from pyrcn.util._util import (new_logger, tud_colors, get_mnist, argument_parser,
                              concatenate_sequences)
from pyrcn.util._feature_extractor import FeatureExtractor

__all__ = ('new_logger',
           'tud_colors',
           'get_mnist',
           'argument_parser',
           'FeatureExtractor',
           'concatenate_sequences'
           )
