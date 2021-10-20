"""
The :mod:`pyrcn.coates` module implements the coates preprocessing.
"""

# See https://github.com/TUD-STKS/PyRCN and for documentation.

from pyrcn.postprocessing._normal_distribution import NormalDistribution
from pyrcn.postprocessing._sequence_to_label import SequenceToLabelClassifier


__all__ = ['NormalDistribution',
           'SequenceToLabelClassifier']
