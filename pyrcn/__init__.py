"""
The :mod:`pyrcn` module includes various reservoir computing algorithms.
"""
# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.
# Author: Peter Steiner <peter.steiner@tu-dresden.de>.
#
# ELMs and ESNs are copyright of their respective owners.
# License: BSD 3-Clause (C) TU Dresden 2021

from pyrcn import (extreme_learning_machine, echo_state_network, base, preprocessing,
                   postprocessing, util)

__all__ = ['extreme_learning_machine',
           'echo_state_network',
           'base',
           'preprocessing',
           'postprocessing',
           'util']
