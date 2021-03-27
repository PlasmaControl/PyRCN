"""
The :mod:`pyrcn` module includes various reservoir computing algorithms.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Author: Michael Schindler <michael.schindler1@mailbox.tu-dresden.de> ,
# Peter Steiner <peter.steiner@tu-dresden.de>. ELMs and ESNs are copyright of their respective owners.
# License: BSD 3-Clause (C) TU Dresden 2020

from pyrcn import extreme_learning_machine, echo_state_network, base, util

__all__ = ['extreme_learning_machine',
           'echo_state_network',
           'base',
           'util'
           ]
