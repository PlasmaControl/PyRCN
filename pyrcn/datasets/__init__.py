"""
The :mod:`pyrcn.datasets` includes toy datasets to quickly develop reference experiments..
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from pyrcn.datasets._base import mackey_glass, load_digits

__all__ = ['mackey_glass',
           'load_digits']
