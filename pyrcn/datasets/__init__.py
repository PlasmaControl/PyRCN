"""
The :mod:`pyrcn.datasets` includes toy datasets to quickly develop reference experiments
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from pyrcn.datasets._base import mackey_glass, load_digits
from pyrcn.datasets._maps_piano_dataset import fetch_maps_piano_dataset
from pyrcn.datasets._ptdb_tug import fetch_ptdb_tug_dataset


__all__ = ['mackey_glass',
           'load_digits',
           'fetch_maps_piano_dataset',
           'fetch_ptdb_tug_dataset']
