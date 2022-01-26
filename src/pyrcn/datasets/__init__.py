"""The :mod:`pyrcn.datasets` includes datasets for reference experiments."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from ._base import mackey_glass, lorenz, load_digits
from ._maps_piano_dataset import fetch_maps_piano_dataset
from ._ptdb_tug import fetch_ptdb_tug_dataset


__all__ = ('mackey_glass',
           'lorenz',
           'load_digits',
           'fetch_maps_piano_dataset',
           'fetch_ptdb_tug_dataset')
