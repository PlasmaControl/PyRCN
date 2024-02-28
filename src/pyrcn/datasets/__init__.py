"""The :mod:`pyrcn.datasets` includes datasets for reference experiments."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from ._base import mackey_glass, lorenz, load_digits


__all__ = 'mackey_glass', 'lorenz', 'load_digits'
