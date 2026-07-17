"""The :mod:`pyrcn.datasets` includes datasets for reference experiments."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from ._base import load_digits, lorenz, mackey_glass

__all__ = 'mackey_glass', 'lorenz', 'load_digits'
