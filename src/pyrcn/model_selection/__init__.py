"""The :mod:`pyrcn.model_selection` to sequentially tune hyper-parameters."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de> and
# Simon Stone <simon.stone@tu-dresden.de>
# License: BSD 3 clause

from ._search import SequentialSearchCV

__all__ = ('SequentialSearchCV',)
