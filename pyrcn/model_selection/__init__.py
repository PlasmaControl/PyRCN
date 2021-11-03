"""The :mod:`pyrcn.model_selection` to sequentially fine-tune estimator-parameters."""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Simon Stone <simon.stone@tu-dresden.de>,
# Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from pyrcn.model_selection._search import SequentialSearchCV

__all__ = ('SequentialSearchCV',)
