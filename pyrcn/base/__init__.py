"""
The :mod:`pyrcn.base` contains various activation functions
and simple object-oriented implementations  of the building blocks for 
Reservoir Computing Networks [#]_.

References
----------
    .. [#] P. Steiner et al., ‘PyRCN: A Toolbox for Exploration and Application 
           of Reservoir Computing Networks’, under review.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Michael Schindler <michael.schindler@maschindler.de>
# License: BSD 3 clause

from pyrcn.base._activations import ACTIVATIONS, ACTIVATIONS_INVERSE, ACTIVATIONS_INVERSE_BOUNDS

__all__ = ['ACTIVATIONS',
           'ACTIVATIONS_INVERSE',
           'ACTIVATIONS_INVERSE_BOUNDS'
           ]
