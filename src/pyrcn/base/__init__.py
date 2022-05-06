"""
The :mod:`pyrcn.base` base functionalities for PyRCN.

It contains activation functions and simple object-oriented implementations
of the building blocks for Reservoir Computing Networks [#]_.

References
----------
    .. [#] P. Steiner et al., ‘PyRCN: A Toolbox for Exploration and Application
    of Reservoir Computing Networks’, under review.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

from ._activations import (ACTIVATIONS, ACTIVATIONS_INVERSE,
                           ACTIVATIONS_INVERSE_BOUNDS)
from ._base import (
    _uniform_random_input_weights, _uniform_random_bias,
    _normal_random_recurrent_weights, _uniform_random_recurrent_weights,
    _normal_recurrent_attention_weights)

__all__ = ('ACTIVATIONS',
           'ACTIVATIONS_INVERSE',
           'ACTIVATIONS_INVERSE_BOUNDS',
           '_uniform_random_input_weights',
           '_uniform_random_bias',
           '_normal_random_recurrent_weights',
           '_uniform_random_recurrent_weights',
           '_normal_recurrent_attention_weights'
           )
