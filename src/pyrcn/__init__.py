"""The :mod:`pyrcn` module includes various reservoir computing algorithms."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

from . import (autoencoder, base, echo_state_network, extreme_learning_machine,
               linear_model, model_selection, postprocessing, preprocessing,
               projection, util)


__all__ = ('autoencoder',
           'base',
           'echo_state_network',
           'extreme_learning_machine',
           'linear_model',
           'model_selection',
           'postprocessing',
           'preprocessing',
           'projection',
           'util')
