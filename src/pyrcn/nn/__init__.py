"""PyTorch building blocks for reservoir computing (``pyrcn.nn``).

The tensor-level, ``torch.nn``-style components that power PyRCN's
scikit-learn estimators, usable directly for pure-PyTorch workflows:

* reservoir layers -- :class:`Reservoir`, :class:`EulerReservoir` -- and their
  single-step cells :class:`LeakyESNCell`, :class:`EulerESNCell`;
* the input feature map :class:`InputFeatureMap`;
* the closed-form ridge readout :class:`IncrementalRidge`.

Weight initializers (spectral-radius normalization, sparse fan-in, the
minimum-complexity topologies, ...) live in :mod:`pyrcn.nn.init`.

Examples
--------
>>> import torch
>>> from pyrcn.nn import InputFeatureMap, IncrementalRidge, Reservoir, init
>>> generator = torch.Generator().manual_seed(0)
>>> feature_map = InputFeatureMap(1, 100)
>>> reservoir = Reservoir(100, spectral_radius=0.9, leakage=0.8)
>>> reservoir.set_recurrent_weights(
...     init.normal_recurrent_weights(100, generator=generator))
>>> x = torch.randn(1, 50, 1)                 # (batch, length, features)
>>> states, final_state = reservoir(feature_map(x[0]).unsqueeze(0))
>>> _ = IncrementalRidge(alpha=1e-3).fit(states[0], torch.randn(50, 1))
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from . import init
from ._input import InputFeatureMap
from ._readout import IncrementalRidge
from ._reservoir import (EulerESNCell, EulerReservoir, LeakyESNCell,
                         Reservoir)

__all__ = ["Reservoir", "EulerReservoir", "LeakyESNCell", "EulerESNCell",
           "InputFeatureMap", "IncrementalRidge", "init"]
