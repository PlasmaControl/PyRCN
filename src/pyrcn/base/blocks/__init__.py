""" The :mod:`autoencoder` contains building blocks for Reservoir Computing."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from __future__ import annotations

from ._input_to_node import (BatchIntrinsicPlasticity, InputToNode,
                             PredefinedWeightsInputToNode)
from ._node_to_node import (EulerNodeToNode, HebbianNodeToNode, NodeToNode,
                            PredefinedWeightsNodeToNode)

__all__ = (
    'InputToNode', 'PredefinedWeightsInputToNode', 'BatchIntrinsicPlasticity',
    'NodeToNode', 'EulerNodeToNode', 'PredefinedWeightsNodeToNode',
    'HebbianNodeToNode')
