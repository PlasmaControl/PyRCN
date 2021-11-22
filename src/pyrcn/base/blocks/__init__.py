""" The :mod:`autoencoder` contains building blocks for Reservoir Computing."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from ._input_to_node import (InputToNode, PredefinedWeightsInputToNode,
                             BatchIntrinsicPlasticity)
from ._node_to_node import (NodeToNode, PredefinedWeightsNodeToNode,
                            HebbianNodeToNode, AttentionWeightsNodeToNode)


__all__ = ('InputToNode',
           'PredefinedWeightsInputToNode',
           'BatchIntrinsicPlasticity',
           'NodeToNode',
           'PredefinedWeightsNodeToNode',
           'AttentionWeightsNodeToNode',
           'HebbianNodeToNode')
