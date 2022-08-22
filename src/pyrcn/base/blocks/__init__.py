""" The :mod:`autoencoder` contains building blocks for Reservoir Computing."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from ._input_to_node import (InputToNode, PredefinedWeightsInputToNode,
                             BatchIntrinsicPlasticity)
from ._node_to_node import (NodeToNode, EulerNodeToNode,
                            PredefinedWeightsNodeToNode, HebbianNodeToNode,
                            AttentionWeightsNodeToNode)


__all__ = ('InputToNode',
           'PredefinedWeightsInputToNode',
           'BatchIntrinsicPlasticity',
           'NodeToNode',
           'EulerNodeToNode',
           'PredefinedWeightsNodeToNode',
           'AttentionWeightsNodeToNode',
           'HebbianNodeToNode')
