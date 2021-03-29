"""
The :mod:`pyrcn.elm` module includes a Cluster Algorithm compareable to KMeans.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Author: Peter Steiner <peter.steiner@tu-dresden.de> and Michael Schindler <michael.schindler1@mailbox.tu-dresden.de>.
# License: BSD 3-Clause (C) TU Dresden 2021

from pyrcn.cluster._kcluster import KCluster

__all__ = ['KCluster'
           ]
