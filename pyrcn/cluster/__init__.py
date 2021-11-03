"""The :mod:`pyrcn.cluster` module includes a Cluster Algorithm comparable to KMeans."""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Author: Peter Steiner <peter.steiner@tu-dresden.de>.
# License: BSD 3-Clause (C) TU Dresden 2021

from pyrcn.cluster._kcluster import KCluster

__all__ = ('KCluster',)
