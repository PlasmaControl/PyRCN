"""
The :mod:`pyrcn.elm` module includes a Cluster Algorithm compareable to KMeans.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Author: Michael Schindler <michael.schindler1@mailbox.tu-dresden.de> with help from
#         the pyrcn community. ELM are copyright of their respective owners.
# License: BSD 3-Clause (C) TU Dresden 2020

from pyrcn.cluster._kcluster import KCluster

__all__ = ['KCluster'
           ]
