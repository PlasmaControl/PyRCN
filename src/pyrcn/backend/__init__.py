"""Private PyTorch compute backend for PyRCN.

This package holds the tensor-level implementation (reservoir engine, input
feature map, readout) behind the scikit-learn-compatible frontend. It is an
implementation detail during the backend redesign and is not part of the public
API yet; a curated public ``pyrcn.nn`` surface is planned once the frontend
reaches parity with the legacy NumPy path.
"""

from ._reservoir import Reservoir

__all__ = ["Reservoir"]
