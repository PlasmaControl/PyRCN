"""
Framework for piecewise regression.
"""

# Author: Michael Schindler <michael.schindler@maschindler.de>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

import numpy as np


class SplitRegressor(Multi):