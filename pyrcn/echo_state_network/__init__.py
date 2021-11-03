"""
The :mod:`pyrcn.echo_state_network`.

It contains  a simple object-oriented implementation of Echo State Networks [#]_ [#]_.

Separate implementations of Classifiers and Regressors as specified by scikit-learn

References
----------
    .. [#] H. Jaeger, ‘The “echo state” approach to analysing
           and training recurrent neural networks – with an
           Erratum note’, p. 48.
    .. [#] M. Lukoševičius, ‘A Practical Guide to Applying Echo
           State Networks’, Jan. 2012, doi: 10.1007/978-3-642-35289-8_36.
"""

# See https://github.com/TUD-STKS/PyRCN for complete
# documentation.

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from pyrcn.echo_state_network._esn import ESNClassifier, ESNRegressor

__all__ = ('ESNClassifier',
           'ESNRegressor',
           )
