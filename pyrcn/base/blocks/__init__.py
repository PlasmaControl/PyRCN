from pkg_resources import parse_version
import warnings
import sklearn
from sklearn.base import BaseEstimator
from pyrcn.base.blocks._input_to_node import InputToNode, PredefinedWeightsInputToNode, BatchIntrinsicPlasticity
from pyrcn.base.blocks._node_to_node import NodeToNode, PredefinedWeightsNodeToNode, HebbianNodeToNode, FeedbackNodeToNode



if parse_version(sklearn.__version__) < parse_version('0.23.1'):
    from sklearn.utils import check_array

    def validate_data(self, X, y=None, *args, **kwargs):
        warnings.warn('Due to scikit version, _validate_data(X, y) returns check_array(X), y.', DeprecationWarning)
        if y:
            return check_array(X, **kwargs), y
        else:
            return check_array(X, **kwargs)

    setattr(BaseEstimator, '_validate_data', validate_data)


__all__ = ['InputToNode',
           'PredefinedWeightsInputToNode',
           'BatchIntrinsicPlasticity',
           ]