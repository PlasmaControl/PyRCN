"""Testing for activation functions in the module (pyrcn.base)."""
import numpy as np
from scipy.special import expit as logistic_sigmoid
from pyrcn.base import ACTIVATIONS, ACTIVATIONS_INVERSE, ACTIVATIONS_INVERSE_BOUNDS


def test_all_activations_callable() -> None:
    print('\test_all_activations_callable():')
    assert any([not callable(func) for func in ACTIVATIONS.values()]) is False


def test_all_inverse_activations_callable() -> None:
    print('\test_all_inverse_activations_callable():')
    assert any([not callable(func) for func in ACTIVATIONS_INVERSE.values()]) is False


def test_inverse_subset_of_activations() -> None:
    print('\test_inverse_subset_of_activations():')
    assert set(ACTIVATIONS_INVERSE.keys()).issubset(set(ACTIVATIONS.keys()))


def test_each_inverse_has_boundaries() -> None:
    print('\test_each_inverse_has_boundaries():')
    assert set(ACTIVATIONS_INVERSE.keys()) == set(ACTIVATIONS_INVERSE_BOUNDS.keys())


def test_each_inverse_boundary_tuple() -> None:
    print('\test_each_inverse_boundary_tuple():')
    assert any([not isinstance(bound, tuple)
                for bound in ACTIVATIONS_INVERSE_BOUNDS.values()]) is False


def test_bounded_relu() -> None:
    print('\test_bounded_relu():')
    X = np.concatenate(([-np.inf], np.arange(-5, 5), [np.inf])).reshape(1, -1)
    X_true = np.minimum(np.maximum(X, 0), 1)
    ACTIVATIONS["bounded_relu"](X)
    np.testing.assert_array_equal(X, X_true)
    X_true = np.minimum(np.maximum(X, 0), 1)
    ACTIVATIONS_INVERSE["bounded_relu"](X)
    np.testing.assert_array_equal(X, X_true)


def test_identity() -> None:
    print('\test_identity():')
    X = np.concatenate(([-np.inf], np.arange(-5, 5), [np.inf])).reshape(1, -1)
    X_true = X
    ACTIVATIONS["identity"](X)
    np.testing.assert_array_equal(X, X_true)
    ACTIVATIONS_INVERSE["identity"](X)
    np.testing.assert_array_equal(X, X_true)


def test_logistic() -> None:
    print('\test_logistic():')
    X = np.concatenate(([-np.inf], np.arange(-5, 5), [np.inf])).reshape(1, -1)
    X_true = logistic_sigmoid(X)
    ACTIVATIONS["logistic"](X)
    np.testing.assert_array_equal(X, X_true)
    X_true = np.negative(np.log(1 - X))
    ACTIVATIONS_INVERSE["logistic"](X)
    np.testing.assert_array_equal(X, X_true)


def test_relu() -> None:
    print('\test_relu():')
    X = np.concatenate(([-np.inf], np.arange(-5, 5), [np.inf])).reshape(1, -1)
    X_true = np.maximum(X, 0)
    ACTIVATIONS["relu"](X)
    np.testing.assert_array_equal(X, X_true)
    X_true = np.maximum(X, 0)
    ACTIVATIONS_INVERSE["relu"](X)
    np.testing.assert_array_equal(X, X_true)


def test_softmax() -> None:
    print('\test_softmax():')
    X = np.arange(-5, 5).reshape(1, -1).astype(float)
    X_true = X - X.max(axis=1)[:, np.newaxis]
    X_true = np.exp(X_true)
    X_true /= X_true.sum(axis=1)[:, np.newaxis]
    ACTIVATIONS["softmax"](X)
    np.testing.assert_array_equal(X, X_true)
    assert any([not X.sum() == 1.0, not X_true.sum() == 1.0,
                not X.sum() == X_true.sum()]) is False


def test_tanh() -> None:
    print('\test_l():')
    X = np.concatenate(([-np.inf], np.arange(-5, 5), [np.inf])).reshape(1, -1)
    X_true = np.tanh(X)
    ACTIVATIONS["tanh"](X)
    np.testing.assert_array_equal(X, X_true)
    X_true = np.arctanh(X)
    ACTIVATIONS_INVERSE["tanh"](X)
    np.testing.assert_array_equal(X, X_true)
