"""Testing for Extreme Learning Machine module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_digits, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion

from pyrcn.base.blocks import InputToNode
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor
from pyrcn.linear_model import IncrementalRegression

X_iris, y_iris = load_iris(return_X_y=True)


def test_elm_get_params() -> None:
    print('\ntest_elm_get_params():')
    elm = ELMClassifier()
    elm_params = elm.get_params()
    print(elm_params)


def test_elm_classifier_fit() -> None:
    print('\ntest_elm_classifier_fit():')
    X, y = load_digits(return_X_y=True)
    elm = ELMClassifier(hidden_layer_size=50)
    elm.fit(X, y)


def test_elm_classifier_partial_fit() -> None:
    print('\ntest_elm_classifier_partial_fit():')
    X, y = load_digits(return_X_y=True)
    elm = ELMClassifier(hidden_layer_size=50)
    for k in range(10):
        elm.partial_fit(X[k:k+1, :], y[k:k+1], classes=np.arange(10))
    print(elm.__sizeof__())
    print(elm.hidden_layer_state(X))
    elm = ELMClassifier(hidden_layer_size=50, regressor=Ridge())
    with pytest.raises(BaseException):
        for k in range(10):
            elm.partial_fit(X[k:k+1, :], y[k:k+1], classes=np.arange(10))


def test_elm_regressor_jobs() -> None:
    print('\ntest_elm_regressor_jobs():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10, random_state=42)
    param_grid = {
        'input_to_node': [
            InputToNode(
                bias_scaling=10., hidden_layer_size=20, random_state=42),
            InputToNode(
                bias_scaling=10., hidden_layer_size=50, random_state=42)],
        'regressor': [IncrementalRegression(alpha=.0001),
                      IncrementalRegression(alpha=.01)],
        'random_state': [42]}
    elm = GridSearchCV(ELMRegressor(), param_grid)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print(f"tests - elm:\n sin | cos \n {y_test-y_elm}")
    print(f"best_params_: {elm.best_params_}")
    print(f"best_score: {elm.best_score_}")
    np.testing.assert_allclose(y_test, y_elm, atol=1e-1)


def test_elm_regressor_chunk() -> None:
    print('\ntest_elm_regressor_chunk():')
    X = np.linspace(0, 10, 2000)
    y = np.hstack((np.sin(X).reshape(-1, 1), np.cos(X).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10, random_state=42)
    param_grid = {
            'input_to_node__hidden_layer_size': [20, 50],
            'input_to_node__input_scaling': [1.],
            'input_to_node__bias_scaling': [10.],
            'input_to_node__activation': ['tanh'],
            'input_to_node__random_state': [42],
            'chunk_size': [500],
            'regressor__alpha': [1e-2, 1e-5],
            'random_state': [42]
    }
    elm = GridSearchCV(ELMRegressor(), param_grid)
    elm.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print(f"tests - elm:\n sin | cos \n {y_test-y_elm}")
    print(f"best_params_: {elm.best_params_}")
    print(f"best_score: {elm.best_score_}")
    np.testing.assert_allclose(y_test, y_elm, atol=1e-1)
    elm.fit(X_train.reshape(-1, 1), y_train)
    y_elm = elm.predict(X_test.reshape(-1, 1))
    print(f"tests - elm:\n sin | cos \n {y_test-y_elm}")
    print(f"best_params_: {elm.best_params_}")
    print(f"best_score: {elm.best_score_}")
    np.testing.assert_allclose(y_test, y_elm, atol=1e-1)
    with pytest.raises(ValueError):
        elm = clone(elm.best_estimator_).set_params(chunk_size=-1)
        elm.fit(X_train.reshape(-1, 1), y_train)


def test_elm_classifier_not_fitted() -> None:
    X, y = load_digits(return_X_y=True)
    with pytest.raises(NotFittedError):
        ELMClassifier(hidden_layer_size=50, verbose=True).predict(X)


def test_iris_ensemble_iterative_regression() -> None:
    print('\ntest_iris_ensemble_iterative_regression():')
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=5, random_state=42)

    cls = ELMClassifier(
        input_to_node=FeatureUnion([
            ('tanh', InputToNode(hidden_layer_size=10, random_state=42,
                                 input_activation='tanh')),
            ('bounded_relu', InputToNode(hidden_layer_size=10, random_state=42,
                                         input_activation='bounded_relu'))]),
        regressor=IncrementalRegression(alpha=.01),
        random_state=42)

    for samples in np.split(np.arange(0, X_train.shape[0]), 5):
        cls.partial_fit(X_train[samples, :], y_train[samples],
                        classes=np.arange(3, dtype=int))
    y_predicted = cls.predict(X_test)

    for record in range(len(y_test)):
        print('predicted: {} \ttrue: {}'
              .format(y_predicted[record], y_test[record]))

    print(f'score: {cls.score(X_test, y_test)}')
    print(f'proba: {cls.predict_proba(X_test)}')
    print(f'log_proba: {cls.predict_log_proba(X_test)}')
    assert cls.score(X_test, y_test) >= 4./5.


def test_elm_classifier_no_valid_params() -> None:
    X, y = load_digits(return_X_y=True)
    with pytest.raises(TypeError):
        ELMClassifier(input_to_node=ELMRegressor()).fit(X, y)
    with pytest.raises(TypeError):
        ELMClassifier(regressor=InputToNode()).fit(X, y)


def test_elm_regressor_partial_fit_verbose() -> None:
    X, y = load_digits(return_X_y=True)
    elm = ELMRegressor(hidden_layer_size=20, verbose=True)
    elm.partial_fit(X[:10], y[:10].astype(float))


class _NonBackableRegression(IncrementalRegression):
    """IncrementalRegression subclass: non-backable (exact-type check) but
    still exposes ``partial_fit`` so the numpy chunk path can be exercised."""


def test_elm_regressor_chunk_numpy() -> None:
    rng = np.random.RandomState(42)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=(30,))
    elm = ELMRegressor(
        hidden_layer_size=10, chunk_size=10,
        regressor=_NonBackableRegression())
    elm.fit(X, y)
    assert elm._use_torch is False
    pred = elm.predict(X)
    assert pred.shape[0] == 30


def test_elm_regressor_invalid_optimizer() -> None:
    X = np.zeros(shape=(10, 2))
    y = np.zeros(shape=(10,))
    with pytest.raises(ValueError):
        ELMRegressor(optimizer="invalid").fit(X, y)


def test_elm_regressor_invalid_loss() -> None:
    X = np.zeros(shape=(10, 2))
    y = np.zeros(shape=(10,))
    with pytest.raises(ValueError):
        ELMRegressor(loss="invalid").fit(X, y)


def test_elm_regressor_invalid_epochs() -> None:
    X = np.zeros(shape=(10, 2))
    y = np.zeros(shape=(10,))
    with pytest.raises(ValueError):
        ELMRegressor(epochs=0).fit(X, y)


def test_elm_regressor_invalid_learning_rate() -> None:
    X = np.zeros(shape=(10, 2))
    y = np.zeros(shape=(10,))
    with pytest.raises(ValueError):
        ELMRegressor(learning_rate=0.).fit(X, y)


def test_elm_hidden_layer_state_not_fitted() -> None:
    elm = ELMRegressor()
    elm.input_to_node = None
    with pytest.raises(NotFittedError):
        elm.hidden_layer_state(np.zeros(shape=(5, 2)))
