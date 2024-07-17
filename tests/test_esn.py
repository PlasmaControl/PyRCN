"""Testing for Echo State Network module."""
import numpy as np
import pytest
from pyrcn.datasets import mackey_glass, load_digits
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError

from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
from pyrcn.metrics import mean_squared_error


def test_esn_get_params() -> None:
    print('\ntest_esn_get_params():')
    esn = ESNClassifier()
    esn_params = esn.get_params()
    print(esn_params)


def test_esn_regressor_jobs() -> None:
    print('\ntest_esn_regressor_jobs():')
    X, y = mackey_glass(n_timesteps=8000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    param_grid = {
        "input_to_node": [
            InputToNode(
                bias_scaling=.1, hidden_layer_size=10,
                input_activation='identity', random_state=42),
            InputToNode(
                bias_scaling=.1, hidden_layer_size=50,
                input_activation='identity', random_state=42)],
        "node_to_node": [
            NodeToNode(
                spectral_radius=0., hidden_layer_size=10, random_state=42),
            NodeToNode(
                spectral_radius=1, hidden_layer_size=50, random_state=42)],
        "regressor": [
            IncrementalRegression(alpha=.0001),
            IncrementalRegression(alpha=.01)],
        'random_state': [42]}
    esn = GridSearchCV(estimator=ESNRegressor(), param_grid=param_grid)
    esn.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    y_esn = esn.predict(X_test.reshape(-1, 1))
    print("tests - esn:\n sin | cos \n {0}".format(y_test-y_esn))
    print("best_params_: {0}".format(esn.best_params_))
    print("best_score: {0}".format(esn.best_score_))
    np.testing.assert_allclose(1, esn.best_score_, atol=1e-1)


def test_esn_regressor_requires_no_sequence() -> None:
    print('\ntest_esn_regressor_requires_sequence():')
    X, y = mackey_glass(n_timesteps=8000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10,
                                                        random_state=42)
    param_grid = {'hidden_layer_size': [20, 50],
                  'input_scaling': [1.],
                  'bias_scaling': [10.],
                  'input_activation': ['identity'],
                  'random_state': [42],
                  'spectral_radius': [0.],
                  'reservoir_activation': ['tanh'],
                  'alpha': [1e-2, 1e-5]
                  }
    esn = GridSearchCV(ESNRegressor(), param_grid)
    esn.fit(X_train.reshape(-1, 1), y_train, n_jobs=2)
    np.testing.assert_equal(esn.best_estimator_.requires_sequence, False)


def test_esn_regressor_requires_sequence() -> None:
    print('\ntest_esn_regressor_requires_sequence():')
    X, y = mackey_glass(n_timesteps=8000)
    X_train = np.empty(shape=(10, ), dtype=object)
    y_train = np.empty(shape=(10, ), dtype=object)
    X_test = np.empty(shape=(10, ), dtype=object)
    y_test = np.empty(shape=(10, ), dtype=object)
    splitter = TimeSeriesSplit(n_splits=10)
    for k, (train_index, test_index) in enumerate(splitter.split(X, y)):
        X_train[k] = X[train_index].reshape(-1, 1)
        y_train[k] = y[train_index]
        X_test[k] = X[test_index].reshape(-1, 1)
        y_test[k] = y[test_index]
    param_grid = {'hidden_layer_size': [20, 50],
                  'input_scaling': [1.],
                  'bias_scaling': [10.],
                  'input_activation': ['identity'],
                  'random_state': [42],
                  'spectral_radius': [0.],
                  'reservoir_activation': ['tanh'],
                  'alpha': [1e-2, 1e-5],
                  }
    esn = GridSearchCV(
        ESNRegressor(), param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False))
    esn.fit(X_train, y_train, n_jobs=2)
    np.testing.assert_equal(esn.best_estimator_.requires_sequence, True)


def test_esn_regressor_wrong_sequence_format() -> None:
    print('\ntest_esn_regressor_requires_sequence():')
    X, y = mackey_glass(n_timesteps=8000)
    X_train = np.empty(shape=(10, 1000, 1))
    y_train = np.empty(shape=(10, 1000, 1))
    splitter = TimeSeriesSplit(n_splits=10)
    for k, (train_index, test_index) in enumerate(splitter.split(X, y)):
        X_train[k, :, :] = X[:1000].reshape(-1, 1)
        y_train[k, :, :] = y[:1000].reshape(-1, 1)
    param_grid = {'hidden_layer_size': 50,
                  'input_scaling': 1.,
                  'bias_scaling': 10.,
                  'input_activation': 'identity',
                  'random_state': 42,
                  'spectral_radius': 0.,
                  'reservoir_activation': 'tanh',
                  'alpha': 1e-5}
    with pytest.raises(ValueError):
        ESNRegressor(verbose=True, **param_grid)\
            .fit(X_train, y_train, n_jobs=2)


def test_esn_output_unchanged() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    shape1 = y[0].shape
    esn = ESNClassifier(hidden_layer_size=50).fit(X, y)
    print(esn)
    shape2 = y[0].shape
    assert (shape1 == shape2)


def test_esn_classifier_sequence_to_value() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    esn = ESNClassifier(hidden_layer_size=50).fit(X, y)
    y_pred = esn.predict(X)
    assert (len(y) == len(y_pred))
    assert (len(y_pred[0]) == 1)
    assert (esn.sequence_to_value is True)
    assert (esn.decision_strategy == "winner_takes_all")
    y_pred = esn.predict_proba(X)
    assert (y_pred[0].ndim == 1)
    y_pred = esn.predict_log_proba(X)
    assert (y_pred[0].ndim == 1)
    esn.sequence_to_value = False
    y_pred = esn.predict(X)
    assert (len(y_pred[0]) == 8)
    y_pred = esn.predict_proba(X)
    assert (y_pred[0].ndim == 2)
    y_pred = esn.predict_log_proba(X)
    assert (y_pred[0].ndim == 2)


def test_esn_classifier_instance_fit() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    esn = ESNClassifier(hidden_layer_size=50).fit(X[0], np.repeat(y[0], 8))
    assert (esn.sequence_to_value is False)
    y_pred = esn.predict_proba(X[0])
    assert (y_pred.ndim == 2)
    y_pred = esn.predict_log_proba(X[0])
    assert (y_pred.ndim == 2)


def test_esn_classifier_partial_fit() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    esn = ESNClassifier(hidden_layer_size=50, verbose=True)
    for k in range(10):
        esn.partial_fit(X[k], np.repeat(y[k], 8), classes=np.arange(10),
                        postpone_inverse=True)
    print(esn.__sizeof__())
    print(esn.hidden_layer_state(X=X))
    esn = ESNClassifier(hidden_layer_size=50, regressor=Ridge())
    with pytest.raises(BaseException):
        for k in range(10):
            esn.partial_fit(X[k], np.repeat(y[k], 8), classes=np.arange(10),
                            postpone_inverse=True)


def test_esn_classifier_not_fitted() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    with pytest.raises(NotFittedError):
        ESNClassifier(hidden_layer_size=50, verbose=True).predict(X)


def test_esn_classifier_no_valid_params() -> None:
    X, y = load_digits(return_X_y=True, as_sequence=True)
    with pytest.raises(TypeError):
        ESNClassifier(input_to_node=ESNRegressor()).fit(X, y)
    with pytest.raises(TypeError):
        ESNClassifier(node_to_node=ESNRegressor()).fit(X, y)
    with pytest.raises(TypeError):
        ESNClassifier(input_to_node=ESNRegressor()).fit(X, y)
    with pytest.raises(ValueError):
        ESNClassifier(requires_sequence="True").fit(X, y)
    with pytest.raises(TypeError):
        ESNClassifier(regressor=InputToNode()).fit(X, y)
