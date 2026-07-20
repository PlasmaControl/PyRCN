"""Model-selection tests for torch-backed ESN estimators.

These prove that GridSearchCV, RandomizedSearchCV and the library's
SequentialSearchCV work with the torch-backed ESN estimators and preserve
the torch fast path through ``clone`` (``best_estimator_._use_torch`` stays
True). Bare-name grids (``spectral_radius``, ``alpha``, ...) are routed to the
sub-blocks by the estimator's ``set_params``. Reservoirs are kept tiny so the
per-timestep recurrence stays fast under repeated CV refits.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     TimeSeriesSplit)

from pyrcn.base.blocks import NodeToNode
from pyrcn.datasets import mackey_glass
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.model_selection import SequentialSearchCV

HIDDEN = 20


def _data() -> tuple[np.ndarray, np.ndarray]:
    """Return small non-sequence mackey-glass data."""
    X, y = mackey_glass(n_timesteps=300)
    return X.reshape(-1, 1), y


def _estimator() -> ESNRegressor:
    """A small torch-backable ESN regressor (hidden size routed to blocks)."""
    return ESNRegressor(hidden_layer_size=HIDDEN, verbose=False)


def test_gridsearch_esn_fastpath_survives_clone() -> None:
    """GridSearchCV fits and keeps the torch fast path via clone."""
    X, y = _data()
    gs = GridSearchCV(
        _estimator(),
        param_grid={'spectral_radius': [0.0, 0.9], 'alpha': [1e-2, 1e-5]},
        cv=TimeSeriesSplit(n_splits=2))
    gs.fit(X, y)
    assert gs.best_estimator_._use_torch is True
    y_pred = gs.predict(X)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_gridsearch_varies_model() -> None:
    """Bare-name grid params really take effect through set_params."""
    X, y = _data()
    gs = GridSearchCV(
        _estimator(),
        param_grid={'spectral_radius': [0.0, 0.9], 'alpha': [1e-2, 1e-5]},
        cv=TimeSeriesSplit(n_splits=2))
    gs.fit(X, y)
    assert len(gs.cv_results_['params']) == 4
    scores = np.asarray(gs.cv_results_['mean_test_score'])
    assert np.unique(np.round(scores, 10)).size >= 2


def test_randomizedsearch_esn_fastpath() -> None:
    """RandomizedSearchCV keeps the torch fast path."""
    X, y = _data()
    rs = RandomizedSearchCV(
        _estimator(),
        param_distributions={'spectral_radius': [0.1, 0.5, 0.9],
                             'alpha': [1e-2, 1e-4]},
        n_iter=3, random_state=42, cv=TimeSeriesSplit(n_splits=2))
    rs.fit(X, y)
    assert rs.best_estimator_._use_torch is True


def test_sequentialsearch_esn_fastpath() -> None:
    """SequentialSearchCV keeps the torch fast path across searches."""
    X, y = _data()
    cv = TimeSeriesSplit(n_splits=2)
    ss = SequentialSearchCV(
        _estimator(),
        searches=[
            ('sr', GridSearchCV, {'spectral_radius': [0.0, 0.9]}, {'cv': cv}),
            ('alpha', GridSearchCV, {'alpha': [1e-2, 1e-5]}, {'cv': cv}),
        ]).fit(X, y)
    assert ss.best_estimator_ is not None
    assert ss.best_estimator_._use_torch is True
    assert 'sr' in ss.all_best_params_
    assert 'alpha' in ss.all_best_params_
    assert isinstance(ss.best_score_, float)


def test_component_swap_grid_fastpath() -> None:
    """Object-valued component swaps still fast-path through clone."""
    X, y = _data()
    gs = GridSearchCV(
        _estimator(),
        param_grid={
            'node_to_node': [
                NodeToNode(hidden_layer_size=HIDDEN, spectral_radius=0.0,
                           random_state=42),
                NodeToNode(hidden_layer_size=HIDDEN, spectral_radius=0.9,
                           random_state=42)],
            'regressor': [IncrementalRegression(alpha=1e-2),
                          IncrementalRegression(alpha=1e-5)]},
        cv=TimeSeriesSplit(n_splits=2))
    gs.fit(X, y)
    assert gs.best_estimator_._use_torch is True
    assert np.all(np.isfinite(gs.predict(X)))
