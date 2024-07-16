"""Testing for model selection module."""

from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from collections.abc import Iterable

from pyrcn.model_selection import SequentialSearchCV, SHGOSearchCV
import pytest


def test_sequentialSearchCV_equivalence() -> None:
    """Test the equivalence of SequentialSearchCV to a manual sequence."""
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    cv = KFold(2, shuffle=True, random_state=42)
    svm1 = SVC(random_state=42)
    svm2 = SVC(random_state=42)
    param_grid1 = {'C': [1, 2], 'kernel': ['rbf', 'linear']}
    param_grid2 = {'shrinking': [True, False]}
    gs1 = GridSearchCV(svm1, param_grid1, cv=cv).fit(X, y)
    gs2 = RandomizedSearchCV(gs1.best_estimator_, param_grid2,
                             cv=cv, random_state=42).fit(X, y)

    ss = SequentialSearchCV(
        svm2, searches=[
            ('gs1', GridSearchCV, param_grid1, {'cv': cv}),
            ('gs2', RandomizedSearchCV, param_grid2,
             {'cv': cv, 'random_state': 42, 'refit': True}),
            ('gs3', GridSearchCV, param_grid1)]).fit(X, y)
    assert gs1.best_params_ == ss.all_best_params_['gs1']
    assert gs2.best_params_ == ss.all_best_params_['gs2']
    assert (isinstance(ss.cv_results_, dict))
    assert (ss.best_estimator_ is not None)
    assert (isinstance(ss.best_score_, float))
    print(ss.best_index_)
    assert (isinstance(ss.n_splits_, int))
    assert (isinstance(ss.refit_time_, float))
    assert (isinstance(ss.multimetric, bool))


@pytest.mark.skip(reason="no way of currently testing this")
def test_SHGOSearchCV() -> None:
    """Test the SHGO search."""
    from sklearn.metrics import accuracy_score
    from sklearn.base import clone, BaseEstimator
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    cv = StratifiedKFold(n_splits=5)
    svm = SVC(random_state=42)

    def func(params: Iterable, param_names: Iterable,
             base_estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
             train: np.ndarray, test: np.ndarray) -> float:
        estimator = base_estimator
        for name, param in zip(param_names, params):
            estimator.set_params(**{name: param})
        mse = []
        for tr, te in zip(train, test):
            est = clone(estimator).fit(X[tr], y[tr])
            y_pred = est.predict(X[te])
            mse.append(-accuracy_score(y[te], y_pred))
        return np.mean(mse)

    params = {'max_iter': (1, 1000)}

    def fun(x: tuple) -> float:
        return max([x[0] - int(x[0])])
    constraints = {'type': 'eq', 'fun': fun}
    search = SHGOSearchCV(
        estimator=svm, func=func, params=params, cv=cv,
        constraints=constraints).fit(X, y)
    y_pred = search.predict(X)
    print(accuracy_score(y_true=y, y_pred=svm.fit(X, y).predict(X)))
    print(accuracy_score(y_true=y, y_pred=y_pred))


if __name__ == '__main__':
    test_SHGOSearchCV()
