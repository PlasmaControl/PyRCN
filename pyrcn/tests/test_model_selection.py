from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from pyrcn.model_selection import SequentialSearchCV


def test_sequentialSearchCV_equivalence():
    # Test the functional equivalence of SequentialSearchCV to a manual sequence
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    cv = KFold(2, shuffle=True, random_state=42)
    svm1 = SVC(random_state=42)
    svm2 = SVC(random_state=42)
    param_grid1 = {'C': [1, 2], 'kernel': ['rbf', 'linear']}
    param_grid2 = {'shrinking': [True, False]}
    gs1 = GridSearchCV(svm1, param_grid1, cv=cv).fit(X, y)
    gs2 = RandomizedSearchCV(gs1.best_estimator_, param_grid2, cv=cv, random_state=42).fit(X, y)

    ss = SequentialSearchCV(svm2,
                            searches=[('gs1', GridSearchCV, param_grid1, {'cv': cv}),
                                      ('gs2', RandomizedSearchCV, param_grid2, {'cv': cv, 'random_state': 42})]
                            ).fit(X, y)
    assert gs1.best_params_ == ss.best_params_['gs1']
    assert gs2.best_params_ == ss.best_params_['gs2']


if __name__ == '__main__':
    test_sequentialSearchCV_equivalence()