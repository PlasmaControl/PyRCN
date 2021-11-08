"""Testing for projection module (pyrcn.projection)."""

import numpy as np
from pyrcn.projection import MatrixToValueProjection


def test_matrix_to_value_projection() -> None:
    print('\ntest_matrix_to_value_projection():')
    r = np.random.RandomState(1234)
    X = r.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = MatrixToValueProjection().fit_transform(X)
    np.testing.assert_equal(y, 0)


def test_matrix_to_value_projection_median() -> None:
    print('\ntest_matrix_to_value_projection_median():')
    trf = MatrixToValueProjection(output_strategy="median")
    r = np.random.RandomState(1234)
    X = r.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = trf.fit_transform(X)
    np.testing.assert_equal(y, np.argmax(np.median(X, axis=0)))


def test_matrix_to_value_projection_lv() -> None:
    print('\ntest_matrix_to_value_projection_lv():')
    r = np.random.RandomState(1234)
    X = r.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    trf = MatrixToValueProjection(output_strategy="last_value").fit(X)
    y = trf.transform(X)
    np.testing.assert_equal(y, np.argmax(X[-1, :]))


def test_matrix_to_value_projection_proba() -> None:
    print('\ntest_matrix_to_value_projection_predict_proba():')
    r = np.random.RandomState(1234)
    X = r.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    trf = MatrixToValueProjection(output_strategy="last_value", needs_proba=True).fit(X)
    y = trf.transform(X)
    np.testing.assert_equal(y, X[-1, :])
