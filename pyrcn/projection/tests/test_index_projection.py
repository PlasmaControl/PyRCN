"""
Testing for Echo State Network module (pyrcn.echo_state_network)
"""
import numpy as np
from pyrcn.projection import MatrixToIndexProjection

import pytest


def test_matrix_to_index_projection():
    print('\ntest_matrix_to_index_projection():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = MatrixToIndexProjection().fit_transform(X)
    np.testing.assert_equal(y, 0)


def test_matrix_to_index_projection_median():
    print('\ntest_matrix_to_index_projection_median():')
    trf = MatrixToIndexProjection(output_strategy="median")
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    y = trf.fit_transform(X)
    np.testing.assert_equal(y, np.argmax(np.median(X, axis=0)))


def test_matrix_to_index_projection_lv():
    print('\ntest_matrix_to_index_projection_lv():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    trf = MatrixToIndexProjection(output_strategy="last_value").fit(X)
    y = trf.transform(X)
    np.testing.assert_equal(y, np.argmax(X[-1, :]))


def test_matrix_to_index_projection_proba():
    print('\ntest_matrix_to_index_projection_predict_proba():')
    X = np.random.rand(5, 3)
    idx_true = np.array([0, 0, 0, 1, 2])
    X[range(5), idx_true] += 1
    trf = MatrixToIndexProjection(output_strategy="last_value", needs_proba=True).fit(X)
    y = trf.transform(X)
    np.testing.assert_equal(y, X[-1, :])


if __name__ == "__main__":
    test_matrix_to_index_projection()
    test_matrix_to_index_projection_median()
    test_matrix_to_index_projection_lv()
    test_matrix_to_index_projection_proba()
