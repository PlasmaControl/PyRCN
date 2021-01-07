import scipy
import numpy as np

import pytest

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from pyrcn.linear_model import IncrementalRegression


def test_linear():
    print('\ntest_linear():')
    rs = np.random.RandomState(42)
    index = range(1000)
    X = np.hstack((np.linspace(0., 10., 1000).reshape(-1, 1),
                   np.linspace(-1., 1., 1000).reshape(-1, 1),
                   rs.random(1000).reshape(-1, 1)))
    transformation = rs.random(size=(3, 2))
    y = np.matmul(X, transformation)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
    reg = IncrementalRegression()

    for prt in np.array_split(index, 3):
        reg.partial_fit(X[prt, :], y[prt, :])

    y_reg = reg.predict(X_test)
    print("test: {0}\nregr: {1}".format(y_test, y_reg))
    np.testing.assert_allclose(y_reg, y_test, rtol=.01, atol=.15)

