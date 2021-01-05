"""
Testing for Extreme Learning Machine module (pyrcn.extreme_learning_machine)
"""
import scipy
import numpy as np

import pytest

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from pyrcn import extreme_learning_machine
from pyrcn.extreme_learning_machine import ELMClassifier, ELMRegressor


def test_iris():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)
    lb = LabelBinarizer().fit(y)
    y_train_numeric = lb.transform(y_train)
    classifier = ELMClassifier(hidden_layer_size=10, random_state=1848)
    classifier.fit(X_train, y_train_numeric)
    y_predicted_numeric = classifier.predict(X_test)
    y_predicted = lb.inverse_transform(y_predicted_numeric)

    for record in range(len(y_test)):
        print('predicted: {0} \ttrue: {1}'.format(y_predicted[record], y_test[record]))


def test_digits():
    X, y = load_digits()
    cls = ELMClassifier()


test_iris()