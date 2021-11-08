"""Testing for Autoenocder module (pyrcn.autoencoder)"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris

from pyrcn.autoencoder import MLPAutoEncoder


X_iris, y_iris = load_iris(return_X_y=True)


def test_ae_full() -> None:
    print('\test_ae_full():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5,
                                                        random_state=42)
    ae = MLPAutoEncoder(transform_type='full', max_iter=2000)
    ae.fit(X_train)

    X_predicted = ae.transform(X_test)

    print('Reconstruction Error: {0}'.format(mean_squared_error(X_test, X_predicted)))

    assert mean_squared_error(X_test, X_predicted) < 1.


def test_ae_only_encode() -> None:
    print('\test_ae_only_encode():')
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=5,
                                                        random_state=42)
    ae = MLPAutoEncoder(transform_type='only_encode', max_iter=2000)
    ae.fit(X_train)

    X_predicted = ae.transform(X_test)

    assert X_predicted.shape == (5, 100)
