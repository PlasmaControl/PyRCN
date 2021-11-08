from pyrcn.echo_state_network import ESNClassifier
from pyrcn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pyrcn.model_selection import SequentialSearchCV


def test_accuracy_score():
    pass
    """
    X = np.empty(shape=(10, ), dtype=object)
    y = np.empty(shape=(10, ), dtype=object)
    for k in range(10):
        X[k], y[k] = make_blobs(n_samples=10*k, n_features=20)
    initially_fixed_params = {'hidden_layer_size': 50,
                              'k_in': 10,
                              'input_scaling': 0.4,
                              'input_activation': 'identity',
                              'bias_scaling': 0.0,
                              'spectral_radius': 1.0,
                              'leakage': 0.1,
                              'k_rec': 10,
                              'reservoir_activation': 'tanh',
                              'bidirectional': False,
                              'alpha': 1e-5,
                              'random_state': 42,
                              'requires_sequence': True}

    step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'spectral_radius': uniform(loc=0, scale=2)}
    step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
    step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
    step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}
    scoring = make_scorer(accuracy_score)

    kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                    'scoring': scoring}
    kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                    'scoring': scoring}
    kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': scoring}
    kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
                    'scoring': scoring}  # TODO: refit=MSE
    searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
                ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

    esn = ESNClassifier(**initially_fixed_params)
    searches = SequentialSearchCV(esn, searches=searches).fit(X, y)
    searches
    """

