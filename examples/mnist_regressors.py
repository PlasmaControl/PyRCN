# MNIST classification using Extreme Learning Machines and Echo State Networks
import numpy as np
import time
from scipy.stats import uniform
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.fixes import loguniform

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.extreme_learning_machine import ELMClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode


# Load the dataset
print("Fetching the data...", end="")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
print("...done.")

# Provide standard split in training and test. Normalize to a range between [-1, 1].
print("Scale and split...", end="")
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X=X)
X_train, X_test = X[:600], X[600:]
y_train, y_test = y[:600].astype(int), y[600:].astype(int)
print("...done.")

# Prepare sequential hyperparameter tuning
initially_fixed_params = {
    'hidden_layer_size': 500,
    'activation': 'tanh',
    'k_in': 10,
    'random_state': 42,
    'bias_scaling': 0.0,
}
step1_params = {'input_to_node__input_scaling': loguniform(1e-5, 1e1)}
kwargs1 = {'random_state': 42, 'verbose': 1}
step2_params = {'input_to_node__bias_scaling': range(5)}
kwargs2 = {'verbose': 10}
i2n = InputToNode(**initially_fixed_params)
elm = ELMClassifier(input_to_node=i2n, regressor=Ridge())

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_params, kwargs1),
            ('step2', GridSearchCV, step2_params, kwargs2)]  # Note that we pass functors, not instances (no '()')!


elm_opt = SequentialSearchCV(elm, searches=searches).fit(X_train, y_train)
print(cross_val_score(elm, X_test, y_test, verbose=10, n_jobs=-1))
print("...done.")
