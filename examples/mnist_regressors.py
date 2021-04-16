# MNIST classification using Extreme Learning Machines

import numpy as np
import time
from scipy.stats import uniform
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.fixes import loguniform

from pyrcn.extreme_learning_machine import ELMClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode


# Load the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# Provide standard split in training and test. Normalize to a range between [-1, 1].


X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X=X)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000].astype(int), y[60000:].astype(int)

# Extreme Learning Machine preparation
param_grid = {'input_to_node__hidden_layer_size': [500],
              'input_to_node__activation': ['tanh'],
              'input_to_node__input_scaling': loguniform(1e-5, 1e1),
              'input_to_node__bias_scaling': [0.0],
              'input_to_node__k_in': [10],
              'input_to_node__random_state': [42] }

random_search = RandomizedSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()),
                                   param_grid, cv=StratifiedKFold(), verbose=0, 
                                   n_jobs=-1, n_iter=24, random_state=0).fit(X=X_train, y=y_train)

# Extreme Learning Machine preparation
param_grid = {'input_to_node__hidden_layer_size': [500],
              'input_to_node__activation': ['tanh'],
              'input_to_node__input_scaling': [random_search.best_params_["input_to_node__input_scaling"]],
              'input_to_node__bias_scaling': uniform(loc=0, scale=5e0),
              'input_to_node__k_in': [10],
              'input_to_node__random_state': [42] }

random_search = RandomizedSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()),
                                   param_grid, cv=StratifiedKFold(), verbose=0, 
                                   n_jobs=-1, n_iter=24, random_state=0).fit(X=X_train, y=y_train)

# Extreme Learning Machine preparation
param_grid = {'input_to_node__hidden_layer_size': [500],
              'input_to_node__activation': ['tanh'],
              'input_to_node__input_scaling': [random_search.best_params_["input_to_node__input_scaling"]],
              'input_to_node__bias_scaling': [random_search.best_params_["input_to_node__bias_scaling"]],
              'input_to_node__k_in': [10],
              'input_to_node__random_state': [42],
              'regressor__alpha': loguniform(1e-5, 1e1),}

random_search = RandomizedSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()),
                                   param_grid, cv=StratifiedKFold(), verbose=0, 
                                   n_jobs=-1, n_iter=24, random_state=0).fit(X=X_train, y=y_train)

param_grid = {'input_to_node__hidden_layer_size': [500],
              'input_to_node__activation': ['tanh'],
              'input_to_node__input_scaling': [random_search.best_params_["input_to_node__input_scaling"]],
              'input_to_node__bias_scaling': [random_search.best_params_["input_to_node__bias_scaling"]],
              'input_to_node__k_in': [10],
              'input_to_node__random_state': [42],
              'regressor__alpha': [random_search.best_params_["regressor__alpha"]],}


grid_search = GridSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=Ridge()),
                             param_grid, cv=StratifiedKFold(n_splits=2), verbose=0, n_jobs=1).fit(X=X_train, y=y_train)
print(grid_search.best_score_)
print(grid_search.refit_time_)
print(grid_search.best_estimator_.__sizeof__())

grid_search = GridSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=IncrementalRegression()),
                             param_grid, cv=StratifiedKFold(n_splits=2), verbose=0, n_jobs=1).fit(X=X_train, y=y_train)
print(grid_search.best_score_)
print(grid_search.refit_time_)
print(grid_search.best_estimator_.__sizeof__())

grid_search = GridSearchCV(ELMClassifier(input_to_node=InputToNode(), regressor=IncrementalRegression(), chunk_size=6000),
                             param_grid, cv=StratifiedKFold(n_splits=2), verbose=0, n_jobs=1).fit(X=X_train, y=y_train)
print(grid_search.best_score_)
print(grid_search.refit_time_)
print(grid_search.best_estimator_.__sizeof__())

