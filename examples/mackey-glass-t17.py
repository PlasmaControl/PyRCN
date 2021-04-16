#!/usr/bin/env python
# coding: utf-8

# # Timeseries prediction of the Mackey-Glass Equation with ESNs 
# 
# ## Introduction
# 
# The Mackey-Glass system is essentially the differential equation, where we set the parameters to $\alpha = 0.2$, $\beta = 10$, $\gamma = 0.1$ and the time delay $\tau = 17$ in  order to have a mildly chaotic attractor. 
# 
# \begin{align}
# \label{eq:MackeyGlass}
# \dot{y}(t) = \alpha y(t-\tau) / (1 + y(t - \tau)^{\beta}) - \gamma y(t)
# \end{align}


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

from pyrcn.base import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.extreme_learning_machine import ELMRegressor
from pyrcn.linear_model import IncrementalRegression


# Load and proprocess the dataset
X = np.loadtxt("./examples/dataset/MackeyGlass_t17.txt")
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X=X.reshape(-1, 1))

# Define Train/Test lengths
trainLen = 2000 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test/run the network

X_train = X[0:trainLen]
y_train = X[1:trainLen+1]
X_test = X[trainLen:trainLen+testLen]
y_test = X[trainLen+1:trainLen+testLen+1]

fig = plt.figure()
plt.plot(X_train, label="Training input", linewidth=1)
plt.plot(y_train, label="Training target", linewidth=1)
plt.xlabel("n")
plt.xlim([0, 200])
plt.ylabel("u[n]")
plt.grid()
plt.legend()
fig.set_size_inches(4, 2.5)
plt.savefig('input_data.pdf', bbox_inches = 'tight', pad_inches = 0)

# initialize an ESNRegressor
esn = ESNRegressor(input_to_node=InputToNode(),
                   node_to_node=NodeToNode(),
                   regressor=Ridge())  # IncrementalRegression()

# initialize an ELMRegressor
elm = ELMRegressor(input_to_node=InputToNode(),
                   regressor=IncrementalRegression())  # Ridge()

# train a model
esn.fit(X=X_train, y=y_train)
elm.fit(X=X_train, y=y_train)

# evaluate the models
y_test_pred = esn.predict(X=X_test)
print(mean_squared_error(y_test, y_test_pred))
y_test_pred = elm.predict(X=X_test)
print(mean_squared_error(y_test, y_test_pred))

# create a unit impulse to record the impulse response of the reservoir
unit_impulse = np.zeros(shape=(100, 1), dtype=int)
unit_impulse[5] = 1

# Echo State Network preparation
param_grid = {'input_to_node__hidden_layer_size': [100],
              'input_to_node__activation': ['identity'],
              'input_to_node__input_scaling': [0.3],
              'input_to_node__bias_scaling': [0.0],
              'input_to_node__k_in': [1],
              'input_to_node__random_state': [42],
              'node_to_node__hidden_layer_size': [100],
              'node_to_node__activation': ['tanh'],
              'node_to_node__spectral_radius': [0.7],
              'node_to_node__leakage': [0.9],
              'node_to_node__bias_scaling': [0.0],
              'node_to_node__bi_directional': [False],
              'node_to_node__k_rec': [10],
              'node_to_node__wash_out': [0],
              'node_to_node__continuation': [True],
              'node_to_node__random_state': [42],
              'regressor__alpha': [1e-5],
              'random_state': [42] }

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

ts_split = TimeSeriesSplit()
grid_search = GridSearchCV(ESNRegressor(), cv=ts_split, param_grid=param_grid, scoring=scorer, n_jobs=-1).fit(X=X_train, y=y_train)

grid_search.best_params_

esn = grid_search.best_estimator_

esn.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=0.3)
plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('BiasScalingESN.pdf', bbox_inches = 'tight', pad_inches = 0)

# Extreme Learning Machine preparation
param_grid = {'input_to_node__hidden_layer_size': [100],
              'input_to_node__activation': ['tanh'],
              'input_to_node__input_scaling': [0.1],
              'input_to_node__bias_scaling': np.linspace(0.0, 1.5, 16),
              'input_to_node__k_in': [1],
              'input_to_node__random_state': [42] }

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

ts_split = TimeSeriesSplit()
grid_search = GridSearchCV(ELMRegressor(), cv=ts_split, param_grid=param_grid, scoring=scorer, n_jobs=-1).fit(X=X_train, y=y_train)

grid_search.best_params_

elm = grid_search.best_estimator_

elm.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(elm._input_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=0.3)
plt.xlim([0, 100])
plt.ylim([0, elm._input_to_node._hidden_layer_state[:, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R\'[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('BiasScalingELM.pdf', bbox_inches = 'tight', pad_inches = 0)


# Training and Prediction. Be careful, this can take a longer time!!!
# 
# The lowest MSE obtained with this settings were \num{5.97e-06} for the training set and \num{43.1e-06} for the test set.


y_train_pred_esn = esn.predict(X=X_train)
y_train_pred_elm = elm.predict(X=X_train)
y_test_pred_esn = esn.predict(X=X_test)
y_test_pred_elm = elm.predict(X=X_test)

test_err_esn = mean_squared_error(y_true=y_test, y_pred=y_test_pred_esn)
test_err_elm = mean_squared_error(y_true=y_test, y_pred=y_test_pred_elm)

print("Test MSE ESN:\t{0}".format(test_err_esn))
print("Test MSE ELM:\t{0}".format(test_err_elm))


# Prediction of the training set.

# In[ ]:


fig = plt.figure()
plt.plot(y_test_pred_esn, label="ESN prediction", linewidth=1)
plt.plot(y_test_pred_elm, label="ELM prediction", linewidth=1)
plt.plot(y_test, label="Test target", linewidth=.5, color="black")
plt.xlabel("n")
plt.xlim([0, 200])
plt.ylabel("u[n]")
plt.grid()
plt.legend()
fig.set_size_inches(4, 2.5)
plt.savefig('test_data.pdf', bbox_inches = 'tight', pad_inches = 0)
