#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction using Echo State Networks
# 
# ## Introduction
# 
# Time Series Prediction is one important regression task, which can be solved using several machine learning techniques.
# 
# In this notebook, we briefly introduce one very basic example for time series prediction:
# 
# Stock Price Prediction
# 
# Disclaimer: We are signal processing experts, not financial advisors. Do not use any of the models presented herein to steer your investments.
# 
# At first, we need to import all required packages

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode


# ## Loading and visualizing stock prices
# 
# A good ressource for stock prices is Yahoo Finance ([https://finance.yahoo.com/](https://finance.yahoo.com/)), where a
# lot of financial data can be downloaded as csv files.
# 
# We have already downloaded several stock prices, which are all stored in the directory "dataset".
# 
# Now, we are working with the gold price in USD (Gold Aug 20(GC=F)) between 2000-02-28 and 2020-05-28.

# In[2]:


X = np.genfromtxt(fname="./examples/dataset/GC=F.csv", usecols=4, skip_header=1,delimiter=",")

unit_impulse = np.zeros(shape=(100, 1), dtype=int)
unit_impulse[5] = 1

plt.figure()
plt.plot(X)
plt.xlim([0, 100])
plt.ylim([0, 1])
plt.xlabel('n')
plt.ylabel('X[n]')
plt.grid()
plt.show()


# Analyzing the dataset, we can analyze the structure. The first column is the date, the second column is the open value,
# e.g. the first value of a day.
# The third column is the highest value of a specific day, the fourth column the lowest and "Close" is the final value of
# a day. The "Adj Close" is a corrected  final value, I DON 'T YET KNOW ANYTHING ABOUT VOLUME!!!
# 
# Here, we just use the "Close" value, the final value of one day. 

# In[3]:


X = X[~np.isnan(X)].reshape(-1, 1)


# As one can see, the stock price trends upwards for the first few thousand days. Then goes down at around $n=3300$ before climbing again after around $n = 4900$. 

# In[4]:


plt.figure(figsize=(4, 2.5))
plt.plot(X)
plt.xlabel("Timestamp")
plt.xlim([0, len(X)])
plt.ylabel("Price")
plt.grid()
plt.tight_layout()
plt.show()


# We pre-processed the dataset by removing undefined values, namely, weekends and public holidays. The remaining values were normalized to be in a range of $[0 1]$.

# In[5]:


train_len = 3000
future_len = 1

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X=X)


# Echo State Network preparation

# In[7]:


base_input_to_nodes = InputToNode(hidden_layer_size=100, activation='identity', k_in=1, input_scaling=0.6, bias_scaling=0.0)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=100, spectral_radius=0.9, leakage=1.0, bias_scaling=0.0, k_rec=10)

esn = ESNRegressor(input_to_node=base_input_to_nodes,
                   node_to_node=base_nodes_to_nodes,
                   regressor=IncrementalRegression(alpha=1e-8), random_state=10)


# Training and Prediction.

# In[8]:

X_train = scaler.transform(X[0:train_len])
y_train = scaler.transform(X[1:train_len+1])
X_test = scaler.transform(X[train_len+1:-1])
y_test = scaler.transform(X[train_len+1+future_len:])

fig = plt.figure()
plt.plot(scaler.transform(X.reshape(-1, 1)))
plt.xlabel("n")
plt.xlim([0, len(X)])
plt.ylabel("u[n]")
plt.grid()
fig.set_size_inches(4, 2.5)
plt.savefig('input_data.pdf', bbox_inches = 'tight', pad_inches = 0)

param_grid = {'input_to_node__hidden_layer_size': [100],
              'input_to_node__input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,  1.1, 1.2, 1.3, 1.4, 1.5],
              'input_to_node__bias_scaling': [0.0],
              'input_to_node__activation': ['identity'],
              'input_to_node__random_state': [42],
              'node_to_node__hidden_layer_size': [100],
              'node_to_node__leakage': [1.0],
              'node_to_node__spectral_radius': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
              'node_to_node__bias_scaling': [0.0],
              'node_to_node__bidirectional': [False],
              'node_to_node__continuation': [True],
              'node_to_node__activation': ['tanh'],
              'node_to_node__wash_out': [0],
              'node_to_node__random_state': [42],
              'regressor__alpha': [1e-5],
              'random_state': [42] }

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

ts_split = TimeSeriesSplit()
grid_search = GridSearchCV(ESNRegressor(), cv=ts_split, param_grid=param_grid, scoring=scorer, n_jobs=-1).fit(X=X_train, y=y_train)

print(grid_search.best_params_)

esn = grid_search.best_estimator_
# esn.set_params(**{'node_to_node__leakage': 0.1})
# esn.fit(X=X_train, y=y_train)
esn.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)
plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(4, 2.5)
plt.savefig('InputScaling_SpectralRadius.pdf', bbox_inches = 'tight', pad_inches = 0)

# esn.fit(X=X_train, y=y_train.ravel())
y_train_pred = esn.predict(X=X_train)
y_test_pred = esn.predict(X=X_test)

train_err = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
test_err = mean_squared_error(y_true=y_test, y_pred=y_test_pred)

print("Train MSE:\t{0}".format(train_err))
print("Test MSE:\t{0}".format(test_err))


# We see that the ESN even captures the downward trend in the test set, although it has not seen any longer downward movement during the training.

# In[9]:

plt.figure()
plt.plot(scaler.inverse_transform(y_test), label='Target')
plt.plot(scaler.inverse_transform(y_test_pred.reshape(-1, 1)), label='Predicted')
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.legend()
plt.show()

# Disclaimer: We are signal processing experts, not financial advisors. Do not use any of the models presented herein to steer your investments.
