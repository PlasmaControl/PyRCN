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

# In[ ]:


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.extreme_learning_machine import ELMRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.datasets import mackey_glass


# Load the dataset and rescale it to a range of [-1, 1]

# In[ ]:


# Load the dataset
X, y = mackey_glass(n_timesteps=20000)
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X=X.reshape(-1, 1))
X = scaler.transform(X=X.reshape(-1, 1))
y = scaler.transform(y.reshape(-1, 1)).ravel()


# Define Train/Test lengths

# In[ ]:


trainLen = 1900 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test/run the network

X_train = X[:trainLen]
y_train = y[:trainLen]
X_test = X[trainLen:trainLen+testLen]
y_test = y[trainLen:trainLen+testLen]


# Visualization

# In[ ]:


fix, axs = plt.subplots()
sns.lineplot(data=X_train.ravel(), ax=axs)
sns.lineplot(data=y_train.ravel(), ax=axs)
axs.set_xlim([0, 1900])
axs.set_xlabel('n')
axs.set_ylabel('u[n]')
plt.legend(["Input", "Target"])


# Training and Prediction using vanilla ESNs and ELMs

# In[ ]:


# initialize an ESNRegressor
esn = ESNRegressor()  # IncrementalRegression()

# initialize an ELMRegressor
elm = ELMRegressor(regressor=Ridge())  # Ridge()

# train a model
esn.fit(X=X_train.reshape(-1, 1), y=y_train)
elm.fit(X=X_train.reshape(-1, 1), y=y_train)

# evaluate the models
y_test_pred = esn.predict(X=X_test)
print(mean_squared_error(y_test, y_test_pred))
y_test_pred = elm.predict(X=X_test)
print(mean_squared_error(y_test, y_test_pred))


# Hyperparameter optimization ESN

# In[ ]:


# Echo State Network sequential hyperparameter tuning
initially_fixed_params = {'hidden_layer_size': 100,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': 1.0,
                          'bi_directional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': False}

step1_esn_params = {'input_scaling': np.linspace(0.1, 5.0, 50),
                    'spectral_radius': np.linspace(0.0, 1.5, 16)}
step2_esn_params = {'leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.5, 16)}

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

kwargs = {'verbose': 5,
          'scoring': scorer,
          'n_jobs': -1,
          'cv': TimeSeriesSplit()}

esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params)

searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs)]


sequential_search_esn = SequentialSearchCV(esn, searches=searches).fit(X_train.reshape(-1, 1), y_train)


# Hyperparameter optimization ELM

# In[ ]:


# Extreme Learning Machine sequential hyperparameter tuning
initially_fixed_elm_params = {'hidden_layer_size': 100,
                              'activation': 'tanh',
                              'k_in': 1,
                              'alpha': 1e-5,
                              'random_state': 42 }

step1_elm_params = {'input_scaling': np.linspace(0.1, 5.0, 50)}
step2_elm_params = {'bias_scaling': np.linspace(0.0, 1.5, 16)}

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

kwargs = {'verbose': 5,
          'scoring': scorer,
          'n_jobs': -1,
          'cv': TimeSeriesSplit()}

elm = ELMRegressor(regressor=Ridge(), **initially_fixed_elm_params)

searches = [('step1', GridSearchCV, step1_elm_params, kwargs),
            ('step2', GridSearchCV, step2_elm_params, kwargs)]

sequential_search_elm = SequentialSearchCV(elm, searches=searches).fit(X_train.reshape(-1, 1), y_train)


# Final prediction and visualization

# In[ ]:


sequential_search_esn.all_best_score_


# In[ ]:


sequential_search_elm.all_best_score_


# In[ ]:


esn = sequential_search_esn.best_estimator_
elm = sequential_search_elm.best_estimator_

y_train_pred_esn = esn.predict(X=X_train)
y_train_pred_elm = elm.predict(X=X_train)
y_test_pred_esn = esn.predict(X=X_test)
y_test_pred_elm = elm.predict(X=X_test)

test_err_esn = mean_squared_error(y_true=y_test, y_pred=y_test_pred_esn)
test_err_elm = mean_squared_error(y_true=y_test, y_pred=y_test_pred_elm)

print("Test MSE ESN:\t{0}".format(test_err_esn))
print("Test MSE ELM:\t{0}".format(test_err_elm))

# Prediction of the test set.
fix, axs = plt.subplots()
sns.lineplot(data=y_test_pred_esn, ax=axs)
sns.lineplot(data=y_test_pred_elm, ax=axs)
axs.set_xlim([0, 1900])
axs.set_xlabel('n')
axs.set_ylabel('u[n]')
plt.legend(["ESN prediction", "ELM prediction"])


# In[ ]:


fig, axs = plt.subplots(1, 2, sharey=True)
sns.heatmap(data=esn.hidden_layer_state[:100, :].T, ax=axs[0], cbar=False)
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Neuron Index")
sns.heatmap(data=elm.hidden_layer_state[:100, :].T, ax=axs[1])
axs[1].set_xlabel("Time Step")


# In[ ]:




