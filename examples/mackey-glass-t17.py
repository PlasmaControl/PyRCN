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
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode


# Load the dataset

# In[ ]:


data = np.loadtxt("examples/dataset/MackeyGlass_t17.txt")


# The first 500 samples are visualized.

# In[ ]:


plt.figure()
plt.plot(data[:500])
plt.xlabel("n")
plt.ylabel("X[n]")
plt.grid()
plt.show()


# Standardization -> From here on, we have a numpy array!!!

# In[ ]:


data = data / (data.max() - data.min())


# Define Train/Test lengths

# In[ ]:


initLen = 100 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = initLen + 1900 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test/run the network


# Echo State Network preparation

# In[ ]:


base_input_to_nodes = InputToNode(hidden_layer_size=500, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0)
base_nodes_to_nodes = NodeToNode(hidden_layer_size=500, spectral_radius=1.2, leakage=1.0, bias_scaling=0.0, k_rec=10)

esn = ESNRegressor(input_to_node=base_input_to_nodes,
                   node_to_node=base_nodes_to_nodes,
                   regressor=IncrementalRegression(alpha=1e-4), random_state=10)


# Training and Prediction. Be careful, this can take a longer time!!!
# 
# The lowest MSE obtained with this settings were \num{5.97e-06} for the training set and \num{43.1e-06} for the test set.

# In[ ]:


train_in = data[None,0:trainLen]
train_out = data[None,0+1:trainLen+1]
test_in = data[None,trainLen:trainLen+testLen]
test_out = data[None,trainLen+1:trainLen+testLen+1]

train_in, train_out = train_in.T, train_out.T
test_in, test_out = test_in.T, test_out.T

esn.fit(X=train_in, y=train_out)
train_pred = esn.predict(X=train_in)
test_pred = esn.predict(X=test_in)

train_err = mean_squared_error(y_true=train_out, y_pred=train_pred)
test_err = mean_squared_error(y_true=test_out, y_pred=test_pred)

print("Train MSE:\t{0}".format(train_err))
print("Test MSE:\t{0}".format(test_err))


# Prediction of the training set.

# In[ ]:


plt.figure()
plt.plot(train_out)
plt.plot(train_pred)
plt.xlabel("n")
plt.ylabel("X[n]")
plt.show()


# Prediction of the test set.
# 

# In[ ]:


plt.figure()
plt.plot(test_out)
plt.plot(test_pred)
plt.xlabel("n")
plt.ylabel("X[n]")
plt.show()


# In[ ]:




