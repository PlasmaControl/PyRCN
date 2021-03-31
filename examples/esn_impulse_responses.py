#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction using Echo State Networks
# 
# ## Introduction
# 
# In this notebook, the impact of different hyper-parameters of an ESN are explained. The notebook depends on just a small variety of packages: numpy, matplotlib, IPython and pyrcn.

# In[1]:


import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from pyrcn.echo_state_network import ESNRegressor
from sklearn.linear_model import Ridge
from pyrcn.base import InputToNode, NodeToNode


# In this notebook, we feed an example impulse through the ESN, treating it as a non-linear filter. The unit impuls starts at n=5.

# In[2]:


X = np.zeros(shape=(100, 1), dtype=int)
X[5] = 1
y = X[:, 0]

plt.figure()
plt.plot(X)
plt.xlim([0, 100])
plt.ylim([0, 1])
plt.xlabel('n')
plt.ylabel('X[n]')
plt.grid()
plt.show()


# At first, show the impact of different input scaling factors.
# 
# Therefore, we neutralize the other hyper-parameters, i.e., no recurrent connections ($\rho = 0$), no bias ($\alpha_{\mathrm{b}} = 0$) and no leakage ($\lambda = 1$). 

# In[3]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=0.1, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.0, leakage=1.0, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# Visualizing this, we can see exactly what we might expect. We have chosen an input scaling factor of 0.1. Thus, the reservoir state is non-zero for exactly one sample. We can see that all reservoir states are zero all the times except for $n=5$, when the impulse is fed into the ESN. 
# 
# The absolute values of the reservoir states lie between 0 and 0.1.

# In[5]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# What happens if we increase the input scaling factor?

# In[7]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.0, leakage=1.0, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# Each reservoir state still has only one non-zero value at $n=5$ as before, just with higher activations up to 0.8. The $\tanh$ non-linearity is damping the reservoir states so that they cannot reach 1.  

# In[8]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# Let's keep the input scaling factor fixed to 1.0 for the next experiments. Echo State Networks have two ways to
# incorporate past information.
# 
# Next, let us analyze the spectral radius. Therefore, we set it to 0.3 in the following example.

# In[9]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.3, leakage=1.0, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# We can observe that the impulse responses are starting at n=5 and decaying until reaching zero after a short time. Obviously, the reservoir states are decaying rather fast, because the recurrent connections are small compared to the input scaling. 

# In[10]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# It is interesting to increase the spectral radius. Therefore, we set it to 0.9.

# In[11]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.9, leakage=1.0, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# The values are still bounded between [0.8, 0.8], but the reservoir states are active over a longer time now ($n\approx 30$).

# In[12]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# We can try to further increase the spectral radius. Therefore, we set it to 1.0.

# In[13]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=1.0, leakage=1.0, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# As we expected, the values are getting slightly larger, but are still bounded because of the tanh()-non-linearities.
# Interestingly, the reservoir states are not completely decaying, although the spectral radius was just increased by 0.1.
# 
# We can see that the reservoir states are decaying very slowly, and they are oscillating with a resonance frequency. For many tasks, it is indeed necessary to preserve the echo state property of reservoir and keep $\rho < 1$. However in some cases, such as time-series prediction in this paper, the spectral radius can be larger than 1. 

# In[14]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# For some use cases, the reservoir should behave more non-linear. Therefore, we can play around with the bias. Here, we
# will show the impact of the bias scaling factor just for one example and increase it to 0.2.
# 
# The spectral radius is decreased to 0.9 in order to fulfil the Echo State Property.

# In[15]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.9, leakage=1.0, bias_scaling=0.2, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# Two impacts of the bias scaling can be mainly observed: (1) The absolute value of the stable states of the reservoir neurons is approximately distributed from 0 to 0.2 and each neuron has its own stable state. When new information from the input is passed to the reservoir neurons, this is the excitation point. (2) Before the impulse arrives in the reservoir ($n=5$), the states are approaching their stable state. Due to the spectral radius, each reservoir neuron is connected to other neurons and thus feeds the constant bias through the network, until each neuron has reached its final state.

# In[16]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# Finally, let us observe the impact of the leakage. Leaky integration is the other way to incorporate past information
# into the reservoir. This works by keeping the previous reservoir states over a longer time.

# In[17]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.9, leakage=0.3, bias_scaling=0.0, k_rec=10),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# The leakage behaves in the same way for all nodes in the reservoir and acts like a low-pass filter. The magnitude is strongly damped, and all reservoir states are decaying exponentially over a longer time. Due to the spectral radius, all neurons have individual decaying times.

# In[18]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# If we would like to incorporate future information into the reservoir, we can pass our input samples for- and backward
# through the reservoir. Therefore, we set bidirectional to True.

# In[19]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.9, leakage=0.3, bias_scaling=0.0, k_rec=10, bi_directional=True),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# We can see the clear effect of usind bidirectional ESNs:
# 
# Because of the additional backward-pass, the number of reservoir states is doubled, and we can see that they decay in
# both, forward and backward direction. This is especially useful for some classification tasks.

# In[20]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# The ESN from the Mackey-Glass dataset with a reduced number of neurons

# In[21]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=1.0, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=1.2, leakage=1.0, bias_scaling=0.0, k_rec=10, bi_directional=False),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# Visualization of the Mackey-Glass-ESN

# In[22]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# The ESN from the Stock-Price dataset with a reduced number of neurons

# In[23]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=0.6, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.9, leakage=1.0, bias_scaling=0.0, k_rec=10, bi_directional=False),
                   regressor=Ridge(alpha=1e-6), random_state=10)

esn.fit(X=X, y=y)
_ = esn.predict(X=X)


# Visualization of the Stock-Price-ESN

# In[24]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# The ESN from the Multipitch dataset with a reduced number of neurons

# In[25]:


esn = ESNRegressor(input_to_node=InputToNode(hidden_layer_size=50, activation='identity', k_in=1, input_scaling=0.6, bias_scaling=0.0),
                   node_to_node=NodeToNode(hidden_layer_size=50, spectral_radius=0.2, leakage=0.3, bias_scaling=0.0, k_rec=10, bi_directional=False),
                   regressor=Ridge(alpha=1e-8), random_state=10)

esn.fit(X=X, y=y, n_jobs=0)
_ = esn.predict(X=X)


# Visualization of the Multipitch-ESN

# In[26]:


plt.figure()
im = plt.imshow(np.abs(esn._node_to_node._hidden_layer_state[:, 1:].T),vmin=0, vmax=1)

plt.xlim([0, 100])
plt.ylim([0, esn._node_to_node._hidden_layer_state[:50, 1:].shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
plt.show()


# In[ ]:




