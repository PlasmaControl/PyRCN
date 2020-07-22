#!/usr/bin/env python
# coding: utf-8

# # Recognizing hand-written digits
# 
# ## Introduction
# 
# This notebook adapts the existing example of applying support vector classification from scikit-learn ([https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)) to PyRCN to demonstrate, how PyRCN can be used to classify hand-written digits.
# 
# The tutorial is based on numpy, scikit-learn and PyRCN. We are using the ESNRegressor, because we further process the outputs of the Echo State Networks.

# In[1]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, accuracy_score

from pyrcn.echo_state_network import ESNRegressor


# ## Load the dataset
# 
# The dataset is already part of scikit-learn and consists of 8x8 images. The dataset has almost 1800 images.

# In[2]:


digits = load_digits()
data = digits.images
print("Number of digits: {0}".format(len(data)))
print("Shape of digits {0}".format(data[0].shape))


# ## Split dataset in training and test
# 
# We use the OneHotEncoder to transform the target output into one-hot encoded values. 
# 
# Afterwards, we split the dataset into training and test sets. We train the ESN using 50% of the digits and test it using the remaining images. 
# 
# We treat each image as a sequence of 8 feature vectors with 8 dimensions.

# In[3]:


# Split data into train and test subsets
enc = OneHotEncoder(sparse=False)
y = enc.fit_transform(X=digits.target.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.5, shuffle=False)
print("Number of digits in training set: {0}".format(len(X_train)))
print("Shape of digits in training set: {0}".format(X_train[0].shape))
print("Shape of output in training set: {0}".format(y_train[0].shape))
print("Number of digits in test set: {0}".format(len(X_test)))
print("Shape of digits in test set: {0}".format(X_test[0].shape))
print("Shape of output in test set: {0}".format(y_test[0].shape))


# ## Set up ESN and a parameter grid to optimize input scaling and spectral radius
# 
# For the hyperparameter optimization, we start to jointly optimize input_scaling and spectral_radius.
# 
# We define a base_reg with deactivated recurrent connections and leaky integration. 
# 
# We define the search space for input_scaling and spectral_radius.

# In[4]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 0.0, bias = 0.0, leakage = 1.0, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 
        'spectral_radius': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
       }


# ## Loop through the grid
# 
# We loop over each combination of the Parameter Grid, set the parameters in reg and fit our model on the training data.
# 
# For each parameter combination, we report the MSE on the training and test set. 
# 
# The lowest MSE in the training and test case is obtained with the combination 

# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    err_train = []
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_train.append(mean_squared_error(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_test.append(mean_squared_error(y, y_pred))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))
        
    


# Optimize bias and leakage

# In[ ]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 1.2, bias = 0.0, leakage = 1.0, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'bias': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 
        'leakage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }


# The same as before

# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    err_train = []
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_train.append(mean_squared_error(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        err_test.append(mean_squared_error(y, y_pred))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))
        
    


# In[ ]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 0.0, bias = 0.0, leakage = 1.0, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 
        'spectral_radius': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
       }


# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_train.append(np.argmax(y))
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_test.append(np.argmax(y))
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
    print('{0}\t{1}'.format(accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_test, Y_pred_test)))


# In[ ]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 1.2, bias = 0.0, leakage = 1.0, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'bias': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 
        'leakage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
       }


# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_train.append(np.argmax(y))
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_test.append(np.argmax(y))
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
    print('{0}\t{1}'.format(accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_test, Y_pred_test)))


# ## Test: Increasing the reservoir size
# 

# In[ ]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 1.2, bias = 0.6, leakage = 0.4, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'beta': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0], 
       }


# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_train.append(np.argmax(y))
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_test.append(np.argmax(y))
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
    print('{0}\t{1}'.format(accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_test, Y_pred_test)))


# In[ ]:


base_reg = ESNRegressor(k_in = 5, input_scaling = 0.1, spectral_radius = 1.2, bias = 0.6, leakage = 0.4, reservoir_size = 50, 
                   k_res = 5, reservoir_activation = 'tanh', teacher_scaling = 1.0, teacher_shift = 0.0, 
                   bi_directional = False, solver = 'ridge', beta = 5e-3, random_state = 1)

grid = {'reservoir_size': [50, 100, 200, 400, 500, 800, 1000], 
        'bi_directional': [False, True]
       }


# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    reg = clone(base_reg)
    reg.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        reg.partial_fit(X=X, y=y, update_output_weights=False)
    reg.finalize()
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_train.append(np.argmax(y))
        Y_pred_train.append(np.argmax(y_pred.sum(axis=0)))
    
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        y_pred = reg.predict(X=X, keep_reservoir_state=False)
        Y_true_test.append(np.argmax(y))
        Y_pred_test.append(np.argmax(y_pred.sum(axis=0)))
    print('{0}\t{1}'.format(accuracy_score(Y_true_train, Y_pred_train), accuracy_score(Y_true_test, Y_pred_test)))

