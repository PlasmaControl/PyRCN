#!/usr/bin/env python
# coding: utf-8

# # Recognizing hand-written digits
# 
# ## Introduction
# 
# This notebook adapts the existing example of applying support vector classification from scikit-learn ([https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)) to PyRCN to demonstrate, how PyRCN can be used to classify hand-written digits.
# 
# The tutorial is based on numpy, scikit-learn and PyRCN. 

# In[ ]:


import numpy as np
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import SeqToLabelESNClassifier
from pyrcn.metrics import accuracy_score
from pyrcn.datasets import load_digits


# ## Load the dataset
# 
# The dataset is already part of scikit-learn and consists of 1797 8x8 images. 
# 
# We are using our dataloader that is derived from scikit-learns dataloader and returns arrays of 8x8 sequences and corresponding labels.

# In[ ]:


X, y = load_digits(return_X_y=True, as_sequence=True)
print("Number of digits: {0}".format(len(X)))
print("Shape of digits {0}".format(X[0].shape))


# ## Split dataset in training and test
# 
# Afterwards, we split the dataset into training and test sets. We train the ESN using 80% of the digits and test it using the remaining images. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print("Number of digits in training set: {0}".format(len(X_train)))
print("Shape of digits in training set: {0}".format(X_train[0].shape))
print("Number of digits in test set: {0}".format(len(X_test)))
print("Shape of digits in test set: {0}".format(X_test[0].shape))


# ## Set up a ESN
# 
# To develop an ESN model for digit recognition, we need to tune several hyper-parameters, e.g., input_scaling, spectral_radius, bias_scaling and leaky integration.
# 
# We follow the way proposed in the introductory paper of PyRCN to optimize hyper-parameters sequentially.
# 
# We define the search spaces for each step together with the type of search (a grid search in this context).
# 
# At last, we initialize a SeqToLabelESNClassifier with the desired output strategy and with the initially fixed parameters.

# In[ ]:


initially_fixed_params = {'hidden_layer_size': 50,
                          'input_activation': 'identity',
                          'k_in': 5,
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': 1.0,
                          'bi_directional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42}

step1_esn_params = {'input_scaling': np.linspace(0.1, 1.0, 10),
                    'spectral_radius': np.linspace(0.0, 1.5, 16)}

step2_esn_params = {'leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]}

kwargs = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs),
            ('step4', GridSearchCV, step4_esn_params, kwargs)]

base_esn = SeqToLabelESNClassifier(output_strategy="winner_takes_all", **initially_fixed_params)


# ## Optimization
# 
# We provide a SequentialSearchCV that basically iterates through the list of searches that we have defined before. It can be combined with any model selection tool from scikit-learn.

# In[ ]:


sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)


# ## Use the ESN with final hyper-parameters
# 
# After the optimization, we extract the ESN with final hyper-parameters as the result of the optimization.

# In[ ]:


base_esn = sequential_search.best_estimator_


# ## Test the ESN
# 
# Finally, we increase the reservoir size and compare the impact of uni- and bidirectional ESNs. Notice that the ESN strongly benefit from both, increasing the reservoir size and from the bi-directional working mode.

# In[ ]:


param_grid = {'hidden_layer_size': [50, 100, 200, 400, 500],
              'bi_directional': [False, True]}

print("CV results\tFit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in ParameterGrid(param_grid):
    esn_cv = cross_validate(clone(base_esn).set_params(**params), X=X_train, y=y_train, scoring=make_scorer(accuracy_score), n_jobs=-1)
    t1 = time.time()
    esn = clone(base_esn).set_params(**params).fit(X_train, y_train)
    t_fit = time.time() - t1
    mem_size = esn.__sizeof__()
    t1 = time.time()
    acc_score = accuracy_score(y_test, esn.predict(X_test))
    t_inference = time.time() - t1
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(esn_cv, t_fit, t_inference, acc_score, mem_size))


# In[ ]:




