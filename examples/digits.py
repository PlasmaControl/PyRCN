#!/usr/bin/env python
# coding: utf-8

# # Recognizing hand-written digits
# 
# ## Introduction
# 
# This notebook adapts the existing example of applying support vector classification from scikit-learn ([https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)) to PyRCN to demonstrate, how PyRCN can be used to classify hand-written digits.
# 
# The tutorial is based on numpy, scikit-learn and PyRCN. We are using the ESNRegressor, because we further process the outputs of the ESN. Note that the same can also be done using the ESNClassifier. Then, during prediction, we simply call "predict_proba".
# 
# This tutorial requires the Python modules numpy, scikit-learn, matplotlib and pyrcn.

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import make_scorer

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import SeqToLabelESNClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.metrics import accuracy_score
from pyrcn.datasets import load_digits


# ## Load the dataset
# 
# The dataset is already part of scikit-learn and consists of 1797 8x8 images. We are using the dataloader from scikit-learn.

# In[ ]:


X, y = load_digits(return_X_y=True, as_sequence=True)
print("Number of digits: {0}".format(len(X)))
print("Shape of digits {0}".format(X[0].shape))


# ## Split dataset in training and test
# 
# We use the OneHotEncoder to transform the target output into one-hot encoded values. 
# 
# Afterwards, we split the dataset into training and test sets. We train the ESN using 50% of the digits and test it using the remaining images. 
# 
# We treat each image as a sequence of 8 feature vectors with 8 dimensions.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print("Number of digits in training set: {0}".format(len(X_train)))
print("Shape of digits in training set: {0}".format(X_train[0].shape))
print("Number of digits in test set: {0}".format(len(X_test)))
print("Shape of digits in test set: {0}".format(X_test[0].shape))


# ## Set up a basic ESN
# 
# To develop an ESN model for digit recognition, we need to tune several hyper-parameters, e.g., input_scaling, spectral_radius, bias_scaling and leaky integration.
# 
# We follow the way proposed in the introductory paper of PyRCN to optimize hyper-parameters sequentially.
# 
# We start to jointly optimize input_scaling and spectral_radius and therefore deactivate bias connections and leaky integration. This is our base_reg.
# 
# We define the search space for input_scaling and spectral_radius. This is done using best practice and background information from the literature: The spectral radius, the largest absolute eigenvalue of the reservoir matrix, is often smaller than 1. Thus, we can search in a space between 0.0 (e.g. no recurrent connections) and 1.0 (maximum recurrent connections). It is usually recommended to tune the input_scaling factor between 0.1 and 1.0. However, as this is strongly task-dependent, we decided to slightly increase the search space.

# In[ ]:


# Prepare sequential hyperparameter tuning
initially_fixed_params = {'input_to_node__hidden_layer_size': 50,
                          'input_to_node__activation': 'identity',
                          'input_to_node__k_in': 5,
                          'input_to_node__random_state': 42,
                          'input_to_node__bias_scaling': 0.0,
                          'node_to_node__hidden_layer_size': 50,
                          'node_to_node__activation': 'tanh',
                          'node_to_node__leakage': 1.0,
                          'node_to_node__bias_scaling': 0.0,
                          'node_to_node__bi_directional': False,
                          'node_to_node__k_rec': 10,
                          'node_to_node__wash_out': 0,
                          'node_to_node__continuation': False,
                          'node_to_node__random_state': 42,
                          'regressor__alpha': 1e-5,
                          'random_state': 42}

step1_esn_params = {'input_to_node__input_scaling': np.linspace(0.1, 1.0, 10),
                    'node_to_node__spectral_radius': np.linspace(0.0, 1.5, 16)}

step2_esn_params = {'node_to_node__leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'node_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}
step4_esn_params = {'regressor__alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]}

kwargs = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs),
            ('step4', GridSearchCV, step4_esn_params, kwargs)]

base_esn = SeqToLabelESNClassifier().set_params(**initially_fixed_params)


# ## Optimize input_scaling and spectral_radius
# 
# We use the ParameterGrid from scikit-learn, which converts the grid parameters defined before into a list of dictionaries for each parameter combination. 
# 
# We loop over each entry of the Parameter Grid, set the parameters in reg and fit our model on the training data. Afterwards, we report the error rates on the training and test set.  
# 
#     The lowest training error rate: 0.536330735; parameter combination: {'input_scaling': 0.1, 'spectral_radius': 1.0}
#     The lowest test error rate: 0.588987764; parameter combination: {'input_scaling': 0.1, 'spectral_radius': 1.0}
# 
# We use the best parameter combination from the training set, because we do not want to overfit on the test set.
# 
# As we can see in the python call, we have modified the training procedure: We use "partial_fit" in order to present the ESN all sequences independently from each other. The function "partial_fit" is part of the scikit-learn API. We have added one optional argument "update_output_weights". By default, it is True and thus, after feeding one sequence through the ESN, output weights are computed.
# 
# However, as this is computationally expensive, we can deactivate computing output weights after each sequence by setting "update_output_weights" to False. Now, we simply collect sufficient statistics for the later linear regression. To finish the training process, we call finalize() after passing all sequences through the ESN.

# In[ ]:


sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train, y_train)


# ## Update parameter of the basic ESN
# 
# After optimizing input_scaling and spectral_radius, we update our basic ESN with the identified values for input_scaling and spectral_radius. 
# 
# For the next optimization step, we jointly optimize bias and leakage.
# 
# We define the search space for bias and leakage. This is again done using best practice and background information from the literature: The bias often lies in a similar value range as the input scaling. Thus we use exactly the same search space as before. The leakage, the parameter of the leaky integration is defined in (0.0, 1.0]. Thus, we tune the leakage between 0.1 and 1.0.

# In[ ]:

final_fixed_params = initially_fixed_params
final_fixed_params.update(sequential_search.all_best_params_["step1"])
final_fixed_params.update(sequential_search.all_best_params_["step2"])
final_fixed_params.update(sequential_search.all_best_params_["step3"])
final_fixed_params.update(sequential_search.all_best_params_["step4"])

base_esn = SeqToLabelESNClassifier().set_params(**final_fixed_params)


# ## Test the ESN
# 
# In the test case, we use a simple variant of sequence classification:
# 
# The ESN computes the output for each sequence. We integrate the outputs over time and find the highest integrated output index. This is the label of the sequence.
# 
# We store all ground truth labels and the predicted labels for training and test. Then, we use the scikit-learn's classification_report and plot a confusion matrix in order to show the classification performance.
# 
# As can be seen, the reservoir size as a very strong impact on the classification result.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000]}

print("CV results\tFit time\tInference time\tAccuracy score\tSize[Bytes]")
for params in reversed(ParameterGrid(param_grid)):
    params["node_to_node__hidden_layer_size"] = params["input_to_node__hidden_layer_size"]
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




