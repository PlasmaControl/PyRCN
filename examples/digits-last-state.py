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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import zero_one_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
get_ipython().run_line_magic('matplotlib', 'inline')

from pyrcn.echo_state_network import ESNClassifier
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode


# ## Load the dataset
# 
# The dataset is already part of scikit-learn and consists of 1797 8x8 images. We are using the dataloader from scikit-learn.

# In[ ]:


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

# In[ ]:


y = digits.target
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.5, shuffle=False)
print("Number of digits in training set: {0}".format(len(X_train)))
print("Shape of digits in training set: {0}".format(X_train[0].shape))
print("Shape of output in training set: {0}".format(y_train[0].shape))
print("Number of digits in test set: {0}".format(len(X_test)))
print("Shape of digits in test set: {0}".format(X_test[0].shape))
print("Shape of output in test set: {0}".format(y_test[0].shape))


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


param_grid = {'input_to_node__hidden_layer_size': [50],
    'input_to_node__input_scaling': np.linspace(start=0.1, stop=1, num=10),
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [1.0],
    'node_to_node__spectral_radius': np.linspace(start=0.0, stop=1, num=11),
    'node_to_node__bias_scaling': [0.0],
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-3],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


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


for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(y, repeats=8, axis=0)
        esn.partial_fit(X=X, y=y.reshape(-1, 1), classes=range(10))
    err_train = []
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_train.append(zero_one_loss(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_test.append(zero_one_loss(y, y_pred))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))


# ## Update parameter of the basic ESN
# 
# After optimizing input_scaling and spectral_radius, we update our basic ESN with the identified values for input_scaling and spectral_radius. 
# 
# For the next optimization step, we jointly optimize bias and leakage.
# 
# We define the search space for bias and leakage. This is again done using best practice and background information from the literature: The bias often lies in a similar value range as the input scaling. Thus we use exactly the same search space as before. The leakage, the parameter of the leaky integration is defined in (0.0, 1.0]. Thus, we tune the leakage between 0.1 and 1.0.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50],
    'input_to_node__input_scaling': [0.1],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': np.linspace(0.1, 1, 10),
    'node_to_node__spectral_radius': [1.0],
    'node_to_node__bias_scaling': np.linspace(0, 1, 11),
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-3],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Optimize bias and leakage
# 
# The optimization workflow is exactly the same as before: We define a ParameterGrid, loop over each entry, set the parameters in reg and fit our model on the training data. Afterwards, we report the error rate on the training and test set.  
# 
#     The lowest training error rate: 0.311804009; parameter combination: {'bias_scaling': 1.0, 'leakage': 0.1}
#     The lowest test error rate: 0.38320356; parameter combination: {'bias_scaling': 0.6, 'leakage': 0.2}
# 
# We use the best parameter combination from the training set, because we do not want to overfit on the test set.
# 
# Note that the bias differs a lot between training and test set. A reason can be that the training set does not completely represent the test set. This should actually be investigated by comparing several train_test_splits, maybe even with other sample sizes.

# In[ ]:


for params in ParameterGrid(grid):
    print(params)
    input_to_node = clone(base_input_to_node)
    node_to_node = clone(base_node_to_node)
    
    node_to_node.set_params(**params)
    esn = ESNClassifier(input_to_node=input_to_node,
                        node_to_node=node_to_node,
                        regressor=FastIncrementalRegression(alpha=5e-3), random_state=10)
    for X, y in zip(X_train, y_train):
        y = np.repeat(y, repeats=8, axis=0)
        esn.partial_fit(X=X, y=y.reshape(-1, 1), classes=range(10))
    err_train = []
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_train.append(zero_one_loss(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_test.append(zero_one_loss(y, y_pred))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))


# ## Update parameter of the basic ESN
# 
# After optimizing bias and leakage, we update our basic ESN with the identified values for bias and leakage. 
# 
# Finally, we would quickly like to see whether the regularization parameter beta lies in the correct range.
# 
# Typically, it is rather difficult to find a proper search range. Here, we use a very rough logarithmic search space.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50],
    'input_to_node__input_scaling': [0.1],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [0.1],
    'node_to_node__spectral_radius': [1.0],
    'node_to_node__bias_scaling': [1.0],
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Optimize alpha
# 
# The optimization workflow is exactly the same as before: We define a ParameterGrid, loop over each entry, set the parameters in reg and fit our model on the training data. Afterwards, we report the error rates on the training and test set.  
# 
#     The lowest training error rate: 0.31096882; parameter combination: {'alpha': 1e-5}
#     The lowest test error rate: 0.396968854; parameter combination: {'alpha': 1e-5}
# 
# We use the best parameter combination from the test set, because the regularization is responsible to prevent overfitting on the training set. In a running system, of course, we should determine the regularization on a separate validation set.

# In[ ]:


for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for X, y in zip(X_train, y_train):
        y = np.repeat(y, repeats=8, axis=0)
        esn.partial_fit(X=X, y=y.reshape(-1, 1), classes=range(10))
    err_train = []
    for X, y in zip(X_train, y_train):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_train.append(zero_one_loss(y, y_pred))
    err_test = []
    for X, y in zip(X_test, y_test):
        y = np.repeat(np.atleast_2d(y), repeats=8, axis=0)
        y_pred = esn.predict(X=X)
        err_test.append(zero_one_loss(y, y_pred))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))


# ## Update parameter of the basic ESN
# 
# After optimizing beta, we update our basic ESN with the identified value for beta.
# 
# Note that we have used almost the ideal value already in the beginning. Thus, the impact is rather small.
# 
# Next, we want to measure the classification accuracy. To do that, we compare several reservoir sizes as well as unidirectional and bidirectional architectures.
# 
# Because this is a rather small dataset, we can use rather small reservoir sizes and increase it up to 5000 neurons.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000],
    'input_to_node__input_scaling': [0.1],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [0.1],
    'node_to_node__spectral_radius': [1.0],
    'node_to_node__bi_directional': [False, True],
    'node_to_node__bias_scaling': [1.0],
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-5],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Test the ESN
# 
# In the test case, we use a simple variant of sequence classification:
# 
# The ESN computes the output for each sequence. For classification, we just use the last output vector of each sequence. The highest index in that output vector is the label of the sequence.
# 
# We store all ground truth labels and the predicted labels for training and test. Then, we use the scikit-learn's classification_report and plot a confusion matrix in order to show the classification performance.
# 
# As can be seen, the reservoir size as a very strong impact on the classification result.

# In[ ]:


for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    esn.node_to_node.hidden_layer_size = params["input_to_node__hidden_layer_size"]
    for X, y in zip(X_train, y_train):
        y = np.repeat(y, repeats=8, axis=0)
        esn.partial_fit(X=X, y=y.reshape(-1, 1), classes=range(10))
    
    Y_true_train = []
    Y_pred_train = []
    for X, y in zip(X_train, y_train):
        y_pred = esn.predict_proba(X=X)
        Y_true_train.append(y)
        Y_pred_train.append(np.argmax(y_pred[-1, :]))
    
    Y_true_test = []
    Y_pred_test = []
    for X, y in zip(X_test, y_test):
        y_pred = esn.predict_proba(X=X)
        Y_true_test.append(y)
        Y_pred_test.append(np.argmax(y_pred[-1, :]))
    cm = confusion_matrix(Y_true_train, Y_pred_train)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
    print("Classification training report for estimator %s:\n%s\n"
      % (esn, classification_report(Y_true_train, Y_pred_train)))
    plt.show()
    
    cm = confusion_matrix(Y_true_test, Y_pred_test)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
    print("Classification test report for estimator %s:\n%s\n"
      % (esn, classification_report(Y_true_test, Y_pred_test)))
    plt.show()


# In[ ]:




