#!/usr/bin/env python
# coding: utf-8

# # Prediction of musical notes
# 
# ## Introduction
# 
# This notebook adapts one reference experiment for note prediction using ESNs from ([https://arxiv.org/abs/1812.11527](https://arxiv.org/abs/1812.11527)) to PyRCN and shows that introducing bidirectional ESNs significantly improves the results in terms of Accuracy, already for rather small networks.
# 
# The tutorial is based on numpy, scikit-learn, joblib and PyRCN. We are using the ESNRegressor, because we further process the outputs of the ESN. Note that the same can also be done using the ESNClassifier. Then, during prediction, we simply call "predict_proba".
# 
# This tutorial requires the Python modules numpy, scikit-learn, matplotlib and pyrcn.

# In[ ]:


import numpy as np
from time import process_time
import os
from joblib import load
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import pandas as pd

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from pyrcn.echo_state_network import FeedbackESNRegressor, ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base import InputToNode, NodeToNode, FeedbackNodeToNode


# ## Load the dataset
# 
# The datasets are online available at ([http://www-etud.iro.umontreal.ca/~boulanni/icml2012](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)). In this notebook, we use the pre-processed piano-rolls. They are coming as a serialized file including a dictionary with training, validation and test partitions. In this example, we are using the "piano-midi.de"-datset, because it is relatively small compared to the other datasets.

# In[ ]:


dataset_path = os.path.normpath(r"E:\MusicPrediction\Piano-midi.de.pickle")
dataset = load(dataset_path)
training_set = dataset['train']
validation_set = dataset['valid']
test_set = dataset['test']
print("Number of sequences in the training, validation and test set: {0}, {1}, {2}".format(len(training_set), len(validation_set), len(test_set)))


# ## Prepare the dataset
# 
# We use the MultiLabelBinarizer to transform the sequences of MIDI pitches into one-hot encoded vectors. Although the piano is restricted to 88 keys, we are initializing the MultiLabelBinarizer with 128 possible pitches to stay more general. Note that this does not affect the performance critically. 
# 
# We can see that the sequences have different lenghts, but consist of vector with 128 dimensions.

# In[ ]:


mlb = MultiLabelBinarizer(classes=range(128))
training_set = [mlb.fit_transform(training_set[k]) for k in range(len(training_set))]
validation_set = [mlb.fit_transform(validation_set[k]) for k in range(len(validation_set))]
test_set = [mlb.fit_transform(training_set[k]) for k in range(len(test_set))]
print("Shape of first sequences in the training, validation and test set: {0}, {1}, {2}".format(training_set[0].shape, validation_set[0].shape, test_set[0].shape))


def optimize_esn(training_set, validation_set, params):
    df_data = params
    esn = clone(base_esn)
    esn.set_params(**params)
    t1 = process_time()
    for X in training_set[:-1]:
        esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=True)
    X = training_set[-1]
    esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=False)
    df_data["Fitting Time"] = process_time() - t1
    err_train = []
    t1 = process_time()
    for X in training_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_train.append(mean_squared_error(X[1:, :], y_pred))
    df_data["Inference Time Training"] = process_time() - t1
    err_validation = []
    t1 = process_time()
    for X in validation_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_validation.append(mean_squared_error(X[1:, :], y_pred))
    df_data["Inference Time Validation"] = process_time() - t1
    df_data["Training Loss"] = np.mean(err_train)
    df_data["Validation Loss"] = np.mean(err_validation)
    return df_data


# ## Set up a basic ESN
# 
# To develop an ESN model for musical note prediction, we need to tune several hyper-parameters, e.g., input_scaling, spectral_radius, bias_scaling and leaky integration.
# 
# We follow the way proposed in the introductory paper of PyRCN to optimize hyper-parameters sequentially.
# 
# We start to jointly optimize input_scaling and spectral_radius and therefore deactivate bias connections and leaky integration. This is our base_esn.
# 
# We define the search space for input_scaling and spectral_radius. This is done using best practice and background information from the literature: The spectral radius, the largest absolute eigenvalue of the reservoir matrix, is often smaller than 1. Thus, we can search in a space between 0.0 (e.g. no recurrent connections) and 1.0 (maximum recurrent connections). It is usually recommended to tune the input_scaling factor between 0.1 and 1.0. However, as this is strongly task-dependent, we decided to slightly increase the search space.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50],
    'input_to_node__input_scaling': [0.4],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [1.0],
    'node_to_node__spectral_radius': [0.5],
    'node_to_node__bias_scaling': [0.0],
    'node_to_node__teacher_scaling': np.linspace(start=0.1, stop=15, num=15),
    'node_to_node__teacher_shift': np.linspace(start=-0.9, stop=0.9, num=19),
    'node_to_node__activation': ['tanh'],
    'node_to_node__output_activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-3],
    'random_state': [42] }

base_esn = FeedbackESNRegressor(input_to_node=InputToNode(), node_to_node=FeedbackNodeToNode(), regressor=IncrementalRegression())

df = pd.DataFrame(columns = list(param_grid.keys()) + ["Fitting Time", "Validation Time Training", "Validation Time Test", "Training Loss", "Validation Loss"])

# ## Optimize input_scaling and spectral_radius
# 
# We use the ParameterGrid from scikit-learn, which converts the grid parameters defined before into a list of dictionaries for each parameter combination. 
# 
# We loop over each entry of the Parameter Grid, set the parameters in esn and fit our model on the training data. Afterwards, we report the MSE on the training and validation set.  
# 
#     The lowest training MSE: 0.000238243207656839; parameter combination: {'input_scaling': 0.4, 'spectral_radius': 0.5}
#     The lowest validation MSE: 0.000223548432343247; parameter combination: {'input_scaling': 0.4, 'spectral_radius': 0.5}
# 
# We use the best parameter combination from the validation set.
# 
# As we can see in the python call, we have modified the training procedure: We use "partial_fit" in order to present the ESN all sequences independently from each other. The function "partial_fit" is part of the scikit-learn API. We have added one optional argument "update_output_weights". By default, it is True and thus, after feeding one sequence through the ESN, output weights are computed.
# 
# However, as this is computationally expensive, we can deactivate computing output weights after each sequence by setting "update_output_weights" to False. Now, we simply collect sufficient statistics for the later linear regression. To finish the training process, we call finalize() after passing all sequences through the ESN.

# In[ ]:


for params in ParameterGrid(param_grid):
    df = df.append(optimize_esn(training_set, validation_set, params), ignore_index=True)




# ## Update parameter of the basic ESN
# 
# After optimizing input_scaling and spectral_radius, we update our basic ESN with the identified values for input_scaling and spectral_radius. 
# 
# For the next optimization step, we jointly optimize bias and leakage.
# 
# We define the search space for bias and leakage. This is again done using best practice and background information from the literature: The bias often lies in a similar value range as the input scaling. Thus we use exactly the same search space as before. The leakage, the parameter of the leaky integration is defined in (0.0, 1.0]. Thus, we tune the leakage between 0.1 and 1.0.

# In[ ]:


param_grid = {'input_to_node__hidden_layer_size': [50],
    'input_to_node__input_scaling': [0.4],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': np.linspace(start=0.1, stop=1, num=10),
    'node_to_node__spectral_radius': 0.5,
    'node_to_node__bias_scaling': np.linspace(start=0.0, stop=1, num=11),
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-3],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Optimize bias and leakage
# 
# The optimization workflow is exactly the same as before: We define a ParameterGrid, loop over each entry, set the parameters in esn and fit our model on the training data. Afterwards, we report the MSE on the training and validation set.  
# 
#     The lowest training MSE: 0.000229618469284352; parameter combination: {'bias': 0.8, 'leakage': 0.2}
#     The lowest validation MSE: 0.000213898523704083; parameter combination: {'bias': 0.1, 'leakage': 0.2}
# 
# We use the best parameter combination from the validation set.

# In[ ]:


for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for X in training_set[:-1]:
        esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=True)
    X = training_set[-1]
    esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=False)
    err_train = []
    for X in training_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_train.append(mean_squared_error(X[1:, :], y_pred))
    err_test = []
    for X in validation_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_test.append(mean_squared_error(X[1:, :], y_pred))
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
    'input_to_node__input_scaling': [0.4],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [0.2],
    'node_to_node__spectral_radius': 0.5,
    'node_to_node__bias_scaling': [0.1],
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Optimize beta
# 
# The optimization workflow is exactly the same as before: We define a ParameterGrid, loop over each entry, set the parameters in esn and fit our model on the training data. Afterwards, we report the MSE on the training and test set.  
# 
#     The lowest training MSE: 0.00012083938686566446; parameter combination: {'beta': 5e-4}
#     The lowest validation MSE: 0.00011885985457347002; parameter combination: {'beta': 5e-3}
# 
# We use the best parameter combination from the validation set, because the regularization is responsible to prevent overfitting on the training set. In a running system, of course, we should determine the regularization on a separate validation set.

# In[ ]:


for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for X in training_set[:-1]:
        esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=True)
    X = training_set[-1]
    esn.partial_fit(X=X[:-1, :], y=X[1:, :], postpone_inverse=False)
    err_train = []
    for X in training_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_train.append(mean_squared_error(X[1:, :], y_pred))
    err_test = []
    for X in validation_set:
        y_pred = esn.predict(X=X[:-1, :])
        err_test.append(mean_squared_error(X[1:, :], y_pred))
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


param_grid = {'input_to_node__hidden_layer_size': [500, 1000, 2000, 4000, 5000],
    'input_to_node__input_scaling': [0.4],
    'input_to_node__bias_scaling': [0.0],
    'input_to_node__activation': ['identity'],
    'input_to_node__random_state': [42],
    'node_to_node__hidden_layer_size': [50],
    'node_to_node__leakage': [0.2],
    'node_to_node__spectral_radius': 0.5,
    'node_to_node__bi_directional': [False, True],
    'node_to_node__bias_scaling': [0.1],
    'node_to_node__activation': ['tanh'],
    'node_to_node__random_state': [42],
    'regressor__alpha': [5e-3],
    'random_state': [42] }

base_esn = ESNClassifier(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())


# ## Test the ESN
# 
# In the test case, we train the ESN using the entire training and validation set as seen before. Next, we compute the predicted outputs on the training, validation and test set and fix a threshold of 0.5, above a note is assumed to be predicted.
# 
# We report the accuracy score for each frame in order to follow the reference paper. 
# 
# As can be seen, the bidirectional mode has a very strong impact on the classification result.

# In[ ]:


from sklearn.metrics import accuracy_score
for params in ParameterGrid(param_grid):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    esn.node_to_node.hidden_layer_size = params["input_to_node__hidden_layer_size"]
    esn.finalize()
    err_train = []
    for X in training_set + validation_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)
        y_pred_bin = np.asarray(y_pred > 0.1, dtype=int)
        err_train.append(accuracy_score(y_true=X[1:, :], y_pred=y_pred_bin))
    err_test = []
    for X in test_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)
        print(np.sum(y_pred, axis=0))
        y_pred_bin = np.asarray(y_pred > 0.1, dtype=int)
        err_test.append(accuracy_score(y_true=X[1:, :], y_pred=y_pred_bin))
    print('{0}\t{1}'.format(np.mean(err_train), np.mean(err_test)))
    


# In[ ]:




