# import required packages
import numpy as np
from joblib import dump, load
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
from pyrcn.model_selection import SequentialSearchCV

# Load the dataset
X = np.loadtxt("./examples/dataset/MackeyGlass_t17.txt")
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X=X.reshape(-1,1))

# Define Train/Test lengths
trainLen = 1900 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test/run the network

X_train = X[0:trainLen].reshape(-1, 1)
y_train = X[1:trainLen+1].reshape(-1, 1)
X_test = X[trainLen:trainLen+testLen].reshape(-1, 1)
y_test = X[trainLen+1:trainLen+testLen+1].reshape(-1, 1)

fig = plt.figure()
plt.plot(X_train, label="Training input")
plt.plot(y_train, label="Training target")
plt.xlabel("n")
plt.xlim([0, 200])
plt.ylabel("u[n]")
plt.grid()
plt.legend()
fig.set_size_inches(4, 2.5)
plt.savefig('input_data.pdf', bbox_inches='tight', pad_inches=0)

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

# Echo State Network sequential hyperparameter tuning
initially_fixed_esn_params = {'input_to_node__hidden_layer_size': 100,
                              'input_to_node__activation': 'identity',
                              'input_to_node__k_in': 10,
                              'input_to_node__random_state': 42,
                              'input_to_node__bias_scaling': 0.0,
                              'input_to_node__k_in': 1,
                              'input_to_node__random_state': 42,
                              'node_to_node__hidden_layer_size': 100,
                              'node_to_node__activation': 'tanh',
                              'node_to_node__leakage': 1.0,
                              'node_to_node__bias_scaling': 0.0,
                              'node_to_node__bi_directional': False,
                              'node_to_node__k_rec': 10,
                              'node_to_node__wash_out': 0,
                              'node_to_node__continuation': False,
                              'node_to_node__random_state': 42,
                              'regressor__alpha': 1e-5,
                              'random_state': 42 }
}
step1_esn_params = {'input_to_node__input_scaling': np.linspace(0.1, 5.0, 50),
                    'node_to_node__spectral_radius': np.linspace(0.0, 1.5, 16)}
step2_esn_params = {'node_to_node__leakage': np.linspace(0.1, 1.0, 10)}
step3_esn_params = {'node_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

kwargs = {'verbose': 5,
          'scoring': scorer,
          'n_jobs': -1}

esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=Ridge()).set_params(**initially_fixed_esn_params)


ts_split = TimeSeriesSplit()
searches = [('step1', GridSearchCV, step1_esn_params, kwargs),
            ('step2', GridSearchCV, step2_esn_params, kwargs),
            ('step3', GridSearchCV, step3_esn_params, kwargs)]


sequential_search_esn = SequentialSearchCV(esn, searches=searches).fit(X_train, y_train)
dump(sequential_search_esn, "sequential_search_esn.joblib")

esn_step1 = sequential_search_esn.all_best_estimator_["step1"]
esn_step2 = sequential_search_esn.all_best_estimator_["step2"]
esn_step3 = sequential_search_esn.all_best_estimator_["step3"]

esn_step1.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(esn_step1.hidden_layer_state[:, 1:].T),vmin=0, vmax=1.0)
plt.xlim([0, 100])
plt.ylim([0, esn_step1.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('InputScaling_SpectralRadius.pdf', bbox_inches = 'tight', pad_inches = 0)

esn_step2.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(esn_step2.hidden_layer_state.T),vmin=0, vmax=1.0)
plt.xlim([0, 100])
plt.ylim([0, esn_step2.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('Leakage.pdf', bbox_inches = 'tight', pad_inches = 0)

esn_step2_1 = esn_step2.set_params(**{"node_to_node__leakage": 0.1}).fit(X_train, y_train)
fig = plt.figure()
im = plt.imshow(np.abs(esn_step2_1.hidden_layer_state.T),vmin=0, vmax=1.0)
plt.xlim([0, 100])
plt.ylim([0, esn_step2_1.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('Leakage_low.pdf', bbox_inches = 'tight', pad_inches = 0)


esn_step3.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(esn_step3.hidden_layer_state.T),vmin=0, vmax=0.3)
plt.xlim([0, 100])
plt.ylim([0, esn_step3.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('BiasScalingESN.pdf', bbox_inches = 'tight', pad_inches = 0)


# Extreme Learning Machine sequential hyperparameter tuning
initially_fixed_elm_params = {'input_to_node__hidden_layer_size': 100,
                              'input_to_node__activation': 'tanh',
                              'input_to_node__k_in': 10,
                              'input_to_node__random_state': 42,
                              'input_to_node__bias_scaling': 0.0,
                              'input_to_node__k_in': 1,
                              'input_to_node__random_state': 42,
                              'regressor__alpha': 1e-5,
                              'random_state': 42 
}
step1_elm_params = {'input_to_node__input_scaling': np.linspace(0.1, 5.0, 50)}
step2_elm_params = {'input_to_node__bias_scaling': np.linspace(0.0, 1.5, 16)}

scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)

kwargs = {'verbose': 10,
          'scoring': scorer,
          'n_jobs': -1}

elm = ELMRegressor(input_to_node=InputToNode(), regressor=Ridge()).set_params(**initially_fixed_elm_params)


ts_split = TimeSeriesSplit()
searches = [('step1', GridSearchCV, step1_elm_params, kwargs),
            ('step2', GridSearchCV, step2_elm_params, kwargs)]


sequential_search_elm = SequentialSearchCV(elm, searches=searches).fit(X_train, y_train)
dump(sequential_search_elm, "sequential_search_elm.joblib")

elm_step1 = sequential_search_elm.all_best_estimator_["step1"]
elm_step2 = sequential_search_elm.all_best_estimator_["step2"]

elm_step1.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(elm_step1.hidden_layer_state.T),vmin=0, vmax=1.0)
plt.xlim([0, 100])
plt.ylim([0, elm_step1.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('InputScaling.pdf', bbox_inches = 'tight', pad_inches = 0)

elm_step2.predict(X=unit_impulse)
fig = plt.figure()
im = plt.imshow(np.abs(elm_step2.hidden_layer_state.T),vmin=0, vmax=1.0)
plt.xlim([0, 100])
plt.ylim([0, elm_step2.hidden_layer_state.shape[1] - 1])
plt.xlabel('n')
plt.ylabel('R[n]')
plt.colorbar(im)
plt.grid()
fig.set_size_inches(2, 1.25)
plt.savefig('BiasScalingELM.pdf', bbox_inches = 'tight', pad_inches = 0)

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
