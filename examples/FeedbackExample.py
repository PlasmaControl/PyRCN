import numpy as np
from matplotlib import pyplot as plt

from pyrcn.base import PredefinedWeightsInputToNode, PredefinedWeightsFeedbackNodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNFeedbackRegressor
from joblib import load

W_i2n = load(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\W_in.joblib")
W_in = W_i2n[:, 1].reshape(1, -1)
W_bias = W_i2n[:, 0]
W_rec = load(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\W_rec.joblib")
W_fb = load(r"C:\Users\Steiner\Documents\Python\PyRCN\examples\W_fb.joblib")


dataset = np.loadtxt(fname=r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\sine_training.csv", delimiter=",", dtype=float)
X = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1]

dataset = np.loadtxt(fname=r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\sine_test.csv", delimiter=",", dtype=float)
X_test = dataset[:, 0].reshape(-1, 1)
y_test = dataset[:, 1]

input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=W_in, predefined_bias_weights=W_bias, hidden_layer_size=200, activation='identity', input_scaling=3., bias_scaling=0.01, random_state=1).fit(X=X)
node_to_node = PredefinedWeightsFeedbackNodeToNode(predefined_recurrent_weights=W_rec.T, predefined_feedback_weights=W_fb, hidden_layer_size=200, sparsity=0.05, activation="tanh", spectral_radius=0.25, leakage=1.0, bias_scaling=0.0, teacher_scaling=1.12, teacher_shift=-0.7, bi_directional=False, output_activation="tanh", random_state=1).fit(X=input_to_node.transform(X=X), y=y.ravel())
reg = IncrementalRegression(alpha=0.0, fit_intercept=False)

esn = ESNFeedbackRegressor(input_to_node=input_to_node, node_to_node=node_to_node, regressor=reg, random_state=1)

esn.fit(X=X, y=y.reshape(-1, 1))
y_pred = esn.predict(X=X_test)

plt.figure(figsize=(10,1.5))
plt.plot(X_test, label='control')
plt.plot(y_test, label='target')
plt.plot(y_pred, label='model')
plt.title('training (excerpt)')
plt.legend()
plt.show()