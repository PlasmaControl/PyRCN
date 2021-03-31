import numpy as np
from matplotlib import pyplot as plt

from pyrcn.base import InputToNode, FeedbackNodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import ESNFeedbackRegressor


dataset = np.loadtxt(fname=r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\sine_training.csv", delimiter=",", dtype=float)
X = dataset[:, 0]
y = dataset[:, 1]

input_to_node = InputToNode(hidden_layer_size=200, activation='identity', input_scaling=3., bias_scaling=0.01, random_state=1)
node_to_node = FeedbackNodeToNode(hidden_layer_size=200, spectral_radius=0.25, sparsity=0.05, teacher_scaling=1.12, teacher_shift=-0.7, random_state=1)
reg = IncrementalRegression(alpha=1e-3)

esn = ESNFeedbackRegressor(input_to_node=input_to_node, node_to_node=node_to_node, regressor=reg, random_state=1)

esn.fit(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
y_pred = esn.predict(X=X.reshape(-1, 1))

plt.figure(figsize=(10,1.5))
plt.plot(X, label='control')
plt.plot(y, label='target')
plt.plot(y_pred, label='model')
plt.title('training (excerpt)')
plt.legend()
plt.show()