import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

from pyrcn.base import InputToNode, FeedbackNodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import FeedbackESNRegressor
from joblib import load


def draw_spectogram(data, ax):
    ax.specgram(data,Fs=4,NFFT=256,noverlap=150,cmap=plt.cm.bone,detrend=lambda x:(x-0.5))
    ax.set_ylim([0,0.5])
    ax.set_ylabel("Frequency")
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_xticks([])


dataset = np.loadtxt(fname=r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\sine_training.csv", delimiter=",", dtype=float)
X = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1]

dataset = np.loadtxt(fname=r"C:\Users\Steiner\Documents\Python\PyRCN\examples\dataset\sine_test.csv", delimiter=",", dtype=float)
X_test = dataset[:, 0].reshape(-1, 1)
y_test = dataset[:, 1]

initially_fixed_esn_params = {'hidden_layer_size': 200,
                              'input_activation': 'identity',
                              'input_scaling': 3.,
                              'bias_scaling': 0.01,
                              'random_state': 1,
                              'reservoir_activation': 'tanh',
                              'leakage': 1.0,
                              'teacher_scaling': 1.12,
                              'teacher_shift': -0.7,
                              'bi_directional': False,
                              'output_activation': 'tanh',
                              'alpha': 1e-3
                              }

param_grid = {'teacher_shift': [-0.4, -0.3, -0.2]}

base_esn = FeedbackESNRegressor(**initially_fixed_esn_params)

for params in ParameterGrid(param_grid):
    esn = clone(base_esn).set_params(**params).fit(X=X, y=y.reshape(-1, 1))

    y_pred = esn.predict(X=X)

    plt.figure(figsize=(10,1.5))
    plt.plot(X, label='Input (Frequency)')
    plt.plot(y, label='Target (Sine)')
    plt.plot(y_pred, label='Predicted (Sine)')
    plt.title('Training')
    plt.xlim([0, len(y_pred)])
    plt.legend()

    f, (ax1, ax2)= plt.subplots(2,1,figsize=(10,3))
    draw_spectogram(y.flatten(), ax1)
    ax1.set_title("Training: Target")
    draw_spectogram(y_pred.flatten(), ax2)
    ax2.set_title("Training: Predicted")

    y_pred = esn.predict(X=X_test)

    plt.figure(figsize=(10,1.5))
    plt.plot(X_test, label='Input (Frequency)')
    plt.plot(y_test, label='Target (Sine)')
    plt.plot(y_pred, label='Predicted (Sine)')
    plt.title('Test')
    plt.xlim([0, len(y_pred)])
    plt.legend()

    f, (ax1, ax2)= plt.subplots(2,1,figsize=(10, 3))
    draw_spectogram(y_test.flatten(), ax1)
    ax1.set_title("Test: Target")
    draw_spectogram(y_pred.flatten(), ax2)
    ax2.set_title("Test: Predicted")
    plt.show()