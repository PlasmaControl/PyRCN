"""
An example of the Coates Idea on the digits dataset.
"""
import os
import numpy as np

from sklearn.decomposition import PCA
from pyrcn.base.blocks import InputToNode

from src.pyrcn.util import tud_colors, get_mnist

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()
example_image_idx = 5
min_var = 3088.6875

# EXPERIMENTS


def input2node_distribution(directory):
    self_name = 'input2node_distribution'
    X, y = get_mnist(directory)

    X /= 255.

    pca = PCA(n_components=784).fit(X)
    X_pca = np.matmul(X, pca.components_.T)

    list_activation = ['tanh', 'relu', 'bounded_relu']
    list_train = [X, X_pca]

    fig, axs = plt.subplots(nrows=2, ncols=3)

    for idx_activation in range(len(list_activation)):
        activation = list_activation[idx_activation]

        for idx_train in range(len(list_train)):
            ax = axs[idx_train, idx_activation]
            train = list_train[idx_train]

            if activation in ['tanh', '']:
                i2n = InputToNode(hidden_layer_size=1, random_state=82, input_scaling=50/784, bias_scaling=0., activation=activation)
            elif activation in ['relu', 'bounded_relu']:
                i2n = InputToNode(hidden_layer_size=1, random_state=82, input_scaling=1., bias_scaling=0., activation=activation)

            node_out = i2n.fit_transform(train, y)
            hist, bin_edges = np.histogram(node_out, bins=20, density=True)

            np.delete(bin_edges[:-1], hist <= 1e-3)
            np.delete(hist, hist <= 1e-3)

            x = bin_edges[:-1]
            width = -np.diff(bin_edges)

            # ax.bar(x=x, height=hist, width=width, label=activation, color=tud_colors['lightblue'], align='edge')
            if activation == 'bounded_relu':
                ax.hist(node_out, label=activation, density=True, bins=[.0, .1, .9, 1.], color=tud_colors['lightblue'])
            else:
                ax.hist(node_out, label=activation, density=True, bins=20, color=tud_colors['lightblue'])

            ax.grid(axis='y')
            ax.set_yscale('log')

            x_ticks = np.min(node_out), np.max(node_out)
            ax.set_xlim(x_ticks)

            # x0, y0, width, height = ax.get_position().bounds
            #  fig.text(x=x0 + width/10, y=y0 + height/2, s='scaling={0:.1e}\nbias={1:.1e}'.format(i2n.input_scaling, i2n.bias_scaling), fontsize='small')
            if activation == 'tanh':
                x_ticks += (0.0, )
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(['{0:.1f}'.format(x_tick) for x_tick in x_ticks])

    axs[0, 0].set_title('tanh, orig.')
    axs[0, 1].set_title('relu, orig.')
    axs[0, 2].set_title('b. relu, orig.')
    axs[1, 0].set_title('tanh, pca')
    axs[1, 1].set_title('relu, pca')
    axs[1, 2].set_title('b. relu, pca')

    # plt.tight_layout()
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'node-out.pdf'), format='pdf')
    fig.savefig(os.path.join(directory, 'node-out.eps'), format='eps')
    plt.rc('pgf', texsystem='pdflatex')
    # fig.savefig(os.path.join(os.environ['PGFPATH'], 'node-out.pgf'), format='pgf')


if __name__ == "__main__":
    directory = os.path.abspath('./examples/input-to-node/')
    input2node_distribution(directory=directory)
