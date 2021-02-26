"""
Plots for numerous reasons
"""
import os

import scipy
import numpy as np
import time

import pickle
import pandas

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from pyrcn.util import tud_colors


directory = 'plots'


def f_x(x, sigma=1, mu=0.):
    return np.multiply(np.divide(1/np.sqrt(2*np.pi), sigma), np.exp(-.5 * np.power(np.divide((x - mu), sigma), 2)))


def f_y(y, sigma=1, mu=0.):
    return np.divide(f_x(np.arctanh(y), sigma, mu), (1 - np.power(y, 2)))


def save_line2d_data(lines: [plt.Line2D], filepath: str):
    data = []
    header = []
    fmt = []
    for line in lines:
        header.append('x({0})'.format(line.get_label()))
        data.append(line.get_xdata())
        fmt.append('%f')
        header.append('y({0})'.format(line.get_label()))
        data.append(line.get_ydata())
        fmt.append('%f')

    # noinspection PyTypeChecker
    np.savetxt(
        fname=filepath,
        X=np.array(data).T,
        fmt=','.join(fmt),
        header=','.join(header),
        comments=''
    )
    return


def plot_activation_variance():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-.9999, .9999, 1000)

    sigma = np.array([.5, .75, 1.])
    mu = np.array([0.])

    fx = np.zeros((len(sigma), len(x)))
    fy = np.zeros((len(sigma), len(y)))

    fig = plt.figure(figsize=(6, 3))
    ax = plt.axes()
    ax.set_xlim((-2, 2))
    ax.set_ylim((0, 1))
    lines = []

    for s in sigma:
        lines += ax.plot(x, f_x(x, sigma=s, mu=mu[0]), color=tud_colors['gray'], linewidth=1.2, alpha=.5, label='fx;sigma={0}'.format(s))
        lines += ax.plot(y, f_y(y, sigma=s, mu=mu[0]), color=tud_colors['lightblue'], linewidth=1.2, label='fy;sigma={0}'.format(s))

    lines[0].set_linestyle('--')
    lines[1].set_linestyle('--')
    lines[2].set_linestyle('-.')
    lines[3].set_linestyle('-.')
    lines[4].set_linestyle(':')
    lines[5].set_linestyle(':')

    ax.legend(lines, ('$f_x, \sigma = .5$', '$f_y, \sigma = .5$', '$f_x, \sigma = .75$', '$f_y, \sigma = .75$','$f_x, \sigma = 1.$', '$f_y, \sigma = 1.$'))
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'plot-distribution-sigma.pdf'))

    save_line2d_data(lines, os.path.join(directory, 'plot-distribution-sigma.csv'))
    return


def plot_activation_mean():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-.9999, .9999, 1000)

    sigma = np.array([.5])
    mu = np.array([0., -.25, -.5])

    fx = np.zeros((len(sigma), len(x)))
    fy = np.zeros((len(sigma), len(y)))

    fig = plt.figure(figsize=(6, 3))
    ax = plt.axes()
    ax.set_xlim((-2, 2))
    ax.set_ylim((0, 1.25))
    lines = []

    for m in mu:
        lines += ax.plot(x, f_x(x, sigma=sigma[0], mu=m), color=tud_colors['gray'], linewidth=1.2, alpha=.5, label='fx;mean={0}'.format(m))
        lines += ax.plot(y, f_y(y, sigma=sigma[0], mu=m), color=tud_colors['lightblue'], linewidth=1.2, label='fy;mean={0}'.format(m))

    lines[0].set_linestyle('--')
    lines[1].set_linestyle('--')
    lines[2].set_linestyle('-.')
    lines[3].set_linestyle('-.')
    lines[4].set_linestyle(':')
    lines[5].set_linestyle(':')

    ax.legend(lines, ('$f_x, \mu = 0$', '$f_y, \mu = 0$', '$f_x, \mu = -.25$', '$f_y, \mu = -.25$','$f_x, \mu = -.5$', '$f_y, \mu = -.75$'))
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'plot-distribution-mean.pgf'), format='pgf')

    save_line2d_data(lines, os.path.join(directory, 'plot-distribution-mean.csv'))
    return


def plot_hyperparameters():
    filepath = os.path.join('./mnist-elm', 'elm_basic.csv')
    df = pandas.read_csv(filepath, sep=',')
    df_tanh2000 = df[
        (df['param_input_to_nodes__activation'] == 'tanh') & (df['param_input_to_nodes__hidden_layer_size'] == 2000)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])
    df_relu500 = df[
        (df['param_input_to_nodes__activation'] == 'relu') & (df['param_input_to_nodes__hidden_layer_size'] == 500)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])
    df_relu2000 = df[
        (df['param_input_to_nodes__activation'] == 'relu') & (df['param_input_to_nodes__hidden_layer_size'] == 2000)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])

    filepath = os.path.join('./mnist-elm', 'elm_preprocessed.csv')
    df = pandas.read_csv(filepath, sep=',')
    df_tanh2000pca = df[
        (df['param_input_to_nodes__activation'] == 'tanh') & (df['param_input_to_nodes__hidden_layer_size'] == 2000)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])
    df_relu500pca = df[
        (df['param_input_to_nodes__activation'] == 'relu') & (df['param_input_to_nodes__hidden_layer_size'] == 500)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])
    df_relu2000pca = df[
        (df['param_input_to_nodes__activation'] == 'relu') & (df['param_input_to_nodes__hidden_layer_size'] == 2000)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])

    n_rows = df_tanh2000.shape[0]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10., 6.))  # , subplot_kw={'projection': '3d'})

    df_dict = {
        0: df_tanh2000,
        1: df_relu500,
        2: df_relu2000,
        3: df_tanh2000pca,
        4: df_relu500pca,
        5: df_relu2000pca
    }

    # colormap
    # cm = ListedColormap(np.linspace(start=tud_colors['red'], stop=tud_colors['lightgreen'], num=255))
    n_upper = 20
    color_array = np.zeros((255, 4))
    color_array[: 255 - n_upper, :] += np.linspace(start=tud_colors['red'], stop=(1., 1., 1., 1.), num=255 - n_upper)
    color_array[255 - n_upper:, :] += np.linspace(start=(1., 1., 1., 1.), stop=tud_colors['darkgreen'], num=n_upper)
    cm = ListedColormap(color_array)

    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            df_loop = df_dict[row*3 + col]
            ax = axs[row][col]

            X_ticks = np.sort(df_loop['param_input_to_nodes__input_scaling'].unique())  # ascending
            Y_ticks = np.sort(df_loop['param_input_to_nodes__bias_scaling'].unique())[::-1]  # descending

            mesh_shape = (len(X_ticks), len(Y_ticks))

            Z_value = df_loop['mean_test_score'].values.reshape(mesh_shape)*100
            # norm = Normalize(vmin=np.mean(Z_value), clip=True) # -np.std(Z_value)

            # surf = ax.plot_surface(
            im = ax.imshow(
                # np.log10(df_loop['param_input_to_nodes__bias_scaling'].values.reshape(mesh_shape)),
                # np.log10(df_loop['param_input_to_nodes__input_scaling'].values.reshape(mesh_shape)),
                Z_value,
                cmap=cm,  # matplotlib.cm.coolwarm
                # norm=norm
            )

            fig.colorbar(im, ax=ax, use_gridspec=True, spacing='proportional')

            # ax.set_xticks(np.log10(X_ticks))
            ax.set_xticks(range(mesh_shape[0]))
            ax.set_xticklabels(['{0:3.3f}'.format(x) for x in X_ticks])
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_xlabel('input scaling')

            # ax.set_yticks(np.log10(Y_ticks))
            ax.set_yticks(range(mesh_shape[1]))
            ax.set_yticklabels(['{0:0.3f}'.format(y) for y in Y_ticks])
            ax.set_ylabel('bias scaling')

            # annotate
            y = np.argmax(Z_value) // len(Z_value)
            x = np.argmax(Z_value) % len(Z_value)
            ax.annotate('{0:0.1f}%'.format(np.max(Z_value)), xy=(x, y), c=(1., 1., 1., 1.), horizontalalignment='center', fontsize='small', fontstretch='ultra-condensed')

    axs[0][0].set_title('tanh, $m=2000$')
    axs[0][1].set_title('relu, $m=500$')
    axs[0][2].set_title('relu, $m=2000$')
    axs[1][0].set_title('pca50, tanh, $m=2000$')
    axs[1][1].set_title('pca50, relu, $m=500$')
    axs[1][2].set_title('pca50, relu, $m=2000$')

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig('/home/michael/Dokumente/Studium/TUD/DA/hyperparameter-relu-tanh.pdf')
    plt.show()


def plot_preprocessed():
    filepath = os.path.join('./mnist-elm', 'elm_preprocessed_relu.csv')
    df = pandas.read_csv(filepath, sep=',')
    df_tanh2000 = df[
        (df['param_input_to_nodes__activation'] == 'relu') & (df['param_input_to_nodes__hidden_layer_size'] == 2000)
    ].sort_values(by=['param_input_to_nodes__bias_scaling', 'param_input_to_nodes__input_scaling'], axis=0, ascending=[False, True])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3., 3.))  # , subplot_kw={'projection': '3d'})

    df_dict = {
        0: df_tanh2000,
    }

    # colormap
    # cm = ListedColormap(np.linspace(start=tud_colors['red'], stop=tud_colors['lightgreen'], num=255))
    n_upper = 20
    color_array = np.zeros((255, 4))
    color_array[: 255 - n_upper, :] += np.linspace(start=tud_colors['red'], stop=(1., 1., 1., 1.), num=255 - n_upper)
    color_array[255 - n_upper:, :] += np.linspace(start=(1., 1., 1., 1.), stop=tud_colors['darkgreen'], num=n_upper)
    cm = ListedColormap(color_array)

    df_loop = df_tanh2000
    ax = axs

    X_ticks = np.sort(df_loop['param_input_to_nodes__input_scaling'].unique())  # ascending
    Y_ticks = np.sort(df_loop['param_input_to_nodes__bias_scaling'].unique())[::-1]  # descending

    mesh_shape = (len(X_ticks), len(Y_ticks))

    Z_value = df_loop['mean_test_score'].values.reshape(mesh_shape)*100

    im = ax.imshow(
        Z_value,
        cmap=cm
    )

    fig.colorbar(im, ax=ax, use_gridspec=True, spacing='proportional')

    ax.set_xticks(range(mesh_shape[0]))
    ax.set_xticklabels(['{0:0.3f}'.format(x) for x in X_ticks])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel('input scaling')

    ax.set_yticks(range(mesh_shape[1]))
    ax.set_yticklabels(['{0:0.3f}'.format(y) for y in Y_ticks])
    ax.set_ylabel('bias scaling')

    # annotate
    y = np.argmax(Z_value) // len(Z_value)
    x = np.argmax(Z_value) % len(Z_value)
    ax.annotate('{0:0.1f}%'.format(np.max(Z_value)), xy=(x, y), c=(1., 1., 1., 1.), horizontalalignment='center', fontsize='small', fontstretch='ultra-condensed')

    ax.set_title('relu, $m=2000$\n450 features')

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig('/home/michael/Dokumente/Studium/TUD/DA/elm_preprocessed_relu-compare.pdf')
    plt.show()


def main():
    if not os.path.exists(directory):
        os.makedirs(directory)

    # plot_activation_variance()
    # plot_activation_mean()
    plot_hyperparameters()
    # plot_preprocessed()
    return


if __name__ == "__main__":
    main()
