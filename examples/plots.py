"""
Plots for numerous reasons
"""
import os

import numpy as np

import pickle
import pandas

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

    fig = plt.figure(figsize=(4, 2))
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

    ax.legend(lines, ('$f_x, \sigma = .5$', '$f_y, \sigma = .5$', '$f_x, \sigma = .75$', '$f_y, \sigma = .75$','$f_x, \sigma = 1.$', '$f_y, \sigma = 1.$'), bbox_to_anchor=(1, .5), loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot-distribution-sigma.pgf'), format='pgf')
    fig.savefig(os.path.join(directory, 'plot-distribution-sigma.pdf'), format='pdf')

    save_line2d_data(lines, os.path.join(directory, 'plot-distribution-sigma.csv'))
    return


def plot_activation_mean():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-.9999, .9999, 1000)

    sigma = np.array([.5])
    mu = np.array([0., -.25, -.5])

    fx = np.zeros((len(sigma), len(x)))
    fy = np.zeros((len(sigma), len(y)))

    fig = plt.figure(figsize=(4, 2))
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

    ax.legend(lines, ('$f_x, \mu = 0$', '$f_y, \mu = 0$', '$f_x, \mu = -.25$', '$f_y, \mu = -.25$','$f_x, \mu = -.5$', '$f_y, \mu = -.5$'), bbox_to_anchor=(1, .5), loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot-distribution-mean.pgf'), format='pgf')
    fig.savefig(os.path.join(directory, 'plot-distribution-mean.pdf'), format='pdf')

    save_line2d_data(lines, os.path.join(directory, 'plot-distribution-mean.csv'))
    return


def plot_ridge():
    rs = np.random.RandomState(1824)

    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x)
    y_noise = y + rs.normal(0., .2, size=x.shape)
    choice = rs.choice(x.size, size=20)
    # choice = np.arange(0, x.size, 75)
    x_sampled = x[choice]
    y_sampled = y[choice]
    y_sampled_noise = y_noise[choice]

    alpha = [0, 1., 100.]

    h = np.array([np.power(x, exponent).T for exponent in range(10)]).T

    fig = plt.figure(figsize=(4, 2))
    ax = plt.axes()
    ax.set_xlim((np.min(x), np.max(x)))
    ax.set_ylim((-1.4, 1.4))
    lines = []

    lines += ax.plot(x, y, label='sin', linestyle='-', color=tud_colors['lightblue'], alpha=.5)

    colors = [tud_colors['gray'], tud_colors['lightgreen'], tud_colors['gray']]
    alphas = [.4, 1., .4]
    linestyles = [':', '-.', '--']

    for a in alpha:
        coefs = np.dot(np.linalg.inv(np.dot(h[choice, :].T, h[choice, :]) + a * np.eye(h.shape[1])), np.dot(h[choice].T, y_noise[choice]))
        y_regression = np.dot(h, coefs)
        lines += ax.plot(x, y_regression, label='lda={0}'.format(a), color=colors.pop(0), alpha=alphas.pop(0), linestyle=linestyles.pop(0))

    lines += ax.plot(x_sampled, y_sampled_noise, label='samples', marker='x', linestyle='None', color=tud_colors['lightblue'], alpha=.3)
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['$0$', '$\pi$', '$2\pi$'])

    ax.legend(bbox_to_anchor=(1, .5), loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot-ridge.pgf'), format='pgf')  # pgf
    fig.savefig(os.path.join(directory, 'plot-ridge.pdf'), format='pdf')  # pgf

    # save_line2d_data(lines, os.path.join(directory, 'plot-ridge.csv'))
    return


def plot_pca():
    df = pandas.read_csv('/home/michael/PycharmProjects/PyRCN/examples/experiments/elm_pca.csv')
    df.sort_values(by='pca_n_components', inplace=True)

    xticks = [10, 20, 50, 100, 200, 500, 784]
    error_yticks = [2e-2, 3e-2, 4e-2, 5e-2]
    ratio_yticks = [.50, .60, .70, .80, .90, 1.00]

    fig = plt.figure(figsize=(5, 3))

    # add grid axis
    ax_error = fig.add_subplot(
        111,
        xscale='log',
        xticks=xticks,
        xticklabels=['{0:.0f}'.format(xtick) for xtick in xticks],
        xlim=(10, 784),
        xlabel='\#components'
    )

    # disable patch
    ax_error.patch.set_visible(False)

    # clone x-axis (the retarded matplotlib-way)
    ax_ratio = ax_error.twinx()

    # plot
    lines = []

    lines += ax_error.plot(
        df['pca_n_components'],
        1 - df['score'],
        color=tud_colors['lightblue'],
        label='test set error rate'
    )

    ax_error.set_yticks(error_yticks)
    ax_error.set_yticklabels(['{0:.1f}%'.format(ytick * 100) for ytick in error_yticks])
    ax_error.set_ylabel('error rate [%]')
    ax_error.set_ylim([2e-2, 5e-2])

    lines += ax_ratio.plot(
        df['pca_n_components'],
        df['pca_explained_variance_ratio'],
        color=tud_colors['gray'],
        label='explained variance ratio',
        linestyle='dashed'
    )

    ax_ratio.set_yticks(ratio_yticks)
    ax_ratio.set_yticklabels(['{0:.1f}%'.format(ytick * 100) for ytick in ratio_yticks])
    ax_ratio.set_ylabel('explained variance ratio [%]')
    ax_ratio.set_ylim([.50, 1.00])

    # grid in background
    ax_error.grid(which='both', axis='both', alpha=.7)

    legend = ax_error.legend(lines, [line._label for line in lines], loc='lower right')  # , bbox_to_anchor=[1., .5]
    fig.tight_layout()

    filename = 'plot_pca'
    fig.savefig(os.path.join('./plots/', '{0}.pdf'.format(filename)), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], '{0}.pgf'.format(filename)), format='pgf')
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

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6., 4.), gridspec_kw={'hspace': 1.1, 'wspace': 1.1, 'left': .1, 'right': .95, 'bottom': .18, 'top': .92})  # , subplot_kw={'projection': '3d'})

    df_tanh2000.attrs.update({'titlestr': 'tanh\n$m=2000$'})
    df_relu500.attrs.update({'titlestr': 'ReLU\n$m=500$'})
    df_relu2000.attrs.update({'titlestr': 'ReLU\n$m=2000$'})
    df_tanh2000pca.attrs.update({'titlestr': 'PCA50, tanh\n$m=2000$'})
    df_relu500pca.attrs.update({'titlestr': 'PCA50, ReLU\n$m=500$'})
    df_relu2000pca.attrs.update({'titlestr': 'PCA50, ReLU\n$m=2000$'})

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
            # ax = axs[col][row]

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
                interpolation='none'
            )

            fig.colorbar(im, ax=ax, use_gridspec=True, shrink=.9)  #spacing='proportional')

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
            ax.annotate('{0:0.1f}%'.format(np.max(Z_value)), xy=(x, y), c=(0., 0., 0., .7), horizontalalignment='center', verticalalignment='center', fontsize='xx-small', fontstretch='ultra-condensed')

            ax.set_title(df_loop.attrs['titlestr'])

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.tight_layout()
    fig.savefig(os.path.join(directory, 'hyperparameter-relu-tanh.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'hyperparameter-relu-tanh.pgf'), format='pgf')  # pgf
    #plt.show()


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
    fig.savefig('/home/michael/Dokumente/Studium/TUD/DA/elm_preprocessed_relu-compare.pdf', format='pdf')
    plt.show()


def plot_hidden_layer_size():
    df = pandas.read_excel(os.path.join('/home/michael/Dokumente/Studium/TUD/DA/hpc-scratch/elm_hidden_layer_size_pca.xlsx'), sheet_name='pca-kmeans-results')

    list_data = [
        {
            'name': 'BP (Chazal et. al.)',
            'x': df.iloc[df[df.type == 'BP ref'].index].hls.values,
            'y': df.iloc[df[df.type == 'BP ref'].index].error_rate.values,
            'color': tud_colors['gray'],
            'linestyle': 'dashed',
        },
        {
            'name': 'ELM (Chazal et. al.)',
            'x': df.iloc[df[df.type == 'ELM ref'].index].hls.values,
            'y': df.iloc[df[df.type == 'ELM ref'].index].error_rate.values,
            'color': tud_colors['gray'],
            'linestyle': 'dashdot',
        },
        {
            'name': 'ELM (PyRCN)',
            'x': df.iloc[df[df.type == 'elm'].index].hls.values,
            'y': df.iloc[df[df.type == 'elm'].index].error_rate.values,
            'color': tud_colors['lightblue'],
            'linestyle': 'solid',
        },
    ]

    """
    pca-elm
    elm
    minibatch
    original
    ELM ref
    BP ref
    stacked
    """

    # plots
    fig, ax = plt.subplots(figsize=(5., 3.))

    for dict_data in list_data:
        ax.plot(dict_data['x'], dict_data['y'], label=dict_data['name'], color=dict_data['color'], linewidth=1., linestyle=dict_data['linestyle'])

    ax.set_xlim([784, 15680])
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')

    yticks = [.02, .03, .04, .05, .06]
    ax.set_yticks(yticks)
    # ax.set_yticklabels([''])
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    xticks = 784 * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20])
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{0:.0f}'.format(xtick/784) for xtick in xticks])

    plt.xlabel('Fan-out')
    plt.ylabel('Error rate [%]')

    plt.tight_layout()
    plt.grid(b=True, which='major', axis='both')
    # plt.show()
    plt.savefig(os.path.join(directory, 'hidden_layer_size.pdf'), format='pdf')
    plt.savefig(os.path.join(os.environ['PGFPATH'], 'hidden_layer_size.pgf'), format='pgf')


def plot_significance():
    directory = './plots'
    filepath = './mnist-elm/significance.xlsx'
    df = pandas.read_excel(filepath, sheet_name='data')

    print(df['type'].unique())
    list_type = [
        'pca+random+activation',
        'pca+kmeans+cosine',
        'pca+kmeans+cosine+norm',
        'pca+avg+kmeans+cosine',
        'pca+kmeans+euclidean',
        'pca+kmeans+euclidean+activation'
    ]
    list_labels = [
        'Random',
        'Cosine',
        'Cosine\n(normed)',
        'Cosine\n(average)',
        'Euclidean',
        'Euclidean\n(activation)'
    ]

    list_error_rates = [df[df.type == i_type].mean_error_rate.values for i_type in list_type]

    fig, ax = plt.subplots(figsize=(4, 3), gridspec_kw={'bottom': .3})
    dict_bplot = ax.boxplot(
        [error_rate * 100 for error_rate in list_error_rates],
        labels=list_labels,
        patch_artist=True,
        boxprops={'facecolor': tud_colors['lightblue'], 'alpha': 1., 'color': tud_colors['gray']},
        medianprops={'color': tud_colors['darkblue']},
        flierprops={'markeredgecolor': tud_colors['orange'], 'marker': 'x', 'alpha': .7},
        whiskerprops={'color': tud_colors['gray'], 'alpha': 1.},
        capprops={'color': tud_colors['gray'], 'alpha': 1.},
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('error rate [%]')
    ax.grid(axis='y')
    # plt.show()
    fig.savefig(os.path.join('./plots/', 'boxplot.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'boxplot.pgf'), format='pgf')


def plot_hls_error_rate():
    directory = '/home/michael/PycharmProjects/PyRCN/examples/plots'
    filepath = './plots/hls.csv'
    df = pandas.read_csv(filepath)

    df.sort_values(by='hls', ascending=True, inplace=True)

    # concatenate name, hls, pca
    df['identifier'] = df['name'].map(str) + df['pca'].map(lambda pca: '{0:.0f}'.format(pca) if not np.isnan(pca) else '')

    list_identifier = list(df['identifier'].unique())
    print(list_identifier)

    # remove labels
    list_identifier.remove('stacked')
    list_identifier.remove('reference-ELM')
    list_identifier.remove('reference-BP')
    list_identifier.remove('original100')
    list_identifier.remove('elm_pca100')
    list_identifier.remove('minibatch100')

    list_identifier = ['elm_basic', 'elm_pca50', 'original50', 'minibatch50']
    dict_linespecs = {
        'elm_basic': {
            'label': 'random ELM',
            'linestyle': '--',
            'color': tud_colors['red'],
            'alpha': .7,
        },
        'elm_pca50': {
            'label': 'random ELM (PCA50)',
            'linestyle': '-',
            'color': tud_colors['lightgreen'],
            'alpha': .7,
        },
        'original50': {
            'label': 'K-Means ELM (PCA50)',
            'linestyle': (0, (5, 5)),
            'color': tud_colors['lightblue'],
            'alpha': .7,
            'cluster_color': tud_colors['darkblue'],
            'cluster_linestyle': (0, (3, 4)),
            'cluster_label': 'K-Means fit time',
        },
        'minibatch50': {
            'label': 'Minibatch ELM (PCA50)',
            'linestyle': (0, (1, 1)),
            'color': tud_colors['lightpurple'],
            'alpha': .7,
            'cluster_color': tud_colors['darkpurple'],
            'cluster_linestyle': (0, (1, 3)),
            'cluster_label': 'Minibatch fit time',
        }
    }

    fig, ax = plt.subplots(figsize=(5, 4), gridspec_kw={'bottom': .15, 'left': .15})

    lines = []
    for identifier in list_identifier:
        lines += ax.plot(
            df[df.identifier == identifier].hls,
            100 * df[df.identifier == identifier].error_rate,
            label=dict_linespecs[identifier]['label'],
            linestyle=dict_linespecs[identifier]['linestyle'],
            color=dict_linespecs[identifier]['color'],
            alpha=dict_linespecs[identifier]['alpha'],
        )

    ax.set_xlim([20, 16000])
    ax.set_xscale('log')
    ax.set_xticks([1e2, 1e3, 1e4])
    ax.set_xlabel('hidden layer size')

    ax.set_yscale('log')
    ax.set_yticks([2., 5., 10.])
    ax.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [2., 5., 10.]])
    ax.set_ylabel('error rate [%]')

    ax.grid(which='both', axis='both')
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(directory, 'hls_error_rate.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'hls_error_rate.pgf'), format='pgf')


def plot_hls_error_rate_pcacompare():
    directory = '/home/michael/PycharmProjects/PyRCN/examples/plots'
    filepath = './plots/hls.csv'
    df = pandas.read_csv(filepath)

    df.sort_values(by='hls', ascending=True, inplace=True)

    # concatenate name, hls, pca
    df['identifier'] = df['name'].map(str) + df['pca'].map(lambda pca: '{0:.0f}'.format(pca) if not np.isnan(pca) else '')

    list_identifier = list(df['identifier'].unique())
    print(list_identifier)

    # remove labels
    list_identifier.remove('stacked')
    list_identifier.remove('reference-ELM')
    list_identifier.remove('reference-BP')
    list_identifier.remove('original100')
    list_identifier.remove('elm_pca100')
    list_identifier.remove('minibatch100')

    list_identifier = ['minibatch50', 'original100', 'minibatch100', 'original50', 'elm_pca100', 'elm_pca50']
    dicts_identifier = {
        'elm_pca50': {
            'label': 'Random ELM (PCA50)',
            'linestyle': (0, (3, 5, 1, 5, 1, 5)),
            'color': tud_colors['lightgreen'],
            'alpha': .7,
        },
        'elm_pca100': {
            'label': 'Random ELM (PCA100)',
            'linestyle': (0, (3, 1, 1, 1, 1, 1)),
            'color': tud_colors['darkgreen'],
            'alpha': .7,
        },
        'original50': {
            'label': 'K-Means ELM (PCA50)',
            'linestyle': (0, (5, 5)),
            'color': tud_colors['lightblue'],
            'alpha': .7,
        },
        'original100': {
            'label': 'K-Means ELM (PCA100)',
            'linestyle': (0, (3, 4)),
            'color': tud_colors['darkblue'],
            'alpha': .7,
        },
        'minibatch50': {
            'label': 'Minibatch ELM (PCA50)',
            'linestyle': (0, (1, 1)),
            'color': tud_colors['lightpurple'],
            'alpha': .7,
        },
        'minibatch100': {
            'label': 'Minibatch ELM (PCA50)',
            'linestyle': (0, (1, 3)),
            'color': tud_colors['darkpurple'],
            'alpha': .7,
        }
    }

    fig, ax = plt.subplots(figsize=(5, 4), gridspec_kw={'bottom': .15, 'left': .15})

    lines = []
    for identifier in dicts_identifier:
        lines += ax.plot(
            df[df.identifier == identifier].hls,
            100 * df[df.identifier == identifier].error_rate,
            label=dicts_identifier[identifier]['label'],
            linestyle=dicts_identifier[identifier]['linestyle'],
            color=dicts_identifier[identifier]['color'],
            alpha=dicts_identifier[identifier]['alpha'],
        )

    ax.set_xlim([20, 16000])
    ax.set_xscale('log')
    ax.set_xticks([1e2, 1e3, 1e4])
    ax.set_xlabel('hidden layer size')

    ax.set_yscale('log')
    ax.set_yticks([2., 5., 10.])
    ax.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [2., 5., 10.]])
    ax.set_ylabel('error rate [%]')

    ax.grid(which='both', axis='both')
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(directory, 'hls_error_rate_pca.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'hls_error_rate_pca.pgf'), format='pgf')


def plot_hls_fittime():
    directory = '/home/michael/PycharmProjects/PyRCN/examples/plots'
    filepath = './plots/hls.csv'
    df = pandas.read_csv(filepath)

    df.sort_values(by='hls', ascending=True, inplace=True)

    # concatenate name, hls, pca
    df['identifier'] = df['name'].map(str) + df['pca'].map(lambda pca: '{0:.0f}'.format(pca) if not np.isnan(pca) else '')

    list_identifier = list(df['identifier'].unique())
    print(list_identifier)

    # remove labels
    list_identifier.remove('stacked')
    list_identifier.remove('reference-ELM')
    list_identifier.remove('reference-BP')
    list_identifier.remove('original100')
    list_identifier.remove('elm_pca100')
    list_identifier.remove('minibatch100')

    list_identifier = ['elm_basic', 'elm_pca50', 'original50', 'minibatch50']
    dict_linespecs = {
        'elm_basic': {
            'label': 'random ELM',
            'linestyle': '--',
            'color': tud_colors['red'],
            'alpha': .7,
        },
        'elm_pca50': {
            'label': 'random ELM (PCA50)',
            'linestyle': '-',
            'color': tud_colors['lightgreen'],
            'alpha': .7,
        },
        'original50': {
            'label': 'K-Means ELM (PCA50)',
            'linestyle': (0, (5, 5)),
            'color': tud_colors['lightblue'],
            'alpha': .7,
            'cluster_color': tud_colors['darkblue'],
            'cluster_linestyle': (0, (3, 4)),
            'cluster_label': 'K-Means fit time',
        },
        'minibatch50': {
            'label': 'Minibatch ELM (PCA50)',
            'linestyle': (0, (1, 1)),
            'color': tud_colors['lightpurple'],
            'alpha': .7,
            'cluster_color': tud_colors['darkpurple'],
            'cluster_linestyle': (0, (1, 3)),
            'cluster_label': 'Minibatch fit time',
        }
    }

    fig, ax = plt.subplots(figsize=(5, 4), gridspec_kw={'bottom': .15, 'left': .15})

    lines = []
    for identifier in list_identifier:
        lines += ax.plot(
            df[df.identifier == identifier].hls,
            df[df.identifier == identifier].fit_time,
            label=dict_linespecs[identifier]['label'],
            linestyle=dict_linespecs[identifier]['linestyle'],
            color=dict_linespecs[identifier]['color'],
            alpha=dict_linespecs[identifier]['alpha'],
        )

        if identifier in ['minibatch50', 'original100', 'minibatch100', 'original50']:
            lines += ax.plot(
                df[df.identifier == identifier].hls,
                1. * df[df.identifier == identifier].kmeans_fit_time,
                label=dict_linespecs[identifier]['cluster_label'],
                linestyle=dict_linespecs[identifier]['cluster_linestyle'],
                color=dict_linespecs[identifier]['cluster_color'],
                alpha=dict_linespecs[identifier]['alpha'],
            )

    ax.set_xlim([20, 16000])
    ax.set_xscale('log')
    ax.set_xticks([1e2, 1e3, 1e4])
    ax.set_xlabel('hidden layer size')

    ax.set_ylim([3e-1, 1e3])
    ax.set_yscale('log')
    # ax.set_yticks([2., 5., 10.])
    # ax.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [2., 5., 10.]])
    ax.set_ylabel('fit time [s]')

    ax.grid(which='both', axis='both')
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(directory, 'hls_fittime.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'hls_fittime.pgf'), format='pgf')


def plot_silhouette_n_clusters():
    filepath = os.path.abspath('./plots/silhouette_n_clusters.csv')
    df = pandas.read_csv(filepath)

    df.sort_values('n_clusters', ascending=True)
    print(df.keys())
    keys = [
        'n_clusters',
        'n_init',
        'variance_threshold',
        'pca_n_components',
        'pca_explained_variance',
        'pca_explained_variance_ratio',
        'silhouette_original',
        'silhouette_variance_threshold',
        'silhouette_pca',
        'fittime_original',
        'fittime_variance_threshold',
        'fittime_pca',
        'inertia_original',
        'inertia_variance_threshold',
        'inertia_pca',
        'n_iter_original',
        'n_iter_variance_threshold',
        'n_iter_pca'
    ]

    fig, ax_score = plt.subplots(figsize=(5., 3.), gridspec_kw={'left': .15, 'right': .85, 'bottom': .2, 'top': .8})

    ax_score.set_xlim([np.min(df['n_clusters'].values), np.max(df['n_clusters'].values)])
    ax_score.set_xlabel(r'\#centroids')
    ax_score.set_xscale('log')
    ax_score.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000])
    ax_score.set_xticklabels(['{0:.0f}'.format(xtick) for xtick in [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]])
    ax_score.grid(axis='x', which='both')

    lines = []

    lines += ax_score.plot(df['n_clusters'].values, 100*df['silhouette_original'].values, color=tud_colors['lightpurple'], alpha=.7)
    lines += ax_score.plot(df['n_clusters'].values, 100*df['silhouette_variance_threshold'].values, color=tud_colors['lightgreen'], alpha=.7)
    lines += ax_score.plot(df['n_clusters'].values, 100*df['silhouette_pca'].values, color=tud_colors['lightblue'], alpha=.7)

    ax_time = ax_score.twinx()

    lines += ax_time.plot(df['n_clusters'].values, df['fittime_original'].values, color=tud_colors['lightpurple'], alpha=.7, linestyle='--')
    lines += ax_time.plot(df['n_clusters'].values, df['fittime_variance_threshold'].values, color=tud_colors['lightgreen'], alpha=.7, linestyle='--')
    lines += ax_time.plot(df['n_clusters'].values, df['fittime_pca'].values, color=tud_colors['lightblue'], alpha=.7, linestyle='--')

    ax_score.set_ylabel('silhouette score [%]')
    ax_score.set_yscale('log')
    ax_score.grid(axis='y', which='major')
    ax_score.set_ylim([1, 10])
    ax_score.set_yticks([1, 2, 5, 10])
    ax_score.set_yticks([], minor=True)
    ax_score.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [1, 2, 5, 10]])
    ax_time.set_ylabel('fit time [s]')
    ax_time.set_yscale('log')

    ax_score.legend(lines[:3], [
        'original',
        'variance\nthreshold = {0:.1f}'.format(df['variance_threshold'][0]),
        'PCA, expl.\nvar ratio = {0:.1f}%'.format(100 * df['pca_explained_variance_ratio'][0])
    ], bbox_to_anchor=(-0.2, 1.05, 1.4, 0), loc="lower left", mode="expand", ncol=3)

    # plt.show()
    fig.savefig('./plots/plot_silhouette_n_clusters.pdf', format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot_silhouette_n_clusters.pgf'), format='pgf')


def silhouette_kmeans_features():
    filepath = os.path.abspath('./plots/silhouette_kmeans20_features.csv')
    df = pandas.read_csv(filepath)

    print(df.keys())
    keys = [
        'nfeatures',
        'fittime_random',
        'fittime_maxvar',
        'fittime_pca',
        'silhouette_random',
        'silhouette_maxvar',
        'silhouette_pca',
        'explainvar_random',
        'explainvar_maxvar',
        'explainvar_pca',
        'explvarrat_random',
        'explvarrat_maxvar',
        'explvarrat_pca',
        'n_clusters'
    ]
    print(np.unique(df['n_clusters'].values))

    df.sort_values('nfeatures', ascending=True)

    fig, ax_score = plt.subplots(figsize=(5., 3.), gridspec_kw={'left': .15, 'right': .85, 'top': .8, 'bottom': .2})

    ax_score.set_xlim([np.min(df['nfeatures'].values), np.max(df['nfeatures'].values)])
    ax_score.set_xscale('log')
    ax_score.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 784])
    ax_score.set_xticklabels(['{0:.0f}'.format(xtick) for xtick in [1, 2, 5, 10, 20, 50, 100, 200, 500, 784]])
    ax_score.grid(axis='x', which='both')
    ax_score.set_xlabel(r'\#features')

    ax_variance = ax_score.twinx()

    lines = []
    lines += ax_score.plot(df['nfeatures'].values, 100*df['silhouette_random'].values, color=tud_colors['lightpurple'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df['nfeatures'].values, 100*df['silhouette_maxvar'].values, color=tud_colors['lightgreen'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df['nfeatures'].values, 100*df['silhouette_pca'].values, color=tud_colors['lightblue'], linestyle='-', alpha=.7)
    lines += ax_variance.plot(df['nfeatures'].values, 100*df['explvarrat_random'].values, color=tud_colors['lightpurple'], linestyle='--', alpha=.7)
    lines += ax_variance.plot(df['nfeatures'].values, 100*df['explvarrat_maxvar'].values, color=tud_colors['lightgreen'], linestyle='--', alpha=.7)
    lines += ax_variance.plot(df['nfeatures'].values, 100*df['explvarrat_pca'].values, color=tud_colors['lightblue'], linestyle='--', alpha=.7)

    ax_score.set_yscale('log')
    ax_score.set_ylim([5, 100])
    ax_score.set_yticks([5, 10, 20, 50, 100])
    ax_score.set_yticks([], minor=True)
    ax_score.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [5, 10, 20, 50, 100]])
    ax_score.set_ylabel('silhouette score [%]')

    ax_variance.set_yscale('log')
    ax_variance.set_ylim([1, 100])
    ax_variance.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax_variance.set_yticks([], minor=True)
    ax_variance.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [1, 2, 5, 10, 20, 50, 100]])
    ax_variance.grid(axis='y', which='both')
    ax_variance.set_ylabel('explained variance ratio [%]')

    ax_score.legend(lines[:3], ['random', 'maximal\nvariance', 'principal\ncomponents'], bbox_to_anchor=(-0.2, 1.05, 1.4, 0), loc="lower left", mode="expand", ncol=3)
    # plt.show()
    fig.savefig('./plots/silhouette_kmeans_features.pdf', format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'silhouette_kmeans_features.pgf'), format='pgf')


def plot_silhouette_subset():
    filepath = os.path.abspath('./plots/silhouette_kmeans_subset_size.csv')
    df = pandas.read_csv(filepath)

    print(df.keys())
    keys = [
        'subset_size',
        'k',
        'n_init',
        'silhouette_raninit',
        'silhouette_preinit',
        'fittime_raninit',
        'fittime_preinit',
        'scoretime_raninit',
        'scoretime_preinit'
    ]
    print(np.unique(df['k'].values))

    df.sort_values('subset_size', ascending=True)

    fig, ax_score = plt.subplots(figsize=(5., 3.), gridspec_kw={'left': .15, 'right': .85, 'top': .8, 'bottom': .2})

    ax_score.set_xlim([np.min(df['subset_size'].values), np.max(df['subset_size'].values)])
    ax_score.set_xscale('log')
    ax_score.set_xticks([250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 60000])
    ax_score.set_xticklabels(['{0:.0f}'.format(xtick) for xtick in [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 60000]])
    ax_score.grid(axis='x', which='both')
    ax_score.set_xlabel(r'subset size')

    ax_time = ax_score.twinx()

    lines = []
    lines += ax_score.plot(df['subset_size'].values, 100*df['silhouette_raninit'].values, color=tud_colors['lightpurple'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df['subset_size'].values, 100*df['silhouette_preinit'].values, color=tud_colors['lightblue'], linestyle='-', alpha=.7)
    lines += ax_time.plot(df['subset_size'].values, df['fittime_raninit'].values, color=tud_colors['lightpurple'], linestyle='--', alpha=.7)
    lines += ax_time.plot(df['subset_size'].values, df['fittime_preinit'].values, color=tud_colors['lightblue'], linestyle='--', alpha=.7)
    lines += ax_time.plot(df['subset_size'].values, df['scoretime_preinit'].values, color=tud_colors['gray'], linestyle='-.', alpha=.5)

    ax_score.set_yscale('log')
    ax_score.set_ylim([8, 11])
    ax_score.set_yticks([8, 9, 10, 11])
    ax_score.set_yticks([], minor=True)
    ax_score.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [8, 9, 10, 11]])
    ax_score.grid(axis='y', which='both')
    ax_score.set_ylabel('silhouette score [%]')

    ax_time.set_yscale('log')
    ax_time.set_ylabel('time [s]')

    ax_score.legend([lines[idx] for idx in [0, 1, 4]], [
        '{0:.0f} random inits'.format(int(np.unique(df['n_init'].values))),
        'preinit',
        'score time'
    ], bbox_to_anchor=(-0.2, 1.05, 1.4, 0), loc="lower left", mode="expand", ncol=3)
    # plt.show()
    fig.savefig('./plots/plot_silhouette_subset.pdf', format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot_silhouette_subset.pgf'), format='pgf')


def plot_silhouette_kcluster():
    filepath = os.path.abspath('./plots/silhouette_kcluster.csv')
    df = pandas.read_csv(filepath)

    print(df.keys())
    keys = [
        'n_clusters',
        'pca_n_components',
        'pca_expl_var',
        'pca_expl_var_ratio',
        'silhouette_kcosine',
        'silhouette_kmeans',
        'fittime_kcosine',
        'fittime_kmeans'
    ]
    print(np.unique(df['pca_n_components'].values))

    df.sort_values('n_clusters', ascending=True)

    fig, ax_score = plt.subplots(figsize=(5., 3.), gridspec_kw={'left': .15, 'right': .85, 'top': .8, 'bottom': .2})

    ax_score.set_xlim([np.min(df['n_clusters'].values), np.max(df['n_clusters'].values)])
    ax_score.set_xscale('log')
    ax_score.set_xticks([10, 20, 50, 100, 200])
    ax_score.set_xticklabels(['{0:.0f}'.format(xtick) for xtick in [10, 20, 50, 100, 200]])
    ax_score.grid(axis='x', which='both')
    ax_score.set_xlabel(r'\#centroids')

    ax_time = ax_score.twinx()

    lines = []
    lines += ax_score.plot(df['n_clusters'].values, 100*df['silhouette_kcosine'].values, color=tud_colors['lightpurple'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df['n_clusters'].values, 100*df['silhouette_kmeans'].values, color=tud_colors['lightblue'], linestyle='-', alpha=.7)
    lines += ax_time.plot(df['n_clusters'].values, df['fittime_kcosine'].values, color=tud_colors['lightpurple'], linestyle='--', alpha=.7)
    lines += ax_time.plot(df['n_clusters'].values, df['fittime_kmeans'].values, color=tud_colors['lightblue'], linestyle='--', alpha=.7)

    ax_score.set_yscale('log')
    ax_score.set_ylim([7, 20])
    ax_score.set_yticks([7, 10, 15, 20])
    ax_score.set_yticks([], minor=True)
    ax_score.set_yticklabels(['{0:.0f}%'.format(ytick) for ytick in [7, 10, 15, 20]])
    ax_score.grid(axis='y', which='both')
    ax_score.set_ylabel('silhouette score [%]')

    ax_time.set_yscale('log')
    ax_time.set_ylabel('time [s]')

    ax_score.legend(lines[:2], [
        'kcosine',
        'kmeans'
    ], bbox_to_anchor=(.1, 1.05, .8, 0), loc="lower left", mode="expand", ncol=3)
    # plt.show()
    fig.savefig('./plots/plot_silhouette_kcluster.pdf', format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot_silhouette_kcluster.pgf'), format='pgf')


def plot_kmeans_subset_phenomen():
    keys = ['n_clusters', 'silhouette_original', 'n_iter_original']

    df_2k = pandas.read_csv('./mnist-elm/silhouette_n_clusters_subset2k.csv').set_index('n_clusters')
    df_5k = pandas.read_csv('./mnist-elm/silhouette_n_clusters_subset5k.csv').set_index('n_clusters')
    df_10k = pandas.read_csv('./mnist-elm/silhouette_n_clusters_subset10k.csv').set_index('n_clusters')

    # force suffix
    df_2k.columns = df_2k.columns.map(lambda x: str(x) + '_2k')

    df = df_10k.join(df_5k, lsuffix='_10k', rsuffix='_5k').join(df_2k)

    print(df.keys())

    df.sort_index(ascending=True, inplace=True)

    fig, ax_score = plt.subplots(figsize=(5., 3.), gridspec_kw={'left': .15, 'right': .85, 'top': .8, 'bottom': .2})

    # df.index = df['n_clusters_2k']
    ax_score.set_xlim([np.min(df.index.values), np.max(df.index.values)])
    ax_score.set_xscale('log')
    ax_score.set_xlim([5, 2000])
    ax_score.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    ax_score.set_xticklabels(['{0:.0f}'.format(xtick) for xtick in [5, 10, 20, 50, 100, 200, 500, 1000, 2000]])
    ax_score.grid(axis='x', which='both')
    ax_score.set_xlabel(r'\#centroids')

    ax_niter = ax_score.twinx()

    lines = []
    lines += ax_score.plot(df.index.values, 100 * df['silhouette_original_2k'].values, color=tud_colors['lightpurple'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df.index.values, 100 * df['silhouette_original_5k'].values, color=tud_colors['lightblue'], linestyle='-', alpha=.7)
    lines += ax_score.plot(df.index.values, 100 * df['silhouette_original_10k'].values, color=tud_colors['lightgreen'], linestyle='-', alpha=.7)

    lines += ax_niter.plot(df.index.values, df['n_iter_original_2k'].values, color=tud_colors['lightpurple'], linestyle='--', alpha=.7)
    lines += ax_niter.plot(df.index.values, df['n_iter_original_5k'].values, color=tud_colors['lightblue'], linestyle='--', alpha=.7)
    lines += ax_niter.plot(df.index.values, df['n_iter_original_10k'].values, color=tud_colors['lightgreen'], linestyle='--', alpha=.7)

    ax_score.grid(axis='y', which='both')
    ax_score.set_ylabel('silhouette score [%]')

    ax_niter.set_yscale('log')
    ax_niter.set_ylabel(r'\#iterations')

    ax_score.legend(lines[:3], [
        '2000',
        '5000',
        '10000'
    ], bbox_to_anchor=(.1, 1.05, .8, 0), loc="lower left", mode="expand", ncol=3)
    # plt.show()
    fig.savefig('./plots/plot_kmeans_subset_phenomen.pdf', format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'plot_kmeans_subset_phenomen.pgf'), format='pgf')


def plot_cluster_critical():
    cluster_centers = np.load('./cluster_critical.npy', allow_pickle=True)

    plt.imsave('./centroid4.png', cluster_centers[99, ...].reshape(28, 28), cmap=plt.cm.gray_r)
    return


def main():
    if not os.path.exists(directory):
        os.makedirs(directory)

    # plot_activation_variance()
    plot_activation_mean()
    # plot_hyperparameters()
    # plot_preprocessed()
    # plot_hidden_layer_size()
    # plot_ridge()
    # plot_pca()
    # plot_significance()
    # plot_hls_error_rate()
    # plot_hls_error_rate_pcacompare()
    # plot_hls_fittime()
    # plot_silhouette_n_clusters()
    # silhouette_kmeans_features()
    # plot_silhouette_subset()
    # plot_silhouette_kcluster()
    # plot_kmeans_subset_phenomen()
    # plot_cluster_critical()
    return


if __name__ == "__main__":
    main()
