"""
An example of the Coates Idea on the digits dataset.
"""
import os
import sys
import logging
import time

import scipy
import numpy as np

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from pyrcn.cluster import KCluster

from pyrcn.util import tud_colors, new_logger,get_mnist

import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

example_image_idx = 5
min_var = 3088.6875

# EXPERIMENTS


def plot_labels(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    # find first digit occurrences
    idx = np.ones((10,)) * -1
    cnt = int(0)
    while np.any(idx == -1):
        if idx[int(y[cnt])] == -1.0:
            idx[int(y[cnt])] = int(cnt)
        cnt += 1

    # display digits
    fig, axs = plt.subplots(2, 5, figsize=(5, 2))

    for i in range(10):
        axs[i // 5][i % 5].imshow(np.resize(X[int(idx[i])], (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
        axs[i // 5][i % 5].set_xticks([0, 27])
        axs[i // 5][i % 5].set_xticklabels([0, 27])
        axs[i // 5][i % 5].set_yticks([0, 27])
        axs[i // 5][i % 5].set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-visualize.pgf'), format='pgf')


def plot_pooling(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    img = X[example_image_idx, :].reshape((28, 28))
    kernel_size = (2, 1)
    img_pooled = np.zeros((int(np.ceil(img.shape[0] / kernel_size[0])), int(np.ceil(img.shape[1] / kernel_size[1]))))

    for x in range(img_pooled.shape[0]):
        for y in range(img_pooled.shape[1]):
            x_min = x*kernel_size[0]
            x_max = x_min + kernel_size[0]
            y_min = y*kernel_size[1]
            y_max = y_min + kernel_size[1]
            img_pooled[x, y] = np.max(img[x_min:x_max, y_min:y_max])

    plt.imsave(os.path.join(directory, 'pooled_max_kernel{0}x{1}.png'.format(kernel_size[0], kernel_size[1])), img_pooled, cmap=plt.cm.gray_r)
    return


def plot_historgram(directory, *args, **kwargs):
    logger = new_logger('plot_historgram', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2),  gridspec_kw={'width_ratios': [1, 1.7]})

    example = np.zeros((28, 28, 3))
    example[..., 0] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # red
    example[..., 1] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # green
    example[..., 2] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # blue

    idx_fringe = (25, 17)
    idx_center = (13, 12)

    example[idx_center[0], idx_center[1], :] = tud_colors['lightblue'][:-1]
    example[idx_fringe[0], idx_fringe[1], :] = tud_colors['orange'][:-1]

    bins = np.array(range(0, 287, 32)).astype(int)

    hist_fringe, bin_edges = np.histogram(X[:, idx_fringe[0] * 28 + idx_fringe[1]], bins=bins)
    hist_center, bin_edges = np.histogram(X[:, idx_center[0] * 28 + idx_center[1]], bins=bins)

    logger.info('validation sum hist_fringe: {0}, sum hist_center: {1}'.format(np.sum(hist_fringe / 1000),
                                                                               np.sum(hist_center / 1000)))

    axs[0].imshow(example, interpolation='none')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].bar(bins[1:] - 32, height=hist_fringe / 1000, width=16, color=tud_colors['orange'], label='fringe',
               align='edge')
    axs[1].bar(bins[1:] - 16, height=hist_center / 1000, width=16, color=tud_colors['lightblue'], label='center',
               align='edge')
    axs[1].tick_params(axis='x', labelrotation=90)
    # axs[1].hist([], bins=range(0, 255, 32), color=[tud_colors['orange'], tud_colors['lightblue']], align='left')

    axs[1].set_xticks(bins)
    # axs[1].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    axs[1].legend(bbox_to_anchor=(1.0, .5), loc="center left")

    # fig.suptitle('Feature distribution in MNIST picture')
    axs[1].set_xlabel('value bins')
    axs[1].set_ylabel('probability')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-histogram.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-pixel-histogram.pgf'), format='pgf')
    # plt.show()
    return


def plot_var(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    scaler = StandardScaler().fit(X)
    pos = range(0, 28)
    meanX = []
    varX = []

    fig, axs = plt.subplots(1, 2, figsize=(5, 2),  gridspec_kw={'width_ratios': [1, 1.4]})

    example = np.zeros((28, 28, 3))
    example[..., 0] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # red
    example[..., 1] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # green
    example[..., 2] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # blue

    for idx in pos:
        example[idx, idx, :] = tud_colors['orange'][:-1]
        meanX.append(scaler.mean_[idx * 28 + idx])
        varX.append(scaler.var_[idx * 28 + idx])

    axs[0].imshow(example, interpolation='none')

    line_var, = axs[1].plot(pos, varX, color=tud_colors['orange'])
    ax_mean = axs[1].twinx()
    line_mean, = ax_mean.plot(pos, meanX, color=tud_colors['lightblue'])

    axs[1].legend((line_var, line_mean), ('$\sigma^2$', '$\mu$'), bbox_to_anchor=(1.2, .5), loc="center left")

    # fig.suptitle('Feature distribution in MNIST picture')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].set_xlim([0, 27])
    axs[1].set_xlabel('position')
    axs[1].set_ylabel('$\sigma^2$', labelpad=-15, loc='top', rotation=0)
    y_ticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(['{0:0.0f}k'.format(y_tick/1000) for y_tick in y_ticks])
    # axs[1].tick_params(axis='x', labelrotation=90)
    ax_mean.set_ylabel('$\mu$', labelpad=-5, loc='top', rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-variance.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-pixel-variance.pgf'), format='pgf')
    # plt.show()
    return


def plot_image_min_var(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    scaler = StandardScaler().fit(X)

    image_size = (28, 28, 3)
    example = np.zeros(X[example_image_idx, ...].shape + (3, ))
    for rgb_idx in range(3):
        example[..., rgb_idx] = 1. - X[example_image_idx, :] / 255.

    p1_1 = 1/10 * 1/10
    p1_2 = 1/10
    var_p1_1 = 255 ** 2 * p1_1 * (1 - p1_1)
    var_p1_2 = 255 ** 2 * p1_2 * (1 - p1_2)

    example_min_var_p1_1 = np.copy(example)
    example_min_var_p1_1[scaler.var_ < var_p1_1, ...] = tud_colors['orange'][:-1]

    example_min_var_p1_2 = np.copy(example)
    example_min_var_p1_2[scaler.var_ < var_p1_2, ...] = tud_colors['orange'][:-1]

    fig, axs = plt.subplots(1, 3, figsize=(5, 2))

    axs[0].imshow(np.reshape(example, image_size), interpolation='none')
    axs[0].set_title('$p_1$=0\noriginal\n$n$={0:d}'.format(len(scaler.var_)))
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].imshow(np.reshape(example_min_var_p1_1, image_size), interpolation='none')
    axs[1].set_title('$p_1$={1:0.2f}\n$\sigma^2$ > {0:0.0f}\n$n$={2:d}'.format(var_p1_1, p1_1, np.sum(scaler.var_ > var_p1_1)))
    axs[1].set_xticks([0, 27])
    axs[1].set_xticklabels([0, 27])
    axs[1].set_yticks([0, 27])
    axs[1].set_yticklabels([0, 27])

    axs[2].imshow(np.reshape(example_min_var_p1_2, image_size), interpolation='none')
    axs[2].set_title('$p_1$={1:0.2f}\n$\sigma^2$ > {0:0.0f}\n$n$={2:d}'.format(var_p1_2, p1_2, np.sum(scaler.var_ > var_p1_2)))
    axs[2].set_xticks([0, 27])
    axs[2].set_xticklabels([0, 27])
    axs[2].set_yticks([0, 27])
    axs[2].set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-img-min-var.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-img-min-var.pgf'), format='pgf')
    # plt.show()
    return


def plot_normalized(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    X = X / 4 + 100
    X_picture_normalization = StandardScaler().fit_transform(X.T).T
    X_feature_normalization = StandardScaler().fit_transform(X)

    fig, axs = plt.subplots(1, 3, figsize=(5, 2))

    img_idx = example_image_idx

    axs[0].imshow(np.resize(X[img_idx, :], (28, 28)).astype(int), interpolation='none', cmap=plt.cm.gray_r,
                  norm=Normalize(vmin=0, vmax=255, clip=True))
    axs[0].set_title('low contrast')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].imshow(np.resize(X_picture_normalization[img_idx, :], (28, 28)), interpolation='none',
                  cmap=plt.cm.gray_r)  # norm=Normalize(vmin=0, vmax=1, clip=True)
    axs[1].set_title('picture\nnormalization')
    axs[1].set_xticks([0, 27])
    axs[1].set_xticklabels([0, 27])
    axs[1].set_yticks([0, 27])
    axs[1].set_yticklabels([0, 27])

    axs[2].imshow(np.resize(X_feature_normalization[img_idx, :], (28, 28)), interpolation='none',
                  cmap=plt.cm.gray_r)  # norm=Normalize(vmin=0, vmax=1, clip=True)
    axs[2].set_title('feature\nnormalization')
    axs[2].set_xticks([0, 27])
    axs[2].set_xticklabels([0, 27])
    axs[2].set_yticks([0, 27])
    axs[2].set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-normalized.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-normalized.pgf'), format='pgf')
    # plt.show()
    return


def plot_variance_mean(directory, *args, **kwargs):
    logger = new_logger('plot_variance_mean', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    image_size = (28, 28)

    scaler = StandardScaler().fit(StandardScaler(with_std=False).fit_transform(X) / 255)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(np.resize(scaler.mean_, image_size), cmap=plt.cm.gray_r, interpolation='none')
    axs[0].imshow(np.resize(scaler.var_, image_size), cmap=plt.cm.gray_r, interpolation='none')

    axs[0].set_title('$\mu$')
    axs[1].set_title('$\sigma^2$')

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-variance-and-mean-avgfree.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-pixel-variance-and-mean-avgfree.pgf'), format='pgf')
    logger.info('np.max(scaler.mean_) = {0}, np.max(scaler.var_) = {1}'.format(np.max(scaler.mean_), np.max(scaler.var_)))
    return


def plot_pca(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    fig, axs = plt.subplots(1, 5, figsize=(6, 1.5), gridspec_kw={'wspace': 0.45, 'left': .05, 'right': .95, 'bottom': .0, 'top': .90})

    n_components_list = [784, 400, 100, 50, 30]
    min_p1_list = [0., .001, .01, .1, .5]
    min_var_list = [255 ** 2 * p1 * (1. - p1) for p1 in min_p1_list]

    # automatic pca
    decomposer = PCA(whiten=False).fit(X)

    sum_explained_variance = np.flip(np.cumsum(np.flip(decomposer.explained_variance_)))

    # original
    axs[0].imshow(np.resize(X[example_image_idx, ...], (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs[0].set_title('original')

    # mean
    axs[1].imshow(np.resize(decomposer.mean_, (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs[1].set_title('average'.format(100))

    # pca 50, average free
    X_avgfree = X - np.mean(X, axis=0)
    M_pca = decomposer.components_[:50, :].T
    M = np.dot(M_pca, M_pca.T)  # transformation and inverse combined

    axs[2].imshow(np.resize(np.dot(X_avgfree[example_image_idx, ...], M), (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs[2].set_title('n={0}\naverage free'.format(M_pca.shape[1]))

    # pca 50, not average free
    axs[3].imshow(np.resize(np.dot(X[example_image_idx, ...], M), (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs[3].set_title('n={0}\nwith average'.format(M_pca.shape[1]))

    # pca 25, not average free
    M_pca = decomposer.components_[:25, :].T
    M = np.dot(M_pca, M_pca.T)  # transformation and inverse combined

    axs[4].imshow(np.resize(np.dot(X[example_image_idx, ...], M), (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs[4].set_title('n={0}\nwith average'.format(M_pca.shape[1]))

    for idx in range(5):
        axs[idx].set_xticks([0, 27])
        axs[idx].set_xticklabels([0, 27])
        axs[idx].set_yticks([0, 27])
        axs[idx].set_yticklabels([0, 27])

    # fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(directory, 'mnist-pca-effects.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-pca-effects.pgf'), format='pgf')
    return


def plot_imbalance(directory):
    self_name = 'plot_imbalance'
    logger = new_logger(self_name, directory)
    X, y = get_mnist(directory)
    logger.info('successfully fetched {0} datapoints'.format(X.shape[0]))

    # y_hist = []
    # [y_hist.append([idx, np.sum(y.astype(int) == idx), np.sum(y[:60000].astype(int) == idx), np.sum(y[60000:].astype(int) == idx)]) for idx in range(10)]
    tp_y_unique = np.unique(y.astype(int), return_counts=True)
    y_unique = tp_y_unique[0][np.argsort(tp_y_unique[0])]
    y_counts = tp_y_unique[1][np.argsort(tp_y_unique[0])]

    tp_y_train = np.unique(y[:60000].astype(int), return_counts=True)
    y_train_unique = tp_y_train[0][np.argsort(tp_y_train[0])]
    y_train_counts = tp_y_train[1][np.argsort(tp_y_train[0])]

    tp_y_test = np.unique(y[60000:].astype(int), return_counts=True)
    y_test_unique = tp_y_test[0][np.argsort(tp_y_test[0])]
    y_test_counts = tp_y_test[1][np.argsort(tp_y_test[0])]
    # y_hist_arr = np.array(y_hist, dtype=float)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6., 2))

    bar_width = .9

    # bars_y = ax.bar(y_hist_arr[:, 0], y_hist_arr[:, 1], label='dataset', color=tud_colors['lightblue'])  # , edgecolor=[(rgba*0.2) for rgba in tud_colors['lightblue']], hatch='x')  # , width=.9)

    bars_train = ax.bar(y_train_unique, y_train_counts, label='train', color=tud_colors['gray'], width=bar_width)  #, edgecolor=['darkgreen'], hatch='//')  # , width=.7)
    bars_test = ax.bar(y_test_unique, y_test_counts, bottom=y_train_counts, label='test', color=tud_colors['lightblue'], width=bar_width)  #, edgecolor=tud_colors['darkpurple'], hatch='\\\\')  # , width=.5)

    for idx in range(y_counts.size):
        plt.text(idx * 1., 3500, '{0:.1f}%'.format(y_counts[idx] / np.sum(y_counts) * 100), color=(1., 1., 1., .2), fontsize='small', horizontalalignment='center')
        # w = bar.get_with()
        # plt.text(bar.get_x() - .04, bar.get_y() + .1, '{0:.1f}%'.format())

    ax.set_xlim([-.5, 9.5])
    ax.set_xticks(y_unique)
    ax.set_xticklabels(['{0:.0f}'.format(idx) for idx in y_unique])

    ax.set_ylim([0, 8000])
    ax.set_yticks([7000], minor=True)
    ax.grid(which='minor', axis='y', alpha=.7, linestyle='--', color=tud_colors['lightgreen'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    ax.legend(bbox_to_anchor=(1, .5), loc='center left')
    fig.tight_layout()
    # fig.patch.set_visible(False)

    fig.savefig(os.path.join(os.environ['PGFPATH'], '{0}.pgf'.format(self_name)), format='pgf')
    fig.savefig(os.path.join(directory, '{0}.pdf'.format(self_name)), format='pdf')
    return


def plot_covariance(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    cov = np.cov((X - np.mean(X, axis=0)).T)
    cov_w, cov_v = np.linalg.eigh(cov)  # covariance matrix is always symmetric -> therefore eigh!
    cov_pca_expl_var = np.matmul(np.matmul(cov_v.T, cov), cov_v)
    # cov_pca_comp = cov_v.T

    n_components = 784

    cov_PCA_alternative = np.flip(np.cov(np.matmul(cov_v.T, X.T)), axis=(0, 1))
    cov_v_ordered = np.flip(cov_v, axis=(0, 1))
    # plt.imsave(os.path.join(directory, 'mnist-cov-pca-alt-db.png'), 20 * np.log10(np.abs(cov_PCA_alternative) + 1.), cmap=plt.cm.gray_r)

    # pca = PCA().fit(X)
    cov_PCA = cov_PCA_alternative  # np.cov(np.matmul(X, pca.components_[:n_components, :].T).T)

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))

    if isinstance(axs, plt.Axes):
        axs = [axs]

    axs[0].imshow(np.resize(cov, (X.shape[1], X.shape[1])), cmap=plt.cm.gray_r, interpolation='none')
    axs[0].set_title('covariance')
    axs[0].set_xticks(np.arange(start=0, stop=785, step=28))
    axs[0].set_xticklabels('{0:.0f}'.format(x) if x in [0, 784] else '' for x in np.arange(start=0, stop=785, step=28))
    axs[0].set_yticks(np.arange(start=0, stop=785, step=28))
    axs[0].set_yticklabels('{0:.0f}'.format(x) if x in [0, 784] else '' for x in np.arange(start=0, stop=785, step=28))

    axs[1].imshow(np.resize(20 * np.log10(np.abs(cov_PCA) + 1.), (n_components, n_components)), cmap=plt.cm.gray_r, interpolation='none')
    axs[1].set_title('after PCA ({0})'.format(n_components))
    axs[1].set_xticks(np.append(np.arange(start=0, stop=n_components, step=28), n_components))
    axs[1].set_xticklabels(np.append(['{0:.0f}'.format(x) if x == 0 else '' for x in np.arange(start=0, stop=n_components, step=28)], '{0}'.format(n_components)))
    axs[1].set_yticks(np.append(np.arange(start=0, stop=n_components, step=28), n_components))
    axs[1].set_yticklabels(np.append(['{0:.0f}'.format(x) if x == 0 else '' for x in np.arange(start=0, stop=n_components, step=28)], '{0}'.format(n_components)))

    axs[2].imshow(20 * np.log10(np.abs(cov_v_ordered.T) + 1.), cmap=plt.cm.gray_r, interpolation='none')
    axs[2].set_title('PCA components')
    axs[2].set_xticks(np.arange(start=0, stop=785, step=28))
    axs[2].set_xticklabels('{0:.0f}'.format(x) if x in [0, 784] else '' for x in np.arange(start=0, stop=785, step=28))
    axs[2].set_yticks(np.arange(start=0, stop=785, step=28))
    axs[2].set_yticklabels('{0:.0f}'.format(x) if x in [0, 784] else '' for x in np.arange(start=0, stop=785, step=28))

    def scale(A):
        return (A - np.min(A)) / (np.max(A) - np.min(A))

    sample_image_idx = 5

    for idx in [0, 1, 2, 3, 4, 5, 20, 50, 100, 200, 400, 600, 701, 783]:
        filepath = os.path.join(directory, '{0}{1}.png'.format('mnist-covariance-eig', idx))
        plt.imsave(filepath, cov_v_ordered.T[idx, ...].reshape(28, 28), cmap=plt.cm.gray_r)
    return

    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(directory, 'mnist-covariance.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-covariance.pgf'), format='pgf')
    plt.imsave(os.path.join(directory, 'mnist-covariance.png'), np.resize(cov, (X.shape[1], X.shape[1])), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-pca.png'), np.resize(cov_PCA, (n_components, n_components)), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-pca-components.png'), cov_v_ordered.T, cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-db.png'), np.resize(20 * np.log10(np.abs(cov) + 1.), (X.shape[1], X.shape[1])), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-pca-db.png'), np.resize(20 * np.log10(np.abs(cov_PCA) + 1.), (n_components, n_components)), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-pca-components-db.png'), 20 * np.log10(np.abs(cov_v_ordered.T) + 1.), cmap=plt.cm.gray_r)
    return


def plot_img_cluster(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    img = X[example_image_idx, :]

    clusterer = KMeans(n_clusters=4, random_state=42)
    img_clusters = clusterer.fit_predict(img.reshape((784, 1))).reshape((28, 28))
    list_cluster_colors = [tud_colors['lightblue'], tud_colors['lightgreen'], tud_colors['lightpurple'], tud_colors['gray']]
    # list_cluster_colors = [tud_colors['lightblue'], tud_colors['red'], tud_colors['gray'], tud_colors['gray']]
    # [tud_colors['lightblue'][:-1] + (.3,), tud_colors['lightgreen'][:-1] + (.3,), tud_colors['lightpurple'][:-1] + (.3,), tud_colors['gray'][:-1] + (.3,)]

    img_cluster_colors = np.zeros((28, 28, 4))

    for x in range(img_cluster_colors.shape[0]):
        for y in range(img_cluster_colors.shape[1]):
            img_cluster_colors[x, y, :] = list_cluster_colors[img_clusters[x, y]]

    # display digits
    fig, axs = plt.subplots(1, 1, figsize=(2, 2))

    # axs.imshow(np.resize(img, (28, 28)), cmap=plt.cm.gray_r, interpolation='none')
    axs.imshow(img_cluster_colors, interpolation='none')
    axs.set_xticks([0, 27])
    axs.set_xticklabels([0, 27])
    axs.set_yticks([0, 27])
    axs.set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'plot-img-clusters.pdf'), format='pdf')
    plt.imsave(os.path.join(directory, 'plot-img-clusters.png'), img_cluster_colors)


def plot_silhouette_dimension_reduction(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_dimension_reduction', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    X, X_trash, y, y_trash = train_test_split(X, y, random_state=42, train_size=1000, shuffle=True, stratify=y)

    # min_var = 6502.5
    min_expl_var = min_var

    scaler = StandardScaler().fit(X)
    whitener = PCA(whiten=False, random_state=42)

    X_avgfree = np.subtract(X, scaler.mean_) / 255
    X_min_var = X_avgfree[..., scaler.var_ > min_var]

    X_whitened_all = whitener.fit_transform(X)
    sum_explained_variance = np.flip(np.cumsum(np.flip(whitener.explained_variance_)))
    X_white_min_var = X_whitened_all[:, sum_explained_variance > min_expl_var]

    logger.info('whitener.explained_variance_ratio_: {0}'.format(
        np.sum(whitener.explained_variance_ratio_[whitener.explained_variance_ > min_expl_var])))
    logger.info('whitener.n_components_: {0}'.format(X_white_min_var.shape[1]))
    logger.info('scaler.var_.shape (> {1}): {0}'.format(min_var, scaler.var_[scaler.var_ > min_var].shape))

    k = np.concatenate((range(5, 20, 1), range(20, 110, 10))).astype(int)

    s_k = [-1]
    s_k_min_var = [-1]
    s_k_white_min_var = [-1]

    # k = [5, 10, 20]

    best_clusterer = None
    metric = 'euclidean'  # 'cosine'

    for n_clusters in k:
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

        pred = clusterer.fit_predict(X)
        s_k.append(silhouette_score(X, pred, metric=metric, random_state=42))

        pred = clusterer.fit_predict(X_min_var)
        s_k_min_var.append(silhouette_score(X_min_var, pred, metric=metric, random_state=42))

        pred = clusterer.fit_predict(X_white_min_var)
        s_k_white_min_var.append(silhouette_score(X_white_min_var, pred, metric=metric, random_state=42))

        logger.info('s_k: {0}'.format(s_k_white_min_var[-1]))

        if s_k_white_min_var[-1] > np.max(s_k_white_min_var[:-1]):
            best_clusterer = clusterer

    s_k.pop(0)
    s_k_min_var.pop(0)
    s_k_white_min_var.pop(0)

    with open('mnist-minibatch-kmeans-silhouette-results.txt', 'w') as results:
        results.write(
            '#features: all={0}, sigma>E(sigma)={1}, pca={2}'.format(X.shape[1], X_min_var[1], X_white_min_var[1]))
        results.write('k: {0}\n'.format(k))
        results.write('s(k): {0}\n'.format(s_k_white_min_var))
        results.write('best_clusterer.cluster_centers_: {0}\n'.format(best_clusterer.cluster_centers_))

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()

    lines = []
    lines += ax.loglog(k, s_k, color=tud_colors['lightpurple'])
    lines += ax.loglog(k, s_k_min_var, color=tud_colors['lightgreen'])
    lines += ax.loglog(k, s_k_white_min_var, color=tud_colors['lightblue'])

    ax.legend(lines, (
        'original ({0})'.format(X.shape[1]),
        'each $\sigma_i^2$ > {0} ({1})'.format(min_var, X_min_var.shape[1]),
        'PCA, $expl. \sigma^2$ > {0} ({1})'.format(min_expl_var, X_white_min_var.shape[1])))

    ax.set_xlim((np.min(k), np.max(k)))
    x_ticks = [5, 10, 15, 20, 50, 100]
    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(['{0:d}'.format(x) for x in x_ticks])
    ax.set_ylim((.055, .09))
    y_ticks = [.06, 0.07, 0.08, 0.09]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{0:0.2f}'.format(y) for y in y_ticks])

    ax.grid(which='both', axis='x')
    ax.grid(which='major', axis='y')
    ax.set_xlabel('#Clusters k')
    ax.set_ylabel('Silhouette score s(k)')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-kmeans-silhouette-k{0}.pdf'.format(np.max(k))))
    return


def plot_silhouette_kcluster(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_kcluster', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    min_expl_var = min_var

    # scaler = StandardScaler().fit(X)
    whitener = PCA(whiten=False, random_state=42)

    # X_avgfree = np.subtract(X, scaler.mean_)/255
    # X_min_var = X_avgfree[..., scaler.var_ > min_var]

    X_whitened_all = whitener.fit_transform(X)
    sum_explained_variance = np.flip(np.cumsum(np.flip(whitener.explained_variance_)))
    X_white_min_var = X_whitened_all[:, sum_explained_variance > min_expl_var]

    # rs = np.random.RandomState(42)
    # n_random = 20
    # X_random = X[:, rs.randint(low=0, high=X.shape[1], size=n_random)]

    logger.info('whitener.explained_variance_ratio_: {0}'.format(
        np.sum(whitener.explained_variance_ratio_[whitener.explained_variance_ > min_expl_var])))
    logger.info('whitener.n_components_: {0}'.format(X_white_min_var.shape[1]))
    # logger.info('scaler.var_.shape (> {1}): {0}'.format(min_var, scaler.var_[scaler.var_ > min_var].shape))

    k = np.concatenate((range(5, 20, 1), range(20, 110, 10))).astype(int)

    s_k_cosine = [-1]
    s_k_euclid = [-1]

    for n_clusters in k:
        clusterer_cosine = KCluster(n_clusters=n_clusters, metric='cosine', random_state=42)
        clusterer_euclid = KMeans(n_clusters=n_clusters, random_state=42)

        pred = clusterer_euclid.fit_predict(X_white_min_var)
        s_k_euclid.append(silhouette_score(X_white_min_var, pred, metric='euclidean', random_state=42))

        pred = clusterer_cosine.fit_predict(X_white_min_var)
        s_k_cosine.append(silhouette_score(X_white_min_var, pred, metric='cosine', random_state=42))

        logger.info('inertia_: {0} n_iter_: {1}'.format(clusterer_cosine.inertia_, clusterer_cosine.n_iter_))
        logger.info('s_k: {0}'.format(s_k_cosine[-1]))

    s_k_cosine.pop(0)
    s_k_euclid.pop(0)

    # noinspection PyTypeChecker
    np.savetxt(
        fname=os.path.join(directory, 'plot_silhouette_kcluster.csv'),
        X=np.hstack((np.array(k, ndmin=2).T, np.array(s_k_cosine, ndmin=2).T, np.array(s_k_euclid, ndmin=2).T)),
        fmt='%f,%f',
        header='k, k-means (cosine), k-means (euclidean)',
        comments='PCA - sum explained variance > {0}\n\n'.format(min_expl_var)
    )

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()

    lines = ax.loglog(k, s_k_cosine, color=tud_colors['lightblue'])
    lines += ax.loglog(k, s_k_euclid, color=tud_colors['lightpurple'])

    ax.legend(lines, ('cosine', 'euclidean'))
    ax.set_xlim((np.min(k), np.max(k)))
    ax.set_xticks(ticks=k)
    ax.set_xticklabels(
        ['5', '', '', '', '', '10', '', '', '', '', '15', '', '', '', '', '20', '', '', '50', '', '', '', '', '100'])
    # ax.set_ylim((.1, .2))
    # ax.set_yticks([.1, .12, .14, .16, .18, .2])
    # ax.set_yticklabels(['0.10', '0.12', '0.14', '0.16', '0.18', '0.20'])

    ax.grid(True)
    ax.set_xlabel('#Clusters k')
    ax.set_ylabel('Silhouette score s(k)')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-kcosine-silhouette-sub{0}.pdf'.format(X.shape[0])))
    return


def plot_silhouette_subset(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_subset', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    subset_sizes = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]  # , 64000]

    # scaler = StandardScaler().fit(X)
    min_expl_var = 6502.5
    whitener = PCA(whiten=False, random_state=42)

    # preprocessing
    X_whitened = whitener.fit_transform(X)
    sum_explained_variance = np.flip(np.cumsum(np.flip(whitener.explained_variance_)))
    indices = sum_explained_variance > min_expl_var
    X_whitended_variance = X_whitened[:, indices]

    logger.info('n_features: {0}'.format(X_whitended_variance.shape[1]))

    logger.info('explained variance: {0}'.format(np.sum(whitener.explained_variance_[indices])))

    logger.info('explained variance ratio: {0}'.format(np.sum(whitener.explained_variance_ratio_[indices])))

    # split subsets
    X_list = []
    y_list = []

    for train_size in subset_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X_whitended_variance, y, random_state=42, train_size=train_size, shuffle=True, stratify=y)

        X_list.append(X_train)
        y_list.append(y_train)

    # run experiment
    k_list = [15, 20, 30]
    s = []
    s_init = []

    for k in k_list:
        s.append([])
        s_init.append([])

        cluster_centers = KMeans(
            n_clusters=k, random_state=42, init='k-means++', n_init=50).fit(X_list[0]).cluster_centers_

        clusterer = KMeans(n_clusters=k, random_state=42)

        for X_subset in X_list:
            clusterer_init = KMeans(n_clusters=k, random_state=42, init=cluster_centers)
            s_init[-1].append(
                silhouette_score(X_subset, clusterer_init.fit_predict(X_subset), metric='euclidean', random_state=42))
            cluster_centers = clusterer_init.cluster_centers_

            if X_subset.shape[0] < 8000:
                s[-1].append(
                    silhouette_score(X_subset, clusterer.fit_predict(X_subset), metric='euclidean', random_state=42))
            else:
                s[-1].append(
                    silhouette_score(X_subset, clusterer.predict(X_subset), metric='euclidean', random_state=42))

            logger.info('s: {0}'.format(s[-1]))

    # plot results
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()

    lines = []
    for idx in range(len(k_list)):
        lines += ax.plot(subset_sizes, s[idx], color=tud_colors['lightblue'],
                         label='k (n_init={1})={0}'.format(k_list[idx], 10), alpha=(idx + 1) / len(k_list))
        lines += ax.plot(subset_sizes, s_init[idx], color=tud_colors['lightgreen'],
                         label='k (preinit)={0}'.format(k_list[idx]), alpha=(idx + 1) / len(k_list))

    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim((np.min(subset_sizes), np.max(subset_sizes)))
    ax.set_xticks(ticks=subset_sizes)
    ax.set_xticklabels(subset_sizes)

    ax.set_yscale('log')
    # ax.set_ylim((.1, .2))
    # ax.set_yticks([.1, .12, .14, .16, .18, .2])
    # ax.set_yticklabels(['0.10', '0.12', '0.14', '0.16', '0.18', '0.20'])

    ax.grid(True)
    ax.set_xlabel('subset size')
    ax.set_ylabel('Silhouette score s')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-kmeans-silhouette-sub{0}.pdf'.format(np.max(subset_sizes))))
    return


def plot_silhouette_features(directory, *args, **kwargs):
    logger = new_logger('plot_silhouette_features', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    scaler = StandardScaler().fit(X)
    variance_indices = np.argsort(scaler.var_)[::-1]
    whitener = PCA(whiten=False, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=1000, shuffle=True, stratify=y)
    X_train_whitened = whitener.fit_transform(X_train)

    logger.info('whitener.explained_variance_ratio_: {0}'.format(np.sum(whitener.explained_variance_ratio_)))
    logger.info('whitener.components_.shape: {0}'.format(whitener.components_.shape))

    n_features_list = np.concatenate((range(1, 10, 1), range(10, 100, 10), range(100, 800, 100))).astype(int)

    rs = np.random.RandomState(42)

    k = 20
    s_rnd = [-1]
    s_var = [-1]
    s_pca = [-1]

    var_rnd = []
    var_var = []
    var_pca = []

    for n_features in n_features_list:
        clusterer = KMeans(n_clusters=k, random_state=42)
        indices = rs.choice(X_train.shape[1], size=n_features)
        pred = clusterer.fit_predict(X_train[:, indices])
        s_rnd.append(silhouette_score(X_train[:, indices], pred, metric='euclidean', random_state=42))
        var_rnd.append(np.sum(scaler.var_[indices]))

        indices = variance_indices[:n_features]
        pred = clusterer.fit_predict(X_train[:, indices])
        s_var.append(silhouette_score(X_train[:, indices], pred, metric='euclidean', random_state=42))
        var_var.append(np.sum(scaler.var_[indices]))

        pred = clusterer.fit_predict(X_train_whitened[:, :n_features])
        s_pca.append(silhouette_score(X_train_whitened[:, :n_features], pred, metric='euclidean', random_state=42))
        var_pca.append(np.sum(whitener.explained_variance_[:n_features]))

        logger.info('silhouette: {0}'.format(s_pca[-1]))

    s_rnd.pop(0)
    s_var.pop(0)
    s_pca.pop(0)

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    ax_var = ax.twinx()

    lines_var = []
    lines_var += ax_var.plot(n_features_list, var_rnd, linestyle='dashed', color=tud_colors['lightpurple'])
    lines_var += ax_var.plot(n_features_list, var_var, linestyle='dashed', color=tud_colors['lightgreen'])
    lines_var += ax_var.plot(n_features_list, var_pca, linestyle='dashed', color=tud_colors['lightblue'])

    lines = []
    lines += ax.plot(n_features_list, s_rnd, color=tud_colors['lightpurple'])
    lines += ax.plot(n_features_list, s_var, color=tud_colors['lightgreen'])
    lines += ax.plot(n_features_list, s_pca, color=tud_colors['lightblue'])

    ax.legend(lines, ['random', 'sorted $\sigma^2$', 'pca expl. $\sigma^2$'], loc='center right')
    ax.set_xscale('log')
    ax.set_xlim((np.min(n_features_list), np.max(n_features_list)))
    x_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(['{:d}'.format(y) for y in x_ticks])

    ax.set_yscale('log')
    ax.set_ylim((.05, 1.))
    y_ticks = [.05, .1, .2, .5, 1.]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{:.2f}'.format(y) for y in y_ticks])

    ax_var.set_yscale('log')
    ax_var.set_ylim((1e4, np.sum(scaler.var_)))
    y_var_ticks = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
    ax_var.set_yticks(y_var_ticks)
    ax_var.set_yticklabels(['{0:0.0f} x 10³'.format(y / 1000) for y in y_var_ticks])

    ax.grid(True)
    ax.set_xlabel('#features')
    ax.set_ylabel('Silhouette score s')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-kmeans-silhouette-n_features{0}.pdf'.format(np.max(n_features_list))))
    return


def main(out_path=os.path.join(os.getcwd(), 'preprocessing-mnist'), function_name='labels'):
    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as error:
            print(error)

    # quick and dirty
    # directory = os.path.join(os.getcwd(), 'preprocessing-mnist')
    directory = out_path

    logger = new_logger('main')
    logger.info('{0} called, entering main'.format(__file__))

    runtime = [time.time()]

    # fetch data
    X, y = get_mnist()

    runtime.append(time.time())
    logger.info('fetch: {0} s'.format(np.diff(runtime[-2:])))
    logger.info('X.shape = {0}, y.shape = {1}'.format(X.shape, y.shape))

    function_dict = {
        'labels': plot_labels,
        'plot_pooling': plot_pooling,
        'histogram': plot_historgram,
        'var': plot_var,
        'normalized': plot_normalized,
        'variance_mean': plot_variance_mean,
        'image_min_var': plot_image_min_var,
        'plot_pca': plot_pca,
        'plot_covariance': plot_covariance,
        'plot_imbalance': plot_imbalance,
        'plot_img_cluster': plot_img_cluster,
        'plot_silhouette_dimension_reduction': plot_silhouette_dimension_reduction,
        'plot_silhouette_subset': plot_silhouette_subset,
        'plot_silhouette_kcluster': plot_silhouette_kcluster,
        'plot_silhouette_features': plot_silhouette_features
    }

    if function_name in function_dict:
        function_dict[function_name](directory)
    else:
        logger.warning('no function {0} found'.format(function_name))

    logger.info('{0} finished, return from main'.format(__file__))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(out_path=sys.argv[1], function_name=sys.argv[2])
    elif len(sys.argv) == 2:
        main(out_path=sys.argv[1])
    else:
        main()
