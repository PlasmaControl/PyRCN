"""
An example of the Coates Idea on the digits dataset.
"""
import os
import sys
import time

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from pyrcn.util import tud_colors, new_logger, get_mnist

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
        axs[i // 5][i % 5].imshow(np.resize(X[int(idx[i])], (28, 28)),
                                  cmap=plt.cm.gray_r, interpolation='none')
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
    img_pooled = np.zeros((int(np.ceil(img.shape[0] / kernel_size[0])),
                           int(np.ceil(img.shape[1] / kernel_size[1]))))

    for x in range(img_pooled.shape[0]):
        for y in range(img_pooled.shape[1]):
            x_min = x*kernel_size[0]
            x_max = x_min + kernel_size[0]
            y_min = y*kernel_size[1]
            y_max = y_min + kernel_size[1]
            img_pooled[x, y] = np.max(img[x_min:x_max, y_min:y_max])

    plt.imsave(os.path.join(directory, 'pooled_max_kernel{0}x{1}.png'
                            .format(kernel_size[0], kernel_size[1])),
               img_pooled, cmap=plt.cm.gray_r)
    return


def plot_poster(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    X /= 255.

    # scale for imsave
    def scale01(X):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    # preprocessing
    pca = PCA(n_components=50).fit(X)
    clusterer = KMeans(n_clusters=20).fit(X[:10000, :])

    # save images
    # example
    plt.imsave(
        os.path.join(os.environ['IMGPATH'], 'example-mnist.png'),
        X[example_image_idx, :].reshape(28, 28), cmap=plt.cm.gray_r, format='png')

    # pca component
    pca_component = scale01(pca.components_[2, :]).reshape(28, 28)
    pca_example = scale01(
        np.matmul(X[example_image_idx, :].reshape(1, -1),
                  np.matmul(pca.components_.T, pca.components_))).reshape(28, 28)

    plt.imsave(
        os.path.join(os.environ['IMGPATH'], 'pca-component3.png'), pca_component,
        cmap=plt.cm.gray_r, format='png')
    plt.imsave(
        os.path.join(os.environ['IMGPATH'], 'pca50-mnist.png'), pca_example,
        cmap=plt.cm.gray_r, format='png')

    # kmeans centroids
    for idx in [0, 4, 9, 14, 19]:
        kmeans_centroid = scale01(clusterer.cluster_centers_[idx, ...]).reshape(28, 28)
        plt.imsave(
            os.path.join(os.environ['IMGPATH'], 'kmeans-centroid{0}.png'.format(idx)),
            kmeans_centroid, cmap=plt.cm.gray_r, format='png')

    # input weights
    T = np.load(os.path.join(os.environ['DATAPATH'], 'pca50+kmeans200_matrix.npy'),
                allow_pickle=True)
    for idx in [0, 49, 99, 149, 199]:
        input_weight = scale01(T[:, idx]).reshape(28, 28)
        plt.imsave(
            os.path.join(os.environ['IMGPATH'], 'input-weight{0}.png'.format(idx)),
            input_weight, cmap=plt.cm.gray_r, format='png')


def plot_historgram(directory, *args, **kwargs):
    logger = new_logger('plot_historgram', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2),
                            gridspec_kw={'width_ratios': [1, 1.7]})

    example = np.zeros((28, 28, 3))
    example[..., 0] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # red
    example[..., 1] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # green
    example[..., 2] = 1. - np.resize(X[example_image_idx, :], (28, 28)) / 255.  # blue

    idx_fringe = (25, 17)
    idx_center = (13, 12)

    example[idx_center[0], idx_center[1], :] = tud_colors['lightblue'][:-1]
    example[idx_fringe[0], idx_fringe[1], :] = tud_colors['orange'][:-1]

    bins = np.array(range(0, 287, 32)).astype(int)

    hist_fringe, bin_edges = np.histogram(X[:, idx_fringe[0] * 28 + idx_fringe[1]],
                                          bins=bins)
    hist_center, bin_edges = np.histogram(X[:, idx_center[0] * 28 + idx_center[1]],
                                          bins=bins)

    logger.info('validation sum hist_fringe: {0}, sum hist_center: {1}'
                .format(np.sum(hist_fringe / 1000), np.sum(hist_center / 1000)))

    axs[0].imshow(example, interpolation='none')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].bar(bins[1:] - 32, height=hist_fringe / 1000, width=16,
               color=tud_colors['orange'], label='fringe', align='edge')
    axs[1].bar(bins[1:] - 16, height=hist_center / 1000, width=16,
               color=tud_colors['lightblue'], label='center', align='edge')
    axs[1].tick_params(axis='x', labelrotation=90)
    # axs[1].hist([], bins=range(0, 255, 32), color=[tud_colors['orange'],
    #                                                tud_colors['lightblue']],
    #             align='left')

    axs[1].set_xticks(bins)
    # axs[1].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand",
    #               ncol=2)
    axs[1].legend(bbox_to_anchor=(1.0, .5), loc="center left")

    # fig.suptitle('Feature distribution in MNIST picture')
    axs[1].set_xlabel('value bins')
    axs[1].set_ylabel('probability')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-histogram.pdf'))
    fig.savefig(
        os.path.join(os.environ['PGFPATH'], 'mnist-pixel-histogram.pgf'),format='pgf')
    # plt.show()
    return


def plot_var(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    scaler = StandardScaler().fit(X)
    pos = range(0, 28)
    meanX = []
    varX = []

    fig, axs = plt.subplots(1, 2, figsize=(5, 2),
                            gridspec_kw={'width_ratios': [1, 1.4]})

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

    axs[1].legend((line_var, line_mean), (r'$\sigma^2$', r'$\mu$'),
                  bbox_to_anchor=(1.2, .5), loc="center left")

    # fig.suptitle('Feature distribution in MNIST picture')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].set_xlim([0, 27])
    axs[1].set_xlabel('position')
    axs[1].set_ylabel(r'$\sigma^2$', labelpad=-15, loc='top', rotation=0)
    y_ticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(['{0:0.0f}k'.format(y_tick/1000) for y_tick in y_ticks])
    # axs[1].tick_params(axis='x', labelrotation=90)
    ax_mean.set_ylabel('$\mu$', labelpad=-5, loc='top', rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-variance.pdf'))
    fig.savefig(
        os.path.join(os.environ['PGFPATH'], 'mnist-pixel-variance.pgf'), format='pgf')
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
    axs[1].set_title(r'$p_1$={1:0.2f}\n$\sigma^2$ > {0:0.0f}\n$n$={2:d}'
                     .format(var_p1_1, p1_1, np.sum(scaler.var_ > var_p1_1)))
    axs[1].set_xticks([0, 27])
    axs[1].set_xticklabels([0, 27])
    axs[1].set_yticks([0, 27])
    axs[1].set_yticklabels([0, 27])

    axs[2].imshow(np.reshape(example_min_var_p1_2, image_size), interpolation='none')
    axs[2].set_title(r'$p_1$={1:0.2f}\n$\sigma^2$ > {0:0.0f}\n$n$={2:d}'
                     .format(var_p1_2, p1_2, np.sum(scaler.var_ > var_p1_2)))
    axs[2].set_xticks([0, 27])
    axs[2].set_xticklabels([0, 27])
    axs[2].set_yticks([0, 27])
    axs[2].set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-img-min-var.pdf'))
    fig.savefig(
        os.path.join(os.environ['PGFPATH'], 'mnist-img-min-var.pgf'), format='pgf')
    # plt.show()
    return


def plot_normalized(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    X = X / 4 + 100
    X_picture_normalization = StandardScaler().fit_transform(X.T).T
    X_feature_normalization = StandardScaler().fit_transform(X)

    fig, axs = plt.subplots(1, 3, figsize=(5, 2))

    img_idx = example_image_idx

    axs[0].imshow(np.resize(X[img_idx, :], (28, 28)).astype(int), interpolation='none',
                  cmap=plt.cm.gray_r, norm=Normalize(vmin=0, vmax=255, clip=True))
    axs[0].set_title('low contrast')
    axs[0].set_xticks([0, 27])
    axs[0].set_xticklabels([0, 27])
    axs[0].set_yticks([0, 27])
    axs[0].set_yticklabels([0, 27])

    axs[1].imshow(np.resize(X_picture_normalization[img_idx, :], (28, 28)),
                  interpolation='none', cmap=plt.cm.gray_r)
    axs[1].set_title('picture\nnormalization')
    axs[1].set_xticks([0, 27])
    axs[1].set_xticklabels([0, 27])
    axs[1].set_yticks([0, 27])
    axs[1].set_yticklabels([0, 27])

    axs[2].imshow(np.resize(X_feature_normalization[img_idx, :], (28, 28)),
                  interpolation='none', cmap=plt.cm.gray_r)
    axs[2].set_title('feature\nnormalization')
    axs[2].set_xticks([0, 27])
    axs[2].set_xticklabels([0, 27])
    axs[2].set_yticks([0, 27])
    axs[2].set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-normalized.pdf'))
    fig.savefig(
        os.path.join(os.environ['PGFPATH'], 'mnist-normalized.pgf'), format='pgf')
    # plt.show()
    return


def plot_variance_mean(directory, *args, **kwargs):
    logger = new_logger('plot_variance_mean', directory)
    logger.info('entering')
    X, y = get_mnist(directory)

    image_size = (28, 28)

    scaler = StandardScaler().fit(StandardScaler(with_std=False).fit_transform(X) / 255)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(np.resize(scaler.mean_, image_size), cmap=plt.cm.gray_r,
                  interpolation='none')
    axs[0].imshow(np.resize(scaler.var_, image_size), cmap=plt.cm.gray_r,
                  interpolation='none')

    axs[0].set_title(r'$\mu$')
    axs[1].set_title(r'$\sigma^2$')

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-pixel-variance-and-mean-avgfree.pdf'))
    fig.savefig(os.path.join(os.environ['PGFPATH'],
                             'mnist-pixel-variance-and-mean-avgfree.pgf'), format='pgf')
    logger.info('np.max(scaler.mean_) = {0}, np.max(scaler.var_) = {1}'
                .format(np.max(scaler.mean_), np.max(scaler.var_)))
    return


def plot_pca(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    fig, axs = plt.subplots(1, 5, figsize=(6, 1.5),
                            gridspec_kw={'wspace': 0.45, 'left': .05, 'right': .95,
                                         'bottom': .0, 'top': .90})

    n_components_list = [784, 400, 100, 50, 30]
    min_p1_list = [0., .001, .01, .1, .5]
    min_var_list = [255 ** 2 * p1 * (1. - p1) for p1 in min_p1_list]

    # automatic pca
    decomposer = PCA(whiten=False).fit(X)

    sum_explained_variance = np.flip(np.cumsum(np.flip(decomposer.explained_variance_)))

    # original
    axs[0].imshow(np.resize(X[example_image_idx, ...], (28, 28)), cmap=plt.cm.gray_r,
                  interpolation='none')
    axs[0].set_title('original')

    # mean
    axs[1].imshow(np.resize(decomposer.mean_, (28, 28)), cmap=plt.cm.gray_r,
                  interpolation='none')
    axs[1].set_title('average'.format(100))

    # pca 50, average free
    X_avgfree = X - np.mean(X, axis=0)
    M_pca = decomposer.components_[:50, :].T
    M = np.dot(M_pca, M_pca.T)  # transformation and inverse combined

    axs[2].imshow(np.resize(np.dot(X_avgfree[example_image_idx, ...], M), (28, 28)),
                  cmap=plt.cm.gray_r, interpolation='none')
    axs[2].set_title('n={0}\naverage free'.format(M_pca.shape[1]))

    # pca 50, not average free
    axs[3].imshow(np.resize(np.dot(X[example_image_idx, ...], M), (28, 28)),
                  cmap=plt.cm.gray_r, interpolation='none')
    axs[3].set_title('n={0}\nwith average'.format(M_pca.shape[1]))

    # pca 25, not average free
    M_pca = decomposer.components_[:25, :].T
    M = np.dot(M_pca, M_pca.T)  # transformation and inverse combined

    axs[4].imshow(np.resize(np.dot(X[example_image_idx, ...], M), (28, 28)),
                  cmap=plt.cm.gray_r, interpolation='none')
    axs[4].set_title('n={0}\nwith average'.format(M_pca.shape[1]))

    for idx in range(5):
        axs[idx].set_xticks([0, 27])
        axs[idx].set_xticklabels([0, 27])
        axs[idx].set_yticks([0, 27])
        axs[idx].set_yticklabels([0, 27])

    # fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(directory, 'mnist-pca-effects.pdf'), format='pdf')
    fig.savefig(
        os.path.join(os.environ['PGFPATH'], 'mnist-pca-effects.pgf'), format='pgf')
    return


def plot_imbalance(directory):
    self_name = 'plot_imbalance'
    logger = new_logger(self_name, directory)
    X, y = get_mnist(directory)
    logger.info('successfully fetched {0} datapoints'.format(X.shape[0]))

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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6., 2.1))

    bar_width = .9

    bars_train = ax.bar(y_train_unique, y_train_counts, label='train',
                        color=tud_colors['gray'], width=bar_width)
    bars_test = ax.bar(y_test_unique, y_test_counts, bottom=y_train_counts,
                       label='test', color=tud_colors['lightblue'], width=bar_width)

    for idx in range(y_counts.size):
        plt.text(idx * 1., 3500, '{0:.1f}%'
                 .format(y_counts[idx] / np.sum(y_counts) * 100),
                 color=(1., 1., 1., .2), fontsize='small', horizontalalignment='center')
        # w = bar.get_with()
        # plt.text(bar.get_x() - .04, bar.get_y() + .1, '{0:.1f}%'.format())

    ax.set_xlim([-.5, 9.5])
    ax.set_xticks(y_unique)
    ax.set_xticklabels(['{0:.0f}'.format(idx) for idx in y_unique])
    ax.set_xlabel('label')

    ax.set_ylim([0, 8000])
    ax.set_yticks([7000], minor=True)
    ax.grid(which='minor', axis='y', alpha=.7, linestyle='--',
            color=tud_colors['lightgreen'])
    ax.set_ylabel(r'\#occurrences')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    ax.legend(bbox_to_anchor=(1, .5), loc='center left')
    fig.tight_layout()
    # fig.patch.set_visible(False)

    fig.savefig(os.path.join(os.environ['PGFPATH'], '{0}.pgf'.format(self_name)),
                format='pgf')
    fig.savefig(os.path.join(directory, '{0}.pdf'.format(self_name)), format='pdf')
    return


def plot_covariance(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    cov = np.cov((X - np.mean(X, axis=0)).T)
    cov_w, cov_v = np.linalg.eigh(cov)
    cov_pca_expl_var = np.matmul(np.matmul(cov_v.T, cov), cov_v)
    # cov_pca_comp = cov_v.T

    n_components = 784

    cov_PCA_alternative = np.flip(np.cov(np.matmul(cov_v.T, X.T)), axis=(0, 1))
    cov_v_ordered = np.flip(cov_v, axis=(0, 1))
    # plt.imsave(os.path.join(directory, 'mnist-cov-pca-alt-db.png'),
    #            20 * np.log10(np.abs(cov_PCA_alternative) + 1.), cmap=plt.cm.gray_r)

    # pca = PCA().fit(X)
    cov_PCA = cov_PCA_alternative

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))

    if isinstance(axs, plt.Axes):
        axs = [axs]

    axs[0].imshow(np.resize(cov, (X.shape[1], X.shape[1])), cmap=plt.cm.gray_r,
                  interpolation='none')
    axs[0].set_title('covariance')
    axs[0].set_xticks(np.arange(start=0, stop=785, step=28))
    axs[0].set_xticklabels('{0:.0f}'.format(x) if x in [0, 784] else ''
                           for x in np.arange(start=0, stop=785, step=28))
    axs[0].set_yticks(np.arange(start=0, stop=785, step=28))
    axs[0].set_yticklabels('{0:.0f}'.format(x) if x in [0, 784] else ''
                           for x in np.arange(start=0, stop=785, step=28))

    axs[1].imshow(np.resize(20 * np.log10(np.abs(cov_PCA) + 1.),
                            (n_components, n_components)),
                  cmap=plt.cm.gray_r, interpolation='none')
    axs[1].set_title('after PCA ({0})'.format(n_components))
    axs[1].set_xticks(np.append(np.arange(start=0, stop=n_components, step=28),
                                n_components))
    axs[1].set_xticklabels(np.append(['{0:.0f}'.format(x) if x == 0 else ''
                                      for x in np.arange(start=0, stop=n_components,
                                                         step=28)],
                                     '{0}'.format(n_components)))
    axs[1].set_yticks(np.append(np.arange(start=0, stop=n_components, step=28),
                                n_components))
    axs[1].set_yticklabels(np.append(['{0:.0f}'.format(x) if x == 0 else ''
                                      for x in np.arange(start=0, stop=n_components,
                                                         step=28)],
                                     '{0}'.format(n_components)))

    axs[2].imshow(20 * np.log10(np.abs(cov_v_ordered.T) + 1.), cmap=plt.cm.gray_r,
                  interpolation='none')
    axs[2].set_title('PCA components')
    axs[2].set_xticks(np.arange(start=0, stop=785, step=28))
    axs[2].set_xticklabels('{0:.0f}'.format(x) if x in [0, 784] else ''
                           for x in np.arange(start=0, stop=785, step=28))
    axs[2].set_yticks(np.arange(start=0, stop=785, step=28))
    axs[2].set_yticklabels('{0:.0f}'.format(x) if x in [0, 784] else ''
                           for x in np.arange(start=0, stop=785, step=28))

    def scale(A):
        return (A - np.min(A)) / (np.max(A) - np.min(A))

    sample_image_idx = 5

    for idx in [0, 1, 2, 3, 4, 5, 20, 50, 100, 200, 400, 600, 701, 783]:
        filepath = os.path.join(directory, '{0}{1}.png'
                                .format('mnist-covariance-eig', idx))
        plt.imsave(filepath, cov_v_ordered.T[idx, ...].reshape(28, 28),
                   cmap=plt.cm.gray_r)

    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(directory, 'mnist-covariance.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-covariance.pgf'),
                format='pgf')
    plt.imsave(os.path.join(directory, 'mnist-covariance.png'),
               np.resize(cov, (X.shape[1], X.shape[1])), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-pca.png'),
               np.resize(cov_PCA, (n_components, n_components)), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-pca-components.png'),
               cov_v_ordered.T, cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-db.png'),
               np.resize(20 * np.log10(np.abs(cov) + 1.), (X.shape[1], X.shape[1])),
               cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-covariance-pca-db.png'),
               np.resize(20 * np.log10(np.abs(cov_PCA) + 1.),
                         (n_components, n_components)), cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-pca-components-db.png'),
               20 * np.log10(np.abs(cov_v_ordered.T) + 1.), cmap=plt.cm.gray_r, vmax=.5)
    return


def plot_img_cluster(directory, *args, **kwargs):
    X, y = get_mnist(directory)

    img = X[example_image_idx, :]

    clusterer = KMeans(n_clusters=4, random_state=42)
    img_clusters = clusterer.fit_predict(img.reshape((784, 1))).reshape((28, 28))
    list_cluster_colors = [tud_colors['lightblue'], tud_colors['lightgreen'],
                           tud_colors['lightpurple'], tud_colors['gray']]

    img_cluster_colors = np.zeros((28, 28, 4))

    for x in range(img_cluster_colors.shape[0]):
        for y in range(img_cluster_colors.shape[1]):
            img_cluster_colors[x, y, :] = list_cluster_colors[img_clusters[x, y]]

    # display digits
    fig, axs = plt.subplots(1, 1, figsize=(2, 2))

    axs.imshow(img_cluster_colors, interpolation='none')
    axs.set_xticks([0, 27])
    axs.set_xticklabels([0, 27])
    axs.set_yticks([0, 27])
    axs.set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'plot-img-clusters.pdf'), format='pdf')
    plt.imsave(os.path.join(directory, 'plot-img-clusters.png'), img_cluster_colors)


def main(out_path=os.path.join(os.getcwd(), 'preprocessing-mnist'),
         function_name='labels'):
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
        'plot_poster': plot_poster,
        'histogram': plot_historgram,
        'var': plot_var,
        'normalized': plot_normalized,
        'variance_mean': plot_variance_mean,
        'image_min_var': plot_image_min_var,
        'plot_pca': plot_pca,
        'plot_covariance': plot_covariance,
        'plot_imbalance': plot_imbalance,
        'plot_img_cluster': plot_img_cluster,
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
