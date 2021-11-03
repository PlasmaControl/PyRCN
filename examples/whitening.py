#!/bin/python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml


def image_whitening_cifar():
    directory = os.path.abspath('./examples/image/')
    data_name = 'CIFAR_10_small'

    npzfilepath = os.path.join(directory, '{0}.npz'.format(data_name))
    data_id = 40926

    if os.path.isfile(npzfilepath):
        npzfile = np.load(npzfilepath, allow_pickle=True)
        X, y = npzfile['X'], npzfile['y']
    else:
        df = fetch_openml(data_id=data_id, as_frame=True)
        X, y = df.data, df.target
        np.savez(npzfilepath, X=X, y=y)

    print('{0} samples loaded'.format(X.shape[0]))

    def scale(X, scaling_range=(0., 1.)):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    X /= 255.

    cov = np.cov(X.T)
    u, s, v = np.linalg.svd(cov)
    print(np.allclose(u.T, v))
    W = np.matmul(u, np.matmul(np.diag(np.sqrt(np.reciprocal(s, where=s != 0.))), u.T))
    X_pca = np.matmul(W, X.T).T

    sample_image_idx = 195
    sample_image = np.transpose(X[sample_image_idx, :].reshape((3, 32, 32)), axes=(1, 2, 0))
    sample_image_pca = np.transpose(X_pca[sample_image_idx, :].reshape((3, 32, 32)), axes=(1, 2, 0))

    list_dict_filter = [
        {
            'offset': 0,
            'color': 'r'
        },
        {
            'offset': 32 * 32,
            'color': 'g'
        },
        {
            'offset': 32 * 32 * 2,
            'color': 'b'
        }
    ]

    for dict_filter in list_dict_filter:
        sample_filter_idx = 32 * 16 + 16 + dict_filter['offset']
        sample_filter = np.transpose(W[:, sample_filter_idx].reshape((3, 32, 32)), axes=(1, 2, 0))
        plt.imsave(os.path.join(directory, 'cifar-filter-{0}-{1}x{2}.png'.format(dict_filter['color'], (sample_filter_idx % (32 ** 2)) // 32, (sample_filter_idx % (32 ** 2)) % 32)), scale(sample_filter))

    plt.imsave(os.path.join(directory, 'cifar-orignal.png'), sample_image)
    plt.imsave(os.path.join(directory, 'cifar-whitened.png'), scale(sample_image_pca))


def image_whitening_mnist():
    directory = os.path.abspath('./examples/image/')
    data_name = 'mnist_784'

    npzfilepath = os.path.join(directory, '{0}.npz'.format(data_name))
    data_id = 554

    if os.path.isfile(npzfilepath):
        npzfile = np.load(npzfilepath, allow_pickle=True)
        X, y = npzfile['X'], npzfile['y']
    else:
        df = fetch_openml(data_id=data_id, as_frame=True)
        X, y = df.data, df.target
        np.savez(npzfilepath, X=X, y=y)

    print('{0} samples loaded'.format(X.shape[0]))

    X /= 255.

    cov = np.cov(X.T)
    u, s, v = np.linalg.svd(cov)
    print(np.allclose(u.T, v))
    W = np.matmul(u, np.matmul(np.diag(np.sqrt(np.reciprocal(s, where=s != 0.))), u.T))

    X_pca = np.matmul(W, X.T).T

    sample_image_idx = 5
    sample_image = X[sample_image_idx, :].reshape((28, 28))
    sample_image_pca = X_pca[sample_image_idx, :].reshape((28, 28))

    sample_filter_idx = 28 * 14 + 14
    sample_filter = W[:, sample_filter_idx].reshape((28, 28))

    plt.imsave(os.path.join(directory, 'mnist-original.png'), sample_image, cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-whitened.png'), sample_image_pca, cmap=plt.cm.gray_r)
    plt.imsave(os.path.join(directory, 'mnist-filter-{0}x{1}.png'.format(sample_filter_idx // 28, sample_filter_idx % 28)), sample_filter, cmap=plt.cm.gray_r)


def main():
    image_whitening_cifar()
    # image_whitening_mnist()
    return


if __name__ == '__main__':
    main()