#!/bin/python

"""
This file contains several functions testing ELMs in different configurations,
optimize them and save the results in data files and pickles
"""

import sys
import os
import glob

import scipy
import numpy as np

import pickle
import csv

import time

import pandas

from scipy.signal import convolve2d

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedShuffleSplit

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import Ridge

from pyrcn.util import new_logger, argument_parser, get_mnist, tud_colors
from pyrcn.base import InputToNode, ACTIVATIONS, BatchIntrinsicPlasticity
from pyrcn.linear_model import IncrementalRegression
from pyrcn.extreme_learning_machine import ELMClassifier

train_size = 60000


def images_filter(images, kernel, stride=1):
    filtered = np.zeros(images.shape)
    for idx in range(images.shape[0]):
        filtered[idx, ...] = convolve2d(images[idx, ...], kernel, mode='same')
    return filtered


def picture_gradient(directory):
    self_name = 'picture_gradient'
    logger = new_logger(self_name, directory=directory)
    X, y = get_mnist(directory)
    logger.info('Loaded MNIST successfully with {0} records'.format(X.shape[0]))

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # scale X so X in [0, 1]
    X /= 255.

    # reshape X
    X_images = X.reshape((X.shape[0], 28, 28))

    list_kernels = [
        {'name': 'laplace', 'kernel': np.array([[-1., -1., -1.], [-1., 8, -1.], [-1., -1., -1.]])},
        {'name': 'mexicanhat', 'kernel': np.array([[0., 0., -1., 0., 0.], [0., -1., -2., -1., 0.], [-1., -2., 16, -2., -1.], [0., -1., -2., -1., 0.], [0., 0., -1., 0., 0.]])},
        {'name': 'v_prewitt', 'kernel': np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])},
        {'name': 'h_prewitt', 'kernel': np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]).T},
        {'name': 'v_sobel', 'kernel': np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])},
        {'name': 'h_sobel', 'kernel': np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).T}]

    example_image_idx = 5

    fig, axs = plt.subplots(1, 4, figsize=(6, 2))
    axs[0].imshow(X_images[example_image_idx], cmap=plt.cm.gray_r, interpolation='none')
    axs[0].set_title('no filter')
    axs[1].imshow(convolve2d(X_images[example_image_idx], list_kernels[0]['kernel'], mode='same'), cmap=plt.cm.gray_r, interpolation='none')
    axs[1].set_title('laplace')
    axs[2].imshow(convolve2d(X_images[example_image_idx], list_kernels[2]['kernel'], mode='same'), cmap=plt.cm.gray_r, interpolation='none')
    axs[2].set_title('vertical\nprewitt')
    axs[3].imshow(convolve2d(X_images[example_image_idx], list_kernels[5]['kernel'], mode='same'), cmap=plt.cm.gray_r, interpolation='none')
    axs[3].set_title('horizontal\nsobel')

    for ax in axs:
        ax.set_xticks([0, 27])
        ax.set_xticklabels([0, 27])
        ax.set_yticks([0, 27])
        ax.set_yticklabels([0, 27])

    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'mnist-image-filters.pdf'), format='pdf')
    fig.savefig(os.path.join(os.environ['PGFPATH'], 'mnist-image-filters.pgf'), format='pgf')


def plot_confusion(directory):
    self_name = 'plot_confusion'
    filepath = os.path.join(os.environ['DATAPATH'], '/coates20210310/est_coates-minibatch-pca50+kmeans16000_matrix-predicted.npz')

    label_encoder = LabelEncoder()

    npzfile = np.load(filepath, allow_pickle=True)
    X_test = np.array(npzfile['X_test'])
    y_test = np.array(npzfile['y_test']).astype(int)
    y_pred = np.array(npzfile['y_pred']).astype(int)

    conf_matrix = np.zeros((10, 10))
    conf_matrix_img = np.zeros((10, 10))

    n = y_test.size

    X_example = X_test[(y_pred == 5) & (y_test == 6), ...]
    img_example = X_example[3, ...]
    imgpath = os.path.join(directory, 'confused6for5.png')
    plt.imsave(imgpath, img_example.reshape(28, 28), format='png', cmap=plt.cm.gray_r)

    for pred_idx in range(conf_matrix.shape[0]):
        for test_idx in range(conf_matrix.shape[1]):
            conf_matrix[pred_idx, test_idx] = int(np.sum((y_pred == pred_idx) & (y_test == test_idx)))

    tpr = np.zeros(10)
    tnr = np.zeros(10)

    for idx in range(10):
        tpr[idx] = conf_matrix[idx, idx] / np.sum(conf_matrix[idx, :])  # row sum
        tnr[idx] = conf_matrix[idx, idx] / np.sum(conf_matrix[:, idx])  # col sum

    fpr = 1 - tpr
    fnr = 1 - tnr

    row_sum = np.linalg.norm(conf_matrix, ord=1, axis=0)  # sum over predicted
    col_sum = np.linalg.norm(conf_matrix, ord=1, axis=1)  # sum over true

    conf_matrix_norm = np.zeros((10, 10))

    # norm row by row! => TPR
    for idx in range(10):
        conf_matrix_norm[idx, :] = conf_matrix[idx, :] / np.sum(conf_matrix[idx, :])
        conf_matrix_norm[idx, idx] = 1 - conf_matrix_norm[idx, idx]

    # colormap
    n_colorsteps = 255  # in promille
    color_array = np.zeros((n_colorsteps, 4))
    lower_margin = 255
    color_array[:lower_margin, :] += np.linspace(start=tud_colors['lightgreen'], stop=tud_colors['red'], num=lower_margin)

    """
    upper_margin = 20
    color_array[n_colorsteps - upper_margin:, :] += np.linspace(start=tud_colors['red'], stop=tud_colors['lightgreen'], num=upper_margin)
    """
    cm = ListedColormap(color_array)

    fig = plt.figure(figsize=(4, 3))

    ax = fig.add_axes([.15, .15, .75, .75])
    img = ax.imshow(conf_matrix_norm * 100, interpolation='none', cmap=cm, origin='lower', alpha=.7)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['{0:.0f}'.format(pred_idx) for pred_idx in np.arange(10)])
    ax.set_xlabel('true')
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(['{0:.0f}'.format(pred_idx) for pred_idx in np.arange(10)])
    ax.set_ylabel('predicted')

    plt.colorbar(img, ax=ax, shrink=1., label='deviation from ideal TPR [%]')

    for pred_idx in range(conf_matrix.shape[0]):
        for test_idx in range(conf_matrix.shape[1]):
            ax.text(x=pred_idx, y=test_idx, s='{0:.0f}'.format(conf_matrix.T[pred_idx][test_idx]), fontsize='xx-small', verticalalignment='center', horizontalalignment='center')

    # plt.show()
    plt.savefig(os.path.join('./experiments/', 'confusion_matrix.pdf'), format='pdf')
    plt.savefig(os.path.join(os.environ['PGFPATH'], 'confusion_matrix.pgf'), format='pgf')


def main(directory, params):
    # workdir
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except PermissionError as e:
            print('mkdir failed due to missing privileges: {0}'.format(e))
            exit(1)

    workdir = directory

    # subfolder for results
    file_dir = os.path.join(directory, 'experiments')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    logger = new_logger('main', directory=file_dir)
    logger.info('Started main with directory={0} and params={1}'.format(directory, params))

    # register parameters
    experiment_names = {
        'picture_gradient': picture_gradient,
        'plot_confusion': plot_confusion,
    }

    # run specified programs
    for param in params:
        if param in experiment_names:
            experiment_names[param](file_dir)
        else:
            logger.warning('Parameter {0} invalid/not found.'.format(param))


if __name__ == '__main__':
    parsed_args = argument_parser.parse_args(sys.argv[1:])
    if os.path.isdir(parsed_args.out):
        main(parsed_args.out, parsed_args.params)
    else:
        main(parsed_args.params)
    exit(0)
