"""
Plots for numerous reasons
"""
import os

import scipy
import numpy as np
import time

import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt


tud_colors = {
    'darkblue': (0 / 255., 48 / 255., 94 / 255.),
    'gray': (114 / 255., 120 / 255., 121 / 255.),
    'lightblue': (0 / 255., 106 / 255., 179 / 255.),
    'darkgreen': (0 / 255., 125 / 255., 64 / 255.),
    'lightgreen': (106 / 255., 176 / 255., 35 / 255.),
    'darkpurple': (84 / 255., 55 / 255., 138 / 255.),
    'lightpurple': (147 / 255., 16 / 255., 126 / 255.),
    'orange': (238 / 255., 127 / 255., 0 / 255.),
    'red': (181 / 255., 28 / 255., 28 / 255.)
}


directory = 'plots'


def f_x(x, sigma=1, mu=0.):
    return np.multiply(np.divide(1/np.sqrt(2*np.pi), sigma), np.exp(-.5 * np.power(np.divide((x - mu), sigma), 2)))


def f_y(y, sigma=1, mu=0.):
    return np.divide(f_x(np.arctanh(y), sigma, mu), (1 - np.power(y, 2)))


def save_plot_data(lines, filepath):
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

    save_plot_data(lines, os.path.join(directory, 'plot-distribution-sigma.csv'))
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

    save_plot_data(lines, os.path.join(directory, 'plot-distribution-mean.csv'))
    return


def main():
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_activation_variance()
    plot_activation_mean()
    return


if __name__ == "__main__":
    main()