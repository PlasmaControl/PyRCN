"""
An example of the Coates Idea on the digits dataset.
"""
import numpy as np
import time

import matplotlib
matplotlib.rc('image', cmap='binary')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pyrcn.util import get_mnist, tud_colors


sns.set_theme()


# define norm
def p2norm(x):
    return np.linalg.norm(x, axis=1, ord=2)


def get_unique(X, y):
    labels = np.unique(y)

    # find first occurrences
    idx = np.ones((len(labels), 2)) * -1
    cnt = 0
    while np.any(idx[:, 0] == -1):
        if idx[int(y[cnt]), 0] == -1.0:
            idx[int(y[cnt]), :] = int(cnt), y[cnt]
        cnt += 1

    # return sorted array
    sorted_index = idx[np.argsort(idx[:, 1]).astype(int), 0].astype(int)
    return X[sorted_index, ...], y[sorted_index, ...]


def main():
    runtime = [time.time()]
    X, y = get_mnist()
    runtime.append(time.time())
    print('fetch: {0} s'.format(np.diff(runtime[-2:])))

    whitener = PCA(50, random_state=42)

    X /= 255.

    X_preprocessed = whitener.fit_transform(X)
    runtime.append(time.time())
    print('preprocessing: {0} s'.format(np.diff(runtime[-2:])))

    cls = KMeans(n_clusters=10, random_state=42).fit(X_preprocessed)
    runtime.append(time.time())
    print('clustering: {0} s'.format(np.diff(runtime[-2:])))

    samples, values = get_unique(X_preprocessed, y)

    # reconstruct cluster centers
    cluster_centers = whitener.inverse_transform(cls.cluster_centers_)

    # normed_samples = (samples.T / np.linalg.norm(samples, axis=1, ord=2)).T
    # calculate distance
    cos_similarity = np.divide(np.dot(samples, cls.cluster_centers_.T),
                               p2norm(samples) * p2norm(cls.cluster_centers_))

    runtime.append(time.time())
    print('calculations: {0} s'.format(np.diff(runtime[-2:])))

    # display digits
    fig = plt.figure(figsize=(6, 3))
    gs_cyphers = gridspec.GridSpec(2, 5, figure=fig, wspace=.4, hspace=.3, top=.97,
                                   bottom=.1, left=.07, right=.95)

    for i in range(10):
        gs_cypher = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cyphers[i],
                                                     height_ratios=[1., .6], hspace=.05)

        ax_centroid = fig.add_subplot(gs_cypher[0, 0])  # axs[(i // 5) * 2, i % 5]
        ax_barchart = fig.add_subplot(gs_cypher[1, 0])  # axs[(i // 5) * 2 + 1, i % 5]

        ax_centroid.imshow(cluster_centers[i, :].reshape(28, 28), interpolation='none')
        ax_centroid.tick_params(left=False, bottom=False, labelleft=False,
                                labelbottom=False)

        ax_barchart.bar(list(map(int, values)), cos_similarity[:, i], tick_label=values,
                        color=tud_colors['lightblue'])
        # ax_barchart.set_xlim([0, 9])
        ax_barchart.grid(which='both', axis='y')
        ax_barchart.set_yticks([-1., 0., 1.], minor=False)
        ax_barchart.set_yticks([-.5, .5], minor=True)
        ax_barchart.set_ylim([-1., 1.])

    # plt.tight_layout()
    plt.savefig('mnist-kmeans-centroids-cos-similarity-pca50.pdf')  # plt.show()
    # plt.savefig(os.path.join(os.environ['PGFPATH'],
    # 'mnist-pca50-kmeans-centroids-cos-similarity.pgf'), format='pgf')
    runtime.append(time.time())
    print('plotting: {0} s'.format(np.diff(runtime[-2:])))


if __name__ == '__main__':
    main()
