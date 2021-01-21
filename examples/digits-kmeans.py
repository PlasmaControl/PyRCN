"""
An example of the Coates Idea on the digits dataset.
"""

import scipy
import numpy as np
import time

import matplotlib
matplotlib.rc('image', cmap='binary')

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_openml
from sklearn.cluster import KMeans


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
    X_digits, y_digits = fetch_openml(data_id=554, data_home='./dataset', return_X_y=True, cache=True, as_frame=False)  # load_digits(return_X_y=True)
    runtime.append(time.time())
    print('fetch: {0} s'.format(np.diff(runtime[-2:])))

    scaler = StandardScaler()
    whitener = PCA()

    X_preprocessed = whitener.fit_transform(scaler.fit_transform(X_digits[:, ...]))
    runtime.append(time.time())
    print('preprocessing: {0} s'.format(np.diff(runtime[-2:])))

    cls = KMeans(n_clusters=10, random_state=42).fit(X_preprocessed)
    runtime.append(time.time())
    print('clustering: {0} s'.format(np.diff(runtime[-2:])))

    samples, values = get_unique(X_preprocessed, y_digits)

    # reconstruct cluster centers
    cluster_centers = scaler.inverse_transform(whitener.inverse_transform(cls.cluster_centers_))
    cluster_center_norm = np.linalg.norm(cluster_centers, axis=1, ord=2)

    # calculate distance
    cos_distance = np.divide(np.dot(samples, np.transpose(cls.cluster_centers_)), cluster_center_norm)

    runtime.append(time.time())
    print('calculations: {0} s'.format(np.diff(runtime[-2:])))

    # display digits
    fig, axs = plt.subplots(4, 5, figsize=(8, 6))
    fig.suptitle('K-Means centroids and their cos-similarity to a series of digits')

    for i in range(10):
        ax_centroid = axs[(i // 5) * 2, i % 5]
        ax_barchart = axs[(i // 5) * 2 + 1, i % 5]

        ax_centroid.imshow(cluster_centers[i, :].reshape(28, 28), interpolation='nearest')
        ax_centroid.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax_barchart.bar(values, cos_distance[:, i], tick_label=values)

    plt.tight_layout()
    plt.savefig("mnist-kmeans-centroids-cos-distance-normalized-whitened.pdf")  # plt.show()
    runtime.append(time.time())
    print('plotting: {0} s'.format(np.diff(runtime[-2:])))


if __name__ == "__main__":
    main()
