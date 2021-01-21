"""
An example of the Coates Idea on the digits dataset.
"""
import scipy
import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt


def get_dataset(debug=True):
    X, y = None, None
    data_source = np.DataSource('.')
    if data_source.exists('./dataset/MNISTX.npy'):
        X = np.load('./dataset/MNISTX.npy')
        y = np.load('./dataset/MNISTy.npy')
    else:
        label_binarizer = LabelBinarizer()

        X_full, y_full = fetch_openml(data_id=554, data_home='./dataset', return_X_y=True, cache=True, as_frame=False)
        X, X_trash, y, y_trash = train_test_split(X_full, label_binarizer.fit_transform(y_full), shuffle=True, random_state=42, train_size=1000)
        np.save('./dataset/MNISTX.npy', X)
        np.save('./dataset/MNISTy.npy', y, allow_pickle=True)
    return X, y


def main():
    runtime = [time.time()]
    X, y = get_dataset(debug=True)
    runtime.append(time.time())
    print('fetch: {0} s'.format(np.diff(runtime[-2:])))
    print('X.shape = {0}, y.shape = {1}'.format(X.shape, y.shape))

    scaler = StandardScaler()
    whitener = PCA()

    X_preprocessed = whitener.fit_transform(scaler.fit_transform(X[:, ...]))
    runtime.append(time.time())
    print('preprocessing: {0} s'.format(np.diff(runtime[-2:])))

    k = np.concatenate((range(5, 20, 1), range(20, 110, 10))).astype(int)
    s_k = [0]
    best_clusterer = None

    for n_clusters in k:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        pred = clusterer.fit_predict(X)
        s_k.append(silhouette_score(X, pred, metric='cosine', random_state=42))  # ! sample size for large data

        if s_k[-1] > np.max(s_k[:-1]):
            best_clusterer = clusterer

        runtime.append(time.time())
        print('- KMeans with k={0}: {1} s'.format(n_clusters, np.diff(runtime[-2:])))

    s_k.pop(0)

    runtime.append(time.time())
    print('clustering: {0} s'.format(np.diff(runtime[-2:])))

    with open('mnist-minibatch-kmeans-silhouette-results.txt', 'w') as results:
        results.write('k: {0}\n'.format(k))
        results.write('s(k): {0}\n'.format(s_k))
        results.write('runtime(k): {0}\n'.format(runtime))
        results.write('best_clusterer.cluster_centers_: {0}\n'.format(best_clusterer.cluster_centers_))

    plt.plot(k, s_k)
    # plt.show()
    plt.suptitle('Silhouette score by MiniBatchKMeans clusters')
    plt.xlabel('#Clusters k')
    plt.ylabel('Silhouette score s(k)')
    plt.tight_layout()
    plt.savefig("mnist-minibatch-kmeans-silhouette-results.pdf")


if __name__ == "__main__":
    main()
