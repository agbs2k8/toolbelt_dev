# K-Modes
# K-Medians -> L1 Norm as the distance measure
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy


def init_centroids(X, k=3):
    return X[np.random.choice(X.shape[0], k)]


def assign_clusters_cosine_dist(X, centroids):
    return np.argmax(cosine_similarity(centroids, X), axis=0)


def assign_clusters_euclidean_dist(X, centroids):
    return np.argmax(euclidean_distances(centroids, X), axis=0)


def update_mode_centroids(X, assigned_clusters, k, old_centroids):
    new_centroids = []
    for i in range(k):
        try:
            new_centroids.append(scipy.stats.mode(X[assigned_clusters == i]).mode[0])
        except IndexError:
            new_centroids.append(old_centroids[i])
    new_centroids = np.array(new_centroids)

    for i, j in combinations(range(k), 2):
        if np.array_equal(new_centroids[i], new_centroids[j]):
            raise ValueError('Two centroids have merged')
    return new_centroids


def update_median_centroids(X, assigned_clusters, k, old_centroids):
    new_centroids = []
    for i in range(k):
        _med = np.median(X[assigned_clusters == i], axis=1)
        if not np.isnan(_med):
            new_centroids.append(_med)
        else:
            new_centroids.append(old_centroids[i])
    new_centroids = np.array(new_centroids)

    for i, j in combinations(range(k), 2):
        if np.array_equal(new_centroids[i], new_centroids[j]):
            raise ValueError('Two centroids have merged')
    return new_centroids


def k_modes(X, k=3):
    _centroids = init_centroids(X, k=k)
    assigned_clusters = assign_clusters_cosine_dist(X, _centroids)

    old_clusters = np.empty(assigned_clusters.shape)
    iterations = 0

    while not np.array_equal(old_clusters, assigned_clusters) and iterations < 300:
        iterations += 1
        old_clusters = assigned_clusters.copy()

        _centroids = update_mode_centroids(X, old_clusters, k, _centroids)
        assigned_clusters = assign_clusters_cosine_dist(X, _centroids)

    return assigned_clusters, _centroids


def k_medians(X, k=3):
    _centroids = init_centroids(X, k=k)
    assigned_clusters = assign_clusters_euclidean_dist(X, _centroids)

    old_clusters = np.empty(assigned_clusters.shape)
    iterations = 0

    while not np.array_equal(old_clusters, assigned_clusters) and iterations < 300:
        iterations += 1
        old_clusters = assigned_clusters.copy()

        _centroids = update_median_centroids(X, old_clusters, k, _centroids)
        assigned_clusters = assign_clusters_euclidean_dist(X, _centroids)

    return assigned_clusters, _centroids
