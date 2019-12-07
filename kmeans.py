"""
Author:
Zhehan Shi      - zs2442@columbia.edu
Guandong Liu    - gl2675@columbia.edu
Yue Wan         - yw3373@columbia.edu
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numpy import random
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn import datasets
from sklearn.cluster import KMeans


def kmeans(dataset, k):
    X, y = dataset  # y is useless, replace with new y got from kmeans
    n = X.shape[0]
    center = [random.normal(size=2) for i in range(k)]
    while True:
        cluster = {i: [] for i in range(k)}
        for i in range(n):
            y[i] = min((linalg.norm(X[i] - c[1]), c[0]) for c in enumerate(center))[1]
            cluster[y[i]].append(X[i])

        newcenter = []
        for key in cluster.keys():
            if len(cluster[key]) == 0:
                newcenter.append(random.normal(size=2))
            else:
                newcenter.append(np.mean(cluster[key], axis=0))

        if np.array_equal(center, newcenter):
            break

        center = newcenter

    return center, X, y


def compute_distance(X):
    dist_vec = pdist(X)
    dist_mat = squareform(dist_vec)
    return dist_mat


def rnn(X, r):
    n = X.shape[0]
    r = min(r, n - 1)
    dist = compute_distance(X)
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argpartition(dist[i, :], r)
        # r + 1 since W[i, i] = 0 is smallest
        W[i, idx[:r + 1]] = 1
        W[idx[:r + 1], i] = 1
        W[i, i] = 0
    return W


def rnn_v2(X, r):
    n = X.shape[0]
    count = np.zeros((n, 1))
    r = min(r, n - 1)
    dist = compute_distance(X)
    mapping = {}
    for i in range(n):
        for j in range(n):
            if i > j:
                mapping[dist[i, j]] = (i, j)
    mapping = dict(sorted(mapping.items()))
    W = np.zeros((n, n))
    for key, value in mapping.items():
        i, j = value
        if count[i] < r and count[j] < r:
            W[i, j] = 1
            W[j, i] = 1
            count[i] += 1
            count[j] += 1
    return W


def compute_matrix(W):
    n = W.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = np.sum(W[i, :])
    L = D - W
    return L


def compute_eigenvectors(L, k):
    n = L.shape[0]
    la, v = linalg.eig(L)
    mapping = {la[i]: v[:, i] for i in range(la.size)}
    mapping = dict(sorted(mapping.items()))
    V = np.zeros((n, k), dtype=complex)
    for i, (key, v) in enumerate(mapping.items()):
        if i >= k:
            break
        V[:, i] = v
    return V


def transform(X, k, r):
    W = rnn(X, r)
    L = compute_matrix(W)
    V = compute_eigenvectors(L, k)
    return V


def show_original(dataset, fname, save=False):
    X, y = dataset
    colors = np.array(list(['#377eb8', '#ff7f00', '#4daf4a']))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    plt.axis('equal')
    ax = plt.gca()
    plt.text(1, 1, 'original', ha='right', va='top', transform=ax.transAxes, fontsize=16)
    if save:
        plt.savefig(fname + '.png')
    plt.show()


def show_kmeans(center, X, y, s, fname, save=False):
    colors = np.array(list(['#377eb8', '#ff7f00', '#4daf4a']))
    center_colors = np.array(list(['#0000CC', '#CC3300', '#669933']))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    for i, c in enumerate(center):
        plt.scatter(c[0], c[1], s=100, color=center_colors[i])
    plt.axis('equal')
    ax = plt.gca()
    plt.text(1, 1, s, ha='right', va='top', transform=ax.transAxes, fontsize=16)
    if save:
        plt.savefig(fname + '.png')
    plt.show()


if __name__ == '__main__':
    # small example to validate the transformation
    # x = np.array([[0, 0], [0, 1], [1, 0], [3, 0], [3, 1]])
    # t = rnn_v2(x, 2)
    # v = transform(x, 2, 10)

    # settings
    np.random.seed(0)
    n_samples = 400
    k = 2
    possible_r = [1, 4, 10, 20, 40, 100, 400]

    # example 1
    circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # example 2
    moons = datasets.make_moons(n_samples=n_samples, noise=.05)

    # example 3
    m = n_samples // 2
    x_raw = random.uniform(-pi, pi, (n_samples, 1))
    x_sin1 = np.sin(x_raw[:m, :]) + 0.5
    x_sin2 = np.sin(x_raw[m:, :]) - 0.5
    x_sin = np.concatenate((x_sin1, x_sin2))
    x_sins = np.concatenate((x_raw, x_sin), axis=1)
    y_sins = np.concatenate((np.zeros(m, dtype=int), np.ones(m, dtype=int)))
    sins = (x_sins, y_sins)

    examples = {'circles': circles, 'moons': moons, 'sins': sins}
    for name, ex in examples.items():
        # show the original data
        show_original(ex, name)
        x_ex, y_ex = ex
        # run kmeans
        c_km, x_km, y_km = kmeans(ex, k)
        # show the data with kmeans
        show_kmeans(c_km, x_ex, y_km, 'kmeans', name + '_km')
        # perform transformation and run kmeans with different r
        for r in possible_r:
            v = transform(x_ex, k, r)
            ex_tf = (v, y_ex)
            c_tf, x_tf, y_tf = kmeans(ex_tf, k)
            # show the data with transformed kmeans
            show_kmeans(c_tf, x_ex, y_tf, 'r = ' + str(r), name + '_' + str(r))

    # run sklearn kmeans to validate
    '''res = KMeans(2).fit_predict(circles_x)
    plt.scatter(circles_x[:, 0], circles_x[:, 1], c=res)
    plt.show()
    v1 = transform(circles_x, k, r)
    res = KMeans(2).fit_predict(v1)
    plt.scatter(circles_x[:, 0], circles_x[:, 1], c=res)
    plt.show()'''
