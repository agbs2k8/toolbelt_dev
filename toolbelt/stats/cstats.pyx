#!python
#cython: language_level=3

import math
from collections import Counter
from scipy.spatial import distance
import scipy.stats as stat
import pandas as pd
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def linreg_cost(X, y, theta):
    return 1/(2*len(y))*np.sum(((X.dot(theta))-y)**2)


def linreg_reg_cost(X, y, theta, l):
    return 1 / (2 * len(y)) * np.sum((((X.dot(theta)) - y) ** 2) + (0.5 * sum(theta[1:] ** 2)))


def linreg(X, y, l=0, regularize = False):
    if not (np.shape(X) != (len(X),) and sum(X[:, 0]) == len(X)):
        X = np.column_stack((np.array([1 for x in range(len(X))]), X))

    if not regularize:
        theta = np.linalg.pinv(X.transpose().dot(X)).dot((X.transpose().dot(y)))
        cost = linreg_reg_cost(X, y, theta, l)
        return theta, cost
    else:
        li = np.identity(X.shape[1]) * l
        li[0, 0] = 0

        theta = np.dot((np.linalg.pinv(np.dot(X.transpose(), X) + li)), (np.dot(X.transpose(), y)))

        cost = linreg_cost(X, y, theta)

        return theta, cost


def bic(kmeans, X: np.array) -> np.float:
    """
    Bayesian Information Criterion for clusters
    :param kmeans: fitted kmeans clustering object
    :param X: numpy array of original X values that were used to fit kmeans
    :return: float BIC
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
                                                           'euclidean') ** 2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d + 1)
    _bic = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                  ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

    return _bic


def cramer_v(x: np.array, y: np.array) -> np.float:
    """
    Cramer's V coefficient - Symmetrical correlation between two categorical variables
    :param x: array
    :param y: array
    :return: float
    """
    # Count of values by each intersection of categorical features
    confusion_matrix = pd.crosstab(x, y)
    # chi2 stat for the hypothesis test of independence of the observed frequencies
    chi2 = stat.chi2_contingency(confusion_matrix)[0]
    # total # of observations
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    # number of unique values in x & y
    r, k = confusion_matrix.shape
    # Bias Correction (see wikipedia if needed)
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    # Calc & return Cramer's V correlation
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def conditional_entropy(x: np.array, y: np.array) -> np.float:
    """
    Calculates the conditional entropy of x given y: S(x|y)
    :param x: array of data
    :param y: array of data
    :return: float of entropy measure
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theil_u(x: np.array, y: np.array) -> np.float:
    """
    Theil's Uncertainty Coefficient - the uncertainty of x given y: value is on the range of [0,1] - where 0 means y
    provides no information about x, and 1 means y provides full information about x
    :param x: array
    :param y: array
    :return: float
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = stat.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def corr_ratio(categories, continuous):
    """
    Given a continuous number, how well can you know to which category it belongs to?
    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
    a category can be determined with absolute certainty.
    :param categories: array of categorical data
    :param continuous: array of continuous data
    :return: float
    """
    # if the input is Pandas Series, convert back to an array
    if isinstance(categories, pd.Series):
        categories = categories.values
    if isinstance(continuous, pd.Series):
        continuous = continuous.values

    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = continuous[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(continuous, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    return eta

def fourier_extrapolation(x, n_harmonics=25, n_predict=0):
    n = x.size                         # number of observations
    t = np.arange(0, n)                # empty "time"
    p = np.polyfit(t, x, 1)            # find linear trend in x
    x_notrend = x - p[0] * t           # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    indexes.sort(key = lambda i: np.absolute(f[i]))  # sort indexes by frequency, lower -> higher
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harmonics * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
