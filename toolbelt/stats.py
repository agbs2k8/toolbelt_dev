# -*- coding: utf-8 -*-
"""
Notes on how to import from the github tracked directory:
    sys.path.append('/Users/ajwilson/GitRepos/toolbelt_dev/')
    from toolbelt.stats import *
"""
import math
from collections import Counter
import pandas as pd
import numpy as np
from scipy.spatial import distance
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_distribution(df):
    """
    Generates a 2x2 plot of KDE, ECDF, Violin and Histogram for passed data series
    will use first column of a provided data frame, or convert a series/list/array to numpy array
    :param df: a data source that can be converted to a numpy array
    """
    fig = plt.figure()

    if isinstance(df, pd.DataFrame):
        x = np.array(df[df.columns.values[0]])
    else:
        x = np.array(df)

    # KDE
    ax1 = fig.add_subplot(221)
    sns.kdeplot(x, ax=ax1)

    # ECDF
    ax2 = fig.add_subplot(222)
    x_ecdf = np.sort(x)
    y_ecdf = np.arange(1, len(x) + 1) / len(x)
    ax2.plot(x_ecdf, y_ecdf, marker='.', linestyle='none')

    # Violinplot
    ax3 = fig.add_subplot(223)
    sns.violinplot(x, ax=ax3)

    # Histogram
    ax4 = fig.add_subplot(224)
    sns.distplot(x, kde=False, ax=ax4)

    plt.tight_layout()

    plt.show()

    return(fig)


# @validate_df
def test_stationarity(timeseries, periods=12, df_test=True):
    # Determing rolling statistics
    rolmean = timeseries.rolling(center=False, window=periods).mean()
    rolstd = timeseries.rolling(center=False, window=periods).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(30, 10))
    orig = plt.plot(timeseries, color='blue', label='Original', linewidth=2)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=2)
    std = plt.plot(rolstd, color='black', label='Rolling Std', linewidth=2)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    """if df_test:
        print('Results of Dickey-Fuller Test:')
        print('---\nHo: A unit-root IS present in the data. \nHa: The data is stationary.\n---')
        print('If p < alpha, then reject Ho.')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
    """


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
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
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
    s_x = stats.entropy(p_x)
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
