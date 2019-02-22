# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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
    return (1 / (2 * len(y)) * np.sum((((X.dot(theta)) - y) ** 2) + (0.5 * sum(theta[1:] ** 2))))


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
