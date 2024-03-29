# -*- coding: utf-8 -*-
from numbers import Number
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

    return fig


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


def describe(values):
    # make sure I have numerical values
    for val in values:
        if not isinstance(val, Number):
            raise ValueError('Only handles numeric values')
    # Make Sure it is a 1D Numpy Array
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    try:
        if values.shape[1] is not None:
            raise ValueError('Must be a 1D array')
    except IndexError:
        pass
    return {'count': float(len(values)),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            '25%': np.percentile(values, 25),
            '50%': np.percentile(values, 50),
            '75%': np.percentile(values, 75),
            'max': values.max()}
