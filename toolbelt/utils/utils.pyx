#!python
#cython: language_level=3
import collections
from collections.abc import Iterable
from scipy.sparse import vstack
import numpy as np
import pandas as pd


def quicksort(xs):
    if not xs:
        return []
    return quicksort([x for x in xs if x < xs[0]]) + [x for x in xs if x == xs[0]] + quicksort(
        [x for x in xs if x > xs[0]])


def batch(iterable, n: int = 1):
    """
    Return a dataset in batches (no overlap)
    :param iterable: the item to be returned in segments
    :param n: length of the segments
    :return: generator of portions of the original data
    """
    for ndx in range(0, len(iterable), n):
        yield iterable[ndx:max(ndx+n, 1)]


def window(seq, n, fillvalue=None, step=1):
    """Return a sliding window of width *n* over the given iterable.

        >>> all_windows = window([1, 2, 3, 4, 5], 3)
        >>> list(all_windows)
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    When the window is larger than the iterable, *fillvalue* is used in place
    of missing values::

        >>> list(window([1, 2, 3], 4))
        [(1, 2, 3, None)]

    Each window will advance in increments of *step*:

        >>> list(window([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2))
        [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

    To slide into the iterable's items, use :func:`chain` to add filler items
    to the left:

        >>> iterable = [1, 2, 3, 4]
        >>> n = 3
        >>> padding = [None] * (n - 1)
        >>> list(window(itertools.chain(padding, iterable), 3))
        [(None, None, 1), (None, 1, 2), (1, 2, 3), (2, 3, 4)]

    """
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        yield tuple()
        return
    if step < 1:
        raise ValueError('step must be >= 1')

    it = iter(seq)
    window = collections.deque([], n)
    append = window.append

    # Initial deque fill
    for _ in range(n):
        append(next(it, fillvalue))
    yield tuple(window)

    # Appending new items to the right causes old items to fall off the left
    i = 0
    for item in it:
        append(item)
        i = (i + 1) % step
        if i % step == 0:
            yield tuple(window)

    # If there are items from the iterable in the window, pad with the given
    # value and emit them.
    if (i % step) and (step - i < n):
        for _ in range(step - i):
            append(fillvalue)
        yield tuple(window)


def extend_timeseries_dates(timeseries, n_new_periods):
    """
    Given a set of datetimes, add n number of new datetimes equally spaced after the existing ones and return it
    :param timeseries: an iterable of datetime objects spaced less than 1 month apart
    :param n_new_periods: an integer of the number of new periods to be added
    :return: a list of the original dates with the new dates appended to the end
    Notes:
        1. Cannot support Month & Year iterations.  Must be divisible into days/hours/minutes/etc.
        2. Dates need to be sorted ahead of time, otherwise the time interval will be very odd
    """
    if isinstance(timeseries[0], pd._libs.tslibs.timestamps.Timestamp):
        timeseries = timeseries.tz_localize(None)
    timeseries = list([np.datetime64(x) for x in timeseries.copy()])
    avg_timedelta = np.mean([x[1]-x[0] for x in window(timeseries, 2)])
    for _ in range(n_new_periods):
        timeseries.append(timeseries[-1]+avg_timedelta)

    return timeseries


def iter_check(x):
    return not isinstance(x, str) and isinstance(x, Iterable)


def flatten(xs):
    if len(xs)>0 and iter_check(xs[0]):
        return flatten([item for sublist in xs for item in sublist])
    else:
        return xs

def flat_map(func, iterable):
    """
    Flat Map - nothing further required
    :param func: function
    :param iterable: iterable
    :return: generator
    """
    for element in flatten([x for x in map(func, iterable)]):
        yield element


def flatten_dict(d, parent_key='', sep='->'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def balanced_random_sample(data: np.array, stratum: np.array, how: str = 'both',
                           fixed_size: int = None, random_state = None, #drop_undersized_classes: bool = False,
                           with_replacement: bool = True) -> (np.array, np.array):
    """
    Create a random sample that balances the classes so that the training data is not wildly skewed to a single class

    Parameters
    ----------
    data : np.array
        the training data (X) values that we are going to sample from. It should line up with the stratum data
        which will be used to select which rows are going to make it into the sample.

    stratum : np.array
        the column/data (y) that is going to be used to determine the classes/groupings that we are sampling for.

    how : str {'undersampe', 'oversample', 'both'}
        'undersample': take a random sample from the larger classes, and use most/all data from the smallest
        'oversample': use most/all of the largest class, and resample from the smaller classes to increase the sample
                      sizes
        'both': 50th percentile of sizes, and oversample/undersample from the larger/smaller classes to balance
                them

    fixed_size : int
        the fixed sample size for the classes.  Overrides the 'how' parameter

    random_state : int or None
        initialization value for the random number generator

    with_replacement : bool
        if True, will sample with replacement.

    #drop_undersized_classes: bool - TODO: build support for this

    Returns
    -------
    (data_sample: np.array, stratum_sample: np.array) - will return a tuple of two arrays, the first is the sampled
        data, and the second is the corresponding stratum labels.


    Examples
    --------
    >>> collections.Counter(y)
    Counter({'class_1': 300,
             'class_2': 100})
    >>> X_sample, y_sample = balanced_random_sample(X, y)
    >>> collections.Counter(y_sample)
    Counter({'class_1': 100,
             'class_2': 100})
    """
    # Handle the random seeding
    if isinstance(random_state, int):
        np.random.seed(random_state)


    stratum_counter = collections.Counter(stratum)
    n_classes = len(stratum_counter.keys())

    # Main switching
    if how == 'undersample':
        class_size = int(np.min(list(stratum_counter.values())))
    elif how == 'oversample':
        class_size = int(np.max(list(stratum_counter.values())))
    elif how == 'both':
        class_size = int(np.percentile(list(stratum_counter.values()), 50))
    else:
        raise ValueError("Invalid value passed for 'how' parameter. Must be in {'undersample', 'oversample', 'both}.")

    if fixed_size is not None:
        class_size = fixed_size

    data_result = None
    stratum_result = None
    # for each class
    for strat in stratum_counter.keys():
        # Get the indexes of the stratum that are part of this class
        idxs = np.where(stratum == strat)[0]
        # Use np.random.choice to select the indexes that we will keep/use
        choice_idxs = np.random.choice(idxs, size=class_size, replace=with_replacement)
        # Add the sampled data in with the final values
        if data_result is None:
            stratum_result = stratum[choice_idxs]
            data_result = data[choice_idxs]
        else:
            stratum_result = np.concatenate((stratum_result, stratum[choice_idxs]))
            data_result = vstack((data_result, data[choice_idxs]))

    # randomly sort the final array (shuffle
    final_idx = np.arange(len(stratum_result))
    np.random.shuffle(final_idx)

    # return the sorted values
    return data_result[final_idx], stratum_result[final_idx]
