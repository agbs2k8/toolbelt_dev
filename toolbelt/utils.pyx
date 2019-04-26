#!python
#cython: language_level=3
# from functools import wraps
from itertools import islice
# import pandas as pd


'''def validate_df(func):
    """
    A decorator function to validate if the input is a pandas data frame
    :param func: a function to validate
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], pd.DataFrame):
            return func(*args, **kwargs)
        else:
            raise TypeError('This tool only supports input as a DataFrame')
    return wrapper


def validate_str(func):
    """
    A decorator function to validate if the input is string
    :param func: a function to validate
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], str):
            return func(*args, **kwargs)
        else:
            raise TypeError('This tool only supports input as a string')
    return wrapper'''


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


def window(sequence, n: int = 5):
    """
    Returns a sliding window of width n over the iterable sequence
    :param sequence: iterable to yield segments from
    :param n: number of items in the window
    :return: generator of windows
    """
    _it = iter(sequence)
    result = tuple(islice(_it, n))
    if len(result) == n:
        yield result
    for element in _it:
        result = result[1:] + (element,)
        yield result
