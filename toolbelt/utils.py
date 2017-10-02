# -*- coding: utf-8 -*-
from functools import wraps
import pandas as pd


def validate_df(func):
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
    return wrapper


# def validate_list(func):
#     """
#    A decorator function to validate if the input is list
#    :param func: a function to validate
#    """
#    @wraps(func)
#    def wrapper(*args, **kwargs):
#        if isinstance(args[0], list):
#            return func(*args, **kwargs)
#        else:
#            raise TypeError('This tool only supports input as a list')
#    return wrapper


def quicksort(xs):
	if not xs:
		return []

    #pivots = [x for x in xs if x==xs[0]]
    #lesser = quicksort([x for x in xs if x < xs[0]])
    #greater = quicksort([x for x in xs if x > xs[0]])

    #return lesser + pivots + greater

	return quicksort([x for x in xs if x < xs[0]]) + [x for x in xs if x==xs[0]] + quicksort([x for x in xs if x > xs[0]])
