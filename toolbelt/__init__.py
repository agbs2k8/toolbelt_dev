# -*- coding: utf-8 -*-
import os

__author__ = """AJ Wilson"""
__email__ = """aj.wilson08@gmail.com"""
__version__ = '0.0.1'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from .stats import visualize_distribution, test_stationarity
from .bsa import fix_bsa_email