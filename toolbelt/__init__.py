# -*- encoding: utf-8 -*-
import os

__author__ = """AJ Wilson"""
__email__ = """aj.wilson08@gmail.com"""
__version__ = '0.0.4'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from .stats import visualize_distribution, test_stationarity, describe
from .stats import sigmoid, linreg_cost, linreg_reg_cost, linreg, bic, cramer_v
from .stats import conditional_entropy, theil_u, corr_ratio
from .stats import MarkovChain, SequenceModel, get_unique_items
from .stats import fourier_extrapolation
from .utils import *
from .trees import *
from .feature_extraction import *
from .cluster import *
from .network import *
