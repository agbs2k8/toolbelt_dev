# -*- coding: utf-8 -*-
import os

__author__ = """AJ Wilson"""
__email__ = """aj.wilson08@gmail.com"""
__version__ = '0.0.4'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from .stats import visualize_distribution, test_stationarity, sigmoid, linreg
from .bsa import fix_bsa_email, multiprocess_fix_emails
from .finance import get_valuation, get_current, export_valuation
from .nlp_tools import remove_punctuation, remove_digits, tokenize_and_stem, tokenize_only
from .nlp_tools import n_grams, sentences
from .address import fill_zip, normalize_street, abbreviate_state, add_abbreviation
