# -*- encoding: utf-8 -*-
import os

__author__ = """AJ Wilson"""
__email__ = """aj.wilson08@gmail.com"""
__version__ = '0.0.4'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

from .stats import visualize_distribution, test_stationarity, bic, cramer_v, theil_u, corr_ratio
from .utils import quicksort, batch, window
from .web_tools import scrape_links_from_url, scrape_links_from_urls
from .trees import Tree, Node
from .process_trees import Master, Host, ProcessTree, Process, find_all_matches
from .process_trees import read_host, read_process_tree, read_master, build_master_from_hosts
#from .nlp_tools import remove_punctuation, remove_digits, tokenize_and_stem, tokenize_only, n_grams, sentences

