# -*- coding: utf-8 -*-

import string
import re
import nltk
from nltk.corpus import stopwords
#from .utils import validate_str


my_stopwords = stopwords.words('english') 
stemmer = nltk.stem.snowball.SnowballStemmer('english')


#@validate_str
def remove_punctuation(text):
    """
    Simple function that will take a string or list of strings, and return them in the 
    same format with all punctuation marks removed
    """
    p_translator = str.maketrans('', '', string.punctuation+'’–')
    if isinstance(text, str):
        return text.translate(p_translator)
    elif isinstance(text, list):
        return [x.translate(p_translator) if isinstance(x, str) else x for x in text]
    else:
        return text


def remove_digits(text):
    """
    Simple function that will take a string or list of strings, and return them in the 
    same format with all numerical digits removed
    """
    d_translator = str.maketrans('', '', string.digits)
    if isinstance(text, str):
        return text.translate(d_translator)
    elif isinstance(text, list):
        return [x.translate(d_translator) if isinstance(x, str) else x for x in text]
    else:
        return text


@validate_str
def tokenize_and_stem(text):
    """
    Given a string, it returns a list of stemmed tokens i.e. the
    derivative of each word, as a list of strings
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


@validate_str
def tokenize_only(text):
    """
    Given a string, it returns a list of tokens i.e. the words, as a list of strings
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


@validate_str
def n_grams(text, ngram=2):
    """
    Uses NLTK functions to return a dictionary of all n-grams from a set of text
    and a count of the iterations of those ngrams
    N-gram = a set of words, of n length, that appear in the given text.

    The ngram words, as a tuple, are the dict keys, and the values are the number of iterations
    of that ngram found in the text
    """
    return dict(nltk.FreqDist(nltk.ngrams(nltk.word_tokenize(remove_punctuation(remove_digits(text.lower()))), ngram)))


@validate_str
def sentences(text):
    """
    Uses NLTK functions to return a dictionary of all sentences from a text and the
    count of iterations of those sentences.

    The dict key is the sentence, and the value is the number of iterations of that sentence
    """
    return dict(nltk.FreqDist(remove_punctuation(nltk.sent_tokenize(remove_digits(text.lower())))))
