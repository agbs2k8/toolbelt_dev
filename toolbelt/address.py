import os
import re
import pickle
from .utils import validate_str
from .nlp_tools import remove_punctuation


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

_abbreviations = pickle.load(open(os.path.join(__location__, '_address_abbreviations.pkl'), 'rb'))
state_abbreviations = pickle.load(open(os.path.join(__location__, '_state_abbreviations.pkl'), 'rb'))


@validate_str
def fill_zip(orig):
    """
    :param orig: zip code as text, potentially missing leading 0's
    :return: recursively filled ZIP with leading 0's
    """
    if len(orig) < 5:
        return fill_zip('0'+orig)
    else:
        return orig


@validate_str
def numeric_abbreviation(in_text):
    """
    :param in_text: string of an ordinal number that uses the digit character (i.e. 1st, 2nd)
    :return: a string of the ordinal number in plain text (i.e. FIRST, SECOND)
    """
    std_street = []
    number = ""
    for x in range(0, len(in_text)):
        if re.match('[0-9]', in_text[x]) is not None:
            number = number+in_text[x]
    if len(number) == 1:
        std_street.append(_abbreviations[str(int(number))][1])
    elif len(number) == 2:
        if number[-1] == '0':
            std_street.append(_abbreviations[str(int(number))][1])
        elif int(number) < 20:
            std_street.append(_abbreviations[str(int(number))][1])
        else:
            std_street.append(_abbreviations[str(int(number[0]))+"0"][0])
            std_street.append(_abbreviations[str(int(number[1]))][1])
    elif len(number) == 3:
        std_street.append(_abbreviations[number[0]][0])
        std_street.append('HUNDRED')

        if float(number) % 10 == 0:
            std_street.append(_abbreviations[str(int(number[-2:]))][1])
        elif float(number[-2:]) < 20:
            std_street.append(_abbreviations[str(int(number[-2:]))][1])
        else:
            std_street.append(_abbreviations[number[-2]+"0"][0])
            std_street.append(_abbreviations[number[-1]][1])
    return ' '.join(std_street)


@validate_str
def normalize_street(street):
    """
    :param street: street address
    :return: standardized street addresses...
    it takes all full words down to their abbreviations,
    removes punctuation and ordinal numbers using digit characters
    """
    std_street = []
    street_elements = remove_punctuation(street).split()
    for element in street_elements:
        if element.isdigit():
            std_street.append(element)
        
        elif element.upper() in _abbreviations.keys():
            std_street.append(_abbreviations[element.upper()])
        
        elif re.match('.*[0-9][a-zA-Z][a-zA-Z]$', element):
            temp = numeric_abbreviation(element)
            std_street.append(temp)
        else:
            std_street.append(element.upper())
    
    return ' '.join(std_street)


@validate_str
def abbreviate_state(state):
    """
    :param state: full text of a state's name
    :return: abbreviation of that state's name
    """
    if len(state) > 2:
        state_abbrev = state_abbreviations[state.upper()]
        return state_abbrev
    elif len(state) == 2:
        return state.upper()
    else:
        return None


@validate_str
def add_abbreviation(key, value):
    """
    :param key: the new word that needs to be added in
    :param value: the new abbreviation for the word
    :return: True if added, False if not added/already present
    """
    if key.upper() not in _abbreviations.keys():
        _abbreviations.update({key.upper(): str(value).upper()})

        with open(os.path.join(__location__, '_address_abbreviations.pkl'), 'wb') as p:
            pickle.dump(_abbreviations, p)
        return True
    else:
        return False
