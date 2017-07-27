# -*- coding: utf-8 -*-

import os
import string
import re
import pandas as pd
from nltk.metrics import edit_distance
from .utils import validate_str
from .distribute_processing import multiprocess_function

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


@validate_str
def fix_bsa_email(email_in, domains = None):
    """Read and correct the various common errors of emails in the BSA servers"""

    # Only work with lower-case emails
    email_in = email_in.lower()

    # Regular Expression To Use:
    regex = r"""^(((([a-zA-Z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-zA-Z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?$"""

    # if passed a valid email address, return it
    if re.match(regex, email_in):
        return email_in

    email_segments = email_in.split()
    for segment in email_segments:
        re_match=re.search(regex,segment)
        if re_match is not None:
            return re_match.group(0)
        else:
            continue

    # Remove common error of first character being a punctuation mark:
    if email_in.lstrip()[0] in string.punctuation:
        email_in = email_in.lstrip()[1:]

    # if a valid email address can be found within the string:
    re_match = re.search(regex, email_in)
    if re_match is not None:
        if re.match(regex, re_match.group(0)):
            return re_match.group(0)
    else:
        # if a valid email address can be found by removing erroneous whitespace
        email_in = "".join(email_in.split())
        re_match = re.search(regex, email_in)
        if re_match is not None:
            if re.match(regex, re_match.group(0)):
                return re_match.group(0)

        else:
            # retrieve list of domains from cwd to compare
            if domains is None:
                domains = pd.read_pickle(os.path.join(__location__,"domains.pickle"))

            # split user input on the '@' character, length must be 2
            _split = email_in.split('@')
            if len(_split) == 2:
                domains['repl_proba'] = (domains['probability']) / (domains['domain'].apply(edit_distance,
                                                                                            s2 = _split[1])**2.71828)
                best_alt_domain = domains.sort_values(by='repl_proba', ascending = False).iloc[0]['domain']

                # if changing domain did not suffice to fix the problem, do not return the new domain
                if re.match(regex, _split[0]+"@"+best_alt_domain):
                    return _split[0]+"@"+best_alt_domain
    return None


# @validate_list
def fix_email_list(email_list, return_dict=None, domains=None):
    if return_dict is not None:
        for email in email_list:
            return_dict[email] = fix_bsa_email(email, domains)
    else:
        fixed_emails = []
        for email in email_list:
            fixed_emails.append(fix_bsa_email(email, domains))
        return fixed_emails


# @validate_list
def multiprocess_fix_emails(email_list, n_jobs=4):
    domains = pd.read_pickle(os.path.join(__location__, "domains.pickle"))
    return multiprocess_function(data_in=email_list, func=fix_email_list, n_jobs=n_jobs, shared_resources=domains)
