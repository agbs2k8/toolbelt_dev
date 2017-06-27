# -*- coding: utf-8 -*-

import os
import string
import re
from jellyfish import levenshtein_distance
from .utils import validate_str

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#@validate_str
def fix_bsa_email(email_in):
	"""
	Read and correct the various common errors of emails in the BSA servers
	"""
	#Regular Expression To Use:
	regex =r"""^(((([a-zA-Z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-zA-Z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?$"""

	#if passed a valid email address, return it
	if re.match(regex, email_in):
		return(email_in)

	#Remove common error of first character being a punctuation mark:
	if email_in.lstrip()[0] in string.punctuation:
		email_in = email_in.lstrip()[1:]

	#if a valid email address can be found within the string: 
	re_match = re.search(regex, email_in)
	if re_match is not None:
		if re.match(regex, re_match.group(0)):
			return(re_match.group(0))
	else:
		#if a valid email address can be found by removing erroneous whitespace
		email_in = "".join(email_in.split())
		re_match = re.search(regex, email_in)
		if re_match is not None:
			if re.match(regex, re_match.group(0)):
				return(re_match.group(0))

		else:
			#retrieve list of domains from cwd to compare
			with open(os.path.join(__location__,"domains_list.txt"),'r') as f:
				domains = f.read().split('\n')
			#split user input on the '@' character, length must be 2
			_split = email_in.split('@')
			if len(_split) == 2:
				best_alt_domain = ''
				_distance = 999
				#look for alternate, possibly correct domain structure in the list
				for domain in domains:
					this_dist = levenshtein_distance(_split[1],domain)
					#replace given domain with closeset by levenshtein_distance
					if this_dist < _distance:
						best_alt_domain = domain
						_distance = this_dist
				#if changing domain did not suffice to fix the problem, do not return the new domain
				if re.match(regex, _split[0]+"@"+best_alt_domain):
					return(_split[0]+"@"+best_alt_domain)

	return(None)