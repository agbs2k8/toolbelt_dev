import time
import json
from ipaddress import ip_address, ip_network, IPv4Address
import requests
import numpy as np
import pandas as pd

from .cipaddresses import find_subnet_key, check_ip


def find_subnets(ip_list: list, existing_subnets: dict = None) -> dict:
    """
    given a list of IP Addresses, call an available API for more data about the IP
    :param ip_list: a list of IP addresses
    :param existing_subnets: optional dict of the results from a prior run of this function
    :return: a dictionary. {IPv4Network: {'ip_list': [...], other api data...}}
    """
    api_url = 'https://api.bgpview.io/ip/{}'

    # make sure we have the right data-type to work from (assuming all are the same data type)
    if isinstance(ip_list[0], str):
        ip_list = [ip_address(ip) for ip in ip_list]
    # make sure the IPs are in order
    ip_list.sort()

    # the return value...
    if existing_subnets is None:
        subnets = dict()
    else:
        subnets = existing_subnets

    # this works better as a sub function with a return than messing with looping over the dictionary
    # and trying to do flow control in the while loop below
    def find_ip_key(ip_obj, subnets_dict):
        for temp_subnet in subnets_dict.keys():
            if ip_obj in temp_subnet:
                return temp_subnet
        return None

    # loop through all of the IP addresses I want to check
    for ip in ip_list:
        # Look and see if an existing subnet would contain this IP address
        existing_subnet = find_ip_key(ip, subnets)
        if existing_subnet:
            subnets[existing_subnet]['ip_list'].append(ip)
        else:
            # Sleep for 1 second before making the API call so I don't get booted
            time.sleep(1)
            # Call API for the data about the subnet that would contain that IP
            r = requests.get(api_url.format(str(ip)))
            if r.status_code == 200:
                content = r.content
                api_return = json.loads(content)['data']
            else:
                continue
            try:
                _subnet = api_return['prefixes'][0]['prefix']
            except IndexError:
                print('API Error on ip: {}'.format(ip))
                continue
            # add data to the return dict
            subnets[ip_network(_subnet)] = {'ip_list': [ip],
                                            '_rir_name': api_return['rir_allocation']['rir_name'],
                                            '_rir_coco': api_return['rir_allocation']['country_code'],
                                            '_rir_sub': api_return['rir_allocation']['prefix'],
                                            '_rir_date': api_return['rir_allocation']['date_allocated']}
            try:
                subnets[ip_network(_subnet)]['_asn_num'] = api_return['prefixes'][0]['asn']['asn']
                subnets[ip_network(_subnet)]['_asn_name'] = api_return['prefixes'][0]['asn']['name']
                subnets[ip_network(_subnet)]['_asn_desc'] = api_return['prefixes'][0]['asn']['description']
                subnets[ip_network(_subnet)]['_asn_country'] = api_return['prefixes'][0]['asn']['country_code']
                subnets[ip_network(_subnet)]['_name'] = api_return['prefixes'][0]['name']
                subnets[ip_network(_subnet)]['_desc'] = api_return['prefixes'][0]['description']
                subnets[ip_network(_subnet)]['_coco'] = api_return['prefixes'][0]['country_code']
            except IndexError:
                pass

    return subnets


def append_network_data(ip_series: pd.Series, subnets: dict, attributes: list) -> pd.DataFrame:
    """
    Given a Pandas Series of IP Addresses (presumably a column from a dataframe) and a dictionary of subnets
    Find out what subnet the IP belongs to, bring back a dataframe of the IP and the listed attributes
    :param ip_series: A Pandas Series of IPv4Addresses or strings which can be converted to them
    :param subnets: the dictionary results from the find_subnets() function
    :param attributes: a list of the keys from the find_subnets() data results that are desired
    :return: a dataframe of the expanded IP data results
    """
    ip_values = np.sort(ip_series[ip_series.notnull()].unique())
    result_rows = []
    prior_subnet = None
    prior_subnet_data = dict()
    for ip in ip_values:
        original_type = type(ip)
        current_subnet = None
        current_subnet_data = None
        result_row = []
        # Validate the IP address
        ip = check_ip(ip, error_bad_ip=False)
        if ip is None:
            continue
        # See if the IP address is in the prior loop-iteration's result since we sorted things
        if prior_subnet is not None:
            if ip in prior_subnet:
                current_subnet = prior_subnet
                current_subnet_data = prior_subnet_data
        # If the IP wasn't in the prior subnet or if we dont have a prior, find our subnet from the list
        if current_subnet is None:
            subnet_key = find_subnet_key(ip, subnets, error_bad_ip=False)
            if subnet_key is None:
                continue
            else:
                current_subnet = subnet_key
                current_subnet_data = subnets[current_subnet]

        # Put the IP Address into the result row as the key. Make sure the type matches the original
        if original_type == str:
            result_row.append(str(ip))
        elif original_type == IPv4Address:
            result_row.append(ip)

        # Put the rest of the items into the result row:
        for attribute in attributes:
            if attribute in current_subnet_data.keys():
                result_row.append(current_subnet_data[attribute])
            else:
                result_row.append(None)

        # Add this result to the master table
        result_rows.append(result_row)

        # Set this run as the prior to hopefully do less overall look-ups
        prior_subnet = current_subnet
        prior_subnet_data = current_subnet_data

    # Turn all of the results into a dataframe with the IP as the key:
    results_df = pd.DataFrame(result_rows, columns=['ip']+attributes).set_index('ip')
    # convert the original series into a dataframe, join the results to it and return all
    return ip_series.to_frame().join(results_df, on=ip_series.name)



