#!python
#cython: language_level=3

from ipaddress import ip_address, IPv4Address, IPv4Network


def check_ip(ipaddr, error_bad_ip: bool=True) -> IPv4Address:
    """Validate provided input that it is an IPv4Address or string that can be converted to one
    :param ipaddr: either a string or IPv4Address object
    :param error_bad_ip: boolean if you want an error thrown when an invalid object is provided for the ip address
                         or only getting None returned
    :return: a valid IPv4Address obbject
    """
    if not isinstance(ipaddr, IPv4Address):
        try:
            ipaddr = ip_address(ipaddr)
            if not isinstance(ipaddr, IPv4Address):
                if error_bad_ip:
                    raise ValueError('Failed to convert the {} to a IPv4Address'.format(ipaddr))
                else:
                    print('Failed to convert the {} to a IPv4Address'.format(ipaddr))
                    return None
        except ValueError:
            if error_bad_ip:
                raise ValueError('Failed to convert the {} to a IPv4Address'.format(ipaddr))
            else:
                print('Failed to convert the {} to a IPv4Address'.format(ipaddr))
                return None
    return ipaddr


def find_subnet_key(ipaddr, subnets:dict, error_bad_ip: bool=True) -> IPv4Network:
    """
    Use the power of Cython to iterate through a dictionary more quickly and find what I'm looking for
    :param ipaddr: IPv4Address or string of an IPv4 address
    :param subnets: the dictionary of subnets with the key being the IPv4Network Object and the values being
                    a dict of network data
    :param error_bad_ip: boolean if you want an error thrown when an invalid object is provided for the ip address
                         or only getting None returned
    :return: IPv4Network object that is one of the keys from the subnets dict
    """
    ip_addr = check_ip(ipaddr, error_bad_ip)
    for subnet in subnets.keys():
        if ip_addr in subnet:
            return subnet
    return None


def get_network_attribute(ipaddr, subnets: dict, attributes: list, error_bad_ip: bool=True) -> dict:
    """
    Find specific network attributes from the subnets dictionary
    :param ipaddr: IPv4Address or string of an IPv4 address
    :param subnets: the dictionary of subnets with the key being the IPv4Network Object and the values being
                    a dict of network data
    :param attributes: a list of the attributes you want back, or None if you want the entire dict of data back
    :param error_bad_ip: boolean if you want an error thrown when an invalid object is provided for the ip address
                         or only getting None returned
    :return: dictionary of the attributes listed
    """
    ipaddr = check_ip(ipaddr, error_bad_ip)

    if isinstance(attributes, str):
        raise ValueError('Attributes must be a list or None for all available attributes')

    # Iterate through the provided dictionary of subnets
    for subnet_obj, subnet_data in subnets.items():
       if ipaddr in subnet_obj:
           if attributes is None:
               return subnet_data
           else:
               retval = dict()
               for attribute in attributes:
                   if attribute in subnet_data.keys():
                       retval[attribute] = subnet_data[attribute]
                   else:
                       retval[attribute] = None
               return retval
    return None
