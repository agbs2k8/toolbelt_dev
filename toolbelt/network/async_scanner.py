import datetime
import numbers
import collections
import ipaddress
import socket
import asyncio


class Ports:
    """A listing of ports for a given host

    Parameters
    ----------
    ports_list : {'default', 'all', or list of ports}
        either a string {'default', 'all'} or a list of ports
        'default' = the most common ports
        'all' = all ports from 1 - 65535
    """
    def __init__(self, ports_list='default'):
        if ports_list == 'default':
            self.ports = [20,  # FTP Data Transfer
                          21,  # FTP C2
                          22,  # SSH login
                          23,  # Telnet
                          25,  # SMTP Routing
                          53,  # DNS service
                          80,  # HTTP
                          110,  # POP3
                          119,  # NNTP
                          123,  # NTP
                          139,  # File and Printer Sharing
                          143,  # IMAP
                          161,  # SNMP
                          194,  # IRC
                          443,  # HTTPS
                          445,  # Active Directory
                          1433,  # MS SQL
                          1521,  # Oracle database default listener
                          3306,  # MySQL
                          3389,  # RDP
                          5432,  # PostgreSQL
                          8080,  # Many
                          8443,  # Many
                          ]
        elif ports_list == 'all':
            self.ports = [x for x in range(1, 65536)]
        elif isinstance(ports_list, collections.Iterable):
            for item in ports_list:
                if not isinstance(item, numbers.Integral):
                    raise ValueError(f'{item} is not a valid port')
                elif item < 1 or item > 65535:
                    raise ValueError(f'{item} is not a valid port')
            self.ports = ports_list
        else:
            raise ValueError('Invalid port input')

        self._scanned = False
        self._results = {port: None for port in self.ports}

    def __iter__(self):
        for port in self.ports:
            yield port

    def __repr__(self):
        return f'<Ports object with {len(self.ports)} items.>'

    def __str__(self):
        return self.__repr__()

    def record_scan_result(self, port, active):
        if active:
            self._results[port] = 'connected'
        else:
            self._results[port] = 'refused'

    def port_status(self):
        return self._results


class Host:
    """ A host object for scanning.
    Parameters
    ----------
    ip : str or ipaddress.IPv4Address/IPv6Address
        a resolvable ip address for the host.  The resolvability will not be
        checked at instantiation

    host_name : str
        a name for the host.  If none is provided, will attempt to retrieve from
        the host at the given IP address

    ports : {'default', 'all'} or list of ports
        either a list of ports to be scanned, or one of the identifier strings.
        Default will scan a selection of very common ports.

    Attributes
    ----------
    ip : ipaddress.IPv4Address/IPv6Address
        The ip address as an instance an ipaddress class.
        Use str() to get the specific address

    host_name : str
        The name of the host

    ports : Ports class object
        The list of all ports for scanning.  Is iterable
    """
    def __init__(self, ip=None, host_name=None, ports='default'):
        self.ip = ip
        self.host_name = host_name
        self.ports = ports

    @property
    def ip(self):
        return self._ip

    @ip.setter
    def ip(self, value):
        if isinstance(value, ipaddress.IPv4Address) or isinstance(value, ipaddress.IPv6Address):
            self._ip = value
        elif value is not None:
            self._ip = ipaddress.ip_address(value)
        else:
            raise ValueError('A valid IP address is required')

    @property
    def host_name(self):
        return self._host_name

    @host_name.setter
    def host_name(self, value):
        if value is None:
            try:
                self._host_name, _, _ = socket.gethostbyaddr(str(self.ip))
            except socket.herror:
                self._host_name = None
        else:
            self._host_name = value

    @property
    def ports(self):
        return self._ports

    @ports.setter
    def ports(self, value):
        if isinstance(value, Ports):
            self._ports = value
        else:
            self._ports = Ports(value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.ip == other.ip and self.host_name == other.host_name
        else:
            raise ValueError('Not a valid Host object.')

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        else:
            raise ValueError('Not a valid Host object.')

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def scan_ports(self):
        for port in self.ports:
            self.ports.record_scan_result(port=port, active=self.check_port(str(self.ip), port))

        return self.ports.port_status()

    @staticmethod
    def check_port(ip, port):
        try:
            s = socket.socket()
            s.connect((ip, port))
            s.close()
            return True
        except ConnectionRefusedError:
            return False

    def async_scan_ports(self):
        asyncio.run(self.async_scan_all_ports())
        return self.ports.port_status()

    async def async_scan_all_ports(self):
        for port in self.ports:
            result = await self.async_check_port(str(self.ip), port)
            self.ports.record_scan_result(port=port, active=result)

    @staticmethod
    async def async_check_port(ip, port):
        try:
            _, writer = await asyncio.open_connection(ip, port)
            writer.close()
            return True
        except ConnectionRefusedError:
            return False


if __name__ == "__main__":
    print('main\n'+('*'*75))
    test_host = Host(ip='192.168.86.32')
    print(test_host.async_scan_ports())
