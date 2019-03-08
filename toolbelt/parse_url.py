import re
import pickle

supported_schemes = ['http', 'https', 'ftp', 'file']

country_domains = pickle.load(open('country_domains.pickle', 'rb'))
top_level_domains = pickle.load(open('tld.pickle', 'rb'))
tld_parts = list(country_domains.keys()) + list(top_level_domains.keys())


class URL:
    # TODO: auth parsing
    scheme = None
    host = None
    port = None
    path = None
    query = None
    query_delim = None
    fragment = None
    host_type = None

    # Domain sub-parts
    www_domain = False
    top_level_domain = None
    primary_domain = None
    sub_domains = None

    def __init__(self, passed_url):
        url = passed_url
        url_parts = url.split('://')

        # Extract Scheme from the URL
        if len(url_parts) == 2 and url_parts[0].lower() in supported_schemes:
            self.scheme = url_parts.pop(0).lower()
            url = url_parts.pop()
        elif len(url_parts) == 1:
            url = url_parts.pop()
        else:
            raise ValueError('Unsupported Scheme')

        # Check for fragment
        url_parts = url.split('#')
        if len(url_parts) == 2:
            self.fragment = url_parts.pop()
            url = url_parts.pop()

        # Separate query from end of the remaining url
        url_parts = url.split('?')
        if len(url_parts) == 2:
            url = url_parts.pop(0)
            query_string = url_parts.pop(0)
            params = {}
            if len(query_string.split('&')) > len(query_string.split(';')):
                self.query_delim = '&'
            else:
                self.query_delim = ';'
            for param in query_string.split(self.query_delim):
                key, value = param.split('=')
                params[key] = value
            self.query = params
        elif len(url_parts) > 2:
            raise ValueError('Error in query')

        # Separate host from path
        url_parts = url.split('/')
        if len(url_parts) > 1:
            self.host = url_parts.pop(0).lower()
            self.path = '/'.join([x.lower() for x in url_parts])
        else:
            self.host = url.lower()

        # Identify any port with the host
        url_parts = self.host.split(':')
        if len(url_parts) == 2:
            self.port = url_parts.pop()
            self.host = url_parts.pop()

        # Is the host an IP address
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        if re.match(ip_pattern, self.host):
            ip_parts = self.host.split('.')
            for part in ip_parts:
                if int(part) < 0 or int(part) > 255:
                    raise ValueError('Invalid IP Address')
            self.host_type = "IP"
        elif self.host == 'localhost':
            self.host_type = "LOCAL"
        else:
            self.host_type = 'URL'

        # Dissect the host to extract the sub domain and
        if self.host_type == 'URL':
            url_parts = self.host.split('.')
            # TODO: Add validation for length & structure
            if url_parts[0] == 'www':
                self.www_domain = True
                url_parts.pop(0)

            tld = []
            is_tld = True
            while is_tld:
                if len(url_parts) > 1 and url_parts[-1] in tld_parts:
                    tld.insert(0, url_parts.pop())
                else:
                    is_tld = False

            self.top_level_domain = '.'.join(tld)
            self.primary_domain = url_parts.pop()
            if len(url_parts) > 0:
                self.sub_domains = '.'.join(url_parts)

    def __repr__(self):
        return f"<type: {self.host_type}; scheme:{self.scheme} host:{self.host} path:{self.path} " \
            f"query:{str(self.query)} port:{self.port}; is_www:{self.www_domain}" \
            f" TLD:{self.top_level_domain} domain:{self.primary_domain} sub_domain:{self.sub_domains}>"

    def __str__(self):
        full_url = ''
        if self.scheme:
            full_url += (self.scheme + '://')
        full_url += self.host
        if self.port:
            full_url += (':' + self.port)
        if self.path:
            full_url += ('/' + self.path)
        if self.query:
            full_url += ('?' + self.query_delim.join([str(x)+'='+str(y) for x, y in self.query.items()]))
        return full_url


if __name__ == '__main__':
    test_urls = ['https://mail.google.com/messages/find?includes=southwest', 'https://www.goOgle.com/',]
    for test_url in test_urls:
        myURL = URL(test_url)
        print(myURL, repr(myURL))
        print()
