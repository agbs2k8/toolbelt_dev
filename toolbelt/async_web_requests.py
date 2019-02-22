import aiohttp

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}


async def get_html(url: str)-> str:
    # These nested with blocks are required for aiohttp
    # Called Controllers - first the session
    async with aiohttp.ClientSession(headers=hdr) as session:

        # then one for the actual request
        async with session.get(url) as r:

            # this makes sure we have a success code (no 400, etc)
            r.raise_for_status()

            # read the content and give it back
            return await r.text()
