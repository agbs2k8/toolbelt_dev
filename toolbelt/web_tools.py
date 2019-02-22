import asyncio
from bs4 import BeautifulSoup
from .async_web_requests import get_html

global loop


def get_links(html: str) -> list:
    soup = BeautifulSoup(html, 'html.parser')
    links = list(set([x['href'] for x in soup.find_all("a", href=True) if x['href'][:4]=='http']))
    return links


async def get_links_from_urls(urls: list):
    tasks = []
    links = []
    for url in urls:
        tasks.append((url, loop.create_task(get_html(url))))

    for src_page, page_html in tasks:
        html = await page_html
        page_links = get_links(html)
        links += page_links
    return links


def scrape_links_from_url(url: str) -> list:
    global loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_links_from_urls([url]))


def scrape_links_from_urls(urls: list) -> list:
    global loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_links_from_urls(urls))

