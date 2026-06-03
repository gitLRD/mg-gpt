"""
data/discover_and_scrape.py — Walk forum index pages, discover new threads,
and scrape any not already in data/corpus/.

Run from project root:
    python data/discover_and_scrape.py --max-index-pages 5 --max-thread-pages 20
"""

import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse

import requests
from lxml import html as lxml_html

from scrape_forums import ForumScraper, USER_AGENT
from bulk_scrape import url_slug

# Forum index (listing) pages to walk
FORUM_INDEXES = [
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/",
    "https://www.mgexp.com/forum/mga-forum.2/",
    "https://www.mgexp.com/forum/mg-performance-forum.83/",
]

# Thread URL pattern on mgexp: /forum/<forum-slug>.<N>/<thread-slug>.<NUMBER>/
# The thread-slug part is hyphenated and has no dots — that distinguishes the
# canonical thread URL from post-deeplinks like .../slug.12345.67890/.
THREAD_URL_REGEX = re.compile(r"^/forum/[^/]+\.\d+/[^/.]+\.\d+/?$")

# Pagination on index pages — same shape as thread pagination
NEXT_PAGE_XPATHS = [
    "//link[@rel='next']/@href",
    "//a[@rel='next']/@href",
    "//a[contains(@class, 'icon-next')]/@href",
    "//a[@class='next']/@href",
    "//li[@class='next']/a/@href",
]


def discover_threads_on_page(page_html, base_url):
    """Extract all thread URLs from a forum index page."""
    tree = lxml_html.fromstring(page_html)
    found = set()
    for href in tree.xpath("//a/@href"):
        if not isinstance(href, str):
            continue
        # Normalise to a path so the regex matches whether it's relative or absolute
        path = urlparse(href).path if "://" in href else href
        if THREAD_URL_REGEX.match(path):
            full = urljoin(base_url, href).split("?")[0].split("#")[0]
            if not full.endswith("/"):
                full += "/"
            found.add(full)
    return found


def find_next_index_page(page_html, current_url):
    tree = lxml_html.fromstring(page_html)
    for xpath in NEXT_PAGE_XPATHS:
        hrefs = tree.xpath(xpath)
        if not hrefs:
            continue
        first = hrefs[0]
        href = first if isinstance(first, str) else first.get("href")
        if not href:
            continue
        nxt = urljoin(current_url, href)
        if nxt != current_url:
            return nxt
    return None


def discover_threads(index_url, max_pages, delay):
    """Walk index pages, collecting all thread URLs seen."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_urls = set()
    current = index_url
    for page in range(max_pages):
        print(f"  [index page {page+1}] {current}")
        try:
            resp = session.get(current, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"    ❌ {e}")
            break

        new_on_page = discover_threads_on_page(resp.text, current)
        new_count = len(new_on_page - all_urls)
        all_urls.update(new_on_page)
        print(f"    found {len(new_on_page)} thread links ({new_count} new, total {len(all_urls)})")

        nxt = find_next_index_page(resp.text, current)
        if not nxt:
            print(f"    no next page link found, stopping")
            break
        current = nxt
        time.sleep(delay)

    return all_urls


def existing_corpus_slugs(corpus_dir="data/corpus"):
    """Slugs of threads already scraped — skip these."""
    slugs = set()
    if not os.path.isdir(corpus_dir):
        return slugs
    for fn in os.listdir(corpus_dir):
        if fn.startswith("forum_") and fn.endswith(".txt"):
            slugs.add(fn[len("forum_"):-len(".txt")])
    return slugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index-pages", type=int, default=5,
                        help="Index pages to walk per forum (default: 5)")
    parser.add_argument("--max-thread-pages", type=int, default=20,
                        help="Pages to scrape per thread (default: 20)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    args = parser.parse_args()

    existing = existing_corpus_slugs()
    print(f"Existing corpus files: {len(existing)} (will skip these slugs)\n")

    # Phase 1: discover
    discovered = set()
    for idx in FORUM_INDEXES:
        print(f"\nDiscovering: {idx}")
        urls = discover_threads(idx, max_pages=args.max_index_pages, delay=args.delay)
        discovered.update(urls)

    new_urls = sorted(u for u in discovered if url_slug(u) not in existing)

    print(f"\n{'='*70}")
    print(f"Discovery summary")
    print(f"  total threads found: {len(discovered)}")
    print(f"  already in corpus:   {len(discovered) - len(new_urls)}")
    print(f"  NEW to scrape:       {len(new_urls)}")
    print(f"{'='*70}\n")

    if not new_urls:
        print("Nothing new to scrape.")
        return

    # Phase 2: scrape
    scraper = ForumScraper(delay=args.delay)
    saved = 0
    for i, url in enumerate(new_urls):
        slug = url_slug(url)
        print(f"\n[{i+1}/{len(new_urls)}] {slug}")
        try:
            posts = scraper.scrape_thread(url, max_pages=args.max_thread_pages)
            if posts:
                scraper.save_posts(posts, thread_id=slug)
                saved += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        if i < len(new_urls) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*70}")
    print(f"✅ Done. {scraper.posts_scraped} posts across {saved} new threads.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
