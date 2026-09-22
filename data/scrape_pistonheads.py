"""
data/scrape_pistonheads.py — PistonHeads /gassing/ forum scraper.

Used by data/discover_and_scrape_pistonheads.py. Can also be run directly to
scrape a single topic by URL.

PH structure:
    Forum index:  forum.asp?h=0&f=<forum_id>            (&i=<offset> for page)
    Topic page:   topic.asp?h=0&f=<forum_id>&t=<topic>  (&i=<offset> for page)
    Posts per page: 20

robots.txt compliance: every fetch is gated through urllib.robotparser
against https://www.pistonheads.com/robots.txt. Disallowed URLs are skipped.
"""

import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser

import requests
from lxml import html as lxml_html

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_URL          = "https://www.pistonheads.com"
ROBOTS_URL        = f"{BASE_URL}/robots.txt"
OUTPUT_DIR        = "data/corpus_pistonheads"
MIN_POST_LENGTH   = 50    # skip very short posts
DEFAULT_DELAY     = 2.0   # PH doesn't set a Crawl-delay; be polite
USER_AGENT        = "MG-GPT/1.0 (+https://mg-gpt.desforgeslabs.com)"

# Post bodies live in <div class="phml msg-body" id="messageN">.
POST_BODY_XPATH = "//div[contains(@class,'phml') and contains(@class,'msg-body')]"

# Per-thread next-page link: the in-thread pager uses &i=<offset> with 20-post
# steps. We detect "more pages" by counting links of the form ...&i=NN where NN
# is the next offset after the current one.
PAGE_OFFSET_STEP = 20

# ---------------------------------------------------------------------------
# Robots-aware fetcher
# ---------------------------------------------------------------------------

class PistonHeadsScraper:
    def __init__(self, delay=DEFAULT_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        # urllib.robotparser handles wildcards, anchors and ordering for us.
        self.rp = RobotFileParser()
        self.rp.set_url(ROBOTS_URL)
        try:
            self.rp.read()
        except Exception as e:
            print(f"⚠️  Could not read robots.txt ({e}); aborting for safety.")
            raise

        self.posts_scraped = 0
        self.urls_visited = set()

    # -- robots.txt gate --
    def can_fetch(self, url):
        return self.rp.can_fetch(USER_AGENT, url)

    def _get(self, url):
        if not self.can_fetch(url):
            print(f"  🚫 robots.txt forbids: {url}")
            return None
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"  ❌ {url}: {e}")
            return None

    # -- post extraction --
    def _extract_posts(self, page_html):
        tree = lxml_html.fromstring(page_html)
        posts = []
        for elem in tree.xpath(POST_BODY_XPATH):
            posts_text = self._elem_to_text(elem)
            if len(posts_text) >= MIN_POST_LENGTH:
                posts.append(posts_text)
        return posts

    @staticmethod
    def _elem_to_text(elem):
        """Flatten an HTML subtree to text, preserving inter-element whitespace.

        lxml's `method='text'` and `text_content()` both concatenate adjacent
        text nodes with no separator, producing word-junctions like
        'block?Head' across paragraph boundaries. Pad block-level elements
        with spaces before flattening so word boundaries survive.
        """
        BLOCK_TAGS = ("br", "p", "div", "li", "tr", "td", "h1", "h2", "h3",
                      "h4", "h5", "h6", "blockquote")
        for tag in BLOCK_TAGS:
            for el in elem.iter(tag):
                if el.text:
                    el.text = " " + el.text
                el.tail = " " + (el.tail or "")
        text = elem.text_content()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -- topic scrape (handles in-thread pagination) --
    def scrape_topic(self, topic_url, max_pages=20):
        """Scrape a single PH topic across its in-thread pages."""
        # Normalise to the i=0 canonical form so we know the page offset
        parsed = urlparse(topic_url)
        qs = parse_qs(parsed.query)
        t = qs.get("t", [None])[0]
        f = qs.get("f", [None])[0]
        if not t:
            print(f"  ⚠️  No t= in URL: {topic_url}")
            return []
        if not f:
            print(f"  ⚠️  No f= in URL: {topic_url}")
            return []

        thread_posts = []
        for page_num in range(max_pages):
            offset = page_num * PAGE_OFFSET_STEP
            page_url = f"{BASE_URL}/gassing/topic.asp?h=0&f={f}&t={t}&i={offset}"

            if page_url in self.urls_visited:
                break
            self.urls_visited.add(page_url)

            print(f"  [page {page_num+1}, i={offset}] {page_url}")
            page_html = self._get(page_url)
            if page_html is None:
                break

            posts = self._extract_posts(page_html)
            if not posts:
                # First page with no posts → topic doesn't exist or is empty
                # Later page with no posts → we ran off the end
                print(f"    no posts on this page, stopping")
                break

            thread_posts.extend(posts)
            self.posts_scraped += len(posts)
            print(f"    extracted {len(posts)} posts (running total {len(thread_posts)})")

            # Heuristic: if a page returns fewer than the page size (20), it's
            # the last page. Saves a redundant request.
            if len(posts) < PAGE_OFFSET_STEP:
                break

            time.sleep(self.delay)

        return thread_posts

    # -- save --
    def save_posts(self, posts, topic_id):
        if not posts:
            return None
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"ph_{topic_id}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for post in posts:
                f.write(post + "\n\n")
        print(f"  💾 {len(posts)} posts → {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape a single PistonHeads topic.")
    parser.add_argument("--url", required=True, help="Topic URL")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = parser.parse_args()

    scraper = PistonHeadsScraper(delay=args.delay)
    parsed = urlparse(args.url)
    qs = parse_qs(parsed.query)
    topic_id = qs.get("t", ["unknown"])[0]
    posts = scraper.scrape_topic(args.url, max_pages=args.max_pages)
    scraper.save_posts(posts, topic_id)
    print(f"\nDone. {scraper.posts_scraped} posts.")


if __name__ == "__main__":
    main()
