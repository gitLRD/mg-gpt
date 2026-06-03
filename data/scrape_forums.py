"""
data/scrape_forums.py — Flexible forum scraper for MG enthusiast forums.

Fetches forum threads, extracts post text, saves to corpus.
Tries common forum HTML patterns (vBulletin, phpBB, Discourse, etc).

Usage:
    python data/scrape_forums.py \
        --urls "https://www.mgexp.com/forums/..." \
        --num-pages 3 \
        --delay 2.0

    python data/scrape_forums.py \
        --urls "https://example.com/thread1" "https://example.com/thread2" \
        --num-pages 1

"""

import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse
from collections import defaultdict

import requests
from lxml import html as lxml_html

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data/corpus"
MIN_POST_LENGTH = 50  # skip very short posts
DEFAULT_DELAY = 2.0   # seconds between requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# CSS selectors tried in order (different forum software uses different classes)
POST_SELECTORS = [
    "div.message-body",      # mgexp.com and similar
    "div.post",              # vBulletin, common
    "article.post",          # modern Discourse-like
    "div[class*='post']",    # contains 'post' in class
    "div.message",           # another common pattern
    "li.post",               # some forums use list items
]

CONTENT_SELECTORS = [
    "div.message-body",
    "div.post-content",
    "div.content",
    "div[class*='content']",
    "div.post-text",
    "div.message-text",
]

# Explicit XPaths — the previous "CSS-like" selectors didn't survive the
# naive split-on-dot conversion, so attribute matchers like `a[rel='next']`
# silently produced invalid XPath and matched nothing.
PAGINATION_XPATHS = [
    "//link[@rel='next']/@href",       # head <link rel="next" href="...">
    "//a[@rel='next']/@href",          # canonical "next" link
    "//a[contains(@class, 'icon-next')]/@href",  # mgexp icon-next button
    "//a[@class='next']/@href",
    "//li[@class='next']/a/@href",
    "//a[contains(@aria-label, 'Next')]/@href",
]

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class ForumScraper:
    def __init__(self, delay=DEFAULT_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.posts_scraped = 0
        self.urls_visited = set()

    def _clean_text(self, text):
        """Clean post text: remove extra whitespace, HTML entities, etc."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common forum artifacts
        text = re.sub(r'(Quote|Posted|Member|Join date|Location).*?:', '', text, flags=re.IGNORECASE)
        return text

    def _extract_posts(self, page_html, page_url):
        """Try to extract posts from a forum page."""
        posts = []
        tree = lxml_html.fromstring(page_html)

        # Try each post selector
        for post_selector in POST_SELECTORS:
            try:
                # Convert CSS-style selectors to XPath
                if '.' in post_selector and '[' not in post_selector:
                    # "div.classname" → "//div[@class='classname']"
                    parts = post_selector.split('.')
                    xpath = f"//{parts[0]}[@class='{parts[1]}']"
                else:
                    xpath = f"//{post_selector}"

                post_elements = tree.xpath(xpath)
                if post_elements:
                    print(f"  Found {len(post_elements)} posts using selector: {post_selector}")

                    for elem in post_elements:
                        # Extract text from this element
                        try:
                            content = lxml_html.tostring(elem, encoding='unicode', method='text')
                        except:
                            continue

                        if content:
                            content = self._clean_text(content)
                            if len(content) >= MIN_POST_LENGTH:
                                posts.append(content)

                    return posts
            except Exception as e:
                continue

        # Fallback: grab all paragraph text (very crude but sometimes works)
        if not posts:
            print(f"  No posts found with selectors. Trying paragraph fallback...")
            try:
                paragraphs = tree.xpath("//p")
                for p in paragraphs[:20]:  # limit to first 20
                    text = lxml_html.tostring(p, encoding='unicode', method='text')
                    text = self._clean_text(text)
                    if len(text) >= MIN_POST_LENGTH:
                        posts.append(text)
            except:
                pass

        return posts

    def _find_next_page(self, page_html, page_url):
        """Find URL of next page."""
        tree = lxml_html.fromstring(page_html)

        for xpath in PAGINATION_XPATHS:
            try:
                hrefs = tree.xpath(xpath)
                if not hrefs:
                    continue
                # Take the first href, normalising element vs str returns
                first = hrefs[0]
                next_href = first if isinstance(first, str) else first.get("href")
                if not next_href:
                    continue
                next_url = urljoin(page_url, next_href)
                # Sanity: don't loop on a self-link
                if next_url == page_url:
                    continue
                return next_url
            except Exception:
                continue

        return None

    def scrape_thread(self, thread_url, max_pages=1):
        """Scrape a single forum thread across multiple pages."""
        print(f"\n{'='*70}")
        print(f"Scraping: {thread_url}")
        print(f"Max pages: {max_pages}")
        print(f"{'='*70}")

        current_url = thread_url
        page_num = 0
        thread_posts = []

        while page_num < max_pages and current_url:
            if current_url in self.urls_visited:
                print(f"Already visited {current_url}, skipping.")
                break

            self.urls_visited.add(current_url)
            page_num += 1

            print(f"\n[Page {page_num}] Fetching: {current_url}")
            try:
                resp = self.session.get(current_url, timeout=10)
                resp.raise_for_status()
            except Exception as e:
                print(f"  ❌ Error fetching: {e}")
                break

            # Extract posts
            posts = self._extract_posts(resp.text, current_url)
            print(f"  ✅ Extracted {len(posts)} posts")
            thread_posts.extend(posts)
            self.posts_scraped += len(posts)

            # Find next page
            current_url = self._find_next_page(resp.text, current_url)

            if current_url:
                print(f"  Found next page: {current_url}")
                time.sleep(self.delay)
            else:
                print(f"  No next page found.")

        return thread_posts

    def save_posts(self, posts, thread_id=None):
        """Save posts to a file in the corpus."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not posts:
            print(f"\nNo posts to save.")
            return

        # Use thread_id or timestamp
        if thread_id is None:
            thread_id = int(time.time())

        filename = f"forum_{thread_id}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for post in posts:
                f.write(post + "\n\n")

        print(f"\n💾 Saved {len(posts)} posts to {filepath}")
        return filepath

    def scrape_and_save(self, thread_urls, max_pages=1):
        """Scrape multiple threads and save."""
        all_posts = []

        for i, url in enumerate(thread_urls):
            posts = self.scrape_thread(url, max_pages=max_pages)
            all_posts.extend(posts)

            # Save after each thread
            thread_id = f"thread_{i+1}_{int(time.time())}"
            self.save_posts(posts, thread_id=thread_id)

            if i < len(thread_urls) - 1:
                time.sleep(self.delay)

        print(f"\n{'='*70}")
        print(f"Total posts scraped: {self.posts_scraped}")
        print(f"Total files saved: {len(thread_urls)}")
        print(f"{'='*70}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape MG forum threads.")
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Forum thread URLs to scrape"
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=1,
        help="Number of pages to scrape per thread (default: 1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})"
    )
    args = parser.parse_args()

    scraper = ForumScraper(delay=args.delay)
    scraper.scrape_and_save(args.urls, max_pages=args.num_pages)

if __name__ == "__main__":
    main()
