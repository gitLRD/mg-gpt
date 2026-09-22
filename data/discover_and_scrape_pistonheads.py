"""
data/discover_and_scrape_pistonheads.py — Walk PistonHeads /gassing/ MG +
Rover sub-forum indexes, discover topics, and scrape any not already in
data/corpus_pistonheads/.

Run from project root:
    python data/discover_and_scrape_pistonheads.py \
        --max-index-pages 20 --max-thread-pages 10

robots.txt compliance is enforced by the scraper itself — every URL is gated
through urllib.robotparser before fetching.
"""

import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse, parse_qs

from scrape_pistonheads import (
    PistonHeadsScraper, BASE_URL, OUTPUT_DIR,
    PAGE_OFFSET_STEP, USER_AGENT,
)
import requests
from lxml import html as lxml_html

# ---------------------------------------------------------------------------
# Target sub-forums
# ---------------------------------------------------------------------------

# f=81 MG — the dedicated MG sub-forum (primary signal)
# f=54 Rover — covers MG Rover era (ZR/ZS/ZT badge-engineered as MG)
SUB_FORUMS = [81, 54]

# Canonical topic URL shape: topic.asp?h=0&f=<N>&t=<N>  (relative form
# as it appears in PH forum-index hrefs). Reject deeplinks with &i=
# (in-thread post offset) and &r= (post anchor) — they're duplicates.
TOPIC_URL_REGEX = re.compile(
    r"(?:^|/)topic\.asp\?h=0&f=\d+&t=\d+$"
)


# ---------------------------------------------------------------------------
# Forum index discovery
# ---------------------------------------------------------------------------

def discover_topics_on_page(page_html):
    """Pull canonical topic URLs out of one forum-index page."""
    tree = lxml_html.fromstring(page_html)
    found = set()
    for href in tree.xpath("//a/@href"):
        if not isinstance(href, str):
            continue
        parsed = urlparse(href)
        path_q = parsed.path + ("?" + parsed.query if parsed.query else "")
        if TOPIC_URL_REGEX.search(path_q):
            # Normalise to absolute, drop fragments
            full = urljoin(BASE_URL + "/gassing/", href).split("#")[0]
            found.add(full)
    return found


def walk_index(scraper, forum_id, max_pages, delay):
    """Walk a PH sub-forum's index pages, collecting topic URLs."""
    all_urls = set()
    for page in range(max_pages):
        offset = page * PAGE_OFFSET_STEP
        if offset == 0:
            page_url = f"{BASE_URL}/gassing/forum.asp?h=0&f={forum_id}"
        else:
            page_url = f"{BASE_URL}/gassing/forum.asp?h=0&f={forum_id}&i={offset}"

        print(f"  [index page {page+1}, i={offset}] {page_url}")
        page_html = scraper._get(page_url)
        if page_html is None:
            print(f"    fetch failed, stopping")
            break

        new = discover_topics_on_page(page_html)
        added = len(new - all_urls)
        all_urls.update(new)
        print(f"    {len(new)} topic links ({added} new, total {len(all_urls)})")

        if added == 0:
            # No new content — likely past the end of populated pages
            print(f"    no new topics on this page, stopping")
            break

        time.sleep(delay)
    return all_urls


# ---------------------------------------------------------------------------
# Skip already-scraped
# ---------------------------------------------------------------------------

def existing_topic_ids(corpus_dir=OUTPUT_DIR):
    """Topic IDs of threads already in data/corpus_pistonheads/."""
    ids = set()
    if not os.path.isdir(corpus_dir):
        return ids
    for fn in os.listdir(corpus_dir):
        if fn.startswith("ph_") and fn.endswith(".txt"):
            ids.add(fn[len("ph_"):-len(".txt")])
    return ids


def topic_id_from_url(url):
    qs = parse_qs(urlparse(url).query)
    return qs.get("t", [None])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index-pages", type=int, default=20,
                        help="Index pages to walk per sub-forum (default: 20)")
    parser.add_argument("--max-thread-pages", type=int, default=10,
                        help="Pages to scrape per topic (default: 10)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Hard cap on topics scraped this run")
    args = parser.parse_args()

    scraper = PistonHeadsScraper(delay=args.delay)
    existing = existing_topic_ids()
    print(f"Existing PH corpus: {len(existing)} topics — will skip these IDs.\n")

    # Phase 1: discover
    discovered = set()
    for fid in SUB_FORUMS:
        print(f"\n== Discovering f={fid} ==")
        urls = walk_index(scraper, fid, args.max_index_pages, args.delay)
        discovered.update(urls)

    # Prioritise f=81 (dedicated MG sub-forum) over f=54 (Rover) so any
    # interruption still leaves us with the densest MG signal. Within each
    # forum, sort by URL for deterministic resume ordering.
    def sort_key(u):
        f = parse_qs(urlparse(u).query).get("f", ["0"])[0]
        forum_priority = SUB_FORUMS.index(int(f)) if int(f) in SUB_FORUMS else 999
        return (forum_priority, u)

    new_urls = sorted(
        (u for u in discovered
         if topic_id_from_url(u) and topic_id_from_url(u) not in existing),
        key=sort_key,
    )

    print(f"\n{'='*70}")
    print(f"Discovery summary")
    print(f"  total topics found:    {len(discovered)}")
    print(f"  already in corpus:     {len(discovered) - len(new_urls)}")
    print(f"  NEW to scrape:         {len(new_urls)}")
    if args.limit:
        new_urls = new_urls[: args.limit]
        print(f"  capped by --limit to:  {len(new_urls)}")
    print(f"{'='*70}\n")

    if not new_urls:
        print("Nothing new to scrape.")
        return

    # Phase 2: scrape
    saved = 0
    for i, url in enumerate(new_urls):
        tid = topic_id_from_url(url)
        print(f"\n[{i+1}/{len(new_urls)}] topic {tid}")
        try:
            posts = scraper.scrape_topic(url, max_pages=args.max_thread_pages)
            if scraper.save_posts(posts, tid):
                saved += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        if i < len(new_urls) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*70}")
    print(f"✅ Done. {scraper.posts_scraped} posts across {saved} topics.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
