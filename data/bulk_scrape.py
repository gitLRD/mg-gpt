"""
data/bulk_scrape.py — Bulk scraper for multiple MG forum threads.

Scrapes all threads in batches with rate limiting to be respectful to mgexp.com.

Usage:
    python data/bulk_scrape.py

Modify THREAD_URLS below to add/remove threads.
"""

import os
import re
import time
import argparse
from urllib.parse import urlparse
from scrape_forums import ForumScraper

# ---------------------------------------------------------------------------
# Thread URLs by category
# ---------------------------------------------------------------------------

MGB_GT_URLS = [
    # General discussion
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/what-did-you-do-with-your-mgb-today.1940738/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/my-budget-restoration-project.5035157/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/just-brought-home-our-first-classic-car.5051919/",

    # Technical
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/1979-mgb-with-twin-sus-do-tapered-air.5053231/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/78-b-gearbox-and-radiator-questions.5053221/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/brake-line-nut-flare-confusion.5053447/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/oil-pressure-gauge-or.5052452/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/how-much-to-mill-off-head.5049119/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/rerouting-my-speedo-cable-for-an-od-cable.5053173/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/strap-installation-for-a-soft-top.5053280/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/source-for-a-good-distributor-rotor.5053286/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/smiths-mushroom-pcv-reinstall-queation.5053369/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/a-reminder-from-mossmotors.5053281/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/another-reason-i-try-not-to-use-moss.5053353/",
    "https://www.mgexp.com/forum/mgb-and-gt-forum.1/mgoc-article-mg-rv8-brake-issues.5053487/",
]

MIDGET_URLS = [
    # General discussion
    "https://www.mgexp.com/forum/mg-midget-forum.3/what-did-you-do-with-your-mg-midget-today.2289300/",

    # Technical
    "https://www.mgexp.com/forum/mg-midget-forum.3/new-coil-and-distributor-no-start.5053563/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/prepping-for-an-engine-overhaul.5053472/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/what-am-i-getting-myself-into.5053094/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/making-the-clutch-pedal-adjustable.5052172/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/76-1500-engine-out-project-managent-advice.5050935/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/76-1500-transmission-cradle-mount.5053551/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/weird-ignition-things-going-on.5053505/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/help-with-bespoke-throw-out-bearing.5053368/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/axle-assembly-thread-stripped.5053274/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/foam-in-pcv.5053288/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/heritage-front-sheet-metal.5050640/",
    "https://www.mgexp.com/forum/mg-midget-forum.3/mg-midget-1500-leaf-spring-plate-bolts.5049807/",
]

MGA_URLS = [
    # General discussion
    "https://www.mgexp.com/forum/mga-forum.2/what-did-you-do-with-your-mga-today.2183776/",

    # Restoration projects
    "https://www.mgexp.com/forum/mga-forum.2/1957-mga-roadster-restoration-with-an.3898051/",
    "https://www.mgexp.com/forum/mga-forum.2/1959-mga-roadster-roadster-restoration.4966012/",
    "https://www.mgexp.com/forum/mga-forum.2/mga-restoration-making-progress.5047722/",
    "https://www.mgexp.com/forum/mga-forum.2/grubeguys-1957-mga-coupe-resto-thread.4769120/",
    "https://www.mgexp.com/forum/mga-forum.2/mga-1958-restoration-project-new-story.4814431/",
    "https://www.mgexp.com/forum/mga-forum.2/restoring-my-vanden-plas-hardtop-finally.5042841/",
    "https://www.mgexp.com/forum/mga-forum.2/new-project-the-inheritance.5052511/",

    # Technical
    "https://www.mgexp.com/forum/mga-forum.2/mga-rear-hub-oil-leaks-and-gasket-sealant.4614116/",
    "https://www.mgexp.com/forum/mga-forum.2/clunking-noise-at-right-rear.5053493/",
    "https://www.mgexp.com/forum/mga-forum.2/flexible-brake-hose-r-and-r.5052052/",
    "https://www.mgexp.com/forum/mga-forum.2/heavy-clutch.5051333/",
    "https://www.mgexp.com/forum/mga-forum.2/high-compression.5052956/",
]

PERFORMANCE_URLS = [
    # Engine swaps and mods (from MG Performance forum if available, or related threads)
    "https://www.mgexp.com/forum/mg-performance-forum.83/",
]

# Combine all
ALL_URLS = MGB_GT_URLS + MIDGET_URLS + MGA_URLS + PERFORMANCE_URLS

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def url_slug(url):
    """Derive a filesystem-safe, stable ID from a thread URL.

    Example: .../1979-mgb-with-twin-sus-do-tapered-air.5053231/
             → '1979-mgb-with-twin-sus-do-tapered-air-5053231'
    """
    path = urlparse(url).path.rstrip("/")
    last = path.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "-", last.lower()).strip("-") or "thread"


def main():
    parser = argparse.ArgumentParser(description="Bulk scrape MG forum threads.")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Max pages per thread (default: 20)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BULK SCRAPER: MG Forum Threads")
    print(f"{'='*70}")
    print(f"Total threads to scrape: {len(ALL_URLS)}")
    print(f"  MGB & GT: {len(MGB_GT_URLS)}")
    print(f"  Midget: {len(MIDGET_URLS)}")
    print(f"  MGA: {len(MGA_URLS)}")
    print(f"Max pages per thread: {args.max_pages}")
    print(f"{'='*70}\n")

    scraper = ForumScraper(delay=args.delay)

    saved = 0
    for i, url in enumerate(ALL_URLS):
        slug = url_slug(url)
        print(f"\n[{i+1}/{len(ALL_URLS)}] {slug}")

        try:
            posts = scraper.scrape_thread(url, max_pages=args.max_pages)
            if posts:
                # Stable thread_id so re-runs overwrite the same file rather
                # than producing dupes with new timestamps.
                scraper.save_posts(posts, thread_id=slug)
                saved += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        # Rate limiting between threads
        if i < len(ALL_URLS) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*70}")
    print(f"✅ Bulk scraping complete!")
    print(f"Total posts scraped: {scraper.posts_scraped}")
    print(f"Total files saved: {saved}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
