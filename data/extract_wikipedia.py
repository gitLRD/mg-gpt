"""
data/extract_wikipedia.py — Extract MG-related articles from a Wikipedia XML dump.

Streams through the full dump using a multiprocessing producer-consumer pipeline:
  - Main process reads and parses XML (I/O bound)
  - Worker processes strip markup and score articles (CPU bound)

Usage:
    python data/extract_wikipedia.py
"""

import os
import re
import multiprocessing as mp
from lxml import etree as ET

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DUMP_PATH = (
    "/Users/lewisdesforges/Downloads/"
    "enwiki-20260401-pages-articles-multistream.xml/"
    "enwiki-20260401-pages-articles-multistream.xml"
)
OUTPUT_DIR   = "data/corpus"
MIN_SCORE    = 15    # minimum keyword hits to keep an article
MAX_ARTICLES = 500   # safety cap

KEYWORDS = [
    "MGB", "MGA", "MGC", "MG Midget", "MG TF", "MG TD", "MG TC",
    "SU carburettor", "SU carburetor",
    "British Leyland", "British Motor Corporation", "BMC",
    "Abingdon", "Austin-Healey",
]

# At least one of these unambiguous multi-word car terms must appear.
# This blocks: Philippine articles (mga = Tagalog), Soviet MGB, Swiss railway MGB,
# cycling BMC, Mumbai BMC, British Mountaineering Council, etc.
CAR_KEYWORDS = [
    "MG Midget", "MG TF", "MG TD", "MG TC",
    "SU carburettor", "SU carburetor",
    "Austin-Healey",
]

NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# ---------------------------------------------------------------------------
# Markup stripping
# ---------------------------------------------------------------------------

def strip_markup(text):
    # 1. Remove templates — repeat to handle nesting
    for _ in range(5):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)

    # 2. Remove image/file links entirely
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # 3. Keep display text from wikilinks: [[British Leyland|BMC]] → BMC
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)

    # 4. Keep plain wikilinks: [[SU carburettor]] → SU carburettor
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)

    # 5. Remove footnotes
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)

    # 6. Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 7. Clean up bold/italic markers
    text = re.sub(r"'{2,3}", '', text)

    # 8. Clean up section headings: ==History== → History
    text = re.sub(r'={2,6}\s*(.*?)\s*={2,6}', r'\1', text)

    # 9. Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_PATTERNS = None

def _build_patterns():
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = [
            (kw, re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE))
            for kw in KEYWORDS
        ]

def score_article(text):
    _build_patterns()
    score = 0
    matched = []

    for keyword, pattern in _PATTERNS:
        count = len(pattern.findall(text))
        if count > 0:
            score += count
            matched.append(f"{keyword}×{count}")

    return score, matched

_CAR_PATTERNS = None

def _build_car_patterns():
    global _CAR_PATTERNS
    if _CAR_PATTERNS is None:
        _CAR_PATTERNS = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in CAR_KEYWORDS
        ]

def has_car_keyword(text):
    _build_car_patterns()
    return any(p.search(text) for p in _CAR_PATTERNS)

# ---------------------------------------------------------------------------
# Worker (runs in a separate process for each CPU core)
# ---------------------------------------------------------------------------

def _worker(work_q, result_q, output_dir, min_score):
    """
    Pull (title, raw_text) from work_q.
    Strip markup, score, write file if above threshold, report to result_q.
    Exit when we receive a None sentinel.
    """
    for item in iter(work_q.get, None):
        title, text = item
        clean = strip_markup(text)
        score, matched = score_article(clean)

        if score >= min_score and has_car_keyword(clean):
            safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            path = os.path.join(output_dir, f"wiki_{safe_name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n\n{clean}")
            result_q.put((title, score, matched))

    result_q.put(None)  # tell main process this worker is finished

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_workers = max(1, mp.cpu_count() - 1)
    # Bounded queue: reader pauses if workers fall behind (prevents RAM blowout)
    work_q   = mp.Queue(maxsize=n_workers * 8)
    result_q = mp.Queue()

    print(f"Starting {n_workers} worker processes...")
    workers = [
        mp.Process(target=_worker, args=(work_q, result_q, OUTPUT_DIR, MIN_SCORE), daemon=True)
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    # --- Reader: stream XML in main process ---
    scanned = 0
    print(f"Scanning: {DUMP_PATH}")
    print(f"Threshold: {MIN_SCORE} keyword hits | Cap: {MAX_ARTICLES} articles\n")

    # tag= means lxml only fires events for <page> elements — child elements
    # (title, ns, text) are never cleared prematurely, so findtext() works correctly.
    for event, elem in ET.iterparse(DUMP_PATH, events=["end"], tag=f"{NS}page"):

        ns    = elem.findtext(f"{NS}ns")
        title = elem.findtext(f"{NS}title")
        text  = elem.findtext(f".//{NS}text") or ""

        # Free memory: clear page element and drop it from the root's child list
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if ns != "0":
            continue

        scanned += 1
        if scanned % 100_000 == 0:
            print(f"  ...scanned {scanned:,} articles")

        work_q.put((title, text))  # blocks if workers are behind — that's intentional

    # Send one None sentinel per worker to shut them down cleanly
    for _ in workers:
        work_q.put(None)

    # --- Drain results until all workers report done ---
    kept = 0
    done = 0
    while done < n_workers:
        item = result_q.get()
        if item is None:
            done += 1
            continue
        title, score, matched = item
        kept += 1
        print(f"  ✅ {title} (score {score}): {', '.join(matched)}")
        if kept >= MAX_ARTICLES:
            print(f"\nReached MAX_ARTICLES cap ({MAX_ARTICLES}).")
            break

    for w in workers:
        w.join()

    print(f"\nDone. Scanned {scanned:,} articles, kept {kept}.")
    print(f"Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
