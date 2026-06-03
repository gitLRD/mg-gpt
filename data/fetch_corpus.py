"""
data/fetch_corpus.py — Bootstrap corpus from Wikipedia.

Fetches the plain text of key MG-related Wikipedia articles and saves them
as .txt files in data/corpus/. Uses only stdlib (urllib) — no extra deps.

Usage:
    python data/fetch_corpus.py
"""

import os
import json
import time
import urllib.request
import urllib.parse

OUTPUT_DIR = "data/corpus"

ARTICLES = [
    "MGB",
    "MG_Cars",
    "SU_carburettor",
    "British_Leyland",
    "MGA",
    "MG_Midget",
    "BMC_B_series_engine",
    "Abingdon-on-Thames",   # where MGs were built
]

API = "https://en.wikipedia.org/w/api.php"

def fetch_article(title: str) -> str:
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,       # plain text, no markup
        "exsectionformat": "plain",
        "format": "json",
    })
    url = f"{API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "mg-gpt-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


os.makedirs(OUTPUT_DIR, exist_ok=True)

total_chars = 0
for title in ARTICLES:
    slug = title.replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"wiki_{slug}.txt")
    if os.path.exists(out_path):
        print(f"  skip (exists): {out_path}")
        continue
    print(f"  fetching: {title} ...", end=" ", flush=True)
    try:
        text = fetch_article(title)
        if not text:
            print("empty — skipped")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{len(text):,} chars → {out_path}")
        total_chars += len(text)
        time.sleep(0.5)   # be polite to Wikipedia's servers
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone. Total new text: {total_chars:,} chars (~{total_chars//4:,} tokens)")
print("Run  python data/prepare.py  to tokenise.")
