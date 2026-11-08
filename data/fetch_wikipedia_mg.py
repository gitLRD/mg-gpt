"""
fetch_wikipedia_mg.py — pull MG-related English Wikipedia articles as plain text.

Licence: Wikipedia text is CC BY-SA 4.0 (and mostly GFDL). We keep an attribution
manifest (title, URL, revision id) so the corpus can be credited. Wikimedia's
terms ask bulk users not to crawl, so this fetches a few hundred named pages
only: serial requests, ~1s apart, `maxlag`, gzip, and a descriptive User-Agent
with a contact address (https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy).

Usage:
    python data/fetch_wikipedia_mg.py --out-dir data/corpus_wikipedia_mg
"""

import argparse
import gzip
import json
import os
import re
import time
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
UA = "MGGPTCorpusBot/0.1 (mailto:lewis.desforges@gmail.com; personal learning project) urllib"
DELAY = 1.2

CATEGORIES = [
    "MG vehicles",
    "MG Motor",
    "MG racing models",
    "MG experimental and prototype models",
]

# Hand-picked context an MG owner's questions touch: the company, the engines,
# the carburettors, the people and places.
EXTRA_TITLES = [
    "MG Cars", "Morris Garages", "Cecil Kimber", "MG Car Club", "Abingdon-on-Thames",
    "MG Rover Group", "MG Motor UK", "Longbridge plant", "British Leyland", "British Motor Corporation",
    "SU carburettor", "BMC B-series engine", "BMC A-series engine", "Rover K-series engine",
    "Rover V8 engine", "Lucas Industries", "Triumph Motor Company", "Austin-Healey", "Morris Minor",
]

DROP_SECTIONS = {"see also", "references", "external links", "notes", "further reading",
                 "sources", "bibliography", "footnotes", "citations"}


def get(params):
    params = dict(params, format="json", maxlag=5, redirects=1)
    req = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
    )
    for attempt in range(5):
        time.sleep(DELAY)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001 — back off and retry on any transient error
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * 2 ** attempt)
            continue
        if data.get("error", {}).get("code") in ("maxlag", "ratelimited"):
            print(f"  server busy ({data['error']['code']}), backing off")
            time.sleep(10 * 2 ** attempt)
            continue
        return data
    raise RuntimeError("giving up after repeated errors")


def category_titles(cat, depth=1):
    titles, subcats = [], []
    cont = {}
    while True:
        d = get(dict(action="query", list="categorymembers", cmtitle="Category:" + cat,
                     cmlimit=500, cmtype="page|subcat", **cont))
        for m in d["query"]["categorymembers"]:
            (subcats if m["ns"] == 14 else titles).append(m["title"])
        if "continue" not in d:
            break
        cont = d["continue"]
    if depth > 0:
        for sc in subcats:
            titles += category_titles(sc.removeprefix("Category:"), depth - 1)
    return titles


def clean(text):
    out, skip = [], False
    for line in text.splitlines():
        m = re.match(r"^=+\s*(.*?)\s*=+$", line)
        if m:
            skip = m.group(1).lower() in DROP_SECTIONS
            if skip:
                continue
        if not skip:
            out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/corpus_wikipedia_mg")
    ap.add_argument("--min-chars", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    titles = []
    for c in CATEGORIES:
        got = category_titles(c)
        print(f"Category:{c}: {len(got)} pages")
        titles += got
    titles += EXTRA_TITLES
    titles = sorted(set(titles))
    print(f"{len(titles)} unique titles")

    manifest, seen_ids = [], set()
    for i, t in enumerate(titles, 1):
        d = get(dict(action="query", prop="extracts|revisions|info", titles=t, explaintext=1,
                     exsectionformat="wiki", rvprop="ids|timestamp", inprop="url"))
        page = next(iter(d["query"]["pages"].values()))
        if "missing" in page or "extract" not in page or page["pageid"] in seen_ids:
            print(f"[{i}/{len(titles)}] skip (missing/duplicate): {t}")
            continue
        seen_ids.add(page["pageid"])
        text = clean(page["extract"])
        if len(text) < args.min_chars:
            print(f"[{i}/{len(titles)}] skip (short, {len(text)} chars): {page['title']}")
            continue
        fname = slug(page["title"]) + ".txt"
        with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"{page['title']}\n\n{text}\n")
        rev = page["revisions"][0]
        manifest.append({
            "title": page["title"], "file": fname, "chars": len(text),
            "url": page["fullurl"], "revid": rev["revid"], "timestamp": rev["timestamp"],
            "licence": "CC BY-SA 4.0",
        })
        print(f"[{i}/{len(titles)}] {page['title']} ({len(text):,} chars)")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"source": "en.wikipedia.org", "licence": "CC BY-SA 4.0 (text)",
                   "retrieved": time.strftime("%Y-%m-%d"), "pages": manifest}, f, indent=1)
    total = sum(p["chars"] for p in manifest)
    print(f"\n{len(manifest)} pages, {total:,} chars (~{total // 4:,} tokens) -> {args.out_dir}")


if __name__ == "__main__":
    main()
