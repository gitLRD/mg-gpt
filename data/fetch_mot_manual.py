"""
fetch_mot_manual.py — pull the UK DVSA "MOT inspection manual for private
passenger and light commercial vehicles" as plain text.

Licence: Open Government Licence v3.0 (GOV.UK footer, "except where otherwise
stated"). It allows copying and adapting, commercial or not, if the source is
acknowledged. Attribution line (also written into manifest.json):
    "Contains public sector information licensed under the Open Government
    Licence v3.0."
The licence does NOT cover logos or third-party material, so we keep TEXT ONLY
(no images, no attachments) and skip the pure-lookup appendices B and C.

Uses GOV.UK's public content API (JSON), serially, ~2s apart, ~15 requests.
The User-Agent is a generic project name on purpose (no personal contact).

Usage:
    python data/fetch_mot_manual.py --out-dir data/corpus_mot_manual
"""

import argparse
import html
import json
import re
import time
import urllib.request
from datetime import date

ROOT = "/guidance/mot-inspection-manual-for-private-passenger-and-light-commercial-vehicles"
API = "https://www.gov.uk/api/content"
PAGE = "https://www.gov.uk"
UA = "mg-gpt-corpus-research/0.1 (personal learning project)"
DELAY = 2.0
SKIP_SLUGS = {"appendix-b-tyre-load-index-tables", "appendix-c-seat-belt-fitment-tables"}
ATTRIBUTION = "Contains public sector information licensed under the Open Government Licence v3.0."


def get(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": UA})
    for attempt in range(4):
        time.sleep(DELAY)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — back off on any transient error
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"giving up on {path}")


def to_text(body_html):
    t = re.sub(r"(?is)<(script|style|table|figure|img)[^>]*>.*?</\1>", " ", body_html)
    t = re.sub(r"(?i)<img[^>]*>", " ", t)
    t = re.sub(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>", lambda m: "\n\n## " + m.group(1) + "\n\n", t)
    t = re.sub(r"(?i)<li[^>]*>", "\n- ", t)
    t = re.sub(r"(?i)</p>|<br\s*/?>", "\n\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = "\n".join(line.strip() for line in re.sub(r"[ \t]+", " ", t).split("\n"))
    # drop the "In this section" link lists (pure navigation)
    t = re.sub(r"## In this section\n+(?:- [^\n]*\n+)+", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    root = get(ROOT)
    sections = [s for g in root["details"]["child_section_groups"] for s in g["child_sections"]]
    manifest = {
        "source": PAGE + ROOT,
        "licence": "Open Government Licence v3.0",
        "licence_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        "attribution": ATTRIBUTION,
        "fetched": date.today().isoformat(),
        "manual_updated_at": root.get("public_updated_at"),
        "note": "text only; images/attachments and appendices B, C skipped",
        "sections": [],
    }
    for s in sections:
        path = s["base_path"]
        slug = path.rsplit("/", 1)[-1]
        if slug in SKIP_SLUGS:
            continue
        d = get(path)
        text = to_text(d["details"].get("body", ""))
        with open(f"{args.out_dir}/{slug}.txt", "w") as f:
            f.write(f"# {s['title']}\n\n{text}\n")
        manifest["sections"].append({"title": s["title"], "url": PAGE + path,
                                     "file": f"{slug}.txt", "chars": len(text)})
        print(f"{s['title']}: {len(text):,} chars")
    with open(f"{args.out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
