"""
extract_tm9_8000.py — turn the US Army manual TM 9-8000 "Principles of Automotive
Vehicles" (25 Oct 1985, Change 1 Nov 1988) into plain-text chapters.

Licence: a work of the US Army, i.e. of US federal government employees in their
official duties, which US law does not copyright (17 U.S.C. 105; see
https://www.usa.gov/government-copyright). The PDF carries no copyright notice and no
distribution restriction (its "Distribution" line is a mailing list). The copy we
use was downloaded from everyspec.com. This is the owner's reading of the
facts, not legal advice; it is also only protected-free inside the US.

Usage:
    python data/extract_tm9_8000.py --pdf path/to/TM_9-8000_CHG-1.pdf --out-dir data/corpus_tm9_8000
"""

import argparse
import json
import os
import re
from datetime import date

import pymupdf

BODY_START_PAGE = 27  # "CHAPTER 1" (0-based); front matter, TOCs and figure lists precede it
INDEX_PAGE = 762      # "INDEX" (0-based); drop it
# Chapters that make no sense for a 1950s-70s petrol sports car (reason in manifest).
EXCLUDE = {
    1: "introduction", 5: "diesel fuels", 6: "LPG", 10: "gas turbines", 18: "radio suppression",
    23: "automatic transmissions", 24: "cross-drive transmissions", 25: "cross-drive X1100",
    27: "transfer cases (4x4)", 31: "tracked-vehicle suspension", 36: "power takeoff, winches",
    37: "refrigeration", 38: "trailers",
}


def clean_block(b):
    t = re.sub(r"\s+", " ", b).strip()
    if not t or "Downloaded from" in t:
        return None
    if re.fullmatch(r"TM 9-8000|[\d\-.]+|Figure [\d\-]+.*|Change \d+", t):
        return None
    if re.search(r"\.{6,}", t):  # table-of-contents dot leaders
        return None
    if len(t) < 25 and not t.endswith("."):  # stray labels from diagrams
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    doc = pymupdf.open(args.pdf)

    chapters, cur = {}, None
    for i in range(BODY_START_PAGE, INDEX_PAGE):
        page = doc[i]
        head = page.get_text()[:200]
        m = re.search(r"CHAPTER\s+(\d+)", head)
        if m:
            cur = int(m.group(1))
        if cur is None:
            continue
        paras = []
        for b in page.get_text("blocks"):
            c = clean_block(b[4])
            if c:
                paras.append(c)
        chapters.setdefault(cur, []).append("\n\n".join(paras))

    manifest = {
        "source": "US Army TM 9-8000 Principles of Automotive Vehicles, 25 Oct 1985, with Change 1 (14 Nov 1988)",
        "obtained_from": "https://everyspec.com/ARMY/TM-Tech-Manual/TM_9-8000_CHG-1_48512/",
        "licence": "US federal government work (17 U.S.C. 105), no copyright notice in the document",
        "extracted": date.today().isoformat(),
        "chapters": [],
    }
    manifest["excluded_chapters"] = {str(k): v for k, v in EXCLUDE.items()}
    for n in sorted(chapters):
        if n in EXCLUDE:
            continue
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(chapters[n])).strip()
        fname = f"chapter_{n:02d}.txt"
        with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"TM 9-8000 Principles of Automotive Vehicles, chapter {n}\n\n{text}\n")
        manifest["chapters"].append({"chapter": n, "file": fname, "chars": len(text)})
        print(f"chapter {n:>2}: {len(text):>8,} chars")
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(c["chars"] for c in manifest["chapters"])
    print(f"total {total:,} chars (~{total // 4:,} tokens)")


if __name__ == "__main__":
    main()
