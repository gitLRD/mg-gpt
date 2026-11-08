"""
data/prepare_pretrain.py — Build a general-English pretraining set from Common Pile.

Common Pile v0.1 (huggingface.co/common-pile) is public-domain / openly licensed
text only, which keeps the pretraining step inside the same robots.txt/TOS
standard as the MG corpus. Shards are streamed over HTTP and the run stops as
soon as each source's token budget is met, so a 200M-token set downloads
a few hundred MB, not the full 21+ GB subsets.

    python data/prepare_pretrain.py --tokens 2e6   --out-dir data/v7_pretrain_smoke   # smoke test
    python data/prepare_pretrain.py --tokens 200e6 --out-dir data/v7_pretrain

Every document carries its own license string; only Creative Commons (no NC/ND),
CC0 and Public Domain are kept. The per-source/per-license counts go to
manifest.json as the attribution record (CC BY / BY-SA need attribution).

Output (in --out-dir): train.bin, val.bin (uint16 GPT-2 BPE, docs joined by <|endoftext|>),
meta.json (same shape train.py reads), manifest.json.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

import numpy as np
import tiktoken

# source -> (share of token budget, number of shards to spread the budget across)
# Modern prose is weighted over pre-1929 books: the target voice is modern British
# hobbyist chat, and Gutenberg's archaic English is a minority flavour.
SOURCES = {
    "wikimedia_filtered": (0.40, 4),
    "doab":               (0.18, 2),
    "libretexts":         (0.10, 1),
    "pressbooks":         (0.10, 1),
    "news":               (0.04, 4),
    "project_gutenberg":  (0.18, 3),
}
HF = "https://huggingface.co"
VAL_PERMILLE = 5          # 0.5% of documents held out
MIN_CHARS = 400
MAX_CHARS = 40_000
GUTENBERG_SKIP, GUTENBERG_TAKE = 8_000, 60_000   # one excerpt per book, past the licence header

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=float, default=200e6, help="total token budget across sources")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
rng = random.Random(args.seed)


def api(path):
    return json.load(urllib.request.urlopen(f"{HF}/api/datasets/common-pile/{path}", timeout=60))


def pick_shards(name, k):
    files = sorted(s["rfilename"] for s in api(f"{name}")["siblings"] if s["rfilename"].endswith(".gz"))
    if len(files) <= k:
        return files
    step = len(files) / k
    return [files[int(i * step)] for i in range(k)]


def stream_docs(name, fname):
    url = f"{HF}/datasets/common-pile/{name}/resolve/main/{fname}"
    with urllib.request.urlopen(url, timeout=120) as resp, gzip.GzipFile(fileobj=resp) as gz:
        for line in gz:
            line = line.strip()
            if line:
                yield json.loads(line)


def license_ok(lic):
    l = (lic or "").lower()
    if "noncommercial" in l or "noderiv" in l or "-nc" in l or "-nd" in l:
        return False
    return ("creative commons" in l) or ("public domain" in l) or l.startswith("cc0")


def clean(text, source):
    if source == "project_gutenberg":
        text = text[GUTENBERG_SKIP:GUTENBERG_SKIP + GUTENBERG_TAKE]
        text = text[text.find("\n\n") + 2:] if "\n\n" in text[:2000] else text   # start on a paragraph
    text = text[:MAX_CHARS]    # no single book section should dominate a source's budget
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def good_text(text):
    if len(text) < MIN_CHARS:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    pipes = text.count("|") / len(text)          # table dumps (pressbooks, wiki tables)
    return letters >= 0.82 and pipes < 0.01


train_chunks, val_chunks = [], []
manifest = {"seed": args.seed, "token_budget": args.tokens, "sources": {}}
total_tokens = 0
t0 = time.time()

for name, (share, k) in SOURCES.items():
    budget = int(args.tokens * share)
    per_shard = budget // k
    stats = {"budget": budget, "tokens": 0, "docs": 0, "val_docs": 0, "skipped_license": 0,
             "skipped_quality": 0, "licenses": {}, "shards": []}
    print(f"\n== {name}: budget {budget:,} tokens over {k} shard(s)", flush=True)
    for fname in pick_shards(name, k):
        stats["shards"].append(fname)
        got = 0
        try:
            for rec in stream_docs(name, fname):
                lic = (rec.get("metadata") or {}).get("license")
                if not license_ok(lic):
                    stats["skipped_license"] += 1
                    continue
                text = clean(rec.get("text", ""), name)
                if not good_text(text):
                    stats["skipped_quality"] += 1
                    continue
                toks = enc.encode_ordinary(text)
                toks.append(EOT)
                is_val = int(hashlib.md5(str(rec.get("id", text[:64])).encode()).hexdigest(), 16) % 1000 < VAL_PERMILLE
                (val_chunks if is_val else train_chunks).append(np.array(toks, dtype=np.uint16))
                stats["docs"] += 1
                stats["val_docs"] += is_val
                stats["licenses"][lic] = stats["licenses"].get(lic, 0) + 1
                got += len(toks)
                if got >= per_shard:
                    break
        except Exception as e:                      # a dropped connection shouldn't lose the other sources
            print(f"   ! {fname}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        stats["tokens"] += got
        print(f"   {fname}: {got:,} tokens  ({time.time() - t0:.0f}s elapsed)", flush=True)
    total_tokens += stats["tokens"]
    manifest["sources"][name] = stats

# Shuffle documents so sources are interleaved through training rather than sequential blocks.
rng.shuffle(train_chunks)
rng.shuffle(val_chunks)
train = np.concatenate(train_chunks) if train_chunks else np.zeros(0, np.uint16)
val = np.concatenate(val_chunks) if val_chunks else np.zeros(0, np.uint16)
train.tofile(os.path.join(args.out_dir, "train.bin"))
val.tofile(os.path.join(args.out_dir, "val.bin"))

meta = {"vocab_size": enc.n_vocab, "train_tokens": int(len(train)), "val_tokens": int(len(val)),
        "split": "by document (md5 of id)", "corpus": "common-pile v0.1 subset (see manifest.json)"}
json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
manifest["train_tokens"], manifest["val_tokens"] = int(len(train)), int(len(val))
json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)

print(f"\n✅ train {len(train):,} | val {len(val):,} tokens -> {args.out_dir}/  ({time.time() - t0:.0f}s)")
for n, s in manifest["sources"].items():
    print(f"   {n:20s} {s['tokens']:>12,} tok  docs={s['docs']:>6,}  skipped(license={s['skipped_license']}, quality={s['skipped_quality']})")
