"""
data/prepare.py — Tokenise the corpus and create train/val splits.

Run this once before training:
    python data/prepare.py
    python data/prepare.py --out-dir data/v7_mgonly   # keep the old bins intact

Output (in --out-dir, default data/):
    train.bin  — token IDs as a memory-mapped numpy array
    val.bin    — same, validation split
    meta.json  — vocab size, split sizes and the held-out thread ids

The split is by *thread*, not by token position. Each forum thread appears in
two corpora (raw dialogue in corpus_pistonheads_qa, LLM case-study in
corpus_synth_qa) under the same filename stem, so both versions of a held-out
thread go to val together. Splitting by position put a synth-only tail in val
while its source threads sat in train as raw dialogue.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Read from multiple corpus directories and concatenate. Mixing the
# forum-format Q&A (noisy, carries voice) with the synth Q&A (clean, factual)
# lets the model learn register from the former and specificity from the latter.
# Missing directories are silently skipped — if you only have one, that's fine.
CORPUS_DIRS = [
    "data/corpus_pistonheads_qa",  # reformatted as USER:/ASSISTANT: dialogue
    "data/corpus_synth_qa",        # LLM case-study synthesis (Phase 2)
]
VAL_FRACTION = 0.1            # ~10% of threads held out for validation
SHUFFLE_SEED = 1337

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="data", help="where to write train.bin/val.bin/meta.json")
args = parser.parse_args()
OUTPUT_DIR = args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Clean forum posts
# ---------------------------------------------------------------------------

def clean_forum_text(text):
    """Remove forum UI boilerplate and metadata."""
    text = re.sub(r"Reply Was this post helpful[^\n]*", "", text)
    text = re.sub(r"Edited \d+ time\(s\)[^\n]*", "", text)
    text = re.sub(r"Last edit at[^\n]*", "", text)
    text = re.sub(r"^\s*\+?\d+\s+(Yes|No|Thank)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text.strip()


def is_val_thread(stem):
    """Deterministic hash split, so the held-out set is stable across reruns."""
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return (h % 1000) < VAL_FRACTION * 1000

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

print("Loading corpus...")
corpus_files = []
for d in CORPUS_DIRS:
    found = sorted(glob.glob(os.path.join(d, "*.txt")))
    if found:
        print(f"  {d}/: {len(found)} files")
        corpus_files.extend(found)

if not corpus_files:
    print(f"\n⚠️  No .txt files found in any of: {CORPUS_DIRS}")
    print("See data/corpus.md for how to build your MG knowledge base.")
    exit(1)

train_texts, val_texts = [], []
val_stems = set()
for path in corpus_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        text = clean_forum_text(f.read().strip())
    if is_val_thread(stem):
        val_texts.append(text)
        val_stems.add(stem)
    else:
        train_texts.append(text)

# Shuffle document order so the PH/synth mix is spread through each split
# rather than sitting in two contiguous blocks.
rng = random.Random(SHUFFLE_SEED)
rng.shuffle(train_texts)
rng.shuffle(val_texts)

# Join with double newline so documents don't bleed into each other
train_text = "\n\n".join(train_texts)
val_text = "\n\n".join(val_texts)
print(f"\nTrain docs: {len(train_texts):,} | Val docs: {len(val_texts):,} "
      f"({len(val_stems):,} held-out threads)")

# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------

print("\nTokenising with GPT-2 BPE...")
enc = tiktoken.get_encoding("gpt2")
train_tokens = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
val_tokens = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")
print(f"Compression ratio: {(len(train_text) + len(val_text)) / (len(train_tokens) + len(val_tokens)):.2f} chars/token")

train_path = os.path.join(OUTPUT_DIR, "train.bin")
val_path   = os.path.join(OUTPUT_DIR, "val.bin")

train_tokens.tofile(train_path)
val_tokens.tofile(val_path)

# Save metadata
meta = {
    "vocab_size": enc.n_vocab,
    "train_tokens": len(train_tokens),
    "val_tokens": len(val_tokens),
    "split": "by thread (md5 of filename stem)",
    "val_threads": sorted(val_stems),
    "corpus_files": [os.path.basename(f) for f in corpus_files],
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Done. Files written to {OUTPUT_DIR}/")
print(f"   vocab_size: {meta['vocab_size']}")

# Sanity check: decode first 100 tokens back
sample = enc.decode(train_tokens[:100].tolist())
print(f"\nFirst 100 tokens decoded:\n---\n{sample}\n---")
