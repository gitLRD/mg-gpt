"""
data/prepare.py — Tokenise the corpus and create train/val splits.

Run this once before training:
    python data/prepare.py

Output:
    data/train.bin  — token IDs as a memory-mapped numpy array
    data/val.bin    — same, validation split
    data/meta.json  — vocab size and split sizes for reference
"""

import os
import json
import glob
import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR  = "data/corpus_qa"   # Q&A-formatted corpus (run data/format_qa.py first)
TRAIN_SPLIT = 0.9                 # 90% train, 10% validation
OUTPUT_DIR  = "data"

# ---------------------------------------------------------------------------
# Clean forum posts
# ---------------------------------------------------------------------------

def clean_forum_text(text):
    """Remove forum UI boilerplate and metadata."""
    import re
    text = re.sub(r"Reply Was this post helpful[^\n]*", "", text)
    text = re.sub(r"Edited \d+ time\(s\)[^\n]*", "", text)
    text = re.sub(r"Last edit at[^\n]*", "", text)
    text = re.sub(r"^\s*\+?\d+\s+(Yes|No|Thank)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

print("Loading corpus...")
corpus_files = glob.glob(os.path.join(CORPUS_DIR, "*.txt"))

if not corpus_files:
    print(f"\n⚠️  No .txt files found in {CORPUS_DIR}/")
    print("See data/corpus.md for how to build your MG knowledge base.")
    print("Quick start: copy any .txt files with MG content into data/corpus/")
    exit(1)

texts = []
for path in sorted(corpus_files):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    text = clean_forum_text(text)
    print(f"  {os.path.basename(path)}: {len(text):,} chars")
    texts.append(text)

# Join with double newline so documents don't bleed into each other
full_text = "\n\n".join(texts)
print(f"\nTotal corpus: {len(full_text):,} characters")

# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------

print("\nTokenising with GPT-2 BPE...")
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(full_text)
tokens = np.array(tokens, dtype=np.uint16)

print(f"Total tokens: {len(tokens):,}")
print(f"Compression ratio: {len(full_text) / len(tokens):.2f} chars/token")

# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

split_idx = int(len(tokens) * TRAIN_SPLIT)
train_tokens = tokens[:split_idx]
val_tokens   = tokens[split_idx:]

print(f"\nTrain tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")

train_path = os.path.join(OUTPUT_DIR, "train.bin")
val_path   = os.path.join(OUTPUT_DIR, "val.bin")

train_tokens.tofile(train_path)
val_tokens.tofile(val_path)

# Save metadata
meta = {
    "vocab_size": enc.n_vocab,
    "train_tokens": len(train_tokens),
    "val_tokens": len(val_tokens),
    "corpus_files": [os.path.basename(f) for f in corpus_files],
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Done. Files written to {OUTPUT_DIR}/")
print(f"   vocab_size: {meta['vocab_size']}")

# Sanity check: decode first 100 tokens back
sample = enc.decode(train_tokens[:100].tolist())
print(f"\nFirst 100 tokens decoded:\n---\n{sample}\n---")
