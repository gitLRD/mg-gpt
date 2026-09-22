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

# Read from multiple corpus directories and concatenate. Mixing the
# forum-format Q&A (noisy, carries voice) with the synth Q&A (clean, factual)
# lets the model learn register from the former and specificity from the latter.
# Missing directories are silently skipped — if you only have one, that's fine.
CORPUS_DIRS = [
    "data/corpus_pistonheads_qa",  # reformatted as USER:/ASSISTANT: dialogue
    "data/corpus_synth_qa",        # LLM case-study synthesis (Phase 2)
]
TRAIN_SPLIT = 0.9             # 90% train, 10% validation
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

texts = []
for path in corpus_files:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    text = clean_forum_text(text)
    texts.append(text)

# Join with double newline so documents don't bleed into each other
full_text = "\n\n".join(texts)
print(f"\nTotal corpus: {len(full_text):,} characters from {len(corpus_files)} files")

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
