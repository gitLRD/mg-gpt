# MG-GPT — Progress & Plans

A from-scratch GPT-2 style transformer trained on MG car knowledge.
Learning exercise focused on understanding transformer internals.

---

## Project structure

```
mg-llm/
├── model/
│   ├── __init__.py
│   ├── gpt.py        # Full transformer architecture (~200 lines)
│   └── config.py     # nano / small / medium size presets
├── data/
│   ├── corpus/       # Raw .txt training files go here
│   ├── prepare.py    # Tokenises corpus → train.bin / val.bin
│   ├── fetch_corpus.py        # Fetches Wikipedia articles via API (small, quick)
│   └── extract_wikipedia.py   # Extracts MG articles from a full Wikipedia XML dump
├── train.py          # Training loop
├── generate.py       # Inference / generation script
├── requirements.txt
└── PROGRESS.md       # This file
```

---

## What's been built / fixed

### Project setup
- [x] Unpacked `own_llm.zip` into correct folder structure
- [x] Added `model/__init__.py` (was missing, would cause import errors)
- [x] Created `requirements.txt` (torch, numpy, tiktoken, matplotlib, lxml)
- [x] Installed all dependencies — PyTorch 2.12, MPS confirmed available

### Training loop fixes (`train.py`)
- [x] Wired up `grad_accum` — was defined but not used. Now runs N micro-batches
  before each optimiser step, giving effective batch size of `batch_size × grad_accum`
- [x] Added loss curve plot — saved to `checkpoints/loss_curve.png` at end of training
- [x] Added `matplotlib` import with `Agg` backend (headless-safe on MPS)

### Corpus tooling
- [x] `data/fetch_corpus.py` — pulls ~8 Wikipedia articles via the API using stdlib only.
  Quick bootstrap. Got ~28K tokens from MGB, MG Cars, SU carburettor, British Leyland, etc.
- [x] `data/extract_wikipedia.py` — full Wikipedia dump extractor.
  Streams a 107GB XML dump without loading it into memory.
  Uses lxml (fast C parser) + multiprocessing (one reader, N-1 worker processes).
  Scores each article by keyword hit count, saves anything above MIN_SCORE=10.
- [x] `data/prepare.py` run — `train.bin`, `val.bin`, `meta.json` written from bootstrap corpus

---

## Corpus status

| Source | Status | Tokens |
|--------|--------|--------|
| Wikipedia API bootstrap | Done | ~28K |
| Wikipedia XML dump extraction | **To verify** — script ran but no files written yet | — |
| MG forum scraper | Planned | — |

Target for nano training: ~100K tokens
Target for small training: ~500K tokens

---

## Hardware

- MacBook Pro, Apple Silicon, 32GB RAM
- Using MPS backend (GPU)
- Training uses float32 throughout (MPS + float16 is unstable)

---

## Next steps

### Immediate
1. **Verify / re-run Wikipedia extractor** — check terminal output, confirm files land in `data/corpus/`
2. **Re-run `data/prepare.py`** once corpus is populated
3. **Run `python train.py --config nano`** — verify pipeline end-to-end, watch loss drop

### Corpus expansion
4. **Build forum scraper** — target mgexp.com and/or mgoc.co.uk forum threads.
   Goal: capture the opinionated, vernacular voice of MG enthusiasts.
   Potentially tens of thousands of posts — much more data than Wikipedia.

### Training progression
5. Once nano confirms the pipeline works → graduate to `--config small`
6. Optionally: overnight run with `--config medium` for better quality output

---

## Architecture notes (for reference)

- GPT-2 style decoder-only transformer
- Token embeddings + positional embeddings → N transformer blocks → LN → linear head
- Weight tying: token embedding matrix shared with LM head
- Pre-norm (LayerNorm before attention, not after)
- Causal mask registered as buffer for device portability
- AdamW with separate decay/no-decay param groups
- Cosine LR schedule with linear warmup
