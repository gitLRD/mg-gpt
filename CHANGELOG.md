# Changelog

A running log of notable changes to MG-GPT. Newest first.

Each entry is one shipped change with a short *why*. For deeper detail, follow
the file refs. README.md tracks the current state; this file tracks the
journey.

---

## In progress

- **Synthetic Q&A via local Qwen3-VL-30B-A3B-Instruct (LM Studio backend).**
  `data/synthesize_qa.py` has a `--lm-studio` flag that swaps the Anthropic
  client for a localhost OpenAI-compatible call. 10-thread test produced
  clean, specific Q&A with real values (~7.6 pairs/thread, 17s/thread).
  Full 893-thread run launched — ~4 hours of background work, $0.
- **Mixed-corpus training plan for v5.** Synth Q&A is clean but bland;
  forum-format Q&A (data/corpus_qa/) is noisy but carries voice. Plan is to
  train on *both* directories combined so the model learns format and
  specificity from synth + register and persona from forum. Implementation:
  small change to `prepare.py` to concat-then-tokenise both dirs.

---

## v4 — Inference & frontend pass *(in progress)*

- **Workshop-manual frontend** (`frontend/index.html`). Cream paper, EB
  Garamond serif, navy ink, MG octagon SVG, FIG.-style message labels. Much
  more on-theme than the generic blue-gradient v3.
- **MG-themed warmup whittisims.** While the backend warms up, an italic
  cycling bubble shows lines like *"Pulling the choke..."*, *"Priming the
  carbs..."*. New every 1s. Replaces the silent multi-second "Thinking..."
  wait so the cold start *feels* like a feature.
- **Smarter inference sampling** (`model/gpt.py`, `lambda/handler.py`).
  Added repetition penalty (1.15), nucleus top-p (0.9), and explicit stop
  token sequences (`\nUSER:`, `\nASSISTANT:`, `\n\n`). Kills the `!!!!`
  mode-collapse loop and lets replies end cleanly mid-budget. Local A/B
  showed clearly better output without retraining.

---

## v3 — Bigger corpus, cleaner data, real personas

- **PII redaction** (`data/strip_pii.py`). Regex-based Macie-style sweep
  before training: 10,676 names, 4,439 URLs, 3,453 forum usernames, 945
  emails, 653 phones, 297 addresses/ZIPs/postcodes. Necessary given the
  chatbot is publicly live and the model can regurgitate training data.
- **Stronger persona priming + opening greetings.** Each persona's priming
  went from 2 example turns (~100 tokens) to 4 turns (~300 tokens), with
  much more distinctive voice. Each persona now also greets the user with
  a hand-written opener on session start — picked from 3 alternatives.
- **Thread discovery + bulk scrape** (`data/discover_and_scrape.py`). Walks
  forum index pages, finds canonical thread URLs, skips ones already in the
  corpus. 968 new threads scraped → **21,380 posts across 1,010 threads**
  (up from 3,578 / 42 threads in v2).
- **Retrained `small` config on the bigger corpus.** Best val loss 4.38
  (iter 4000) vs 6.11 in v2 — first run where the model actually
  generalised instead of just memorising. Trained on PII-stripped,
  Q&A-formatted 2.5M-token corpus.

## v2 — Q&A format, decoding fixes, config-from-checkpoint

- **Q&A reformat** (`data/format_qa.py`). Converts each thread file from
  raw alternating posts to `USER:`/`ASSISTANT:` dialogue so the from-scratch
  GPT learns the chat pattern, not just forum prose. Strips boilerplate
  ("Reply Was this post helpful...", attachment lines, "In reply to #..."
  quote headers) along the way.
- **Config read from checkpoint** (`lambda/handler.py`). Stopped hardcoding
  `small_config()` — the handler now reads block_size/n_layer/n_embd from
  `checkpoint["config"]` so any nano/small/medium ckpt loads correctly.
  Avoids the shape-mismatch class of bugs entirely.
- **Decoding & truncation fixes.** Slice generated tokens by index (not
  decoded-prompt char length — BPE round-trips aren't exact), strip leading
  punctuation noise, truncate prompts cleanly when `block_size` is small.
- **Stable scraper filenames** (`data/bulk_scrape.py`). Switched from
  timestamp-based filenames to URL-slug-based so re-runs overwrite rather
  than dupe.

## v1 — Initial deploy

- **Lambda WebSocket async self-invoke pattern** (`lambda/handler.py`,
  `terraform/main.tf`). API Gateway WS integrations cap at 29s; the
  handler now splits into a fast ack path (DynamoDB write + "thinking"
  message) and an async self-invoked worker that runs the slow generation
  and posts back via apigatewaymanagementapi. No SQS, no extra cost.
- **PyTorch bump 2.0.1 → 2.1.2** (`lambda/requirements.txt`). Fixed
  `LayerNorm.__init__() got an unexpected keyword argument 'bias'` —
  the `bias` kwarg landed in torch 2.1.0.
- **Cloudflare token loader in `deploy.sh`.** Reads from
  `~/.config/cloudflare/token` if `CLOUDFLARE_API_TOKEN` isn't set in env,
  so Terraform's Cloudflare provider can run unattended. Token stays out
  of git.
- **Forum scraper pagination fix** (`data/scrape_forums.py`). CSS-style
  selectors like `a[rel='next']` were being naively split into invalid
  XPath (`//a[rel='next']` instead of `//a[@rel='next']`), silently
  matching nothing. Switched to explicit XPath. ~7.5× more posts per
  scrape afterwards.
- **First deploy.** Lambda + DynamoDB + S3 + CloudFront + API Gateway WS
  + Cloudflare DNS, all via Terraform. Live at mg-gpt.desforgeslabs.com.
