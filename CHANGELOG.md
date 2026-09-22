# Changelog

A running log of notable changes to MG-GPT. Newest first.

Each entry is one shipped change with a short *why*. For deeper detail, follow
the file refs. README.md tracks the current state; this file tracks the
journey.

---

## In progress

- **TransformerLens exploration notebook** (`notebooks/transformer_lens_explore.ipynb`).
  Loads `ckpt_best.pt` into a `HookedTransformer` via the built-in
  `convert_nanogpt_weights` adapter (our state-dict layout already matches
  nanoGPT exactly). Includes activation cache, per-head attention pattern
  viz via CircuitsVis, and a logit-lens cell. Pulled in
  `transformer_lens`, `circuitsvis`, `ipywidgets`; pip resolver pinned
  torch back from 2.12.0 → 2.7.1 to stay inside TransformerLens 3.4's
  range. Doesn't touch Lambda — the deploy image keeps its own torch pin.

## v6 — TOS-clean rebuild, live again *(shipped)*

- **Trained `medium` config (51.5M params) on the PH-only corpus**
  (`train.py`). 2.4M train / 267K val tokens, 10K iters target. Best val
  loss **3.14** at iter 3000 (vs v5's 3.08 on a larger but TOS-tainted
  corpus) — small regression on the loss metric, but the corpus is
  qualitatively richer (case-study Q&A on classic MG + MG Rover era).
  Memory pressure on MPS reduced via `batch_size=4`, `grad_accum=4` —
  same effective batch (16), ~4× less activation memory.
- **`weights_only=False` on checkpoint resume** (`train.py:117`). PyTorch
  2.6 flipped the `torch.load` default and refused to unpickle our
  `GPTConfig` object. We wrote the checkpoint ourselves, so explicit
  opt-in is the right fix.
- **Lambda handler, Dockerfile, requirements, and frontend restored**
  from the v5.1 baseline + new ckpt baked into the image. Holding page
  removed; chat UI is live again at https://mg-gpt.desforgeslabs.com.

## v6 prep — TOS-clean corpus rebuild *(shipped)*

- **Phase 1 — PistonHeads scrape.** New scraper
  (`data/scrape_pistonheads.py`, `data/discover_and_scrape_pistonheads.py`)
  walks the dedicated MG sub-forum (f=81) and the Rover sub-forum (f=54).
  Every fetch is gated through `urllib.robotparser` against PH's
  `robots.txt`. 2,864 topics scraped in ~45 min → 16,500 posts, ~8MB raw.
  An early extraction bug (block-element word-junctions like "block?Head")
  was caught in smoke test and fixed by padding block elements with
  whitespace before flattening.
- **PII redaction** (`data/strip_pii.py`). Same regex sweep as v3:
  1,944 URLs, 3,836 names, 88 phones, 62 emails, 88 ZIPs, 45 postcodes,
  11 addresses.
- **Phase 2 — case-study Q&A synthesis.** Rewrote the `synthesize_qa.py`
  system prompt to extract symptom → cause → check → fix case studies as
  natural prose (3–6 sentences per answer), expanded the persona to cover
  MG Rover era (ZR/ZS/ZT, K-series). 1,625 eligible threads processed via
  local Qwen3-VL-30B-A3B-Instruct-MLX (LM Studio); 1,622 successful pairs
  saved (3 from a smoke test). 2.6M input + 769K output tokens, 7 hours,
  $0. Output quality is markedly more diagnostic than the prior bare-Q&A
  format — answers reference specific parts (Samco hose, Mikalor clamps)
  and concrete checks where the thread surfaced them.
- **Tokenisation** (`data/prepare.py`). Merged PH dialogue + synth Q&A
  into a single 2.67M-token training set (2.4M train / 267K val).
  Comparable in volume to the withdrawn v5 corpus, materially higher
  signal-to-noise.



## v5.2 — Withdrawn pending clean-corpus rebuild *(shipped)*

- **Model withdrawn.** A robots.txt re-read showed mgexp.com (and
  mg-rover.org) explicitly disallow `anthropic-ai`, `ClaudeBot`, `GPTBot`,
  etc. — clear "no AI training" intent. v5 was trained on that corpus.
  Strict reading: don't keep serving it.
- **Lambda handler replaced with a stub** (`lambda/handler.py`,
  `lambda/Dockerfile`, `lambda/requirements.txt`). New image ships no model
  code, no checkpoint, and no torch. The WS lifecycle wiring (`$connect`,
  `$disconnect`, `$default`) stays valid so infra holds; any user message
  gets a static "closed for refit" reply. v6 deploy restores the real
  handler.
- **Frontend replaced with a holding page** (`frontend/index.html`). Same
  Workshop Manual aesthetic — octagon, EB Garamond, cream paper — no chat
  UI. Explains the rebuild in two sentences.
- **mgexp corpus & v5 checkpoints archived** (`data/_archive_mgexp/`,
  `checkpoints/_archive_v5/`). Kept on disk for reference; removed from
  the training and deploy paths. Also archived `discover_and_scrape.py`
  and `bulk_scrape.py` since they were mgexp-specific.

## v5.1 — Reply polish *(shipped)*

- **Capitalise first letter of replies** (`lambda/handler.py`). The model
  often emits a lowercase continuation token first because the forum corpus
  is full of mid-thread starts. Post-processing fixes it so answers read
  cleanly without retraining.
- **Trim dangling tail to last sentence boundary** (`lambda/handler.py`).
  If a reply ends without `.!?` and there's ≥40 chars of clean text before
  the last terminator, cut back to it. Avoids leaving half-thoughts on
  screen when the token budget runs out mid-sentence.
- **Slower warmup cycle** (`frontend/index.html`). Whittisims now rotate
  every 10s instead of every 1s — feels less frantic, gives each line time
  to read.

## v5 — Mixed-corpus training *(shipped)*

- **Multi-directory corpus loading** (`data/prepare.py`). Concatenates files
  from `data/corpus_qa/` (forum format) and `data/corpus_synth_qa/` (LLM
  extracted) so the model learns voice from one and specificity from the
  other. Skips missing directories silently — works fine with just one.
- **Retrained small config on the mixed 2.9M-token corpus.** Best val loss
  **3.08** (iter 9000, perplexity ~22) — clearly best result yet, vs v3's
  4.38 (iter 4000) on the 2.5M synth-free corpus. Mixed signal both
  improved generalisation *and* delayed overfitting by ~5K iterations.
  Train/val gap stayed healthy throughout (2.69 / 3.08 at the best ckpt).

## v4.1 — Synthetic Q&A via local Qwen *(shipped)*

- **Local-LLM extraction pipeline** (`data/synthesize_qa.py --lm-studio`).
  Added an OpenAI-compatible LM Studio backend alongside the Anthropic one,
  with `/no_think` appended for Qwen3 (skip reasoning trace), retry+backoff,
  `<think>` block stripping, and a connectivity preflight that fails fast
  if the server isn't up.
- **Full Q&A synthesis on 893 threads via Qwen3-VL-30B-A3B-Instruct (MLX
  Q4).** Ran locally on M-series via LM Studio. 883/883 success rate.
  Output: **7,014 clean Q&A pairs**, 2.25M input tokens, 356K output tokens.
  5 hours elapsed, $0 cost. Quality vastly better than the alternating-posts
  format — specific questions, grounded answers with real values, no
  chitchat. `data/corpus_synth_qa/` now exists ready for v5 training.

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
