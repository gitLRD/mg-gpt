# Changelog

A running log of notable changes to MG-GPT. Newest first.

Each entry is one shipped change with a short *why*. For deeper detail, follow
the file refs. README.md tracks the current state; this file tracks the
journey.

---

## In progress

- **v8 fine-tune on augmented data** (`data/v8_augmented`: +2,713 extra forum Q&A, 53 Wikipedia
  articles, 671 wiki Q&A; train 2.44M -> 2.93M tokens, val identical). Best val 3.65 @3000 vs v7 3.69.
  Judge, 3 seeds x 2 prompts x 2 judges (qwen3-vl, qwen3-coder): v8 vs v7 = 2.76/2.54 vs 2.75/2.56
  (vl), 2.80/2.58 vs 2.70/2.67 (coder). No reliable gain, so v8 is not deployed. 14 threads were
  skipped (HTTP 400 from the model server). Runs in `eval_runs/v8_aug_*`, `v7_ft_*`, `judge2_*`.
- **More MG text, licence-checked** (research 2026-11-06). Not usable: workshop manuals and
  reprints (commercial, copyrighted), MG Car Club *Safety Fast!* (permission needed, members only),
  Chicagoland MG Club pages (all rights reserved), Stack Exchange (terms exclude AI training), NC
  open textbooks, US Army TM 9-8000 (public-domain status not stated by the host; skipped), Gutenberg
  1907-15 motoring books (public domain but dated and low value). Usable: English Wikipedia MG
  articles, CC BY-SA 4.0 (attribution + share-alike; the terms don't mention AI training either way).
- **`data/fetch_wikipedia_mg.py`**: 53 MG-related Wikipedia articles (~98K tokens) via the API,
  serial 1.2 s apart, `maxlag`, gzip, User-Agent with a contact address, per Wikimedia's etiquette
  (they ask bulk users not to crawl). `manifest.json` records title, URL and revision id for attribution.
  Modern SAIC-era models and a town article were dropped as off-topic for a classic-MG bot.
- **`data/synthesize_wiki_qa.py`**: grounded Q&A from those articles (answers only from the text, local Qwen).
- **`data/synthesize_qa_extra.py`**: second-pass Q&A on forum threads >= 3000 chars (first pass averaged
  3.4 pairs/thread vs a 3-8 target); val threads skipped.
- **`data/prepare.py` `EXTRA_TRAIN_DIRS`**: new sources are train-only and added after the original
  shuffles, so val is unchanged and v7 numbers stay comparable.
- **Deployed: persona priming removed from the live prompt** (`lambda/handler.py`, Lambda version 21).
  Plain prompt scored equal or better in 4/4 runs and cuts ~500 prompt tokens; persona is now the greeting only.
  Live probe: 5/5 non-empty. Done by hand (`docker push` + `aws lambda update-function-code`) as the
  Cloudflare token still fails `terraform apply`.
- **Persona-priming test** (`eval_mg.py` new `short_nospace` mode; runs in `eval_runs/priming_*`):
  fine-tuned model, 30 Qs x 4 runs (3 personas, 2 seeds for one). Judge mean: plain 2.70,
  full priming (live) 2.51, one-pair priming 2.56. Plain >= full priming in 4/4 runs, but gaps
  (~0.2) are about the size of the seed-to-seed swing (plain 2.73 vs 2.60), so: lean, not proof.
  Live handler unchanged.
- **Deployed v7 to Lambda** (2026-11-06): fine-tuned model (iter 3750, val 3.69) plus the
  trailing-space handler fix. Live probe: 5/5 replies non-empty (v6 was ~67% empty). Slim
  `checkpoints/ckpt_best.pt` = weights + config only (228 MB). Previous image kept in ECR as
  `v6-rollback` (the repo lifecycle policy expires untagged images after 1 day).
  `deploy.sh` stopped at `terraform apply` (Cloudflare token rejected, status 400), so the Lambda
  was updated by hand with `aws lambda update-function-code`; fix the token before the next full deploy.
- **v7 plan: general-English pretrain → MG fine-tune** (tracked in Notion: "MG-GPT v7").
  MG-only data (2.7M tokens) can't teach English *and* MG to a 51M model; v6
  overfit ~68 epochs (val 3.14 @ iter 3000 → 3.79 @ 10000).
- **Val split fixed to be by thread** (`data/prepare.py`). The old split cut by
  token position, so val was 100% synth Q&A while 1,588/1,625 of those threads
  were also in train as raw dialogue. v6's 3.14 is not comparable to new numbers.
  New data goes to `data/v7_*/` (gitignored); old bins untouched.
- **`train.py` flags**: `--data-dir --checkpoint-dir --max-iters --lr --dropout
  --patience --init-from` (weights-only init for fine-tuning).
- **`eval_mg.py`**: val loss + 20 fixed MG prompts + held-out questions, reports
  empty-reply rate and cheap coherence proxies; raw generations saved to `eval_runs/`.
- **`data/prepare_pretrain.py`**: streams a license-filtered slice of Common Pile
  (public-domain / CC only) with an attribution manifest. Smoke-tested at 2M tokens;
  full 200M run not yet launched.
- **MG-only baseline on the clean split** (from scratch, early stopped): best val
  **4.26** @ iter 1750, vs v6's leaked 3.14. Memorises fast (train 2.82, val ~4.3).
  Checkpoint + log in `checkpoints/v7_mgonly/`, `data/logs/train_v7_mgonly.log`.
- **Pretrain done** (2026-11-06): 196M-token Common Pile slice (`data/v7_pretrain/`,
  manifest.json = licence record), 12K iters, ~8.7h. General val 3.75 best, no overfit.
  `train.py --save-interval` added to keep checkpoint disk use sane.
- **MG fine-tune from the pretrain** (`checkpoints/v7_finetune/`, lr 1e-4, dropout 0.1):
  best val **3.69** @ 3750 vs 4.26 for MG-only (same held-out threads; full-sweep 3.73 vs 4.33).
  Keyword-hit 95% (bare) / 75% (handler) vs 85% / 60%.
- **`judge_mg.py`**: blind 1-5 "makes sense" scoring of `eval_mg.py` replies by a local
  Qwen (LM Studio). Bare prompt mean: v6 2.80, MG-only 2.30, fine-tuned 2.70; handler
  prompt: 1.97 / 2.37 / 2.43. On the 20 fixed questions the fine-tune leads (3.05 v6 2.85
  MG-only 2.45); v6 only wins on held-out Qs it trained on. n=30, one judge, noisy.
- **Found: live empty replies come from the prompt's trailing space**
  (`lambda/handler.py`, `"ASSISTANT: "`). BPE attaches the space to the next word, so
  the model sees an off-distribution token. v6 ckpt: 0/30 empty without the space,
  13/20 with it. Fix pending, not deployed.

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
