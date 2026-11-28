# MG-GPT

A small transformer written from scratch, trained on classic MG forum posts and served as a live chatbot from AWS Lambda. Built for learning: every line is meant to be read and understood.

**Live demo:** https://mg-gpt.desforgeslabs.com
**Case study:** https://desforgeslabs.com/work/mg-gpt.html
**Write-up:** [WRITEUP.md](WRITEUP.md): what broke, what was measured, what didn't help.

The model has about 51M parameters (the `medium` preset), trained on roughly 2.4M tokens of MG text after pretraining on general English. It is fluent, not knowledgeable: it still states MG facts confidently and wrongly.

## What this is

The end-to-end stack:

1. **Scrape** the PistonHeads MG (`f=81`) and Rover (`f=54`) sub-forums, `robots.txt`-aware via `urllib.robotparser`. I first used the dedicated MG enthusiast forums, then found their `robots.txt` asks AI crawlers not to, so I withdrew that model and rebuilt the corpus here. About a fifth of the threads are from the Rover sub-forum, so "classic MG" is the aim, not a strict description. (`data/scrape_pistonheads.py`, `data/discover_and_scrape_pistonheads.py`)
2. **Strip PII** with Macie-style regex redaction (`data/strip_pii.py`)
3. **Format as Q&A** dialogue so the model learns the chat pattern (`data/format_qa.py`)
4. **Synthesise case studies**: local Qwen3 (via LM Studio) or Claude reads each thread and extracts symptom, cause and fix Q&A pairs (`data/synthesize_qa.py`)
5. **Tokenise** with GPT-2 BPE into a memory-mapped numpy array (`data/prepare.py`)
6. **Train** a `medium` from-scratch GPT (`model/gpt.py`, `train.py`)
7. **Deploy** the trained checkpoint to a containerised Lambda behind API Gateway WebSocket + CloudFront + Cloudflare DNS (`lambda/`, `terraform/`, `deploy.sh`)
8. **Chat** from the frontend at the live URL: three pickable personas (Sunday Driver, Workshop Veteran, Restorer), each opening the conversation with a hand-written greeting (`frontend/index.html`, `lambda/handler.py`)

## Architecture notes worth knowing

**WebSocket 29s timeout workaround.** API Gateway WebSocket integrations are capped at 29 seconds. The Lambda handler splits the request: the WS-triggered invocation does fast bookkeeping (DynamoDB write, "thinking" ack), then async-invokes itself (`InvocationType="Event"`) to run the slow generation. The worker branch posts the response back via `apigatewaymanagementapi`. This avoids needing SQS for ~$0 extra cost.

**Config-from-checkpoint.** `lambda/handler.py` reads the GPT architecture config (block_size, n_layer, n_embd) from the checkpoint itself rather than hardcoding it. Train with `--config nano` or `--config small` or whatever else and the handler matches automatically, with no shape mismatches at load time.

**Personas.** Three personas (Sunday Driver, Workshop Veteran, Restorer), each with hand-written opening greetings. The old multi-turn priming block is no longer sent to the model (a plain prompt scored equal or better); it stays in `PERSONAS` only so `eval_mg.py` can still compare against it. Personas are selected once per session via the WS picker overlay and persist in DynamoDB keyed on `connection_id` (15-minute TTL).

**Frontend feel.** The UI is themed as a Haynes-style workshop manual: cream paper, navy ink, EB Garamond serif, FIG.-style message labels, MG octagon logo. While the model warms up, an italic "warming up" bubble cycles through classic-car-themed status lines ("Pulling the choke...", "Priming the carbs...", "Setting the tappet clearances..."). Source: `frontend/index.html`.

**PII redaction is regex-only.** It catches the high-volume cases (emails, names, URLs, forum usernames, phone numbers, addresses, postcodes) but is not security-grade. For anything more sensitive, swap in Presidio + spaCy NER.

## Evaluation

`eval_mg.py` runs a fixed-prompt eval (validation loss, empty-reply rate, coherence proxies). `judge_mg.py` has a local Qwen model, via LM Studio, score the replies blind on a 1 to 5 scale.

**Topic evals** (`evals/`, `python -m pytest`): a keyword lexicon says which of 9 topics a question or reply is about; `topic_report` shows how often each topic leaks into replies to other-topic questions; `release_gate` is the pass/fail gate for a new version (intrusion at most 1.5x a topic's question share, empty replies at most 5%, val loss within +0.05, judge within noise, plus a judge-vs-human agreement bar of kappa 0.6; Qwen scored 0.29 and Haiku 0.26 against hand scores, so judge results are indicative only). Question sets are in `data/eval/` (143 balanced questions, 36 short symptom questions). Live v7 baseline: 3 seeds in `eval_runs/bal_v7_*` and `sym_v7_*`, judge mean 2.68/5. Candidates tried, none deployed: v10 (51M + 5,088 grounded mechanics Q&A) cut leakage on the balanced set; an 82M model (`mid` preset, same data and recipe as v7) did about the same, so size alone did not help. Gate files: `eval_runs/gate_*_vs_v7.txt`.

[WRITEUP.md](WRITEUP.md) covers two bugs found by measuring: a validation split cut by position rather than by thread, and a trailing space in the prompt that made the live bot reply with nothing about two-thirds of the time.

## Project structure

```
mg-gpt/
├── README.md             ← you are here (current state)
├── CHANGELOG.md          ← reverse-chronological log of shipped changes
├── CLAUDE.md             ← instructions for AI sessions working on this repo
├── DEPLOYMENT.md         ← AWS infra deep-dive
├── GUIDE.md              ← longer-form learning guide
├── WRITEUP.md            ← what broke, what was measured, what didn't help
│
├── data/
│   ├── corpus_pistonheads/      ← raw scraped PH threads (one .txt per topic)
│   ├── corpus_pistonheads_clean/← PII-redacted PH corpus
│   ├── corpus_pistonheads_qa/   ← reformatted as USER:/ASSISTANT: dialogue
│   ├── corpus_synth_qa/         ← LLM case-study synthesis (Phase 2)
│   ├── corpus_synth_qa_extra/   ← second-pass Q&A from the same threads (train-only)
│   ├── corpus_wikipedia_mg/     ← MG Wikipedia articles, CC BY-SA 4.0 (manifest.json = attribution)
│   ├── corpus_wikipedia_qa/     ← grounded Q&A from those articles (train-only)
│   ├── corpus_wikipedia_mechanics{,_b}/ ← car-mechanics Wikipedia articles, CC BY-SA 4.0 (not yet trained on)
│   ├── corpus_tm9_8000/         ← US Army TM 9-8000 automotive principles (US government work; not yet trained on)
│   ├── eval/                    ← eval question sets (balanced, written top-ups, symptom-style)
│   ├── _archive_mgexp/          ← v5-era mgexp corpus + scrapers (withdrawn)
│   ├── scrape_forums.py         ← generic forum-scraper engine
│   ├── scrape_pistonheads.py    ← PH-specific scraper (robots.txt-aware)
│   ├── discover_and_scrape_pistonheads.py ← walk PH indexes f=81 (MG) + f=54 (Rover)
│   ├── strip_pii.py             ← regex-based PII redaction (Macie-style)
│   ├── format_qa.py             ← turn forum posts into Q&A pairs
│   ├── synthesize_qa.py         ← LLM extracts case-study Q&A (Qwen / Claude)
│   ├── prepare.py               ← tokenise MG corpus → train.bin / val.bin (split by thread; also adds the train-only extra dirs)
│   ├── prepare_pretrain.py      ← stream a general-English pretraining set from Common Pile (license-filtered)
│   ├── fetch_wikipedia_mg.py    ← MG-related Wikipedia articles (CC BY-SA 4.0) with attribution manifest
│   ├── synthesize_wiki_qa.py    ← grounded Q&A from those articles (local Qwen)
│   ├── synthesize_mech_qa.py    ← grounded Q&A from mechanics text, MOT manual, TM 9-8000 (local Qwen; output `data/corpus_mech_qa/` is not committed, licences)
│   └── synthesize_qa_extra.py   ← second-pass extra Q&A from forum threads (train-only)
│
├── model/
│   ├── gpt.py                   ← the transformer (~200 lines)
│   └── config.py                ← nano / small / medium / mid (82M) presets
│
├── lambda/
│   ├── Dockerfile               ← container image for Lambda (model code + ckpt baked in)
│   ├── handler.py               ← WebSocket handler: personas, generation, reply clean-up
│   └── requirements.txt         ← torch, tiktoken, numpy
│
├── checkpoints/
│   └── _archive_v5/             ← v5 training checkpoints (withdrawn)
│
├── terraform/                   ← all AWS + Cloudflare infra as code
├── frontend/index.html          ← chat UI (Workshop Manual theme)
├── notebooks/
│   └── transformer_lens_explore.ipynb  ← load ckpt into TransformerLens; viz attention + logit lens
├── deploy.sh                    ← build → push → terraform apply → S3 sync
├── train.py                     ← training loop: cosine LR, early stopping, --init-from for fine-tuning
├── evals/                       ← topic lexicon, topic_report, question builder, Qwen labeller, release_gate (tests in tests/)
├── eval_mg.py                   ← fixed-prompt eval: val loss, empty-reply rate, coherence proxies
├── judge_mg.py                  ← blind LLM-judge (local Qwen via LM Studio) scoring eval_mg replies 1-5
└── generate.py                  ← interactive local inference
```

The scraped forum text, the checkpoints and the archived v5 data are not in this repo. The forum text belongs to the people who wrote it, so re-scrape it with the scripts above if you want to rebuild the corpus.

## Quickstart (local training only)

```bash
pip install -r requirements.txt

# Build the corpus (skip if you already have data/corpus/*.txt)
python data/discover_and_scrape_pistonheads.py --max-index-pages 5 --max-thread-pages 20

# Redact PII, format as Q&A, tokenise
python data/strip_pii.py
python data/format_qa.py
python data/prepare.py

# Train
python train.py --config small

# Chat locally
python generate.py --prompt "USER: My MGB starts hard in winter.\nASSISTANT: "
```

## Deploy to AWS

```bash
# Requires: AWS CLI authed, terraform installed,
# CLOUDFLARE_API_TOKEN set or ~/.config/cloudflare/token present
./deploy.sh
```

`deploy.sh` builds the Docker image (with `checkpoints/ckpt_best.pt` baked in; for v7 this is a weights-only copy of `checkpoints/v7_finetune/ckpt_best.pt`), pushes to ECR, applies Terraform, force-updates the Lambda function code, uploads the frontend HTML to S3, and invalidates the CloudFront cache. See `DEPLOYMENT.md` for the infra deep-dive. If `terraform apply` fails on the Cloudflare token, the manual route works: docker build and push, `aws lambda update-function-code --publish`, then `aws s3 cp` the HTML (with `__WS_ENDPOINT__` substituted) and a CloudFront invalidation.

**Question logging.** The chat page has a "log questions for training and eval" tick box, ticked by default. When ticked, the question text only is written to the Lambda's CloudWatch log as `QLOG {...}`; read it with `aws logs filter-log-events --log-group-name /aws/lambda/mg-gpt-handler --filter-pattern QLOG`.

**Licence note.** The UK MOT inspection manual text, if used, carries: "Contains public sector information licensed under the Open Government Licence v3.0."


## Hardware

| Config   | Params | Hardware       | ~Time to coherence |
|----------|--------|----------------|--------------------|
| `nano`   | 1.5M   | CPU            | 30 min             |
| `small`  | 10M    | MPS (M-series) | 2–4 hrs            |
| `medium` | ~51M   | MPS (M-series) | overnight          |
| `mid`    | ~82M   | MPS (M-series) | ~16 h pretrain + ~4 h fine-tune |

Nano works on CPU. Small benefits from MPS acceleration. The model genuinely needs ~2M+ training tokens before `small` stops overfitting badly.
