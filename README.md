# MG-GPT 🚗
### A tiny transformer trained on classic MG knowledge, deployed as a live chatbot

A from-scratch GPT trained on classic MG forum posts, served from AWS Lambda
behind a WebSocket-based chat UI. Built for learning — every line is meant to
be read and understood — and live at **https://mg-gpt.desforgeslabs.com**.

---

## What this is

The end-to-end stack:

1. **Scrape** the mgexp.com forums (`data/scrape_forums.py`, `bulk_scrape.py`, `discover_and_scrape.py`)
2. **Strip PII** with Macie-style regex redaction (`data/strip_pii.py`)
3. **Format as Q&A** dialogue so the model learns the chat pattern (`data/format_qa.py`)
4. **Tokenise** with GPT-2 BPE into a memory-mapped numpy array (`data/prepare.py`)
5. **Train** a small from-scratch GPT (`model/gpt.py`, `train.py`)
6. **Deploy** the trained checkpoint to a containerised Lambda behind API Gateway
   WebSocket + CloudFront + Cloudflare DNS (`lambda/`, `terraform/`, `deploy.sh`)
7. **Chat** from the frontend at the live URL — three pickable personas
   (Sunday Driver, Workshop Veteran, Restorer), each opens the conversation
   with a hand-written greeting (`frontend/index.html`, `lambda/handler.py`)

---

## Project structure

```
mg-llm/
├── README.md             ← you are here (current state)
├── CHANGELOG.md          ← reverse-chronological log of shipped changes
├── CLAUDE.md             ← instructions for AI sessions working on this repo
├── DEPLOYMENT.md         ← AWS infra deep-dive
├── GUIDE.md              ← longer-form learning guide
│
├── data/
│   ├── corpus/           ← raw scraped forum threads (one .txt per thread)
│   ├── corpus_clean/     ← PII-redacted corpus
│   ├── corpus_qa/        ← reformatted as USER:/ASSISTANT: dialogue
│   ├── scrape_forums.py  ← single-thread scraper (paginated, polite)
│   ├── bulk_scrape.py    ← scrape a hand-curated URL list
│   ├── discover_and_scrape.py ← walk forum indexes, find new threads
│   ├── strip_pii.py      ← regex-based redaction (Macie-style)
│   ├── format_qa.py      ← turn forum posts into Q&A pairs
│   ├── prepare.py        ← tokenise to train.bin / val.bin
│   └── meta.json         ← split sizes, vocab info
│
├── model/
│   ├── gpt.py            ← the transformer (~200 lines)
│   └── config.py         ← nano / small / medium presets
│
├── lambda/
│   ├── Dockerfile        ← container image for Lambda
│   ├── handler.py        ← WebSocket handler, personas, async self-invoke
│   └── requirements.txt  ← torch, tiktoken, numpy
│
├── terraform/            ← all AWS + Cloudflare infra as code
├── frontend/index.html   ← static chat UI (Haynes-manual themed) via S3 + CloudFront
├── deploy.sh             ← build → push → terraform apply → S3 sync
├── train.py              ← training loop with cosine LR, eval, checkpoints
└── generate.py           ← interactive local inference
```

---

## Quickstart (local training only)

```bash
pip install -r requirements.txt

# Build the corpus (skip if you already have data/corpus/*.txt)
python data/discover_and_scrape.py --max-index-pages 5 --max-thread-pages 20

# Redact PII, format as Q&A, tokenise
python data/strip_pii.py
python data/format_qa.py
python data/prepare.py

# Train
python train.py --config small

# Chat locally
python generate.py --prompt "USER: My MGB starts hard in winter.\nASSISTANT: "
```

---

## Deploy to AWS

```bash
# Requires: AWS CLI authed, terraform installed,
# CLOUDFLARE_API_TOKEN set or ~/.config/cloudflare/token present
./deploy.sh
```

`deploy.sh` builds the Docker image (with the latest `checkpoints/ckpt_best.pt`
baked in), pushes to ECR, applies Terraform, force-updates the Lambda function
code, uploads the frontend HTML to S3, and invalidates the CloudFront cache.
See `DEPLOYMENT.md` for the infra deep-dive.

---

## Architecture notes worth knowing

**WebSocket 29s timeout workaround.** API Gateway WebSocket integrations are
capped at 29 seconds. The Lambda handler splits the request: the WS-triggered
invocation does fast bookkeeping (DynamoDB write, "thinking" ack), then
async-invokes itself (`InvocationType="Event"`) to run the slow generation.
The worker branch posts the response back via `apigatewaymanagementapi`. This
avoids needing SQS for ~$0 extra cost.

**Config-from-checkpoint.** `lambda/handler.py` reads the GPT architecture
config (block_size, n_layer, n_embd) from the checkpoint itself rather than
hardcoding it. Train with `--config nano` or `--config small` or whatever else
— the handler matches automatically. No more shape mismatches at load time.

**Personas.** Three personas (Sunday Driver, Workshop Veteran, Restorer), each
with a multi-turn priming example block and hand-written opening greetings.
Personas are selected once per session via the WS picker overlay and persist
in DynamoDB keyed on `connection_id` (15-minute TTL).

**Frontend feel.** The UI is themed as a Haynes-style workshop manual: cream
paper, navy ink, EB Garamond serif, FIG.-style message labels, MG octagon
logo. While the model warms up, an italic "warming up" bubble cycles through
classic-car-themed status lines ("Pulling the choke...", "Priming the
carbs...", "Setting the tappet clearances..."). Source: `frontend/index.html`.

**PII redaction is regex-only.** Catches the high-volume cases (emails, names,
URLs, forum usernames, phone numbers, addresses, postcodes) but is not
security-grade. For anything more sensitive, swap in Presidio + spaCy NER.

---

## Hardware

| Config   | Params | Hardware       | ~Time to coherence |
|----------|--------|----------------|--------------------|
| `nano`   | 1.5M   | CPU            | 30 min             |
| `small`  | 10M    | MPS (M-series) | 2–4 hrs            |
| `medium` | 50M    | MPS (M-series) | overnight          |

Nano works on CPU. Small benefits from MPS acceleration. The model genuinely
needs ~2M+ training tokens before `small` stops overfitting badly.
