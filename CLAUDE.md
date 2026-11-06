# Instructions for Claude sessions working on this repo

## How to explain things to the owner

Explain like the owner is about 10 and has played with LLMs a bit: they know
what a prompt, a token and "training" are, but not the maths or the jargon
around them.

- Say what happened and *why it matters* in plain words first. Technical
  detail comes after, if at all.
- Use an everyday picture where it helps (cars and workshops fit this project).
- The first time a term shows up (val loss, overfitting, BPE, warm start),
  explain it in one short sentence.
- Keep it accurate. Simple does not mean vague: still give the real numbers,
  and say clearly when something is a guess or not checked.
- This is for chat. Code, README, CHANGELOG and docs keep their normal style.

## Keep README.md and CHANGELOG.md up to date

Two living docs, two different jobs:

- **README.md** — "what is this project right now". Always reflects the
  current state. Overwrite/edit in place.
- **CHANGELOG.md** — "what we shipped and why". Append-only log, newest
  entries at the top under the matching section ("In progress" while
  unshipped, then promoted into a versioned section when shipped). Each
  entry is one line of *what* and one line of *why* — keep it tight, link
  to file paths for detail.

After any shipped change that's user-visible or architecturally notable,
update both. The list below covers when:

## When to update README.md

`README.md` is the canonical "what is this project right now" document. After
any of the following changes, update it in the same session before ending:

- **New script in `data/`, `lambda/`, `model/`, `terraform/`, or `frontend/`** —
  add it to the project-structure tree with a one-line description.
- **New pipeline step** (e.g. adding a deduper between scrape and PII strip) —
  update the numbered pipeline list at the top.
- **Architecture or deployment change** — update the "Architecture notes worth
  knowing" section. WS protocol changes, IAM changes, new AWS resources,
  Cloudflare changes, Docker changes, config-loading changes all count.
- **New persona, or change to how personas work** — update the personas note.
- **New `--config` preset or significant training behaviour change** — update
  the hardware table and any nearby prose.
- **Live URL changes** — update the top of the README.

Treat README.md as part of "definition of done" for changes that touch the
above. Don't end a working session with `git status` showing source changes
without also updating README.md if the change is user-facing.

## When to update CHANGELOG.md

Append a line to the **In progress** section at the top *while* working on
something non-trivial. When that work ships (deployed, merged, or otherwise
"done"), move the line into the latest version section (or start a new one
if the change is significant enough to warrant it).

Entries should be:
- **What changed** in one line, with file refs in parens
- **Why** in one short sentence

Don't log internal scratch work (debugging steps, dead ends, refactors that
got reverted). Do log: shipped features, architectural decisions, bug fixes
with non-obvious causes, dependency bumps, config changes that future-you
would want to understand.

The CHANGELOG is for the *owner* — not a marketing changelog. Brevity > polish.

## What goes in README vs DEPLOYMENT.md vs GUIDE.md

- **README.md** — current-state, scannable: what the project is, how it's
  organised, how to run it end-to-end, key architecture decisions in 1–2
  sentences each.
- **DEPLOYMENT.md** — AWS infra deep-dive: which resources, why, how to
  bootstrap from scratch, gotchas.
- **GUIDE.md** — longer-form learning material: explanations of the
  transformer architecture, why decisions were made.

If you're adding documentation, put it in the right place. Don't bloat README
with deep dives — link out instead.

## Project context

This is a personal learning project ("a stupid project to share with a couple
of friends" per the owner). Keep changes proportional to that scope:

- Don't add enterprise patterns (DI containers, factories, abstract base
  classes) where a function would do.
- Don't add observability beyond what's useful for one user debugging.
- Cost matters more than scale — favour serverless-cheap over horizontally
  scalable. The site is at https://mg-gpt.desforgeslabs.com and runs on a
  Lambda + DynamoDB + S3 + CloudFront budget.
- The model is intentionally tiny. Don't suggest swapping to a hosted LLM
  unless explicitly asked — the point is to train from scratch.

## Security constraints

- The Cloudflare API token lives at `~/.config/cloudflare/token` (gitignored
  user-level config). Never commit it. `deploy.sh` loads it from there or
  from the `CLOUDFLARE_API_TOKEN` env var.
- The corpus contains scraped forum content. PII is redacted by
  `data/strip_pii.py` before training — don't bypass that step when retraining
  for deployment.
