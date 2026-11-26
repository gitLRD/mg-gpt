# Design: topic-collapse evals and a fair size comparison (plan A)

Status: **draft, awaiting owner approval.** Nothing in this doc has been built.
Date: 2026-11-26. Hardware: M2 Pro, 32 GB, PyTorch MPS, fp32. No cloud GPU for now.

## Intent and why

mg-llm is becoming a case study in *evaluating* LLMs, before release (is this
version good enough to ship?) and after (is it still good, did anything change?).
The owner's own testing of live v7 found it "loves fuel and nothing else". None of
the existing metrics (val loss, keyword hits, empty-reply rate, 1-5 judge) flagged
it. Before spending ~20 h of compute on a bigger model, find out what is actually
wrong, so the bigger model is a test of a hypothesis rather than a hope.

## What we measured (read-only, from saved generations and the corpora)

Keyword-topic counts, crude (regex lexicon in the session scratchpad, not in the
repo yet). 180 v7 replies (3 seeds x 30 questions x 2 prompt modes), 30 distinct questions.

1. **Collapse is real but not fuel-only.** Mentions per 1,000 words:

   | | fuel | cooling | electrical | gearbox | body | suspension | brakes |
   |---|---|---|---|---|---|---|---|
   | test questions | 7.1 | 5.3 | 5.3 | 7.1 | 7.1 | 1.8 | 1.8 |
   | v7 replies | **14.9** | **14.8** | 7.7 | 7.0 | 7.0 | 3.6 | 1.8 |
   | training: synthetic Q&A answers | 5.6 | 8.5 | 6.5 | 5.8 | 4.9 | 7.2 | 2.9 |
   | training: forum threads | 2.2 | 2.1 | 2.2 | 3.1 | 2.2 | 3.1 | 1.4 |

   About 23% of v7's replies to non-fuel questions contain fuel words (v6 23%,
   MG-only 14%, v8 23%, v9 26%). So fuel *and* cooling are over-produced (2-3x
   the training rate); suspension is under-produced (half).
2. **The data does not explain it.** Fuel is no more common than cooling or
   electrical in the forum threads, and in the synthetic Q&A cooling is higher
   than fuel. The model amplifies a couple of common topics beyond what it saw.
3. **It did not change across versions** (v6, v7, v8, v9 all ~23-26%), so it is
   not something the experiments so far moved.

Caveats: keyword lexicon is crude ("tank", "jet", "lean" may inflate fuel);
n = 30 questions; question-side baseline is rough. Temperature 0.6 in both eval
and live site, so sampling is consistent.

## Goals

- A **topic-spread metric**: distribution of reply topics vs question topics, plus
  an "off-topic intrusion rate" (e.g. fuel words in answers to non-fuel questions).
- A **bigger, balanced question set** (target ~150, each labelled with its topic),
  drawn from held-out threads, so one topic cannot hide in 30 questions.
- **Judge validation**: hand-score ~50 replies, report judge-vs-human agreement,
  so judge numbers have a stated trust level.
- A **fair size comparison**: ~51M (existing v7 recipe) vs ~124M, same data and
  same token budget, scored on all of the above.
- A **pass/fail release gate** written down before the results are seen.

## Non-goals

- Cloud GPU, hosted-LLM swap, or a sizes sweep beyond two (a middle ~90M size only
  if time allows; not timed yet).
- Changing the live site or deploying anything.
- Post-release monitoring (separate, later phase; needs this gate first).

## Anti-goals (would make this a failure even if it "works")

- Picking the metric after seeing which model wins.
- Declaring a size effect from one seed, or from judge gaps inside the noise (~0.2).
- Retraining on test questions (held-out stays held-out; val split stays by thread).

## Approaches considered

- **A (chosen): diagnose, then size A/B on the Mac.** Cheapest route to a real answer.
- B: just train 124M and look afterwards. Rejected: would repeat v9 (lots of compute,
  no way to explain the result), and the weakest eval story.
- C (fallback): port training to MLX for bf16 speed. Only if A shows we need more
  speed; try PyTorch MPS autocast for an hour first. Needs weights converted back to
  PyTorch for Lambda, and a 51M both-ways run to prove the stacks agree.

## Measured budget (timing test, 2026-11-26, LM Studio open, so indicative)

| model | params | s/step | tokens/s | MPS memory |
|---|---|---|---|---|
| medium (live v7) | 51.5M | 2.54 | 6,461 | 11.6 GB |
| 12 layers / 12 heads / 768 dim / 1024 ctx | 124.4M | 5.48 | 2,991 | 18.5 GB |

One step = 16,384 tokens (batch 4 x accum 4 x block 1024).
- **Pretrain 124M on the same 196M tokens** (12,000 steps): ~18.3 h. (51M took 8.7 h:
  12,000 x 2.54 s matches.)
- **MG fine-tune 124M**, ~3,750 steps like v7: ~5.7 h. Total ~24 h for the big model.
- Memory 18.5 GB of 32 GB: fits, but quit LM Studio during runs (it also holds GPU
  memory and the judge needs it, so judging is a separate phase).
- Disk: 75 GB free. Checkpoints with optimizer state ~1.5 GB each; use
  `--save-interval 1000` and prune: ~15 GB worst case.
- 51M baseline needs no retraining (v7 exists); extra seeds of the fine-tune only
  (~2.6 h each at 51M, ~5.7 h at 124M).

## Proposed pass/fail gate (to be confirmed before any results)

A model version ships only if, against the previous live version, on the balanced
question set with 3 seeds:
1. Off-topic intrusion rate for any single topic is no higher than 1.5x that topic's
   share in the questions.
2. Judge mean >= previous live (within the measured seed noise counts as a tie, not a win).
3. Empty-reply rate <= 5%; val loss no worse than +0.05.
4. Judge agrees with hand scores well enough to be quoted (threshold set after the
   50-reply check; below it, judge numbers are reported as indicative only).

## Open decisions for the owner

1. **Approve or edit** this design and the gate thresholds.
2. **Middle size (~90M)?** Adds a point on the curve; cost not yet measured.
3. **Fix the CHANGELOG?** It says the 12K-step pretrain "covered a quarter" of the
   data; the log and code say ~1 pass (12,000 x 16,384 = 196.6M tokens). If so, v9's
   "continued pretrain" was ~2 passes, not 1 vs 0.25. I would correct it after checking
   with you.
4. **Start the 124M pretrain** (~18 h) once approved, with LM Studio closed.

## Left to the builder

Exact lexicon vs an LLM topic labeller for the metric (start with lexicon + a spot
check), question-set construction details, seed counts if time is tight, plot design.
