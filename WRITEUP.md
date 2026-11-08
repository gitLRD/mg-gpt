# MG-GPT: what I learned making a tiny from-scratch LLM less wrong

A 51M-parameter GPT trained from scratch on classic-MG forum text, served live
at https://mg-gpt.desforgeslabs.com from an AWS Lambda (CPU only). This note
covers the v7 round of work (Nov 2026): what was broken, what I measured, and
what did not work. Numbers are small-sample (30 questions, a few seeds); where
a difference is inside the noise I say so.

## Summary

- Two real bugs, both found by measuring rather than guessing: a **leaky
  validation split** and a **trailing-space prompt bug** that made the live
  bot reply with nothing about two-thirds of the time.
- **Pretraining on general English first, then fine-tuning on MG, beat
  training on MG alone** (val loss 3.69 vs 4.26 on identical held-out threads).
- Three follow-ups (more MG data, longer pretraining, removing the persona
  priming block) produced **no reliable gain**. v7 stayed live.
- The model still states MG facts confidently and wrongly. It is fluent, not
  knowledgeable: 51M parameters and 2.4M tokens of MG text are the limit.

## 1. Leaky validation split

The original split cut the token stream by position. Validation turned out to
be 100% synthetic Q&A, and 1,588 of those 1,625 threads were also in the
training set as raw dialogue. The reported val loss (3.14) measured memory, not
generalisation. The split is now **by thread** (md5 of the file name), so a
thread is wholly train or wholly val. Every later number uses that split, and
the val set stayed byte-identical across all experiments so they compare.

Lesson: if val loss looks too good, check what val actually contains.

## 2. Empty live replies: a trailing space

The live bot returned empty replies about 67% of the time. I extracted the
checkpoint from the deployed image and reproduced it locally. The handler
ended the prompt with `"ASSISTANT: "` (trailing space). GPT-2 BPE normally
fuses the space onto the next word (` The`), so a prompt ending in a bare space
token puts the model in a spot it almost never saw in training. Dropping the
space took empty replies from 67% to 0% on the same checkpoint.

## 3. Pretrain, then fine-tune

An MG-only model has to learn English and MG from 2.4M tokens. Instead:

| Run | Held-out val loss |
|---|---|
| MG-only, from scratch | 4.26 |
| General-English pretrain (196M-token licence-filtered Common Pile slice), then MG fine-tune | **3.69** |

The pretrain used only public-domain / CC-licensed text, with a manifest.
Stack Exchange was ruled out (its dump terms exclude LLM training).

I judged answers with a blind 1-5 "makes sense" rubric using a local Qwen
model. On the plain prompt the fine-tune scored 2.70 vs 2.30 for MG-only. The
old v6 model scored 2.80, but it had trained on those held-out questions, so that number
is inflated; on the fixed questions it hadn't seen, the fine-tune led.

## 4. Things that did not help

All compared on 30 held-out questions x 3 seeds x 2 prompt styles, scored by
**two** judges (qwen3-vl-30b and qwen3-coder-30b) because the same Qwen wrote
the synthetic training data and might favour its own style.

| Change | Result |
|---|---|
| **More MG data** (+2,713 extra forum Q&A, 53 Wikipedia articles with grounded Q&A; +20% train tokens) | Val 3.65 vs 3.69; judge means within 0.1 of v7 in all four settings. No reliable gain. |
| **Longer pretrain** (12K -> 24K steps; general val 3.75 -> 3.55) then fine-tune | Judge **lower** in all four settings (about -0.15 to -0.35). Better general English did not carry to MG answers. |
| **Persona priming block** in the live prompt | Plain prompt scored >= priming in 4/4 runs, though the gap (~0.2) is near seed noise. Removed anyway: it also saves ~500 prompt tokens. |

One finding worth noting: the first pretrain only read about a quarter of its
196M-token dataset (12K steps x 4 x 1024 tokens), so it was compute-limited,
not data-limited. More pretraining did lower general loss, which made the
fine-tune result a clean negative rather than a "not enough training" excuse.
Why the fine-tune got worse is a guess (a more settled start moved less before
early stopping), not something I tested.

## 5. Caveats

- n = 30 questions per cell and a ~0.2 seed-to-seed swing: only gaps larger
  than that mean anything.
- LLM-as-judge is crude and the judges are small local models. They agree with
  each other, which helps, but neither is checked against a human.
- Synthetic Q&A written by a small model can be confidently wrong; I spot-checked
  a few Wikipedia-grounded pairs (small inconsistencies found) but not the forum ones.
- The Wikipedia text is CC BY-SA 4.0; `data/corpus_wikipedia_mg/manifest.json`
  records title, URL and revision for attribution. The forum-derived data is
  not published in the repo.

## 6. Hosting notes

The model fits the existing Lambda (3 GB, CPU-only) comfortably: 228 MB of
weights. A ~120M model would also fit, but I dropped it: training time on one
laptop is the limit and the experiments above give no sign that more general
pretraining helps MG answers.

## Reproducing

See `README.md` for the pipeline and `CHANGELOG.md` for every run, with
file paths for logs and eval outputs (`eval_runs/`, `data/logs/`).
