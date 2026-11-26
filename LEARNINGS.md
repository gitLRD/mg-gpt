# Learnings

Raw notes for extending WRITEUP.md later. Newest at the bottom. Plain facts: what we
saw, why it matters, and what is a guess. Not a changelog (that is CHANGELOG.md).

## 2026-11-26: starting the eval-focused phase

**Goal change.** mg-llm is now also a case study in *evaluating* LLMs: pre-release
gates (is this version good enough to ship?) and post-release monitoring (is it still
good, and did anything change?). The owner's own use of the live site found the v7
model "loves fuel and nothing else", which the existing metrics did not catch.

**What the existing evals already caught (keep for the write-up).**
- Val split leaked (cut by token position, so val was synth Q&A of threads also in
  train). v6's val 3.14 was not comparable to later numbers.
- More general-English pretraining (v9) lowered general val loss but made MG answers
  worse (judge -0.15 to -0.35). Loss on one text type does not predict quality on another.
- Judge noise: gaps of ~0.2 are the same size as the seed-to-seed swing, n=30.

**Blind spot found.** Val loss, keyword hits, empty-reply rate and a 1-5 "makes sense"
judge all missed topic collapse (answering everything with fuel). Lesson: a metric only
catches what it was designed to look for. Needs a topic-spread check and a data audit.

**Where size might (not) matter. Unverified.** The fine-tune set is ~2.9M tokens for a
51M model. Data volume/balance may be the limit as much as parameter count. Plan A
tests this with a fair size comparison on the Mac rather than assuming.

**Hardware facts.** M2 Pro, 32 GB, fp32 on MPS. 51M pretrain: ~8.7 h for 12K iters,
~49M tokens seen (~1,500 tok/s, estimate from notes). A ~120M model is estimated at
~2.3x the compute per token, so ~20 h for the same reading (guess until timed).
Cloud GPU deliberately avoided for now (owner decision).

**Measured: the "fuel" collapse (from saved generations, keyword lexicon, crude).**
v7 replies say fuel words ~15x per 1,000 words vs 7.1 in the test questions and 5.6 in
the synthetic Q&A it trained on; cooling is amplified the same way (14.8 vs 8.5);
suspension is under-produced. ~23% of replies to non-fuel questions contain fuel words,
and it is the same for v6, v7, v8 and v9. So it is not only fuel, and the data does not
explain it (fuel is no more common than cooling in the corpora). A small model amplifies
a few common topics beyond their training share. Lesson for the write-up: the 20 fixed
questions + keyword hits + a 1-5 judge could not see this; a topic-distribution metric
can. Sampling is the same (temperature 0.6) in eval and live, so that is ruled out as
the difference.

**Measured: training speed (replaces my guess).** 51M: 2.54 s/step, 6,461 tok/s, 11.6 GB.
124M (12 layer, 768 dim): 5.48 s/step, 2,991 tok/s, 18.5 GB. My earlier "~1,500 tok/s"
came from a wrong premise (below). Pretraining 124M on 196M tokens ~18 h, fine-tune
~6 h. LM Studio was open during timing, so treat as indicative.

**Likely error in the changelog (not fixed yet).** It says the 12K-step pretrain
"only covered a quarter" of the 196M-token set. The log says batch 4 x accum 4 x 1024 =
16,384 tokens/step, and 12,000 steps took 31,208 s, so it saw ~196M tokens, about one
pass (random windows, so ~63% unique). If so, v9's extra 12K steps was ~2 passes, not
"1 vs 0.25". Check with the owner before editing the changelog.

**Process notes.** `rsync --checksum` over an SMB mount reads every file back over the
network; for a 4 GB offload, comparing file count and total bytes was enough and fast.
Deleted extracted frame JPEGs but kept the small manifest/detection files (26 MB), which
are costly to regenerate.

**Housekeeping.** Disk was 96% full (40 GB free). Freed space in pothole-reporter
(frame JPEGs, build outputs, proxies, datasets deleted; rfdetr checkpoints offloaded to
the NAS with a checksum verify before local deletion).
