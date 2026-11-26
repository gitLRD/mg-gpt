# Topic Evals and Size Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a topic-spread eval, a balanced labelled question set, a validated judge and a coded release gate, then use them to compare 51M / 82M / 124M models (plus a car-data arm) at a fixed 196M-token pretrain budget.

**Architecture:** Pure-Python eval helpers live in a new `evals/` package with pytest tests in `tests/`. Existing scripts (`eval_mg.py`, `judge_mg.py`, `train.py`, `data/prepare.py`) get small flag additions, not rewrites. The gate is a script that reads eval/judge JSONL and a manifest, with thresholds fixed in code and committed before any new results exist.

**Tech Stack:** Python 3, numpy, pytest (new dev dependency), PyTorch MPS (fp32), tiktoken, LM Studio (local Qwen `qwen/qwen3-vl-30b`, OpenAI-compatible API on `localhost:1234`).

**Spec:** `docs/superpowers/specs/2026-11-26-size-comparison-eval-design.md` (approved 2026-11-26). Read it first; this plan does not repeat its reasoning.

## Global Constraints

- Hardware: M2 Pro 32 GB, PyTorch MPS, fp32. No cloud GPU.
- One step = 16,384 tokens (batch 4 x accum 4 x block 1024). Pretrain budget is fixed: `--max-iters 12000` = ~196M tokens, data `data/v7_pretrain`, for every size.
- Model configs: 51M = existing `medium` (8L/8H/512); 82M = 10L/10H/640/ctx 1024; 124M = 12L/12H/768/ctx 1024. dropout 0.1 in preset (pretrain run with `--dropout 0.0`, as v7).
- Use `--save-interval 1000` on every long run; prune checkpoints as in Task 5 (disk 68 GB free, ~1.5 GB per full ckpt).
- Quit LM Studio before every training run. LM Studio is only open for Q&A synthesis and judging, which are separate phases.
- Sampling for all evals: temperature 0.6, top-k 50, top-p 0.9, rep-penalty 1.15 (eval_mg.py defaults), mode `bare` (the live prompt has no persona priming since Lambda v21), 3 seeds (0, 1, 2).
- Held-out stays held-out: val split is by thread (`val_threads` in `meta.json`); nothing from a val thread enters any train set, question synthesis for training, or rebalancing.
- Gate thresholds are fixed in code and committed BEFORE any candidate result is generated (Task 3). No metric is changed after seeing results.
- A single seed is never reported as a size effect. Judge gaps inside the measured seed noise are reported as ties.
- Attribution: README and manifests carry the OGL v3.0 line for the MOT manual and the Wikipedia revision ids (CC BY-SA 4.0). Derived Q&A stays CC BY-SA 4.0.
- No personal email or contact address in any file, User-Agent, or request. Use a generic project name.
- Not in scope: deploying, touching the live site, the Lambda image, cloud GPU, MLX port. (Lambda reads architecture from the checkpoint; whether a 124M model fits the current Lambda memory is a separate question for any future deploy.)
- Repo `CLAUDE.md`: README.md edited in place, CHANGELOG.md append at top (one line what, one line why), no enterprise patterns, plain-words chat explanations to the owner.

## Review Focus

Inputs and conditions the spec implies but does not spell out, most likely first. Each has a test in the named task.

1. Questions that match two topics or none (e.g. "brake fluid leaking onto the fuel pipe") must be dropped from the balanced set, not forced into one topic. Task 2.
2. Replies that are empty or mention no topic at all must not crash or divide by zero in the topic metrics; a topic with no non-topic questions yields "n/a", not 0%. Task 1.
3. Lexicon false positives the spec calls out ("tank", "jet", "lean"): "septic tank", "jetty", "lean-burn" must not count as fuel. Task 1.
4. The gate must fail closed: missing seed files, fewer than 2 judge files for a model, or an unset agreement threshold exits non-zero with a message, never "PASS". Task 3.
5. Rebalancing must never copy a val-thread document into train, and repeat factors are capped. Task 7.

## File Structure

| Path | Responsibility |
|---|---|
| `evals/__init__.py` | package marker |
| `evals/topics.py` | topic lexicon, `tag_topics`, `primary_topic`, rates, intrusion rates |
| `evals/topic_report.py` | CLI: eval JSONL -> topic table (text + JSON) |
| `evals/questions.py` | build + load the balanced labelled question set |
| `evals/judge_agreement.py` | sample replies for hand scoring; compute judge-vs-human agreement |
| `evals/release_gate.py` | pass/fail gate, thresholds as constants |
| `evals/rebalance.py` | pure function: topic counts -> repeat factors |
| `evals/plot_topics.py` | bar chart of topic spread per model |
| `scripts/prune_checkpoints.py` | keep ckpt_best + N newest numbered checkpoints |
| `data/synthesize_mechanics_qa.py` | grounded Q&A from MOT manual + Wikipedia mechanics (local Qwen) |
| `data/build_rebalanced_extra.py` | writes upsampled train-only docs dir |
| `data/eval/balanced_questions.jsonl` | the ~150 labelled held-out questions (output of Task 2) |
| `tests/` | pytest tests, one file per `evals/` module plus config/prune/synth tests |
| modify `eval_mg.py` | `--questions`, topic in JSONL rows, `--summary-json` |
| modify `model/config.py`, `train.py` | `mid82`/`large124` configs, `--seed` |
| modify `data/prepare.py` | `--extra-dirs` |

---

### Task 1: Topic-spread metric and intrusion rate

**Files:**
- Create: `evals/__init__.py` (empty), `evals/topics.py`, `evals/topic_report.py`, `tests/test_topics.py`
- Modify: `requirements.txt` (add `pytest>=8.0`)

**Interfaces:**
- Produces:
  - `TOPICS: dict[str, re.Pattern]` with keys `fuel, cooling, brakes, electrical, oil, body, gearbox, susp, price` (compiled, case-insensitive).
  - `tag_topics(text: str) -> set[str]`
  - `primary_topic(text: str) -> str | None` : topic with the strictly highest match count; `None` if no match or a tie.
  - `rates_per_1000_words(texts: list[str]) -> dict[str, float]` : mentions per 1,000 words over all texts (0.0 for all topics if no words).
  - `intrusion_rates(rows: list[dict]) -> dict[str, float | None]` : rows have `topic` (question label) and `clean` (reply). For each topic T: fraction of replies to questions whose `topic != T` that contain T; `None` if there are no such rows.
  - `question_shares(rows: list[dict]) -> dict[str, float]` : fraction of rows labelled with each topic.

Port the lexicon from the scratchpad `topics.py` (9 topics, same regex idea) with these changes: drop bare `tank`, `jet`, `lean`, `earth`; use `fuel tank`, `petrol tank`, `jets?` only when preceded by `main|slow|needle|su ` , `lean (mixture|running|burn)` only as phrase "lean mixture", `earth (strap|lead|wire|fault)|earthing`. Keep word boundaries on every term.

- [ ] **Step 1: Write failing tests in `tests/test_topics.py`**
  - `test_fuel_false_positives`: `tag_topics("the septic tank by the jetty is lean-burn free")` has no `fuel`.
  - `test_fuel_true_positives`: `"my SU carb main jet and fuel pump"` -> contains `fuel`.
  - `test_primary_topic_tie_and_none`: `primary_topic("hello there")` is `None`; `primary_topic("brake fluid near the fuel pipe")` is `None` (1 brakes vs 1 fuel hit... adjust text so counts tie exactly); `primary_topic("brakes caliper pads")` is `"brakes"`.
  - `test_rates_empty_and_zero_words`: `rates_per_1000_words([])` and `rates_per_1000_words([""])` return all zeros.
  - `test_rates_value`: 2 fuel words in a 1000-word text -> `fuel == 2.0`.
  - `test_intrusion_rate`: rows = 4 replies, 2 to `brakes` questions, 2 to `fuel` questions; 1 of the brakes-question replies mentions fuel -> `intrusion_rates(rows)["fuel"] == 0.5`; and for a topic with no non-topic rows (all rows labelled `fuel`) -> `intrusion_rates(rows)["fuel"] is None`.
  - `test_question_shares_sum`: shares over labelled rows sum to 1.0.
- [ ] **Step 2:** `python -m pytest tests/test_topics.py -v` -> FAIL (module missing).
- [ ] **Step 3:** Implement `evals/topics.py` per Interfaces. Count matches with `findall` for `primary_topic` and rates.
- [ ] **Step 4:** `python -m pytest tests/test_topics.py -v` -> all PASS.
- [ ] **Step 5:** Write `evals/topic_report.py`: `python -m evals.topic_report RUN.jsonl [RUN2.jsonl ...] [--json out.json]` prints, per file, rates per 1,000 words per topic, intrusion rate per topic, and the question shares; rows lacking `topic` are skipped for intrusion with a printed note. Sanity-run on the old v7 file: `python -m evals.topic_report eval_runs/v7_ft_s0.jsonl` (no `topic` field yet, so only the rates print). Expected: fuel ~15 and cooling ~15 per 1,000 words, matching the spec table within a few points (lexicon was tightened, so not identical; note the difference in LEARNINGS in Task 9).
- [ ] **Step 6: Commit** `git add evals tests requirements.txt && git commit -m "feat(evals): topic-spread metric and intrusion rate"`

---

### Task 2: Balanced labelled held-out question set

**Files:**
- Create: `evals/questions.py`, `tests/test_questions.py`, `data/eval/balanced_questions.jsonl` (generated)
- Modify: `eval_mg.py` (flags, row fields), `tests` as needed

**Interfaces:**
- Consumes: `evals.topics.primary_topic`, `TOPICS`.
- Produces:
  - `build_balanced(candidates: list[tuple[str, str]], per_topic: int, seed: int) -> list[dict]` : `candidates` are `(thread_stem, question)`; keeps questions whose `primary_topic` is not `None`, takes up to `per_topic` per topic (deterministic shuffle by `seed`), at most 2 questions per thread, returns `{"q", "topic", "thread"}` dicts. Never raises on shortfall; returns what exists.
  - `load_questions(path: str) -> list[tuple[str, list[str], str]]` : `(question, [], topic)` triples.
  - `eval_mg.py --questions PATH` : replaces PROMPTS with this set (no keyword hits for it); every JSONL row gains `"topic"` (None for the old prompts).
  - `eval_mg.py --summary-json PATH` : writes `{"ckpt","val_loss","modes":{mode:{"n","empty_rate"}}}` per run.

Candidates come from `val_threads` in `data/v7_mgonly/meta.json` (same list as `data/meta.json`; check both are identical first), reading every `USER:` turn in `data/corpus_synth_qa/{stem}.txt`. Target `per_topic=17` over the 9 topics (~153 questions).

- [ ] **Step 1: Write failing tests** in `tests/test_questions.py`:
  - `test_ambiguous_dropped`: candidate `"brake fluid leaked onto the fuel pipe"` is not in output.
  - `test_per_topic_cap_and_thread_cap`: 30 brake questions from 3 threads, `per_topic=5` -> exactly 5 brakes, no thread appears more than twice.
  - `test_deterministic`: same inputs and seed -> identical output; different seed -> different order.
  - `test_load_questions_roundtrip`: write 2 rows to a tmp JSONL, `load_questions` returns `(q, [], topic)`.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement `build_balanced`, `load_questions`, and a `__main__` that reads val threads, writes `data/eval/balanced_questions.jsonl`, and prints the per-topic count table.
- [ ] **Step 4:** run tests, expect PASS.
- [ ] **Step 5:** Edit `eval_mg.py` for the three additions above (small diff; `prompts` tuples become `(q, kws, topic)` with `topic=None` for the legacy lists, update the unpack in the loop and `heldout_questions`).
- [ ] **Step 6:** Build the set: `python -m evals.questions`. Expected: roughly 17 per topic. If any topic has fewer than 12, do not lower the bar silently; report the table to the owner (this is a data-availability question, not a STOP for a long run) and proceed with the shortfall noted in the file's header comment in the README.
- [ ] **Step 7:** Spot-check labels: print 30 random rows (question + topic) and confirm by reading that at least 27 of 30 are labelled sensibly. If fewer, tighten the lexicon (tests first) and rebuild; if still below, stop and ask the owner about switching to an LLM labeller (spec "left to the builder").
- [ ] **Step 8:** Smoke-run eval on the live model: Do not run the full set here (long). Run `eval_mg.py` with `--ckpt checkpoints/v7_finetune/ckpt_best.pt --modes bare --no-val` on a 3-question file made with `head -3` (output to the scratchpad dir) and confirm rows have `topic` and the summary file is written. (Use the scratchpad dir, not `/tmp`, if running as an agent.)
- [ ] **Step 9: Commit** (include `data/eval/balanced_questions.jsonl`; it is synthetic Q&A derived text, PII-stripped upstream by `data/strip_pii.py`; confirm no `[URL]`-style or personal strings with `grep -E "@|http" data/eval/balanced_questions.jsonl` returning nothing).

---

### Task 3: Release gate script (thresholds fixed before results)

**Files:**
- Create: `evals/release_gate.py`, `tests/test_release_gate.py`

**Interfaces:**
- Consumes: `evals.topics.intrusion_rates`, `question_shares`; eval JSONL rows (`topic`, `clean`); judge JSONL rows (`score`); `--summary-json` files from Task 2; agreement JSON from Task 5 (`{"kappa": float}`).
- Constants at top of file (the pre-registered gate, copied from spec): `INTRUSION_MAX_RATIO = 1.5`, `EMPTY_MAX = 0.05`, `VAL_LOSS_MAX_WORSE = 0.05`, `MIN_SEEDS = 3`, `MIN_KAPPA = None` (owner sets in Task 5; `None` makes the gate refuse to run), `JUDGE_NOISE = "stdev of the previous version's per-seed judge means"`.
- Produces: `evaluate(manifest: dict) -> dict` returning `{"checks": {name: {"status": "PASS"|"FAIL"|"INDICATIVE", "detail": str}}, "verdict": "PASS"|"FAIL"|"INCOMPLETE"}`; CLI `python -m evals.release_gate MANIFEST.json` prints a table, exits 0 PASS, 1 FAIL, 2 INCOMPLETE.
- Manifest shape: `{"candidate": {"name", "eval": [3 jsonl], "judge": [3 jsonl], "summary": [3 json]}, "previous": {same}, "agreement": "path.json"}`.

Checks (per spec): (1) for every topic T, candidate's mean intrusion rate (pooled over seeds) <= 1.5 x T's share in the questions; topics with `None` intrusion are skipped. (2) candidate judge mean (mean of per-seed means) >= previous mean minus noise, where noise = sample stdev of previous's per-seed means; reported as WIN if above previous plus noise, TIE if within, FAIL if below minus noise. (3) mean empty-reply rate <= 5% AND candidate val loss (seed-0 summary) <= previous + 0.05. (4) agreement kappa >= `MIN_KAPPA`; if below, check 2 becomes `INDICATIVE` (not a PASS) and the verdict stays FAIL-free but is labelled "judge indicative only".

- [ ] **Step 1: Write failing tests** (`tests/test_release_gate.py`, build tiny manifests in `tmp_path`):
  - `test_pass_case`: all four checks satisfied -> verdict `PASS`.
  - `test_intrusion_fail`: candidate fuel intrusion 0.23 with fuel question share 0.11 -> check 1 `FAIL`, verdict `FAIL`.
  - `test_judge_tie_within_noise`: candidate mean 0.1 below previous but noise 0.2 -> check 2 status `PASS` with detail containing `TIE`.
  - `test_empty_and_valloss_fail` (two asserts).
  - `test_missing_seed_incomplete`: only 2 eval files -> verdict `INCOMPLETE`, CLI exit code 2.
  - `test_fewer_than_two_judge_files_incomplete` (noise cannot be measured).
  - `test_min_kappa_unset_incomplete`: `MIN_KAPPA is None` -> `INCOMPLETE` with message mentioning Task 5.
  - `test_low_kappa_indicative`: kappa below threshold -> check 2 `INDICATIVE`.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement. **Step 4:** run, expect PASS.
- [ ] **Step 5:** Review the constants block with the owner (**STOP point 1: owner confirms gate thresholds**, spec open decision 1). Then commit: `git commit -m "feat(evals): pre-registered release gate"`. This commit must exist before Task 4 produces any candidate data.

---

### Task 4: Baseline run of live v7 on the balanced set

**Files:** outputs only: `eval_runs/bal_v7_s{0,1,2}.jsonl`, `eval_runs/bal_v7_s{0,1,2}.summary.json`, `eval_runs/bal_v7_topics.json`

- [ ] **Step 1:** Quit LM Studio. Confirm free memory.
- [ ] **Step 2:** For S in 0 1 2: `python eval_mg.py --ckpt checkpoints/v7_finetune/ckpt_best.pt --questions data/eval/balanced_questions.jsonl --modes bare --seed S --out eval_runs/bal_v7_sS.jsonl --summary-json eval_runs/bal_v7_sS.summary.json` (add `--no-val` for S=1,2; the gate reads val loss from seed 0). Expected: 3 files of ~150 rows each, no crash.
- [ ] **Step 3:** `python -m evals.topic_report eval_runs/bal_v7_s*.jsonl --json eval_runs/bal_v7_topics.json`. Expected: fuel and cooling intrusion clearly above the other topics (matches the spec's ~23% finding). If not, report; do not retune the lexicon to make it so.
- [ ] **Step 4:** Open LM Studio (model `qwen/qwen3-vl-30b`), judge each seed separately: `python judge_mg.py --runs v7_sS=eval_runs/bal_v7_sS.jsonl --mode bare --out eval_runs/bal_judge_v7_sS.jsonl` for S in 0 1 2. (Separate runs per seed, as `judge_mg.py` keys replies by question.)
- [ ] **Step 5: Commit** the small summary/topic JSON only; the JSONL runs follow the existing convention for `eval_runs/` (check `git ls-files eval_runs | head`; commit them if the old runs are tracked).

---

### Task 5: Judge-vs-human validation

**Files:**
- Create: `evals/judge_agreement.py`, `tests/test_judge_agreement.py`, `data/eval/human_sheet.csv` (generated), `data/eval/agreement.json` (generated)

**Interfaces:**
- Produces:
  - `sample_sheet(sources: dict[str, list[dict]], counts: dict[str, int], seed: int) -> list[dict]` : rows `{"id": int, "src": label, "q", "reply", "human_score": ""}`, shuffled, ids 1..N, no model name anywhere except `src` (kept in a separate hidden column file, see Step 3). Empty replies are included as the string `(empty reply)`, as the judge sees them.
  - `agreement(human: list[int], judge: list[int]) -> dict` : `{"n", "exact", "within1", "kappa_qw", "mean_diff"}` where `kappa_qw` is quadratic-weighted Cohen's kappa over scores 1-5 and `mean_diff` = mean(judge - human).
- CLI: `python -m evals.judge_agreement sample ...` and `... score HUMAN.csv --judge-files ... --out data/eval/agreement.json`.

Sample 50 replies: 40 from `bal_v7_s0..2` (balanced across topics, about 4-5 per topic), and 10 from the weaker `eval_runs/v7_mgonly.jsonl` (`bare` mode) so human scores cover a wider quality range than v7 alone. Note this in the sheet README line.

- [ ] **Step 1: Write failing tests:** `test_kappa_perfect` (identical lists -> 1.0), `test_kappa_chance_like` (anti-correlated -> < 0), `test_within1_and_mean_diff` on a 5-item hand-computed example, `test_sample_hides_source_and_is_deterministic`, `test_sample_includes_empty_reply_marker`.
- [ ] **Step 2:** run, FAIL. **Step 3:** implement; `sample` writes `human_sheet.csv` (columns `id,q,reply,human_score`) and a separate `human_sheet_key.json` mapping `id -> {src, q}` that the owner does not open. **Step 4:** tests PASS.
- [ ] **Step 5:** Generate the sheet and key. Commit code and tests (not yet the filled sheet).
- [ ] **STOP point 2: owner hand-scores 50 replies.** Hand the owner `data/eval/human_sheet.csv` with the same 1-5 rubric text as `judge_mg.py` `RUBRIC`, scored blind (do not show judge scores). Expected effort ~30-45 min. Do not continue until the `human_score` column is full.
- [ ] **Step 6:** Judge the sampled replies (LM Studio open): run `judge_mg.py` over the same source files already judged in Task 4 and `eval_runs/v7_mgonly.jsonl` (judge scores for sampled replies are looked up by `(src, q)`; if the mgonly replies were not judged with the same judge model, judge them now).
- [ ] **Step 7:** `python -m evals.judge_agreement score data/eval/human_sheet.csv ... --out data/eval/agreement.json`. Report n, exact, within-1, quadratic kappa, mean bias to the owner.
- [ ] **STOP point 3: owner sets `MIN_KAPPA`.** Suggest the owner choose it (a common reading: >= 0.6 "usable", 0.4-0.6 "indicative only"); this is the owner's call per the spec ("threshold set after the 50-reply check"). Edit `MIN_KAPPA` in `evals/release_gate.py`, rerun `tests/test_release_gate.py`, commit with the agreement file.

---

### Task 6: Model-size configs, `--seed`, checkpoint pruning

**Files:**
- Modify: `model/config.py`, `train.py`
- Create: `scripts/prune_checkpoints.py`, `tests/test_configs.py`, `tests/test_prune.py`

**Interfaces:**
- Produces: `mid82_config() -> GPTConfig` (block 1024, 10L, 10H, 640, dropout 0.1); `large124_config() -> GPTConfig` (block 1024, 12L, 12H, 768, dropout 0.1); `train.py --config {nano,small,medium,mid82,large124}`; `train.py --seed INT` (default `None` = current behaviour; when set: `torch.manual_seed`, `np.random.seed`); `prune_checkpoints.prune(dir: str, keep_numbered: int = 2) -> list[str]` returns deleted paths, never deletes `ckpt_best.pt`, sorts numbered checkpoints by integer iteration.

- [ ] **Step 1: Tests first.** `test_configs.py`: build each model on CPU with `vocab_size=50257`; `GPT(cfg).num_params()` within 0.5M of 82.1M and 124.4M; `medium_config` unchanged at 51.5M. `test_prune.py`: tmp dir with `ckpt_best.pt`, `ckpt_1000.pt`, `ckpt_2000.pt`, `ckpt_10000.pt`, `ckpt_3000.pt` -> `prune(dir, keep_numbered=2)` leaves best, `ckpt_3000`, `ckpt_10000` (numeric, not alphabetical, order), returns the two deleted.
- [ ] **Step 2:** run, FAIL. **Step 3:** implement all three. **Step 4:** PASS.
- [ ] **Step 5:** Quick train smoke (short, allowed): `python train.py --config mid82 --data-dir data/v7_pretrain --checkpoint-dir /private/tmp/claude-501/-Users-lewisdesforges-Documents-TechDev-mg-llm/7f82603a-c323-4ec5-85ab-3c861037545f/scratchpad/smoke --max-iters 3 --seed 1` runs 3 steps and exits (LM Studio closed). Repeat for `large124`. This is the only training allowed before the STOP points below. Expected: prints `MG-GPT initialised: ~82M` / `~124M parameters`.
- [ ] **Step 6: Commit.**

---

### Task 7: Data arm (car-specific data, 51M fine-tune)

**Files:**
- Create: `data/synthesize_mechanics_qa.py`, `data/build_rebalanced_extra.py`, `evals/rebalance.py`, `tests/test_rebalance.py`, `tests/test_mech_synth.py`
- Modify: `data/prepare.py` (add `--extra-dirs`), `.gitignore` (`data/v10_*/`), README attribution (Task 10)
- Output dirs: `data/corpus_mechanics_qa/`, `data/corpus_rebalanced_extra/`, `data/v10_mech/`

**Interfaces:**
- Consumes: `evals.topics.primary_topic`; `valid`, `chunks` from `data/synthesize_wiki_qa.py` (import via `sys.path`); `topic` field in `data/corpus_wikipedia_mechanics/manifest.json` (`body_tyres` etc.) for article-level topic; MOT chapter file names for topic.
- Produces:
  - `synthesize_mechanics_qa.py`: for each text in `data/corpus_mot_manual/*.txt` and `data/corpus_wikipedia_mechanics/*.txt` writes `data/corpus_mechanics_qa/{stem}.txt` of `USER:/ASSISTANT:` pairs. Prompt differs from the MG one: general car-component questions ("How does...", "What does an MOT tester check on..."), answers only from the text, no mention of manual/article/Wikipedia/section numbers. Wrapper `clean_pairs(text: str) -> str` extends `valid` by rejecting pairs matching `\b(manual|section|appendix|the text|passage)\b`.
  - `evals/rebalance.py`: `repeat_factors(topic_tokens: dict[str, int], cap: int = 3) -> dict[str, int]` : target = the median topic's token count; factor = `min(cap, max(1, round(target / tokens)))` for each topic; topics above target get 1 (no downsampling; YAGNI).
  - `build_rebalanced_extra.py`: counts tokens per `primary_topic` over the train-side Q&A (`corpus_synth_qa`, `corpus_synth_qa_extra`, `corpus_wikipedia_qa`, `corpus_mechanics_qa`, excluding every `val_threads` stem), then writes extra copies (`{stem}__rep{n}.txt`, n = 2..factor) of documents of under-weight topics into `data/corpus_rebalanced_extra/`. Prints a before/after topic-share table.
  - `prepare.py --extra-dirs a,b` appends these dirs to `EXTRA_TRAIN_DIRS` (train-only, after the original shuffles, so val stays byte-identical).

- [ ] **Step 1: Tests first.** `test_mech_synth.py`: `clean_pairs` keeps a clean pair, drops pairs saying "the manual", "section 5.1", "Wikipedia". `test_rebalance.py`: `repeat_factors({"a":100,"b":100,"c":25})` -> c gets 3 (cap), a/b get 1; cap respected at `cap=2`; a topic with 0 tokens gets `cap` (no division by zero); and an integration test with a fake tmp corpus where one doc stem is in `val_threads`: `build_rebalanced_extra` output never contains that stem or its `__rep` copies.
- [ ] **Step 2:** run, FAIL. **Step 3:** implement. **Step 4:** PASS.
- [ ] **Step 5 (needs LM Studio):** Open LM Studio, run `python data/synthesize_mechanics_qa.py` (log to `data/logs/synth_mechanics_qa.log`, resumable like `synthesize_wiki_qa.py`: skip existing outputs). Check: the pair count, 10 random pairs read by eye for hallucinated brand claims and mentions of "manual"; token count of the output. Close LM Studio.
- [ ] **Step 6:** `python data/build_rebalanced_extra.py`; confirm the before/after table: brakes, suspension, electrical, body shares rise; fuel and cooling do not rise. Then `python data/prepare.py --out-dir data/v10_mech --extra-dirs data/corpus_mechanics_qa,data/corpus_rebalanced_extra`. Check `val.bin` is byte-identical to `data/v7_mgonly/val.bin`: `cmp data/v10_mech/val.bin data/v7_mgonly/val.bin` prints nothing.
- [ ] **Step 7: Commit** scripts, tests, manifests. Add the mechanics-Q&A directory to git the same way `data/corpus_wikipedia_qa` is handled (check `git ls-files data/corpus_wikipedia_qa | head -1`).
- [ ] **STOP point 4: owner go-ahead for the data-arm fine-tune (~2.6 h, LM Studio closed).**
- [ ] **Step 8:** `python train.py --config medium --data-dir data/v10_mech --init-from checkpoints/v7_pretrain/ckpt_best.pt --checkpoint-dir checkpoints/v10_mech_finetune --max-iters 4000 --lr 1e-4 --dropout 0.1 --patience 4 --save-interval 1000 2>&1 | tee data/logs/train_v10_mech_finetune.log` (same recipe as `data/logs/train_v7_finetune.log`; only the data differs). Run in the background with a log; do not poll in a loop.
- [ ] **Step 9:** Prune: `python scripts/prune_checkpoints.py checkpoints/v10_mech_finetune`. Evaluate with Task 8's procedure (name `v10mech`).

---

### Task 8: Evaluation procedure (reused by every candidate)

Not a code task; a fixed recipe the later tasks reference as "Evaluate NAME CKPT".

- [ ] **Step 1:** LM Studio closed. For S in 0 1 2: `python eval_mg.py --ckpt CKPT --questions data/eval/balanced_questions.jsonl --modes bare --seed S --out eval_runs/bal_NAME_sS.jsonl --summary-json eval_runs/bal_NAME_sS.summary.json` (`--no-val` for S=1,2).
- [ ] **Step 2:** `python -m evals.topic_report eval_runs/bal_NAME_s*.jsonl --json eval_runs/bal_NAME_topics.json`.
- [ ] **Step 3:** Open LM Studio; per seed `python judge_mg.py --runs NAME_sS=eval_runs/bal_NAME_sS.jsonl --mode bare --out eval_runs/bal_judge_NAME_sS.jsonl`.
- [ ] **Step 4:** Write `eval_runs/gate_NAME.json` manifest (previous = `v7`) and run `python -m evals.release_gate eval_runs/gate_NAME.json | tee eval_runs/gate_NAME.txt`. Record exit code. A FAIL is a legitimate result; do not re-run with different seeds or thresholds to change it.

---

### Task 9: Size runs 82M and 124M

**Files:** outputs only (`checkpoints/v11_82m_*`, `checkpoints/v12_124m_*`, `data/logs/train_*.log`, `eval_runs/bal_*`).

Time budget from the spec: 82M pretrain ~12.7 h + fine-tune ~4 h; 124M pretrain ~18.3 h + fine-tune ~5.7 h. 51M is the existing v7 (`checkpoints/v7_pretrain`, `checkpoints/v7_finetune`), no retraining. Run sequentially, never two at once (memory 14.7 / 18.5 GB each).

- [ ] **STOP point 5: owner go-ahead to start the 82M pretrain (~12.7 h).** Confirm LM Studio quit and the Mac set not to sleep (`caffeinate -i` prefix).
- [ ] **Step 1 (82M pretrain):** `caffeinate -i python train.py --config mid82 --data-dir data/v7_pretrain --checkpoint-dir checkpoints/v11_82m_pretrain --max-iters 12000 --lr 3e-4 --dropout 0.0 --save-interval 1000 --seed 0 2>&1 | tee data/logs/train_v11_82m_pretrain.log` (same flags as `data/logs/train_v7_pretrain.log`, only `--config` and `--seed` differ). Background with a log. Expected: loss falls from ~10.9; general val best near or below 3.75. If interrupted: `--resume checkpoints/v11_82m_pretrain/ckpt_N.pt` with the same other flags.
- [ ] **Step 2:** Every ~4 hours, or on completion notice: `python scripts/prune_checkpoints.py checkpoints/v11_82m_pretrain` (keeps `ckpt_best.pt` plus 2 newest). Check `df -h .` stays above 30 GB.
- [ ] **Step 3 (82M fine-tune, ~4 h):** `python train.py --config mid82 --data-dir data/v7_mgonly --init-from checkpoints/v11_82m_pretrain/ckpt_best.pt --checkpoint-dir checkpoints/v11_82m_finetune --max-iters 4000 --lr 1e-4 --dropout 0.1 --patience 4 --save-interval 1000 --seed 0 2>&1 | tee data/logs/train_v11_82m_finetune.log`. Note: `data/v7_mgonly` is the data dir whose val split matches v7_finetune (confirm in `data/logs/train_v7_finetune.log`, "Train tokens: 2,442,543 | Val tokens: 224,857"; if v7_finetune used a different dir, use that one). Prune afterwards.
- [ ] **Step 4:** Evaluate `v11_82m` per Task 8 with `CKPT=checkpoints/v11_82m_finetune/ckpt_best.pt`.
- [ ] **STOP point 6: owner go-ahead to start the 124M pretrain (~18.3 h, then ~5.7 h fine-tune, ~24 h total).** Show the 82M gate result first; the owner may stop here if 82M already answers the question.
- [ ] **Step 5 (124M pretrain):** as Step 1 with `--config large124`, `--checkpoint-dir checkpoints/v12_124m_pretrain`, log `data/logs/train_v12_124m_pretrain.log`. Prune as in Step 2 (checkpoints are the biggest here; worst case ~15 GB total).
- [ ] **Step 6 (124M fine-tune, ~5.7 h):** as Step 3 with `--config large124`, init from `v12_124m_pretrain/ckpt_best.pt`, `--checkpoint-dir checkpoints/v12_124m_finetune`, same lr 1e-4 and `--patience 4` (early stopping is the guard against overfitting the ~2.9M MG tokens; the recipe is deliberately unchanged so size is the only variable). Prune. Record the iteration where best val occurred; if best is within the first 500 steps, say so in the write-up (overfit-fast signal).
- [ ] **Step 7:** Evaluate `v12_124m` per Task 8.
- [ ] **STOP point 7 (optional): extra fine-tune seeds.** If a size or data effect is within seed noise, ask the owner before spending ~2.6 h (51M) / ~4 h (82M) / ~5.7 h (124M) per extra `--seed` fine-tune. Never report a size effect from one training seed without this caveat.

---

### Task 10: Comparison, plot, docs

**Files:**
- Create: `evals/plot_topics.py`
- Modify: `README.md`, `CHANGELOG.md`, `WRITEUP.md`, `LEARNINGS.md`

- [ ] **Step 1:** `evals/plot_topics.py`: `python -m evals.plot_topics eval_runs/bal_v7_topics.json eval_runs/bal_v10mech_topics.json eval_runs/bal_v11_82m_topics.json eval_runs/bal_v12_124m_topics.json --out eval_runs/topic_spread.png` draws grouped bars of intrusion rate per topic per model, with the 1.5x question-share cap as a horizontal marker. Verify it writes a non-empty PNG for the available files (smoke test with the v7 file alone).
- [ ] **Step 2: README.md** (edit in place): add `evals/`, `tests/`, `scripts/prune_checkpoints.py`, `data/synthesize_mechanics_qa.py`, `data/build_rebalanced_extra.py`, `data/eval/` to the project-structure tree with one-line descriptions; add the new `--config mid82`/`large124` presets to the hardware table with measured s/step and memory from the spec; add a short "How we evaluate" paragraph (topic-spread, balanced set, judge trust level, release gate) linking to the spec; add the attribution line: MOT manual under the Open Government Licence v3.0 (text only), Wikipedia mechanics articles CC BY-SA 4.0 with revision ids in `data/corpus_wikipedia_mechanics/manifest.json`.
- [ ] **Step 3: CHANGELOG.md** top "In progress": one what-line and one why-line each for: topic-spread metric + balanced set, judge validation (with kappa), release gate, data arm result, each size result (gate verdict + judge mean +/- noise), and the replaced "Car-specific data fetched, not yet used" entry. Promote shipped lines out of "In progress" per repo CLAUDE.md. Nothing is deployed, so no version section is created.
- [ ] **Step 4: WRITEUP.md:** add the evaluation case study: what the old metrics missed, the topic-spread table before/after, the judge trust level (kappa, n=50), the fixed-budget caveat (196M tokens = ~1.6 to 3.8 tokens per parameter, not compute-optimal), the gate and its verdicts, honest "no size effect" if that is the result.
- [ ] **Step 5: LEARNINGS.md:** append dated facts: lexicon tightening vs the scratchpad numbers, label spot-check accuracy, any topic shortfall in the question set, measured agreement, actual run times vs the spec's estimates, disk used, best-val iteration for each fine-tune.
- [ ] **Step 6:** `python -m pytest -q` passes; `git status` shows no unrelated changes and no files containing a personal email (`git grep -nE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}" -- . ':!LICENSE'` shows nothing new).
- [ ] **Step 7: Commit** locally. Do not push or deploy; deployment of any new model is a separate decision for the owner.

---

## Self-Review Notes

- Spec coverage: topic-spread + intrusion (T1), ~150 balanced set (T2), judge validation (T5), gate coded (T3), data arm (T7), three sizes at fixed budget with exact commands, save interval, pruning, LM Studio closed (T6, T9), docs (T10). 51M uses existing v7.
- STOP points: 1 gate thresholds, 2 hand-scoring, 3 kappa threshold, 4 data-arm fine-tune, 5 82M pretrain, 6 124M run, 7 optional extra seeds.
- The gate's check 1 follows the spec literally: intrusion rate vs 1.5x the topic's share among questions. With 9 balanced topics the share is ~11%, so the cap is ~17%; v7's ~23% fuel figure would fail it. That is likely the intent, but it is an interpretation.
