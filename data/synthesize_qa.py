"""
data/synthesize_qa.py — Use an LLM to turn raw forum threads into clean Q&A pairs.

The "alternate-posts-as-USER/ASSISTANT" trick in format_qa.py gives the model
the *shape* of dialogue but the *content* is noisy — half of any forum thread
is chitchat, replies-to-replies, and parts numbers without context. Synthetic
Q&A flips that: a stronger LLM reads each thread and extracts only the
practical knowledge as clean, self-contained Q&A pairs.

Two backends supported:
  - Anthropic Claude (default) — needs ANTHROPIC_API_KEY
  - Local LM Studio (--lm-studio) — needs LM Studio running with a Qwen3 model

Usage:
    # Dry run — print one prompt and estimated cost, don't call any API
    python data/synthesize_qa.py --dry-run

    # Sample output quality on a few threads (safe to run, cheap)
    python data/synthesize_qa.py --limit 10

    # Full run via Claude Haiku (paid)
    python data/synthesize_qa.py

    # Full run via local LM Studio Qwen3 (free, slower)
    python data/synthesize_qa.py --lm-studio --limit 10
    python data/synthesize_qa.py --lm-studio
"""

import os
import sys
import time
import glob
import argparse

INPUT_DIR  = "data/corpus_pistonheads_clean"   # PII-stripped PH corpus
OUTPUT_DIR = "data/corpus_synth_qa"

CLAUDE_MODEL = "claude-haiku-4-5"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "local-model"   # LM Studio uses whatever's loaded; name is informational
MAX_OUTPUT_TOKENS = 2000

# Skip threads too short to extract useful Q&A from. (Posts < this many chars
# tend to be one-liners with no real content.)
MIN_THREAD_CHARS = 1000

# Truncate threads larger than this so we don't burn tokens on outliers
# (the 5 longest restoration diaries can be 200K+ chars). 40K chars is
# roughly 10K tokens, plenty for Claude to find the gold.
MAX_THREAD_CHARS = 40_000

# Haiku 4.5 pricing per million tokens (approximate, used only for cost estimates)
PRICE_INPUT_PER_M  = 1.00
PRICE_OUTPUT_PER_M = 5.00

SYSTEM_PROMPT = """You are an expert MG mechanic. Your range covers classic MG (MGA, MGB, MGC, Midget, MG TF, MG RV8) and the MG Rover era (ZR, ZS, ZT, MG TF post-2001, K-series engines). You're reading a PistonHeads forum thread to extract its practical technical knowledge as workshop case studies — for training a small chatbot to diagnose and advise.

For each thread, extract 3–8 case studies in conversational form. Each is a USER turn (symptom or problem, as an owner would describe it) and an ASSISTANT turn (your diagnostic reasoning + recommended action).

Each ASSISTANT answer should follow this implicit structure — but written as natural prose, not as labelled sections:
1. **Cause** — the likely fault(s), most-probable first.
2. **Check** — what to inspect or test to confirm.
3. **Fix** — the action to take, with specific values, parts, or torque settings if the thread mentions them.

Rules:
- USER turns: 1–2 sentences, specific car details if the thread gives them ("My '78 MGB idles lumpily when warm"), realistic phrasing.
- ASSISTANT turns: 3–6 sentences. Confident but not hand-wavy. If the thread surfaces multiple opinions, summarise the consensus or the strongest-evidenced one — don't list every poster's view.
- Use specific values when the thread provides them (e.g. "0.015\" tappet clearance hot", "30 ft-lb head bolt torque"). Don't invent numbers the thread doesn't support.
- Skip pure chitchat, jokes, and "thanks for the help". Only extract pairs with real technical or practical content.
- If the same fault appears in multiple posts with different conclusions, prefer the answer that was thanked, accepted, or appears to be the resolved fix.
- If a thread is mostly off-topic or has no recoverable knowledge, return only the few good pairs (even just 1) or nothing at all.
- Output ONLY the pairs in this exact format. No preamble, no commentary, no numbering, no headers:

USER: <symptom or question>
ASSISTANT: <cause, check, and fix as natural prose>

USER: <symptom or question>
ASSISTANT: <cause, check, and fix as natural prose>
"""

USER_PROMPT_TEMPLATE = """Here is the forum thread. Extract the Q&A pairs as instructed.

---
{thread}
---"""

# Qwen3 supports a `/no_think` directive that disables its thinking-mode chain
# of thought. For extraction tasks we don't need the reasoning trace and it
# would blow up the output budget without helping quality. Appended to the
# user prompt when using LM Studio.
NO_THINK_SUFFIX = "\n\n/no_think"


def estimate_tokens(text):
    """Rough char-to-token estimate (~4 chars/token for English)."""
    return len(text) // 4


def collect_threads():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    eligible = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if len(text) < MIN_THREAD_CHARS:
            continue
        if len(text) > MAX_THREAD_CHARS:
            text = text[:MAX_THREAD_CHARS]
        eligible.append((path, text))
    return files, eligible


def already_processed(input_path):
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def write_output(input_path, content):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


class Usage:
    """Lightweight usage struct so both backends produce a uniform shape."""
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def call_claude(client, thread_text, max_retries=3):
    user_prompt = USER_PROMPT_TEMPLATE.format(thread=thread_text)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
            return resp.content[0].text, usage
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  ⚠️  Claude API error: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Claude API failed after {max_retries} retries: {last_err}")


def call_lm_studio(url, thread_text, max_retries=3):
    """Hit LM Studio's OpenAI-compatible /v1/chat/completions endpoint."""
    import requests
    user_prompt = USER_PROMPT_TEMPLATE.format(thread=thread_text) + NO_THINK_SUFFIX
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.4,   # extraction is mostly deterministic — keep it tight
        "stream": False,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            # LM Studio can take a while on large prompts; give it real time.
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {})
            usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return text, usage
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  ⚠️  LM Studio error: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LM Studio failed after {max_retries} retries: {last_err}")


def strip_qwen_thinking(text):
    """Qwen3 sometimes still emits a <think>...</think> block even with
    /no_think. Drop it so we only persist the actual Q&A pairs."""
    if "<think>" in text and "</think>" in text:
        end = text.index("</think>") + len("</think>")
        text = text[end:].lstrip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the prompt and cost estimate, don't call any API")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N eligible threads")
    parser.add_argument("--lm-studio", action="store_true",
                        help="Use local LM Studio instead of Claude (free, slower)")
    parser.add_argument("--lm-studio-url", default=LM_STUDIO_URL,
                        help=f"LM Studio endpoint (default: {LM_STUDIO_URL})")
    args = parser.parse_args()

    backend = "LM Studio" if args.lm_studio else "Claude Haiku"

    all_files, eligible = collect_threads()
    print(f"Found {len(all_files)} corpus files, {len(eligible)} eligible "
          f"(>= {MIN_THREAD_CHARS} chars, capped at {MAX_THREAD_CHARS}).")

    # Skip already-processed threads (resume support)
    to_process = [(p, t) for p, t in eligible if not already_processed(p)]
    already = len(eligible) - len(to_process)
    if already:
        print(f"  {already} already processed in {OUTPUT_DIR}/ — skipping")

    if args.limit is not None:
        to_process = to_process[:args.limit]
        print(f"  --limit {args.limit} → processing {len(to_process)} threads")

    if not to_process:
        print("Nothing to do.")
        return

    # Token estimate (used by both cost projection and the dry-run summary)
    total_in_tokens = sum(estimate_tokens(t) for _, t in to_process) + len(SYSTEM_PROMPT) // 4 * len(to_process)
    est_out_tokens  = len(to_process) * 800  # rough — 800 output tokens per thread average
    print(f"\nBackend:           {backend}")
    print(f"Estimated tokens:  {total_in_tokens:,} in + {est_out_tokens:,} out")
    if not args.lm_studio:
        est_cost = (total_in_tokens / 1_000_000) * PRICE_INPUT_PER_M + \
                   (est_out_tokens   / 1_000_000) * PRICE_OUTPUT_PER_M
        print(f"Estimated cost:    ${est_cost:.2f} (Haiku 4.5)")
    else:
        print(f"Estimated cost:    $0.00 (local model)")
    print()

    if args.dry_run:
        sample_path, sample_text = to_process[0]
        print(f"--- DRY RUN: example prompt for {os.path.basename(sample_path)} ---")
        print(f"\n[SYSTEM]\n{SYSTEM_PROMPT[:300]}...\n")
        user_p = USER_PROMPT_TEMPLATE.format(thread=sample_text[:500] + '...')
        if args.lm_studio:
            user_p += NO_THINK_SUFFIX
        print(f"[USER]\n{user_p[:700]}")
        print("\n(no API call made)")
        return

    # Set up the chosen backend
    if args.lm_studio:
        # Ping LM Studio before kicking off a long loop — fail fast if it's not up.
        import requests
        ping_url = args.lm_studio_url.rsplit("/", 1)[0] + "/models"
        try:
            requests.get(ping_url, timeout=5).raise_for_status()
        except Exception as e:
            print(f"❌ Can't reach LM Studio at {args.lm_studio_url}: {e}")
            print("   Make sure LM Studio is running and 'Start Server' is on (default port 1234).")
            sys.exit(1)
        caller = lambda text: call_lm_studio(args.lm_studio_url, text)
    else:
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ anthropic SDK not installed. Run: pip install anthropic")
            sys.exit(1)
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        client = Anthropic()
        caller = lambda text: call_claude(client, text)

    actual_in_tokens  = 0
    actual_out_tokens = 0
    saved = 0
    started = time.time()

    for i, (path, thread_text) in enumerate(to_process, 1):
        slug = os.path.basename(path)
        print(f"[{i}/{len(to_process)}] {slug}  ({len(thread_text):,} chars)", end=" ", flush=True)
        t0 = time.time()
        try:
            output, usage = caller(thread_text)
            actual_in_tokens  += usage.input_tokens
            actual_out_tokens += usage.output_tokens
        except Exception as e:
            print(f"❌ {e}")
            continue

        output = strip_qwen_thinking(output)
        pair_count = output.count("USER:")
        if pair_count == 0:
            print(f"(no pairs extracted, skipping)")
            continue

        write_output(path, output)
        saved += 1
        print(f"→ {pair_count} pairs  [{time.time()-t0:.1f}s]")

    elapsed = time.time() - started
    print(f"\n{'='*60}")
    print(f"✅ Saved {saved}/{len(to_process)} threads to {OUTPUT_DIR}/")
    print(f"   backend:     {backend}")
    print(f"   tokens used: {actual_in_tokens:,} in + {actual_out_tokens:,} out")
    if not args.lm_studio:
        actual_cost = (actual_in_tokens / 1_000_000) * PRICE_INPUT_PER_M + \
                      (actual_out_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
        print(f"   actual cost: ${actual_cost:.2f}")
    print(f"   elapsed:     {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
