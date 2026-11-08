"""
data/synthesize_qa_extra.py — second pass: ask the local LLM for *additional* Q&A
pairs from threads the first pass (synthesize_qa.py) already covered.

The first pass averaged 3.4 pairs/thread against a 3-8 target, so most threads
still hold unused facts. This pass shows the model the thread AND the pairs
already extracted and asks for different ones.

Held-out (val) threads are skipped on purpose: val must stay byte-identical so
numbers stay comparable with the v7 runs. Output goes to a separate directory
that prepare.py only ever adds to train.

Usage:
    python data/synthesize_qa_extra.py --limit 5            # try a few
    python data/synthesize_qa_extra.py --min-chars 3000     # the meatier threads
"""

import argparse
import glob
import hashlib
import json
import os
import re
import time
import urllib.request

INPUT_DIR = "data/corpus_pistonheads_clean"
FIRST_PASS_DIR = "data/corpus_synth_qa"
OUTPUT_DIR = "data/corpus_synth_qa_extra"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"
VAL_FRACTION = 0.1  # keep in step with data/prepare.py
MAX_THREAD_CHARS = 40_000

SYSTEM_PROMPT = """You are an expert MG mechanic (MGA, MGB, MGC, Midget, MG TF, MG RV8, and the MG Rover era ZR/ZS/ZT with K-series engines). You are reading a PistonHeads forum thread to extract practical knowledge as workshop Q&A for training a small chatbot.

A first pass already extracted some Q&A pairs from this thread (shown below the thread). Write 2-5 NEW pairs about DIFFERENT facts, faults, parts, values or procedures that the thread supports and the first pass did not cover. If the thread has nothing further worth extracting, output nothing at all.

Rules:
- Use ONLY what the thread says or clearly implies. Never invent part numbers, torque values or clearances.
- USER turns: 1-2 sentences, how an owner would phrase it, with car details if the thread gives them.
- ASSISTANT turns: 3-6 sentences of natural prose covering likely cause, what to check, and the fix.
- Skip chitchat, jokes and anything already covered by the existing pairs.
- Output ONLY pairs in exactly this format, no preamble or numbering:

USER: <question>
ASSISTANT: <answer>

USER: <question>
ASSISTANT: <answer>
"""


def is_val_thread(stem):
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 < VAL_FRACTION * 1000


def ask(thread, existing):
    user = (f"THREAD:\n---\n{thread}\n---\n\nPAIRS ALREADY EXTRACTED:\n---\n{existing}\n---\n\n"
            "Write the new pairs now.\n\n/no_think")
    body = {"model": MODEL, "temperature": 0.5, "max_tokens": 1800, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
    req = urllib.request.Request(LM_STUDIO_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    return text.strip()


def valid(text):
    """Keep only well-formed USER:/ASSISTANT: pairs."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    ok = [b for b in blocks if re.match(r"^USER:.+\nASSISTANT:.+", b, re.S) and b.count("USER:") == 1]
    return "\n\n".join(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=3000, help="skip shorter threads")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    todo = []
    for path in sorted(glob.glob(os.path.join(FIRST_PASS_DIR, "*.txt"))):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        src = os.path.join(INPUT_DIR, name)
        if is_val_thread(stem) or not os.path.exists(src) or os.path.exists(os.path.join(OUTPUT_DIR, name)):
            continue
        if os.path.getsize(src) < args.min_chars:
            continue
        todo.append((name, src, path))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} threads to process")

    t0, pairs = time.time(), 0
    for i, (name, src, first) in enumerate(todo, 1):
        thread = open(src, encoding="utf-8").read()[:MAX_THREAD_CHARS]
        try:
            out = valid(ask(thread, open(first, encoding="utf-8").read().strip()))
        except Exception as e:  # noqa: BLE001 — skip this thread, keep the run alive
            print(f"[{i}/{len(todo)}] {name}: error {e}")
            continue
        # An empty file marks "nothing more to extract" so a rerun skips it.
        with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(out + "\n" if out else "")
        n = len(re.findall(r"^USER:", out, re.M))
        pairs += n
        rate = (time.time() - t0) / i
        print(f"[{i}/{len(todo)}] {name}: +{n} pairs (total {pairs}, {rate:.0f}s/thread, "
              f"~{rate * (len(todo) - i) / 60:.0f}m left)", flush=True)


if __name__ == "__main__":
    main()
