"""
data/synthesize_mech_qa.py — turn the car-mechanics text into grounded Q&A.

Input:  the INPUTS dirs below (Wikipedia mechanics articles CC BY-SA 4.0; TM 9-8000, a US
        government work; UK MOT inspection manual, OGL v3.0)
Output: data/corpus_mech_qa/*.txt   USER:/ASSISTANT: pairs, one file per source file

Every answer must come from the source text. A local Qwen (LM Studio) writes the pairs.
Train-only: prepare.py must never put these in the validation split.

Derived Wikipedia text stays CC BY-SA 4.0; the MOT text needs the OGL line from the README.

Usage:
    python data/synthesize_mech_qa.py --limit 3     # smoke test
    python data/synthesize_mech_qa.py               # everything (skips files already done)
"""

import argparse
import glob
import json
import os
import re
import time
import urllib.request

INPUTS = [
    "data/corpus_wikipedia_mechanics",
    "data/corpus_wikipedia_mechanics_b",
    "data/corpus_tm9_8000",
    "data/corpus_mot_manual",
]
OUTPUT_DIR = "data/corpus_mech_qa"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"
CHUNK_CHARS = 5000

SYSTEM_PROMPT = """You write training data for a small chatbot that talks like a friendly, knowledgeable classic-car enthusiast who owns old MGs.

Read the technical extract (about how cars work, wear, fail and are inspected) and write 3-5 Q&A pairs about the facts in it.

Rules:
- Use ONLY facts stated in the extract. Never add facts from your own memory. If it lacks enough facts, write fewer pairs or none.
- Spread the questions over different facts. Mix "how does X work" questions with owner-style symptom or check questions ("my brakes feel spongy, why?", "how do I tell if the bushes are worn?") when the extract supports an answer.
- USER turns: one or two sentences, how an owner would ask, standing alone (name the part or system).
- ASSISTANT turns: 2-4 sentences of natural conversational prose, with the specifics from the extract. Do not open with "The most likely cause". Do not mention "the extract", "the article", "the manual" or Wikipedia. Do not mention military vehicles.
- Output ONLY pairs in exactly this format, no preamble or numbering:

USER: <question>
ASSISTANT: <answer>

USER: <question>
ASSISTANT: <answer>
"""

BAD = r"\b(extract|the article|wikipedia|the manual|military|army|soldiers?)\b"


def ask(title, chunk):
    user = f"Source: {title}\n\nExtract:\n---\n{chunk}\n---\n\nWrite the pairs now.\n\n/no_think"
    body = {"model": MODEL, "temperature": 0.4, "max_tokens": 1500, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
    req = urllib.request.Request(LM_STUDIO_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    return text.strip()


def valid(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    ok = [b for b in blocks if re.match(r"^USER:.+\nASSISTANT:.+", b, re.S) and b.count("USER:") == 1
          and not re.search(BAD, b, re.I)]
    return "\n\n".join(ok)


def chunks(text, size=CHUNK_CHARS):
    """Split on paragraph boundaries into ~size-char pieces."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (smoke test)")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for d in INPUTS for f in glob.glob(os.path.join(d, "*.txt")))
    if args.limit:
        files = files[: args.limit]
    t0, total = time.time(), 0
    for i, path in enumerate(files, 1):
        name = os.path.basename(os.path.dirname(path)).replace("corpus_", "") + "__" + os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(out_path):
            continue
        text = open(path, encoding="utf-8").read()
        title, body = text.split("\n\n", 1) if "\n\n" in text else (name, text)
        pairs = []
        for ch in chunks(body):
            try:
                got = valid(ask(title, ch))
            except Exception as e:  # noqa: BLE001 — skip this chunk
                print(f"  {name}: chunk error {e}")
                continue
            if got:
                pairs.append(got)
        out = "\n\n".join(pairs)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out + "\n" if out else "")
        n = len(re.findall(r"^USER:", out, re.M))
        total += n
        print(f"[{i}/{len(files)}] {title[:50]}: {n} pairs (total {total}, {(time.time() - t0) / 60:.0f}m)", flush=True)


if __name__ == "__main__":
    main()
