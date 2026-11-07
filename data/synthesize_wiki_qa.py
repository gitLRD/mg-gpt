"""
data/synthesize_wiki_qa.py — turn fetched Wikipedia MG articles into grounded Q&A.

Input:  data/corpus_wikipedia_mg/*.txt   (data/fetch_wikipedia_mg.py; CC BY-SA 4.0)
Output: data/corpus_wikipedia_qa/*.txt    USER:/ASSISTANT: pairs, one file per article

Every answer must come from the article text, so facts (years, engines, model
differences) are as accurate as Wikipedia is. The raw articles are also trained
on as plain prose. prepare.py only ever adds both to the train split.

Derived text stays CC BY-SA 4.0 — keep data/corpus_wikipedia_mg/manifest.json
with anything that ships this data.

Usage:
    python data/synthesize_wiki_qa.py
"""

import argparse
import glob
import json
import os
import re
import time
import urllib.request

INPUT_DIR = "data/corpus_wikipedia_mg"
OUTPUT_DIR = "data/corpus_wikipedia_qa"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"
CHUNK_CHARS = 5000

SYSTEM_PROMPT = """You write training data for a small chatbot that talks like a friendly, knowledgeable classic-MG enthusiast.

Read the encyclopedia extract and write 3-6 Q&A pairs about the facts in it (model years, engines, body styles, changes between versions, history of the company, who designed or built what, production numbers).

Rules:
- Use ONLY facts stated in the extract. Never add facts from your own memory. If it lacks enough facts, write fewer pairs or none.
- USER turns: one or two sentences, phrased the way an owner or curious buyer would ask, and mention the model by name so the question stands alone.
- ASSISTANT turns: 2-5 sentences of natural conversational prose, with the specific numbers and years from the extract. Do not mention "the extract", "the article" or Wikipedia.
- Output ONLY pairs in exactly this format, no preamble or numbering:

USER: <question>
ASSISTANT: <answer>

USER: <question>
ASSISTANT: <answer>
"""


def ask(title, chunk):
    user = f"Article: {title}\n\nExtract:\n---\n{chunk}\n---\n\nWrite the pairs now.\n\n/no_think"
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
          and not re.search(r"\b(extract|the article|wikipedia)\b", b, re.I)]
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
    argparse.ArgumentParser().parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    t0, total = time.time(), 0
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(out_path):
            continue
        text = open(path, encoding="utf-8").read()
        title, body = text.split("\n\n", 1)
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
        print(f"[{i}/{len(files)}] {title}: {n} pairs (total {total}, {(time.time() - t0) / 60:.0f}m)", flush=True)


if __name__ == "__main__":
    main()
