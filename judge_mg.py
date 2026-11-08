"""
judge_mg.py — blind LLM-judge for eval_mg.py generations.

Reads the JSONL files eval_mg.py wrote, takes the replies for one prompt mode,
and asks a local model (LM Studio, OpenAI-compatible API) to score each reply
1-5 for whether it makes sense as an answer to the question. The judge never
sees which model wrote a reply, and the order is shuffled per call.

Usage:
    python judge_mg.py --runs v6=eval_runs/v6_baseline.jsonl \
        mgonly=eval_runs/v7_mgonly.jsonl finetune=eval_runs/v7_finetune.jsonl
"""

import argparse
import json
import random
import re
import statistics
import sys
import urllib.request

RUBRIC = """You are marking answers from a very small, home-made language model that was
asked a question about classic MG cars (MGB, MGF, ZR, Midget, etc).
Score the ANSWER from 1 to 5 for whether it makes sense as an answer to the QUESTION:

5 = on topic, factually plausible, coherent, would help a real owner
4 = mostly sensible, small slips or vagueness
3 = on topic but partly wrong, rambling or self-contradicting
2 = mostly nonsense, with a few relevant words
1 = incoherent, off topic or empty

Judge sense and accuracy, not style or length. Reply with ONLY a JSON object:
{"score": <1-5>, "reason": "<max 15 words>"}"""


def ask(base_url, model, question, answer):
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer}"},
        ],
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r'"score"\s*:\s*([1-5])', text)
    return (int(m.group(1)) if m else None), text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.jsonl ...")
    ap.add_argument("--mode", default="bare")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="qwen/qwen3-vl-30b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_runs/judge.jsonl")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        runs[label] = {r["q"]: r["clean"] for r in rows if r["mode"] == args.mode}

    questions = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"{len(questions)} shared questions, mode={args.mode}, judge={args.model}")

    rng = random.Random(args.seed)
    scores = {label: [] for label in runs}
    with open(args.out, "w") as out:
        for q in questions:
            labels = list(runs)
            rng.shuffle(labels)  # judge order is irrelevant to it, but keeps logs unbiased
            for label in labels:
                answer = runs[label][q].strip() or "(empty reply)"
                score, raw = ask(args.base_url, args.model, q, answer)
                scores[label].append(score)
                out.write(json.dumps({"label": label, "q": q, "score": score, "judge": raw}) + "\n")
                out.flush()

    print("\nlabel        n   mean  median  %>=4  unparsed")
    for label, s in scores.items():
        ok = [x for x in s if x is not None]
        print(f"{label:<10} {len(ok):3d}  {statistics.mean(ok):.2f}  {statistics.median(ok):5.1f}  "
              f"{100 * sum(x >= 4 for x in ok) / len(ok):4.0f}  {len(s) - len(ok)}")


if __name__ == "__main__":
    sys.exit(main())
