"""
label_questions.py — ask a local LLM (LM Studio) which ONE topic each eval question is about.

Why: the keyword lexicon mislabels ~15% of questions ("diesel" in a brakes question counts as
fuel). The lexicon still scores replies (cheap, repeatable); only the QUESTION labels come
from the model, once, saved into the file so the eval stays deterministic.

Usage: python -m evals.label_questions data/eval/balanced_questions.jsonl
Keeps the old keyword label as `topic_lex`; `topic` becomes the model's label. Questions the
model calls "none" are dropped (they are not about any of the 9 topics).
"""
import json
import sys
import urllib.request

from evals.topics import TOPICS

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"
HELP = {
    "fuel": "fuel, petrol, carburettors, fuel injection, fuel pump/tank/lines",
    "cooling": "overheating, radiator, coolant, thermostat, water pump, head gasket",
    "brakes": "brakes, discs, pads, calipers, servo, handbrake, brake fluid",
    "electrical": "battery, alternator, starter, wiring, fuses, ignition, lights, switches",
    "oil": "engine oil, oil pressure, oil leaks, seals, oil filter",
    "body": "rust, bodywork, panels, paint, bumpers, hood, trim, chassis",
    "gearbox": "gearbox, clutch, differential, overdrive, driveshaft",
    "susp": "suspension, springs, dampers, steering, tyres, handling, wheels",
    "price": "price, value, what to pay, running costs, buying advice",
}


def ask(q):
    menu = "\n".join(f"- {k}: {v}" for k, v in HELP.items())
    prompt = (f"A forum user asks about a classic MG car:\n\"\"\"{q}\"\"\"\n\n"
              f"Which ONE topic is the question mainly about?\n{menu}\n- none: none of these\n\n"
              "Answer with just the topic word.")
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 8,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
    w = out.strip().lower().strip(".'\" ")
    return w if w in TOPICS or w == "none" else "none"


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        r.setdefault("topic_lex", r["topic"])
        r["topic"] = ask(r["q"])
    kept = [r for r in rows if r["topic"] != "none"]
    agree = sum(r["topic"] == r["topic_lex"] for r in rows)
    print(f"{len(rows)} questions, model agrees with keyword label on {agree}, 'none' on {len(rows)-len(kept)}")
    for t in TOPICS:
        print(f"  {t:11}{sum(r['topic']==t for r in kept):>3}")
    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
