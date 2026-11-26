"""
topic_report.py — print topic spread for eval runs.

    python -m evals.topic_report RUN.jsonl [RUN2.jsonl ...] [--json out.json]

Per file: mentions per 1,000 words by topic, and (when rows carry a question `topic` label)
the question shares and how often each topic leaks into replies to other-topic questions.
With several files for one model (e.g. 3 seeds) the JSON also holds the pooled numbers.
"""

import argparse
import json

from evals.topics import TOPICS, intrusion_rates, question_shares, rates_per_1000_words


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows):
    rates = rates_per_1000_words([r.get("clean", "") for r in rows])
    labelled = [r for r in rows if r.get("topic")]
    return {
        "n": len(rows),
        "n_labelled": len(labelled),
        "rates_per_1000_words": rates,
        "intrusion": intrusion_rates(rows) if labelled else None,
        "question_shares": question_shares(rows) if labelled else None,
    }


def fmt(v, pct=False):
    if v is None:
        return "   n/a"
    return f"{100 * v:5.1f}%" if pct else f"{v:6.1f}"


def print_summary(name, s):
    print(f"\n== {name}  (n={s['n']}, labelled={s['n_labelled']})")
    print(f"{'topic':12}{'per 1000w':>10}{'q share':>9}{'intrusion':>11}{'cap 1.5x':>10}")
    for t in TOPICS:
        share = (s["question_shares"] or {}).get(t)
        intr = (s["intrusion"] or {}).get(t)
        cap = None if share is None else 1.5 * share
        print(f"{t:12}{fmt(s['rates_per_1000_words'][t]):>10}{fmt(share, True):>9}"
              f"{fmt(intr, True):>11}{fmt(cap, True):>10}")
    if s["intrusion"] is None:
        print("(no `topic` labels in these rows: intrusion skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", help="write all summaries (and pooled, if >1 file) here")
    args = ap.parse_args()
    out, pooled = {}, []
    for p in args.files:
        rows = load(p)
        pooled += rows
        out[p] = summarize(rows)
        print_summary(p, out[p])
    if len(args.files) > 1:
        out["pooled"] = summarize(pooled)
        print_summary("pooled", out["pooled"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
