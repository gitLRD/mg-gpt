"""
questions.py — build and load the balanced, topic-labelled held-out question set.

Questions come from the USER turns of synthetic Q&A written for held-out (val) threads, so
none of them was ever trained on. A question is kept only if exactly one topic wins the
lexicon (ambiguous or topic-less questions are dropped, not forced). Each topic gets at
most `per_topic` questions and each thread at most 2, so no thread or topic dominates.

    python -m evals.questions      # writes data/eval/balanced_questions.jsonl and prints counts
"""

import argparse
import json
import os
import random
import re

from evals.topics import TOPICS, primary_topic

MAX_PER_THREAD = 2


def build_balanced(candidates, per_topic, seed):
    """candidates: (thread_stem, question). Returns [{"q","topic","thread"}], possibly short."""
    rng = random.Random(seed)
    cands = list(dict.fromkeys(candidates))  # drop exact duplicates, keep order
    rng.shuffle(cands)
    taken, per_thread, seen_q, out = {t: 0 for t in TOPICS}, {}, set(), []
    for thread, q in cands:
        topic = primary_topic(q)
        if topic is None or q in seen_q:
            continue
        if taken[topic] >= per_topic or per_thread.get(thread, 0) >= MAX_PER_THREAD:
            continue
        taken[topic] += 1
        per_thread[thread] = per_thread.get(thread, 0) + 1
        seen_q.add(q)
        out.append({"q": q, "topic": topic, "thread": thread})
    return out


def top_up(rows, written, per_topic):
    """Tag rows as held-out, then add hand-written questions to topics still below per_topic.

    Written questions were never in any training data, so they are as 'held out' as the rest,
    but the file keeps `source` so results can be reported with and without them.
    """
    out = [dict(r, source="held-out") for r in rows]
    have = {r["q"] for r in out}
    for r in written:
        n = sum(x["topic"] == r["topic"] for x in out)
        if n < per_topic and r["q"] not in have:
            out.append(dict(r, source="written"))
            have.add(r["q"])
    return out


def load_questions(path):
    """(question, [], topic) triples, the shape eval_mg.py uses."""
    with open(path, encoding="utf-8") as f:
        return [(r["q"], [], r["topic"]) for r in map(json.loads, filter(str.strip, f))]


def val_candidates(meta_path="data/v7_mgonly/meta.json", qa_dir="data/corpus_synth_qa"):
    stems = json.load(open(meta_path))["val_threads"]
    out = []
    for s in stems:
        p = os.path.join(qa_dir, f"{s}.txt")
        if os.path.exists(p):
            text = open(p, encoding="utf-8").read()
            out += [(s, q.strip()) for q in re.findall(r"USER:\s*(.+?)\nASSISTANT:", text, re.S)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--written", default="data/eval/written_questions.jsonl",
                    help="hand-written questions that top up topics the held-out threads cannot fill")
    ap.add_argument("--out", default="data/eval/balanced_questions.jsonl")
    args = ap.parse_args()
    cands = val_candidates()
    rows = build_balanced(cands, args.per_topic, args.seed)
    if args.written and os.path.exists(args.written):
        written = [json.loads(l) for l in open(args.written, encoding="utf-8") if l.strip()]
        rows = top_up(rows, written, args.per_topic)
    else:
        rows = [dict(r, source="held-out") for r in rows]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(cands)} candidate questions -> {len(rows)} kept -> {args.out}")
    for t in TOPICS:
        n = sum(r["topic"] == t for r in rows)
        w = sum(r["topic"] == t and r["source"] == "written" for r in rows)
        print(f"  {t:11}{n:>3}  ({w} written){'  <-- under 12' if n < 12 else ''}")


if __name__ == "__main__":
    main()
