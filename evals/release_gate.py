"""
release_gate.py — pass/fail gate for a candidate model version vs the previous live one.

The thresholds below were agreed with the owner on 2026-11-26 BEFORE any candidate data
existed (spec: docs/superpowers/specs/2026-11-26-size-comparison-eval-design.md). Do not edit
them after seeing results; if they are wrong, change them in a commit that says so.

    python -m evals.release_gate MANIFEST.json     # exit 0 PASS, 1 FAIL, 2 INCOMPLETE

Manifest: {"candidate": V, "previous": V, "agreement": "agreement.json"} where
V = {"name", "eval": [3 jsonl], "judge": [3 jsonl], "summary": [3 json]}.
"""
import json
import statistics
import sys

from evals.topics import intrusion_rates, question_shares

INTRUSION_MAX_RATIO = 1.5      # a topic may leak into replies to other-topic questions at most 1.5x its question share
EMPTY_MAX = 0.05               # at most 5% empty replies
VAL_LOSS_MAX_WORSE = 0.05      # val loss may be at most 0.05 worse than previous
MIN_SEEDS = 3
MIN_KAPPA = 0.6                # judge-vs-human agreement bar, set by owner 2026-11-26 (Qwen scored 0.29 on 32 hand-scored replies => judge indicative only)
# judge noise = sample stdev of the previous version's per-seed judge means


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _judge_means(paths):
    out = []
    for p in paths:
        s = [r["score"] for r in _jsonl(p) if r.get("score") is not None]
        out.append(statistics.mean(s))
    return out


def evaluate(m):
    cand, prev = m["candidate"], m["previous"]
    problems = []
    for label, v in (("candidate", cand), ("previous", prev)):
        for key in ("eval", "judge", "summary"):
            if len(v.get(key, [])) < MIN_SEEDS:
                problems.append(f"{label} has {len(v.get(key, []))} {key} files, need {MIN_SEEDS}")
    if MIN_KAPPA is None:
        problems.append("MIN_KAPPA is not set: the owner sets it after the judge-vs-human check (Task 5)")
    if problems:
        return {"checks": {}, "verdict": "INCOMPLETE", "problems": problems}

    checks = {}
    # 1. intrusion: pooled over seeds, per topic, against the absolute cap
    rows = [r for p in cand["eval"] for r in _jsonl(p)]
    shares, intr = question_shares(rows), intrusion_rates(rows)
    bad = []
    for t, rate in intr.items():
        if rate is None or t not in shares:
            continue
        cap = INTRUSION_MAX_RATIO * shares[t]
        if rate > cap:
            bad.append(f"{t} {rate:.0%} > cap {cap:.0%}")
    checks["intrusion"] = {"status": "FAIL" if bad else "PASS",
                           "detail": "; ".join(bad) or "every topic under its cap"}

    # 2. judge: mean of per-seed means vs previous, within noise = TIE
    cm, pm = _judge_means(cand["judge"]), _judge_means(prev["judge"])
    noise = statistics.stdev(pm)
    diff = statistics.mean(cm) - statistics.mean(pm)
    verdict = "WIN" if diff > noise else "FAIL" if diff < -noise else "TIE"
    status = "FAIL" if verdict == "FAIL" else "PASS"
    detail = f"{verdict}: candidate {statistics.mean(cm):.2f} vs previous {statistics.mean(pm):.2f} (noise {noise:.2f})"
    with open(m["agreement"]) as f:
        kappa = json.load(f)["kappa"]
    if kappa < MIN_KAPPA:
        status, detail = "INDICATIVE", detail + f"; judge indicative only (kappa {kappa:.2f} < {MIN_KAPPA})"
    checks["judge"] = {"status": status, "detail": detail}

    # 3. empty replies and val loss
    sums = [json.load(open(p)) for p in cand["summary"]]
    empties = [mode["empty_rate"] for s in sums for mode in s["modes"].values()]
    e = statistics.mean(empties)
    checks["empty"] = {"status": "PASS" if e <= EMPTY_MAX else "FAIL", "detail": f"{e:.1%} (max {EMPTY_MAX:.0%})"}
    cv = next((s["val_loss"] for s in sums if s.get("val_loss") is not None), None)
    pv = next((s["val_loss"] for s in (json.load(open(p)) for p in prev["summary"]) if s.get("val_loss") is not None), None)
    if cv is None or pv is None:
        return {"checks": checks, "verdict": "INCOMPLETE", "problems": ["val loss missing in a summary"]}
    checks["val_loss"] = {"status": "PASS" if cv <= pv + VAL_LOSS_MAX_WORSE else "FAIL",
                          "detail": f"{cv:.3f} vs previous {pv:.3f} (max +{VAL_LOSS_MAX_WORSE})"}

    verdict = "FAIL" if any(c["status"] == "FAIL" for c in checks.values()) else "PASS"
    return {"checks": checks, "verdict": verdict}


def main(argv):
    r = evaluate(json.load(open(argv[0])))
    for name, c in r["checks"].items():
        print(f"{c['status']:11}{name:11}{c['detail']}")
    for p in r.get("problems", []):
        print("INCOMPLETE:", p)
    print("VERDICT:", r["verdict"])
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[r["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
