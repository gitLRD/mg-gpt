import json

from evals.questions import build_balanced, load_questions


def test_ambiguous_dropped():
    out = build_balanced([("t1", "brake fluid leaked onto the fuel pipe")], per_topic=5, seed=0)
    assert out == []


def test_per_topic_cap_and_thread_cap():
    cands = [(f"t{i % 3}", f"my brake pads squeal {i}") for i in range(30)]
    out = build_balanced(cands, per_topic=5, seed=0)
    assert len(out) == 5 and all(r["topic"] == "brakes" for r in out)
    # 3 threads x 2 questions max = 6 >= 5, and no thread more than twice
    per_thread = {}
    for r in out:
        per_thread[r["thread"]] = per_thread.get(r["thread"], 0) + 1
    assert max(per_thread.values()) <= 2


def test_thread_cap_limits_total():
    cands = [("only", f"my brake pads squeal {i}") for i in range(10)]
    assert len(build_balanced(cands, per_topic=5, seed=0)) == 2


def test_shortfall_does_not_raise():
    out = build_balanced([("a", "radiator leaking coolant")], per_topic=17, seed=0)
    assert [r["topic"] for r in out] == ["cooling"]


def test_deterministic_and_seed_changes_order():
    cands = [(f"t{i}", f"my brake pads squeal {i}") for i in range(40)]
    a = build_balanced(cands, per_topic=10, seed=1)
    assert a == build_balanced(cands, per_topic=10, seed=1)
    assert a != build_balanced(cands, per_topic=10, seed=2)


def test_duplicate_questions_dropped():
    cands = [("a", "my brake pads squeal"), ("b", "my brake pads squeal")]
    assert len(build_balanced(cands, per_topic=5, seed=0)) == 1


def test_load_questions_roundtrip(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"q": "q1", "topic": "fuel", "thread": "a"}, {"q": "q2", "topic": "brakes", "thread": "b"}]) + "\n")
    assert load_questions(str(p)) == [("q1", [], "fuel"), ("q2", [], "brakes")]
