import json

import pytest

from evals import release_gate as g


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _eval_rows(fuel_leak):
    """20 rows: 10 fuel questions, 10 brakes questions. fuel_leak = share of brakes replies mentioning fuel."""
    rows = [{"topic": "fuel", "clean": "check the carburettor"} for _ in range(10)]
    n_leak = round(10 * fuel_leak)
    rows += [{"topic": "brakes", "clean": "check the fuel line" if i < n_leak else "bleed the brakes"} for i in range(10)]
    return rows


def _version(tmp_path, name, fuel_leak=0.0, judge=(3.0, 3.0, 3.0), empty=0.0, val=3.7, n_seeds=3, n_judge=3):
    ev = [_write(tmp_path / f"{name}_e{i}.jsonl", _eval_rows(fuel_leak)) for i in range(n_seeds)]
    ju = [_write(tmp_path / f"{name}_j{i}.jsonl", [{"score": s}, {"score": s}]) for i, s in enumerate(judge[:n_judge])]
    su = []
    for i in range(n_seeds):
        p = tmp_path / f"{name}_s{i}.json"
        p.write_text(json.dumps({"seed": i, "val_loss": val if i == 0 else None,
                                 "modes": {"bare": {"n": 20, "empty_rate": empty}}}))
        su.append(str(p))
    return {"name": name, "eval": ev, "judge": ju, "summary": su}


def _manifest(tmp_path, cand, prev, kappa=0.8):
    ag = tmp_path / "agree.json"
    ag.write_text(json.dumps({"kappa": kappa}))
    return {"candidate": cand, "previous": prev, "agreement": str(ag)}


@pytest.fixture(autouse=True)
def kappa_set(monkeypatch):
    monkeypatch.setattr(g, "MIN_KAPPA", 0.6)


def test_pass_case(tmp_path):
    m = _manifest(tmp_path, _version(tmp_path, "c"), _version(tmp_path, "p", judge=(3.0, 3.1, 2.9)))
    r = g.evaluate(m)
    assert r["verdict"] == "PASS", r


def test_intrusion_fail(tmp_path):
    # fuel questions are 50% of the set, so cap is 75%; make the cap bite with a smaller fuel share
    rows = [{"topic": "fuel", "clean": "carb"}] * 2 + [{"topic": "brakes", "clean": "fuel fuel"}] * 8
    ev = [_write(tmp_path / f"e{i}.jsonl", rows) for i in range(3)]
    cand = _version(tmp_path, "c")
    cand["eval"] = ev
    r = g.evaluate(_manifest(tmp_path, cand, _version(tmp_path, "p")))
    assert r["checks"]["intrusion"]["status"] == "FAIL" and r["verdict"] == "FAIL"


def test_judge_tie_within_noise(tmp_path):
    prev = _version(tmp_path, "p", judge=(3.0, 3.4, 3.2))   # stdev 0.2
    cand = _version(tmp_path, "c", judge=(3.1, 3.1, 3.1))   # 0.1 below the mean 3.2
    r = g.evaluate(_manifest(tmp_path, cand, prev))
    assert r["checks"]["judge"]["status"] == "PASS" and "TIE" in r["checks"]["judge"]["detail"]


def test_judge_fail_below_noise(tmp_path):
    prev = _version(tmp_path, "p", judge=(3.0, 3.4, 3.2))
    cand = _version(tmp_path, "c", judge=(2.0, 2.0, 2.0))
    assert g.evaluate(_manifest(tmp_path, cand, prev))["checks"]["judge"]["status"] == "FAIL"


def test_empty_and_valloss_fail(tmp_path):
    prev = _version(tmp_path, "p")
    r = g.evaluate(_manifest(tmp_path, _version(tmp_path, "c", empty=0.10), prev))
    assert r["checks"]["empty"]["status"] == "FAIL"
    r = g.evaluate(_manifest(tmp_path, _version(tmp_path, "c2", val=3.8), prev))
    assert r["checks"]["val_loss"]["status"] == "FAIL"


def test_missing_seed_incomplete(tmp_path, capsys):
    cand = _version(tmp_path, "c", n_seeds=2)
    m = _manifest(tmp_path, cand, _version(tmp_path, "p"))
    assert g.evaluate(m)["verdict"] == "INCOMPLETE"
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m))
    assert g.main([str(mp)]) == 2


def test_fewer_than_two_judge_files_incomplete(tmp_path):
    prev = _version(tmp_path, "p", n_judge=1)
    assert g.evaluate(_manifest(tmp_path, _version(tmp_path, "c"), prev))["verdict"] == "INCOMPLETE"


def test_min_kappa_unset_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "MIN_KAPPA", None)
    r = g.evaluate(_manifest(tmp_path, _version(tmp_path, "c"), _version(tmp_path, "p")))
    assert r["verdict"] == "INCOMPLETE" and "Task 5" in json.dumps(r)


def test_low_kappa_indicative(tmp_path):
    r = g.evaluate(_manifest(tmp_path, _version(tmp_path, "c"), _version(tmp_path, "p"), kappa=0.3))
    assert r["checks"]["judge"]["status"] == "INDICATIVE"
    assert r["verdict"] == "PASS"
