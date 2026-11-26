from evals.topics import (TOPICS, tag_topics, primary_topic, rates_per_1000_words,
                          intrusion_rates, question_shares)


def test_topic_keys():
    assert set(TOPICS) == {"fuel", "cooling", "brakes", "electrical", "oil", "body",
                           "gearbox", "susp", "price"}


def test_fuel_false_positives():
    assert "fuel" not in tag_topics("the septic tank by the jetty is lean-burn free")


def test_fuel_true_positives():
    assert "fuel" in tag_topics("my SU carb main jet and fuel pump")
    assert "fuel" in tag_topics("there is a leak from the petrol tank")


def test_earth_needs_context():
    assert "electrical" not in tag_topics("the earth is round")
    assert "electrical" in tag_topics("check the earth strap")


def test_primary_topic_tie_and_none():
    assert primary_topic("hello there") is None
    assert primary_topic("brake fluid near the fuel pipe") is None  # 1 brakes vs 1 fuel
    assert primary_topic("brakes caliper pads") == "brakes"


def test_rates_empty_and_zero_words():
    for r in (rates_per_1000_words([]), rates_per_1000_words([""])):
        assert set(r) == set(TOPICS) and all(v == 0.0 for v in r.values())


def test_rates_value():
    text = "fuel fuel " + " ".join(["word"] * 998)
    assert rates_per_1000_words([text])["fuel"] == 2.0


def test_intrusion_rate():
    rows = [
        {"topic": "brakes", "clean": "check the fuel pump"},
        {"topic": "brakes", "clean": "bleed the pads"},
        {"topic": "fuel", "clean": "clean the carb"},
        {"topic": "fuel", "clean": "set the jets"},
    ]
    assert intrusion_rates(rows)["fuel"] == 0.5
    only_fuel = [{"topic": "fuel", "clean": "carb"}]
    assert intrusion_rates(only_fuel)["fuel"] is None


def test_intrusion_empty_reply_and_unlabelled_rows():
    rows = [{"topic": "brakes", "clean": ""}, {"topic": None, "clean": "fuel"}]
    r = intrusion_rates(rows)
    assert r["fuel"] == 0.0  # unlabelled row ignored, empty reply counts as no intrusion


def test_question_shares_sum():
    rows = [{"topic": "fuel"}, {"topic": "fuel"}, {"topic": "brakes"}, {"topic": None}]
    s = question_shares(rows)
    assert abs(sum(s.values()) - 1.0) < 1e-9
    assert abs(s["fuel"] - 2 / 3) < 1e-9
