"""
topics.py — a crude keyword lexicon that says which car topics a piece of text is about,
plus the "topic spread" numbers built on it.

Why it exists: the live model answers too many questions with fuel and cooling words, and
none of our older metrics (val loss, keyword hits, 1-5 judge) noticed. These helpers measure
that directly: how often each topic shows up in replies (rates), and how often a topic
leaks into replies to questions about OTHER topics (intrusion).

It is a lexicon, so it is crude. Ambiguous words ("tank", "jet", "lean", "earth") only count
in car-specific phrases, so "septic tank" or "jetty" do not count as fuel.
"""

import re

_F = re.IGNORECASE


def _c(pattern):
    return re.compile(pattern, _F)


TOPICS = {
    "fuel": _c(
        r"\b(fuel|petrol|diesel|gasoline|carb(?:s|uretto?rs?)?|injectors?|injection|octane|ethanol|e10"
        r"|rich mixture|lean mixture|float bowl|su carbs?"
        r"|(?:fuel|petrol) tanks?|(?:main|slow|needle|su) jets?)\b"
    ),
    "cooling": _c(
        r"\b(coolant|antifreeze|radiators?|thermostat|overheat\w*|head gasket|water pump"
        r"|temperature gauge|cooling)\b"
    ),
    "brakes": _c(r"\b(brake\w*|calipers?|master cylinder|bleed\w*|servo|discs?|pads?)\b"),
    "electrical": _c(
        r"\b(batter(?:y|ies)|alternator|dynamo|starter|wiring|fuses?|relays?|ignition|coils?"
        r"|distributor|spark plugs?|points|earthing|earth (?:strap|lead|wire|fault))\b"
    ),
    "oil": _c(r"\b(oil|20w|zddp|sump|seals?)\b"),
    "body": _c(r"\b(rust\w*|sills?|floors?|arch(?:es)?|chassis|panels?|weld\w*|bumpers?|paint|body)\b"),
    "gearbox": _c(r"\b(gearbox|clutch|synchro\w*|overdrive|gears?|transmission|diff\w*)\b"),
    "susp": _c(
        r"\b(suspension|dampers?|shocks?|shock absorbers?|springs?|anti-roll|bush\w*|steering"
        r"|tyres?|handling)\b"
    ),
    # £ has no word boundary, so it sits outside the \b group
    "price": _c(r"(£|\b(?:price|value|worth|cost)\b)"),
}


def _count(topic, text):
    return len(TOPICS[topic].findall(text or ""))


def tag_topics(text):
    """Set of topics mentioned at least once in text."""
    return {t for t in TOPICS if _count(t, text) > 0}


def primary_topic(text):
    """Topic with strictly the most matches; None if nothing matches or the top is a tie."""
    counts = {t: _count(t, text) for t in TOPICS}
    best = max(counts.values())
    if best == 0:
        return None
    top = [t for t, n in counts.items() if n == best]
    return top[0] if len(top) == 1 else None


def rates_per_1000_words(texts):
    """Mentions per 1,000 words over all texts, per topic (all 0.0 if there are no words)."""
    words = sum(len((t or "").split()) for t in texts)
    if words == 0:
        return {t: 0.0 for t in TOPICS}
    return {t: 1000.0 * sum(_count(t, x) for x in texts) / words for t in TOPICS}


def intrusion_rates(rows):
    """For each topic T: share of replies to questions NOT about T that mention T.

    rows: dicts with `topic` (the question's label) and `clean` (the reply). Rows without a
    label are ignored. None when no row has a different topic (nothing to measure).
    """
    labelled = [r for r in rows if r.get("topic")]
    out = {}
    for t in TOPICS:
        others = [r for r in labelled if r["topic"] != t]
        out[t] = None if not others else sum(_count(t, r.get("clean")) > 0 for r in others) / len(others)
    return out


def question_shares(rows):
    """Fraction of labelled rows carrying each topic label (unlabelled rows ignored)."""
    labelled = [r["topic"] for r in rows if r.get("topic")]
    if not labelled:
        return {}
    return {t: labelled.count(t) / len(labelled) for t in set(labelled)}
