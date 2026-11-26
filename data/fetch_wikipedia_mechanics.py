"""
fetch_wikipedia_mechanics.py — pull general car-mechanics Wikipedia articles,
aimed at the topics v7 under-produces (brakes, suspension, electrical, body,
tyres) plus classic-era parts. Same method, licence (CC BY-SA 4.0) and
attribution manifest as fetch_wikipedia_mg.py, which it reuses.

The User-Agent is a generic project name on purpose (no personal contact).
Every title is a guess at an article name; missing ones are skipped and listed.

Usage:
    python data/fetch_wikipedia_mechanics.py --out-dir data/corpus_wikipedia_mechanics
"""

import argparse
import json
import os
import time

import fetch_wikipedia_mg as w

w.UA = "mg-gpt-corpus-research/0.1 (personal learning project; urllib)"

TITLES = {
    "brakes": [
        "Brake", "Drum brake", "Disc brake", "Brake fluid", "Master cylinder", "Brake bleeding",
        "Brake pad", "Brake servo", "Handbrake", "Hydraulic brake", "Brake lining", "Brake fade",
        "Lockheed Corporation", "Girling", "Anti-lock braking system", "Brake booster",
    ],
    "suspension": [
        "Suspension (vehicle)", "Leaf spring", "Shock absorber", "Anti-roll bar", "Wishbone suspension",
        "Double wishbone suspension", "Live axle", "Independent suspension", "Coil spring", "Wheel alignment",
        "Camber angle", "Toe (automotive)", "Caster angle", "Rack and pinion", "Recirculating ball",
        "Power steering", "Ball joint", "Kingpin", "Trunnion", "Rubber bushing", "Lever arm damper",
    ],
    "electrical": [
        "Alternator (automotive)", "Dynamo", "Car battery", "Lead–acid battery", "Ignition system",
        "Distributor", "Ignition coil", "Spark plug", "Contact breaker", "Starter motor", "Car alarm",
        "Fuse (electrical)", "Automotive lighting", "Positive earth", "Negative earth", "Magneto",
        "Electronic ignition", "Wiring harness", "Voltage regulator", "Headlamp",
    ],
    "body_tyres": [
        "Corrosion", "Rust", "Galvanization", "Monocoque", "Body-on-frame", "Chassis", "Car body style",
        "Roadster (automobile)", "Convertible", "Tire", "Radial tire", "Bias-ply tire", "Wire wheel",
        "Wheel", "Alloy wheel", "Spare tire", "Tire code", "Underbody", "Sill (automotive)", "Welding",
        "Car restoration", "Car rust prevention", "Vehicle bodywork", "Paint", "Chrome plating",
    ],
    "engine_drivetrain_classic": [
        "Clutch", "Gearbox", "Manual transmission", "Synchronizer", "Differential (mechanical device)",
        "Propeller shaft", "Hypoid gear", "Overdrive (mechanics)", "Hardy Spicer", "Universal joint",
        "Head gasket", "Cylinder head", "Camshaft", "Crankshaft", "Piston ring", "Engine oil", "Oil pressure",
        "Carburetor", "Fuel pump", "Choke (valve)", "Radiator (engine cooling)", "Thermostat",
        "Water pump", "Engine knocking", "Cooling system", "Exhaust system", "Catalytic converter",
        "Overheating (engine)", "Compression ratio", "Valve clearance", "Tappet", "Timing chain",
        "Hydrolock", "Oil filter", "Engine tuning", "Twin carburetor", "Weber carburetor",
    ],
    "classic_car_ownership": [
        "Classic car", "Vintage car", "Car club", "Motor vehicle service", "MOT test", "Vehicle inspection",
        "Roadworthiness", "Pre-war car", "Sports car", "Lotus Elan", "Triumph TR4", "Triumph Spitfire",
        "Austin-Healey Sprite", "Morgan Plus 4", "Jaguar E-Type", "Austin Seven", "Riley Motor",
        "Wolseley Motors", "Austin Motor Company", "Nuffield Organization", "Leyland Motors",
        "Rover Company", "Standard Motor Company", "Rootes Group", "Rubery Owen",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/corpus_wikipedia_mechanics")
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true", help="only list titles")
    args = ap.parse_args()
    titles = sorted({t for ts in TITLES.values() for t in ts})
    print(f"{len(titles)} titles")
    if args.dry_run:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    topic_of = {t: k for k, ts in TITLES.items() for t in ts}

    manifest, seen_ids, missing = [], set(), []
    for i, t in enumerate(titles, 1):
        d = w.get(dict(action="query", prop="extracts|revisions|info", titles=t, explaintext=1,
                       exsectionformat="wiki", rvprop="ids|timestamp", inprop="url"))
        page = next(iter(d["query"]["pages"].values()))
        if "missing" in page or "extract" not in page:
            missing.append(t)
            print(f"[{i}/{len(titles)}] missing: {t}")
            continue
        if page["pageid"] in seen_ids:
            continue
        seen_ids.add(page["pageid"])
        text = w.clean(page["extract"])
        if len(text) < args.min_chars:
            print(f"[{i}/{len(titles)}] short ({len(text)}): {page['title']}")
            continue
        fname = w.slug(page["title"]) + ".txt"
        with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"{page['title']}\n\n{text}\n")
        rev = page["revisions"][0]
        manifest.append({"title": page["title"], "file": fname, "chars": len(text),
                         "topic": topic_of.get(t), "url": page["fullurl"],
                         "revid": rev["revid"], "timestamp": rev["timestamp"],
                         "licence": "CC BY-SA 4.0"})
        print(f"[{i}/{len(titles)}] {page['title']} ({len(text):,} chars)")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"source": "en.wikipedia.org", "licence": "CC BY-SA 4.0 (text)",
                   "retrieved": time.strftime("%Y-%m-%d"), "pages": manifest,
                   "missing_titles": missing}, f, indent=1)
    total = sum(p["chars"] for p in manifest)
    print(f"\n{len(manifest)} pages, {total:,} chars (~{total // 4:,} tokens), "
          f"{len(missing)} missing -> {args.out_dir}")


if __name__ == "__main__":
    main()
