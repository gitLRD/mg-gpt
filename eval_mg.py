"""
eval_mg.py — Fixed-prompt evaluation for MG-GPT checkpoints.

Usage:
    python eval_mg.py --ckpt checkpoints/ckpt_best.pt --data-dir data/v7_mgonly
    python eval_mg.py --ckpt X.pt --modes bare,handler --heldout 20 --out eval_runs/v7.jsonl

Reports, per prompt mode:
  - val loss on <data-dir>/val.bin (full non-overlapping sweep, deterministic)
  - empty rate:  replies that post-process to "" (the handler would show its canned fallback)
  - cheap coherence proxies: keyword hit rate, [URL]/[NAME] leak rate, repeated-trigram rate

Modes isolate where empty replies come from:
  bare             "USER: q\\nASSISTANT:"                       (matches training format)
  handler          persona priming + "USER: q\\nASSISTANT: "    (exactly what lambda/handler.py sends)
  handler_nospace  persona priming + "USER: q\\nASSISTANT:"     (handler minus the trailing space)
  short_nospace    first priming pair only + "USER: q\\nASSISTANT:"

Raw generations go to --out (JSONL) so they can be read, not just counted.
"""

import argparse
import ast
import json
import os
import random
import re
import sys

import numpy as np
import tiktoken
import torch

from model.gpt import GPT

# (question, keywords a sensible answer would plausibly touch — crude relevance proxy)
PROMPTS = [
    ("My MGB GT runs rough and the temperature gauge climbs in traffic. What should I check?",
     ["coolant", "radiator", "thermostat", "fan", "water", "head gasket", "hose", "timing", "mixture"]),
    ("What's the difference between a chrome bumper and a rubber bumper MGB?",
     ["rubber", "chrome", "1974", "bumper", "ride height", "1975", "heavier", "emission"]),
    ("How often should I adjust the tappets on a B-series engine?",
     ["tappet", "clearance", "thou", "feeler", "valve", "mile", "adjust", "rocker"]),
    ("My MG ZR won't hold coolant, where do I start?",
     ["coolant", "head gasket", "k-series", "k series", "hose", "radiator", "pressure test", "thermostat", "housing"]),
    ("What spark plugs should I fit to an MGB?",
     ["plug", "champion", "ngk", "gap", "n9y", "bp", "thou", "heat range"]),
    ("My SU carbs are hunting at idle. What's wrong?",
     ["su", "carb", "jet", "needle", "mixture", "float", "dashpot", "oil", "idle", "linkage"]),
    ("Is it worth converting my MGB to an electric fan?",
     ["fan", "electric", "thermostat", "switch", "radiator", "relay", "cooling"]),
    ("What oil should I use in a classic MG?",
     ["20w50", "20w-50", "oil", "multigrade", "zddp", "classic", "castrol", "viscosity"]),
    ("The brakes on my MG Midget feel spongy. Where should I look first?",
     ["brake", "fluid", "bleed", "hose", "master cylinder", "caliper", "air", "dot"]),
    ("Does an MG TF 135 suffer from the same head gasket problem as the ZR?",
     ["head gasket", "k-series", "k series", "tf", "135", "cooling", "coolant", "engine"]),
    ("What should I look for when buying a rusty MGB roadster?",
     ["sill", "rust", "floor", "arch", "chassis", "inner wing", "wing", "panel", "weld"]),
    ("Why does my MG ZS diesel struggle to start on cold mornings?",
     ["glow", "plug", "diesel", "battery", "starter", "cold", "fuel", "injector"]),
    ("How do I set the ignition timing on an MGB with a distributor?",
     ["timing", "degree", "btdc", "strobe", "distributor", "vacuum", "advance", "points", "gap"]),
    ("My MGB has an oil leak from the rear of the engine. Common culprit?",
     ["rear", "seal", "oil", "crankshaft", "gasket", "sump", "leak", "rocker"]),
    ("Is the MG Metro 6R4 a sensible daily driver?",
     ["metro", "6r4", "rally", "group b", "v6", "engine"]),
    ("What upgrades make an MGB handle better?",
     ["anti-roll", "bar", "damper", "shock", "bush", "suspension", "polyurethane", "spring", "tyre"]),
    ("My MG ZT's gearbox crunches going into second. Any ideas?",
     ["gearbox", "synchro", "oil", "clutch", "second", "gear", "synchromesh", "transmission"]),
    ("How do I know if my MGB's overdrive is working properly?",
     ["overdrive", "solenoid", "oil", "switch", "gearbox", "laycock", "relay", "fuse"]),
    ("What's a fair price for an MGB GT in good condition?",
     ["price", "£", "condition", "gt", "pound", "value", "mgb"]),
    ("Why won't my MG start when it's hot but starts fine cold?",
     ["fuel", "heat", "vapour", "coil", "starter", "hot", "pump", "ignition"]),
]

STOP = None  # filled in after the tokenizer is built

# --- helpers ---------------------------------------------------------------

def load_personas():
    """Pull PERSONAS out of lambda/handler.py without importing boto3."""
    src = open("lambda/handler.py", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PERSONAS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("PERSONAS not found in lambda/handler.py")


def build_prompt(mode, persona, q):
    if mode == "bare":
        return f"USER: {q}\nASSISTANT:"
    priming = persona["priming"]
    if mode == "handler":
        return f"{priming}USER: {q}\nASSISTANT: "
    if mode == "handler_nospace":
        return f"{priming}USER: {q}\nASSISTANT:"
    if mode == "short_nospace":
        # Only the first USER/ASSISTANT pair of the priming (~1/4 the tokens).
        first_pair = priming[: priming.index("USER:", 1)]
        return f"{first_pair}USER: {q}\nASSISTANT:"
    raise ValueError(mode)


def postprocess(text):
    """Mirror of the handler's cleanup, enough to decide 'would this be empty?'."""
    for marker in ("\nUSER:", "\nASSISTANT:", "USER:", "ASSISTANT:"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    return re.sub(r"^[\s!?.,;:\-*]+", "", text).strip()


def repeated_trigram_rate(text):
    w = text.lower().split()
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0.0 if not tri else 1 - len(set(tri)) / len(tri)


@torch.no_grad()
def val_loss(model, val_bin, block, device):
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    losses = []
    for i in range(0, len(data) - block - 1, block):
        x = torch.from_numpy(data[i:i + block].astype(np.int64))[None].to(device)
        y = torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64))[None].to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses)), len(losses)


def heldout_questions(data_dir, n, seed=0):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    stems = [s for s in meta.get("val_threads", [])
             if os.path.exists(f"data/corpus_synth_qa/{s}.txt")]
    random.Random(seed).shuffle(stems)
    out = []
    for s in stems[:n]:
        m = re.match(r"USER:\s*(.+?)\nASSISTANT:", open(f"data/corpus_synth_qa/{s}.txt", encoding="utf-8").read(), re.S)
        if m:
            out.append((m.group(1).strip(), []))
    return out


# --- main ------------------------------------------------------------------

def main():
    global STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--persona", default="workshop_veteran")
    ap.add_argument("--modes", default="bare,handler,handler_nospace")
    ap.add_argument("--heldout", type=int, default=0, help="also use N questions from held-out threads")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSONL of raw generations")
    ap.add_argument("--no-val", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    STOP = [enc.encode("\nUSER:"), enc.encode("\nASSISTANT:"), enc.encode("\n\n")]

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    cfg.vocab_size = 50257
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"ckpt {args.ckpt}: {cfg.n_layer}L/{cfg.n_embd}D block={cfg.block_size} "
          f"iter={ck.get('iter_num')} best_val={ck.get('best_val_loss')} device={device}")

    if not args.no_val:
        vl, nwin = val_loss(model, os.path.join(args.data_dir, "val.bin"), cfg.block_size, device)
        print(f"val loss on {args.data_dir}/val.bin: {vl:.4f}  ({nwin} windows of {cfg.block_size})")

    persona = load_personas()[args.persona]
    prompts = list(PROMPTS) + (heldout_questions(args.data_dir, args.heldout) if args.heldout else [])
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    if out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for mode in args.modes.split(","):
        torch.manual_seed(args.seed)
        n = empty = leak = kw_hit = kw_n = 0
        lens, reps = [], []
        for q, kws in prompts:
            ptoks = enc.encode(build_prompt(mode, persona, q))
            max_prompt = cfg.block_size - args.max_new
            ptoks = ptoks if len(ptoks) <= max_prompt else ptoks[-max_prompt:]
            x = torch.tensor(ptoks, dtype=torch.long)[None].to(device)
            with torch.no_grad():
                gen = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p,
                                     repetition_penalty=args.rep_penalty, stop_token_sequences=STOP)
            raw = enc.decode(gen[0].tolist()[len(ptoks):])
            clean = postprocess(raw)
            n += 1
            empty += (clean == "")
            leak += bool(re.search(r"\[(URL|NAME|EMAIL|PHONE|ADDRESS)\]", clean))
            lens.append(len(clean.split()))
            reps.append(repeated_trigram_rate(clean))
            if kws:
                kw_n += 1
                kw_hit += any(k in clean.lower() for k in kws)
            if out:
                out.write(json.dumps({"mode": mode, "q": q, "raw": raw, "clean": clean}) + "\n")
        print(f"\n[{mode}] n={n}  empty={empty / n:.0%}  [TAG] leak={leak / n:.0%}  "
              f"keyword-hit={(kw_hit / kw_n if kw_n else float('nan')):.0%}  "
              f"mean words={np.mean(lens):.0f}  repeated-trigram={np.mean(reps):.2f}")
    if out:
        out.close()
        print(f"\nraw generations -> {args.out}")


if __name__ == "__main__":
    main()
