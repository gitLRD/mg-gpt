"""
generate.py — Interactive generation from a trained MG-GPT checkpoint.

Usage:
    python generate.py
    python generate.py --prompt "The SU carburettor on an MGB"
    python generate.py --checkpoint checkpoints/ckpt_best.pt --temperature 0.8
"""

import argparse
import torch
import tiktoken

from model.gpt import GPT
from model.config import GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   default="checkpoints/ckpt_best.pt")
parser.add_argument("--prompt",       default="The MGB GT was")
parser.add_argument("--max_tokens",   type=int,   default=200)
parser.add_argument("--temperature",  type=float, default=0.8,
                    help="1.0=neutral, <1=focused, >1=wild")
parser.add_argument("--top_k",        type=int,   default=50,
                    help="Sample from top-k tokens only (reduces incoherence)")
parser.add_argument("--num_samples",  type=int,   default=3)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print(f"Loading checkpoint: {args.checkpoint}")
with torch.serialization.safe_globals([GPTConfig]):
    ckpt = torch.load(args.checkpoint, map_location=device)
config = ckpt["config"]
config.dropout = 0.0   # Disable dropout at inference

model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)

print(f"Model: {model.num_params()/1e6:.1f}M params  |  "
      f"Trained to step {ckpt['iter_num']}  |  "
      f"Best val loss: {ckpt['best_val_loss']:.4f}")

# ---------------------------------------------------------------------------
# Tokenise prompt and generate
# ---------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")
prompt_tokens = enc.encode(args.prompt)
x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

print(f"\nPrompt: \"{args.prompt}\"")
print(f"Temperature: {args.temperature}  |  top_k: {args.top_k}\n")
print("=" * 60)

for i in range(args.num_samples):
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    output = enc.decode(y[0].tolist())
    print(f"\n[Sample {i+1}]\n{output}")
    print("-" * 60)
