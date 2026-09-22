"""
train.py — Training loop for MG-GPT.

Usage:
    python train.py                     # uses 'small' config
    python train.py --config nano       # faster, for testing
    python train.py --config medium     # overnight run

Checkpoints are saved to checkpoints/ every eval_interval steps.
Resume from a checkpoint with --resume checkpoints/ckpt_1000.pt
"""

import os
import sys
import math
import time
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — safe on MPS/remote
import matplotlib.pyplot as plt

from model.gpt import GPT
from model.config import nano_config, small_config, medium_config

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="small", choices=["nano", "small", "medium"])
parser.add_argument("--resume", default=None, help="Path to checkpoint .pt file")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Training
max_iters      = 10_000      # Total training steps
batch_size     = 4           # Sequences per batch (small for MPS memory headroom)
grad_accum     = 4           # micro-batch × grad_accum = effective batch (16)

# Optimiser
learning_rate  = 3e-4
weight_decay   = 0.1
beta1, beta2   = 0.9, 0.95
grad_clip      = 1.0         # Gradient clipping (prevents exploding gradients)

# Learning rate schedule (cosine decay with warmup)
warmup_iters   = 100
lr_decay_iters = max_iters
min_lr         = learning_rate / 10

# Evaluation
eval_interval  = 250         # Evaluate on val set every N steps
eval_iters     = 50          # Number of val batches to average
log_interval   = 10          # Print loss every N steps
save_interval  = 500         # Save checkpoint every N steps

# Paths
data_dir       = "data"
checkpoint_dir = "checkpoints"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device: {device}")
dtype = torch.float32  # MPS doesn't support bfloat16 reliably yet

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_tokens(split):
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        print(f"\n❌ {path} not found. Run: python data/prepare.py")
        sys.exit(1)
    return np.memmap(path, dtype=np.uint16, mode='r')

train_data = load_tokens("train")
val_data   = load_tokens("val")

with open(os.path.join(data_dir, "meta.json")) as f:
    meta = json.load(f)

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

configs = {"nano": nano_config, "small": small_config, "medium": medium_config}
config = configs[args.config]()
config.vocab_size = meta["vocab_size"]

model = GPT(config).to(device)

iter_num = 0
best_val_loss = float('inf')

if args.resume:
    print(f"Resuming from {args.resume}")
    # weights_only=False — we wrote this checkpoint ourselves and need the
    # GPTConfig object back. PyTorch 2.6 flipped the default.
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    iter_num = ckpt["iter_num"]
    best_val_loss = ckpt.get("best_val_loss", float('inf'))

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

# Separate parameters into those that should and shouldn't have weight decay.
# Embeddings and layer norms shouldn't decay — only weight matrices.
decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
optim_groups = [
    {"params": decay_params,    "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

if args.resume and "optimizer" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_batch(split):
    data = train_data if split == "train" else val_data
    block_size = config.block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def get_lr(it):
    """Cosine decay with linear warmup."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

os.makedirs(checkpoint_dir, exist_ok=True)
print(f"\nTraining MG-GPT ({args.config} config, {model.num_params()/1e6:.1f}M params)")
print(f"Max iters: {max_iters}  |  Batch size: {batch_size}  |  Block size: {config.block_size}\n")

t0 = time.time()
model.train()
train_losses = []   # per-iter losses for the loss curve plot
val_checkpoints = []  # (iter, val_loss) pairs

for iter_num in range(iter_num, max_iters + 1):

    # Update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate periodically
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {iter_num:5d} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        val_checkpoints.append((iter_num, losses['val']))

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            path = os.path.join(checkpoint_dir, "ckpt_best.pt")
            torch.save(ckpt, path)
            print(f"  ✅ New best val loss — saved to {path}")

    if iter_num % save_interval == 0 and iter_num > 0:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        }
        path = os.path.join(checkpoint_dir, f"ckpt_{iter_num}.pt")
        torch.save(ckpt, path)

    if iter_num == max_iters:
        break

    # Forward + backward with gradient accumulation.
    # Accumulate gradients over `grad_accum` micro-batches before stepping,
    # giving an effective batch size of batch_size * grad_accum.
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for micro_step in range(grad_accum):
        X, Y = get_batch("train")
        _, loss = model(X, Y)
        loss = loss / grad_accum   # normalise so sum == mean over all micro-batches
        loss.backward()
        accum_loss += loss.item()

    # Gradient clipping — prevents a single bad batch from destroying weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = (t1 - t0) / max(iter_num, 1)
        remaining = dt * (max_iters - iter_num) / 60
        print(f"  iter {iter_num:4d} | loss {accum_loss:.4f} | "
              f"~{remaining:.0f}m remaining")
    train_losses.append(accum_loss)

print("\n🏁 Training complete.")

# ---- loss curve ----
if train_losses:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, alpha=0.4, linewidth=0.8, label="train (per iter)")
    if val_checkpoints:
        v_iters, v_losses = zip(*val_checkpoints)
        ax.plot(v_iters, v_losses, "o-", linewidth=1.5, label="val")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"MG-GPT training — {args.config} config")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {plot_path}")
