"""
MG-GPT: A minimal GPT implementation for learning.

Architecture: GPT-2 style decoder-only transformer.
Read this top to bottom — it's the whole model.

References:
  - Attention Is All You Need (Vaswani et al., 2017)
  - GPT-2 (Radford et al., 2019)
  - nanoGPT (Karpathy, 2022) — https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (look-left-only) mask.

    Each token can attend to all *previous* tokens but not future ones.
    This is what makes it a language model rather than a bidirectional encoder.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "Embedding dim must be divisible by number of heads"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Single linear projects input to Q, K, V all at once (3x for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — a triangular matrix that prevents attending to future tokens.
        # Registered as a buffer so it moves to the right device automatically.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Compute Q, K, V in one shot then split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into (batch, heads, seq_len, head_dim)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Scale by 1/sqrt(head_dim) to keep gradients stable
        scale = 1.0 / math.sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * scale          # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v                                        # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with a GELU activation between them.
    Expands to 4x the embedding dim internally — a GPT-2 convention.
    This is where the model does most of its 'thinking'.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """
    One transformer block = LayerNorm → Attention → residual + LayerNorm → MLP → residual.

    The residual connections are critical: they let gradients flow directly
    back through the network without vanishing. Pre-norm (norm before attention)
    is used here, which trains more stably than the original post-norm paper.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention sub-layer
        x = x + self.mlp(self.ln_2(x))    # feed-forward sub-layer
        return x


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT: a stack of transformer blocks preceded by embeddings.

    Token embeddings + positional embeddings → N blocks → layer norm → linear head.
    The linear head projects back to vocabulary size to produce logits.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language model head: projects embeddings → vocab logits
        # Weight tying: shares weights with the token embedding matrix.
        # Intuition: the same vector that means "MGB" as input should score
        # high when predicting "MGB" as output.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Initialise weights
        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2 * n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"MG-GPT initialised: {self.num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Build position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Run through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute cross-entropy loss over all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference: only need logits for the last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, repetition_penalty=1.0, stop_token_sequences=None):
        """
        Autoregressively generate tokens given a prompt.

        temperature:        >1 = more random, <1 = more focused
        top_k:              keep only the top-k highest-prob tokens before sampling
        top_p:              nucleus sampling — keep smallest token set with
                            cumulative prob >= top_p (applied after top_k)
        repetition_penalty: divide positive logits of already-generated tokens
                            by this factor (and multiply negative ones). 1.0
                            disables. ~1.1–1.3 is the useful range — kills the
                            mode-collapse loops a small model is prone to.
        stop_token_sequences: list of token-id lists; if the generated tail
                            matches any of these, generation halts and the
                            stop sequence is stripped from the output.
        """
        stop_seqs = stop_token_sequences or []
        prompt_len = idx.size(1)

        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is getting long
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                           else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Repetition penalty (CTRL formulation): only applied to tokens that
            # have actually been generated this turn, not to the prompt itself —
            # otherwise priming words would be unfairly suppressed.
            if repetition_penalty != 1.0 and idx.size(1) > prompt_len:
                generated = idx[0, prompt_len:].unique()
                if generated.numel() > 0:
                    token_logits = logits[0, generated]
                    token_logits = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
                    logits[0, generated] = token_logits

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Nucleus filter — composes after top_k. Always keeps at least one
            # token (the highest-prob one) so we never end up with no choices.
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative > top_p
                # Shift right so we always include the first token that crosses the threshold
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, remove_mask)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop sequence check — match against the generated tail only.
            for stop in stop_seqs:
                if 0 < len(stop) <= idx.size(1) - prompt_len:
                    tail = idx[0, -len(stop):].tolist()
                    if tail == stop:
                        return idx[:, :-len(stop)]

        return idx
