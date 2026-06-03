"""
Model configurations for MG-GPT.

Start with 'nano' to verify your pipeline works, then graduate to 'small'.
The difference in output quality between nano and small is significant.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Sequence / vocabulary
    block_size: int = 512       # Max context window (tokens)
    vocab_size: int = 50257     # GPT-2 BPE tokeniser vocabulary size

    # Architecture
    n_layer: int = 6            # Number of transformer blocks
    n_head: int = 6             # Number of attention heads per block
    n_embd: int = 384           # Embedding dimension (must be divisible by n_head)

    # Regularisation
    dropout: float = 0.1        # Dropout probability (set to 0 for inference)
    bias: bool = True           # Use bias in linear layers and layer norms


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def nano_config() -> GPTConfig:
    """~1.5M params. Runs on CPU. Good for checking the pipeline works."""
    return GPTConfig(
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )


def small_config() -> GPTConfig:
    """~10M params. Runs well on Apple MPS. Recommended starting point."""
    return GPTConfig(
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    )


def medium_config() -> GPTConfig:
    """~50M params. Overnight on M-series. Much better language quality."""
    return GPTConfig(
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
    )
