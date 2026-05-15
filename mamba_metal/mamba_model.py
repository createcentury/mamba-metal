"""Full Mamba causal language model — stack of (RMSNorm → MambaBlock + residual) layers
followed by a final norm and a tied LM head.

Attribute layout mirrors HuggingFace's `state-spaces/mamba-130m-hf` so that
weight loading reduces to stripping the "backbone." prefix and a single
conv1d-weight transpose:

    HF                                            ours
    backbone.embeddings.weight                 ↔ embeddings.weight
    backbone.layers.{i}.norm.weight            ↔ layers.{i}.norm.weight
    backbone.layers.{i}.mixer.<X>              ↔ layers.{i}.mixer.<X>
    backbone.norm_f.weight                     ↔ norm_f.weight
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mamba_metal.mamba_block import MambaBlock


@dataclass
class MambaConfig:
    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 50280
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None  # auto = ceil(d_model / 16) if None
    rms_norm_eps: float = 1e-5


class MambaResidualBlock(nn.Module):
    """Pre-norm + residual wrapper around a MambaBlock."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mixer = MambaBlock(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dt_rank=cfg.dt_rank,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.mixer(self.norm(x))


class MambaModel(nn.Module):
    """Stack of MambaResidualBlocks with tied LM head."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [MambaResidualBlock(cfg) for _ in range(cfg.n_layer)]
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """input_ids: (B, L) of int → logits (B, L, vocab_size)."""
        x = self.embeddings(input_ids)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        # Tied LM head: logits = x @ embeddings.weight.T
        return x @ self.embeddings.weight.T
