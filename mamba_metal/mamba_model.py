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

    def prefill(self, x):
        y, conv_state, ssm_state = self.mixer.prefill(self.norm(x))
        return x + y, conv_state, ssm_state

    def step(self, x_token, conv_state, ssm_state):
        y, new_conv, new_ssm = self.mixer.step(self.norm(x_token), conv_state, ssm_state)
        return x_token + y, new_conv, new_ssm


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

    def init_state(self, batch_size: int = 1) -> tuple[list[mx.array], list[mx.array]]:
        """Zero-initialised conv & SSM state for incremental decoding."""
        d_inner = self.cfg.expand * self.cfg.d_model
        conv_states = [
            mx.zeros((batch_size, d_inner, self.cfg.d_conv), dtype=mx.float32)
            for _ in self.layers
        ]
        ssm_states = [
            mx.zeros((batch_size, d_inner, self.cfg.d_state), dtype=mx.float32)
            for _ in self.layers
        ]
        return conv_states, ssm_states

    def prefill(
        self, input_ids: mx.array
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        """Process a (B, L) prompt in parallel.

        Returns (logits: (B, L, vocab), conv_states, ssm_states).
        Use ``logits[:, -1, :]`` as the next-token distribution and the
        returned state lists to continue with ``step`` for O(1)/tok decode.
        """
        x = self.embeddings(input_ids)
        conv_states, ssm_states = [], []
        for layer in self.layers:
            x, cs, ss = layer.prefill(x)
            conv_states.append(cs)
            ssm_states.append(ss)
        x = self.norm_f(x)
        logits = x @ self.embeddings.weight.T
        return logits, conv_states, ssm_states

    def step(
        self,
        input_ids: mx.array,            # (B, 1) of int
        conv_states: list[mx.array],
        ssm_states: list[mx.array],
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        """Process a single new token, carrying state across calls. O(1) per step."""
        x = self.embeddings(input_ids)
        new_conv, new_ssm = [], []
        for layer, cs, ss in zip(self.layers, conv_states, ssm_states):
            x, cs_new, ss_new = layer.step(x, cs, ss)
            new_conv.append(cs_new)
            new_ssm.append(ss_new)
        x = self.norm_f(x)
        logits = x @ self.embeddings.weight.T
        return logits, new_conv, new_ssm
