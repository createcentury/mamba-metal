"""One Mamba block built on top of the Metal selective scan kernel.

Layout (single layer, inference-only):

    x: (B, L, d_model)
    ↓ in_proj (Linear)
    [x_main, z]: each (B, L, d_inner)
    ↓ conv1d (causal, depth-wise)  →  SiLU
    ↓ x_proj → split → (dt_pre, B_ssm, C_ssm)
    ↓ dt_proj on dt_pre
    ↓ selective_scan with A = -exp(A_log), D, z, delta_softplus=True
    ↓ out_proj
    out: (B, L, d_model)

This mirrors the official Mamba block (without the residual stream / LayerNorm
that wraps it; those are typically part of the outer model).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mamba_metal.selective_scan import selective_scan


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # left+right; right side is sliced off for causality
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log parameterises A = -exp(A_log), so A is negative.
        # Init: A[d, n] = n+1, then store log so the parameter is unconstrained.
        A_init = mx.broadcast_to(
            mx.arange(1, d_state + 1, dtype=mx.float32),
            (self.d_inner, d_state),
        )
        self.A_log = mx.log(A_init)
        self.D = mx.ones(self.d_inner, dtype=mx.float32)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, d_model)
        _, L, _ = x.shape

        xz = self.in_proj(x)                                  # (B, L, 2*d_inner)
        x_main, z = mx.split(xz, 2, axis=-1)                  # each (B, L, d_inner)

        # Causal depth-wise conv: pad d_conv-1 on both sides, drop the trailing
        # d_conv-1 outputs so each timestep depends only on earlier ones.
        x_main = self.conv1d(x_main)[:, :L, :]                # (B, L, d_inner)
        x_main = nn.silu(x_main)

        # Selective parameters from the post-conv state
        x_dbl = self.x_proj(x_main)                           # (B, L, dt_rank + 2*d_state)
        dt = x_dbl[..., : self.dt_rank]
        B_ssm = x_dbl[..., self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_dbl[..., self.dt_rank + self.d_state :]
        dt = self.dt_proj(dt)                                 # (B, L, d_inner)

        A = -mx.exp(self.A_log)                               # (d_inner, d_state)

        # selective_scan expects channel-first: (B, dim, seqlen)
        y = selective_scan(
            u=mx.transpose(x_main, (0, 2, 1)),
            delta=mx.transpose(dt, (0, 2, 1)),
            A=A,
            B=mx.transpose(B_ssm, (0, 2, 1)),
            C=mx.transpose(C_ssm, (0, 2, 1)),
            D=self.D,
            z=mx.transpose(z, (0, 2, 1)),
            delta_softplus=True,
        )
        y = mx.transpose(y, (0, 2, 1))                        # back to (B, L, d_inner)

        return self.out_proj(y)
