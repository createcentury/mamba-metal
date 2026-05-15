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

    def prefill(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Process a prompt of length L in parallel and return the per-state
        caches needed for incremental decode.

        Returns (y, conv_state, ssm_state) where
          y          : (B, L, d_model) — block output
          conv_state : (B, d_inner, d_conv) — last d_conv pre-conv values
          ssm_state  : (B, d_inner, d_state) — h^{(s)} at position L-1
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_pre_conv, z = mx.split(xz, 2, axis=-1)              # each (B, L, d_inner)

        # Conv-state buffer = the d_conv pre-conv values ending at position L-1.
        # Left-pad with zeros when L < d_conv (causal conv convention).
        if L >= self.d_conv:
            conv_state = x_pre_conv[:, L - self.d_conv : L, :]
        else:
            pad = mx.zeros(
                (B, self.d_conv - L, self.d_inner), dtype=x_pre_conv.dtype
            )
            conv_state = mx.concatenate([pad, x_pre_conv], axis=1)
        conv_state = mx.transpose(conv_state, (0, 2, 1))      # (B, d_inner, d_conv)

        x_main = self.conv1d(x_pre_conv)[:, :L, :]
        x_main = nn.silu(x_main)

        x_dbl = self.x_proj(x_main)
        dt = x_dbl[..., : self.dt_rank]
        B_ssm = x_dbl[..., self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_dbl[..., self.dt_rank + self.d_state :]
        dt = self.dt_proj(dt)

        A = -mx.exp(self.A_log)

        y, ssm_state = selective_scan(
            u=mx.transpose(x_main, (0, 2, 1)),
            delta=mx.transpose(dt, (0, 2, 1)),
            A=A,
            B=mx.transpose(B_ssm, (0, 2, 1)),
            C=mx.transpose(C_ssm, (0, 2, 1)),
            D=self.D,
            z=mx.transpose(z, (0, 2, 1)),
            delta_softplus=True,
            return_state=True,
        )
        y = mx.transpose(y, (0, 2, 1))                        # (B, L, d_inner)
        y = self.out_proj(y)
        return y, conv_state, ssm_state

    def step(
        self,
        x_token: mx.array,        # (B, 1, d_model)
        conv_state: mx.array,     # (B, d_inner, d_conv)
        ssm_state: mx.array,      # (B, d_inner, d_state)
    ) -> tuple[mx.array, mx.array, mx.array]:
        """O(1) per-token forward — updates carried state in place of running the scan.

        Returns (y, new_conv_state, new_ssm_state).
        """
        xz = self.in_proj(x_token)                            # (B, 1, 2*d_inner)
        x_main, z = mx.split(xz, 2, axis=-1)
        x_main = x_main[:, 0, :]                              # (B, d_inner)
        z = z[:, 0, :]                                        # (B, d_inner)

        # Causal conv via sliding window: shift left, append new value.
        new_conv = mx.concatenate(
            [conv_state[:, :, 1:], x_main[:, :, None]], axis=2
        )                                                     # (B, d_inner, d_conv)
        # conv1d.weight shape is (d_inner, d_conv, 1) after our HF transpose.
        w = self.conv1d.weight.squeeze(-1)                    # (d_inner, d_conv)
        x_conv = (new_conv * w[None, :, :]).sum(axis=-1) + self.conv1d.bias
        x_conv = nn.silu(x_conv)                              # (B, d_inner)

        x_dbl = self.x_proj(x_conv)                           # (B, dt_rank + 2*d_state)
        dt_pre = x_dbl[:, : self.dt_rank]
        B_ssm = x_dbl[:, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_dbl[:, self.dt_rank + self.d_state :]
        dt = self.dt_proj(dt_pre)                             # (B, d_inner)
        dt = nn.softplus(dt)                                  # match selective_scan's softplus

        A = -mx.exp(self.A_log)                               # (d_inner, d_state)

        # SSM step: h_new = exp(dt*A) * h_prev + dt*x*B
        a = mx.exp(dt[:, :, None] * A[None, :, :])            # (B, d_inner, d_state)
        b = (dt * x_conv)[:, :, None] * B_ssm[:, None, :]     # (B, d_inner, d_state)
        new_ssm = a * ssm_state + b

        y = (new_ssm * C_ssm[:, None, :]).sum(axis=-1)        # (B, d_inner)
        y = y + self.D * x_conv                               # D skip
        y = y * nn.silu(z)                                    # z gate

        y = self.out_proj(y)[:, None, :]                      # (B, 1, d_model)
        return y, new_conv, new_ssm
