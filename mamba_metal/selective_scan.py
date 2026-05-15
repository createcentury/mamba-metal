"""High-level Python entry point for the selective scan Metal kernel."""

import mlx.core as mx

from mamba_metal._loader import load_kernel


TG_SIZE = 1024
MAX_DSTATE = 64


_kernel = load_kernel(
    name="selective_scan_chunked",
    input_names=[
        "u", "delta", "A", "B", "C", "D", "z",
        "batch", "dim", "dstate", "seqlen",
        "apply_softplus", "use_D", "use_z",
    ],
    output_names=["y"],
)


def selective_scan(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    *,
    D: mx.array | None = None,
    z: mx.array | None = None,
    delta_softplus: bool = False,
) -> mx.array:
    """Mamba selective scan with optional D / z / softplus.

    Shapes:
      u, delta: (batch, dim, seqlen)
      A: (dim, dstate)
      B, C: (batch, dstate, seqlen)
      D: (dim,) — additive skip connection coefficient (optional)
      z: (batch, dim, seqlen) — gating signal; applied as SiLU(z) elementwise (optional)
      delta_softplus: if True, delta is mapped through softplus before use
      returns y: (batch, dim, seqlen)
    """
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    assert delta.shape == u.shape
    assert B.shape == (batch, dstate, seqlen) == C.shape
    assert dstate <= MAX_DSTATE

    use_D = D is not None
    use_z = z is not None

    # Pass zero buffers when features are off (the kernel ignores them anyway,
    # but MLX still binds the buffers and the kernel reads D[dim_id] unconditionally).
    if D is None:
        D = mx.zeros((dim,), dtype=u.dtype)
    if z is None:
        z = mx.zeros((1,), dtype=u.dtype)  # tiny dummy; kernel guarded by use_z

    (y,) = _kernel(
        inputs=[
            u, delta, A, B, C, D, z,
            mx.array(batch, dtype=mx.uint32),
            mx.array(dim, dtype=mx.uint32),
            mx.array(dstate, dtype=mx.uint32),
            mx.array(seqlen, dtype=mx.uint32),
            mx.array(1 if delta_softplus else 0, dtype=mx.uint32),
            mx.array(1 if use_D else 0, dtype=mx.uint32),
            mx.array(1 if use_z else 0, dtype=mx.uint32),
        ],
        output_shapes=[u.shape],
        output_dtypes=[u.dtype],
        grid=(TG_SIZE, batch, dim),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return y
