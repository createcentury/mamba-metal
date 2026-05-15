"""High-level Python entry point for the selective scan Metal kernel."""

import mlx.core as mx

from mamba_metal._loader import load_kernel


TG_SIZE = 1024
MAX_DSTATE = 64


_kernel = load_kernel(
    name="selective_scan_chunked",
    input_names=["u", "delta", "A", "B", "C", "batch", "dim", "dstate", "seqlen"],
    output_names=["y"],
)


def selective_scan(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
) -> mx.array:
    """Mamba selective scan (chunked).

    Shapes:
      u, delta: (batch, dim, seqlen)
      A: (dim, dstate)
      B, C: (batch, dstate, seqlen)
      returns y: (batch, dim, seqlen)

    Constraints: dstate <= 64, fp32, no z gate / D / softplus / complex.
    """
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    assert delta.shape == u.shape
    assert B.shape == (batch, dstate, seqlen) == C.shape
    assert dstate <= MAX_DSTATE, f"dstate {dstate} exceeds MAX_DSTATE={MAX_DSTATE}"

    (y,) = _kernel(
        inputs=[
            u, delta, A, B, C,
            mx.array(batch, dtype=mx.uint32),
            mx.array(dim, dtype=mx.uint32),
            mx.array(dstate, dtype=mx.uint32),
            mx.array(seqlen, dtype=mx.uint32),
        ],
        output_shapes=[u.shape],
        output_dtypes=[u.dtype],
        grid=(TG_SIZE, batch, dim),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return y
