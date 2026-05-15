"""Step 3: Selective Scan — minimal version.

For each (batch, dim), compute:
    y[t] = sum_{s=0..N-1} h_t^{(s)} * C[s, t]
    h_t^{(s)} = exp(delta[t] * A[s]) * h_{t-1}^{(s)} + delta[t] * u[t] * B[s, t]

Kernel layout:
  - grid = (TG_SIZE, batch, dim)
  - one threadgroup per (batch, dim) pair
  - inside the threadgroup, threads handle different t (seqlen positions)
  - state dimension is iterated sequentially in the inner loop

Constraints (lifted in later steps):
  - seqlen <= TG_SIZE (1024)
  - fp32, no complex
  - kIsVariableB = kIsVariableC = true (selective B, C)
  - no z gate, no D skip, no delta_softplus
  - kNRows = 1
"""

import numpy as np
import mlx.core as mx


TG_SIZE = 1024


SELECTIVE_SCAN_SRC = """
    uint t = thread_position_in_threadgroup.x;
    uint batch_id = threadgroup_position_in_grid.y;
    uint dim_id = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint n_sg = simdgroups_per_threadgroup;

    threadgroup float warp_a[32];
    threadgroup float warp_b[32];

    bool in_range = (t < seqlen);

    // Load u[batch_id, dim_id, t] and delta[batch_id, dim_id, t]
    uint udx = batch_id * dim * seqlen + dim_id * seqlen + t;
    float u_t = in_range ? u[udx] : 0.0;
    float delta_t = in_range ? delta[udx] : 0.0;

    float y_t = 0.0;

    for (uint s = 0; s < dstate; ++s) {
        // Ensure all threads finished the previous iteration's reads of warp_*
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // a_t = exp(delta_t * A[dim_id, s]),  b_t = delta_t * u_t * B[batch_id, s, t]
        float A_ds = A[dim_id * dstate + s];
        uint bcdx = batch_id * dstate * seqlen + s * seqlen + t;
        float B_st = in_range ? B[bcdx] : 0.0;
        float C_st = in_range ? C[bcdx] : 0.0;

        float a = in_range ? exp(delta_t * A_ds) : 1.0;
        float b = in_range ? (delta_t * u_t * B_st) : 0.0;

        // SIMD-level inclusive scan over pair composition
        for (uint d = 1u; d < 32u; d <<= 1) {
            float a_prev = simd_shuffle_up(a, d);
            float b_prev = simd_shuffle_up(b, d);
            if (lane >= d) {
                b = a * b_prev + b;
                a = a * a_prev;
            }
        }

        // Bridge across SIMD groups via threadgroup memory
        if (lane == 31u) {
            warp_a[sg] = a;
            warp_b[sg] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float ta = (lane < n_sg) ? warp_a[lane] : 1.0;
            float tb = (lane < n_sg) ? warp_b[lane] : 0.0;
            for (uint d = 1u; d < 32u; d <<= 1) {
                float ta_prev = simd_shuffle_up(ta, d);
                float tb_prev = simd_shuffle_up(tb, d);
                if (lane >= d) {
                    tb = ta * tb_prev + tb;
                    ta = ta * ta_prev;
                }
            }
            if (lane < n_sg) {
                warp_a[lane] = ta;
                warp_b[lane] = tb;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg > 0u) {
            float ca = warp_a[sg - 1u];
            float cb = warp_b[sg - 1u];
            b = a * cb + b;
            a = a * ca;
        }

        // b now holds h_t for this state_idx s (since h_{-1} = 0)
        // Accumulate y_t += h * C
        if (in_range) {
            y_t += b * C_st;
        }
    }

    if (in_range) {
        y[batch_id * dim * seqlen + dim_id * seqlen + t] = y_t;
    }
"""


k_selective_scan = mx.fast.metal_kernel(
    name="selective_scan",
    input_names=["u", "delta", "A", "B", "C", "batch", "dim", "dstate", "seqlen"],
    output_names=["y"],
    source=SELECTIVE_SCAN_SRC,
)


def selective_scan_metal(
    u: mx.array, delta: mx.array, A: mx.array, B: mx.array, C: mx.array
) -> mx.array:
    """Shapes:
      u, delta: (batch, dim, seqlen)
      A: (dim, dstate)
      B, C: (batch, dstate, seqlen)
      returns y: (batch, dim, seqlen)
    """
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    assert delta.shape == u.shape
    assert B.shape == (batch, dstate, seqlen) == C.shape
    assert seqlen <= TG_SIZE, "Step 3 limited to seqlen <= 1024 (Step 5 lifts this)"

    (y,) = k_selective_scan(
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


def selective_scan_ref(
    u: np.ndarray, delta: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Numpy reference. Faithful to the recurrence, no fused tricks."""
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    y = np.zeros_like(u)
    for bi in range(batch):
        for di in range(dim):
            h = np.zeros(dstate, dtype=np.float32)
            for t in range(seqlen):
                a_t = np.exp(delta[bi, di, t] * A[di])  # (N,)
                b_t = delta[bi, di, t] * u[bi, di, t] * B[bi, :, t]  # (N,)
                h = a_t * h + b_t
                y[bi, di, t] = np.dot(h, C[bi, :, t])
    return y


def run_test(batch: int, dim: int, dstate: int, seqlen: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    # delta typically small positive (post softplus). Use uniform(0.01, 0.1).
    delta = rng.uniform(0.01, 0.1, size=(batch, dim, seqlen)).astype(np.float32)
    # A is typically negative diagonal
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)

    y_ref = selective_scan_ref(u, delta, A, B, C)
    y_metal = np.array(
        selective_scan_metal(
            mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C)
        )
    )
    mx.synchronize()

    abs_err = np.max(np.abs(y_metal - y_ref))
    rel_err = np.max(np.abs(y_metal - y_ref) / (np.abs(y_ref) + 1e-6))
    label = f"B={batch} D={dim} N={dstate} T={seqlen}"
    print(f"  {label:<28}  abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    print("Selective scan (minimal MSL impl) vs numpy reference\n")
    run_test(batch=1, dim=1, dstate=4, seqlen=32)
    run_test(batch=1, dim=1, dstate=16, seqlen=128)
    run_test(batch=2, dim=4, dstate=16, seqlen=256)
    run_test(batch=2, dim=8, dstate=16, seqlen=1000)
    run_test(batch=2, dim=8, dstate=16, seqlen=1024)


if __name__ == "__main__":
    main()
