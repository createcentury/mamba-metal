"""Step 5: selective scan with chunking — handles arbitrary seqlen.

For seqlen > TG_SIZE (1024), we process in chunks of TG_SIZE. Between chunks
the per-state running prefix (a, b) is carried in threadgroup memory.

This mirrors Mamba's smem_running_prefix mechanism in
selective_scan_fwd_kernel.cuh — state is kept in SRAM, not bounced through
global memory between chunks.

For each (batch, dim, state) the carry holds the cumulative pair through
all earlier chunks. After scanning chunk c, the new carry is the chunk's
block-total pair composed with the previous carry.
"""

import numpy as np
import mlx.core as mx


TG_SIZE = 1024
MAX_DSTATE = 64  # threadgroup memory budget for per-state carries


CHUNKED_SRC = """
    uint t = thread_position_in_threadgroup.x;
    uint batch_id = threadgroup_position_in_grid.y;
    uint dim_id = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint n_sg = simdgroups_per_threadgroup;

    threadgroup float warp_a[32];
    threadgroup float warp_b[32];
    threadgroup float carry_a[64];  // MAX_DSTATE
    threadgroup float carry_b[64];

    // Initialize per-state carries to identity (1, 0)
    if (t < dstate) {
        carry_a[t] = 1.0;
        carry_b[t] = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk_size = 1024u;
    uint n_chunks = (seqlen + chunk_size - 1u) / chunk_size;

    for (uint c = 0; c < n_chunks; ++c) {
        uint global_t = c * chunk_size + t;
        bool in_range = global_t < seqlen;

        uint udx = batch_id * dim * seqlen + dim_id * seqlen + global_t;
        float u_t = in_range ? u[udx] : 0.0;
        float delta_t = in_range ? delta[udx] : 0.0;

        float y_t = 0.0;

        for (uint s = 0; s < dstate; ++s) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float A_ds = A[dim_id * dstate + s];
            uint bcdx = batch_id * dstate * seqlen + s * seqlen + global_t;
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
                float ca_intra = warp_a[sg - 1u];
                float cb_intra = warp_b[sg - 1u];
                b = a * cb_intra + b;
                a = a * ca_intra;
            }

            // Combine with the inter-chunk carry from previous chunks
            float ca = carry_a[s];
            float cb = carry_b[s];
            b = a * cb + b;
            // (a * ca would be the new per-thread cumulative a, but we don't
            //  need it again — only the block-total a matters for the carry.)

            // h_t = b after combining with carry (initial state assumed 0)
            if (in_range) {
                y_t += b * C_st;
            }

            // Update carry to (block_total ∘ old_carry) for next chunk
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0u) {
                float block_a = warp_a[n_sg - 1u];
                float block_b = warp_b[n_sg - 1u];
                carry_a[s] = block_a * ca;
                carry_b[s] = block_a * cb + block_b;
            }
        }

        if (in_range) {
            y[batch_id * dim * seqlen + dim_id * seqlen + global_t] = y_t;
        }
    }
"""


k_chunked = mx.fast.metal_kernel(
    name="selective_scan_chunked",
    input_names=["u", "delta", "A", "B", "C", "batch", "dim", "dstate", "seqlen"],
    output_names=["y"],
    source=CHUNKED_SRC,
)


def selective_scan_metal(
    u: mx.array, delta: mx.array, A: mx.array, B: mx.array, C: mx.array
) -> mx.array:
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    assert dstate <= MAX_DSTATE
    (y,) = k_chunked(
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


def selective_scan_ref(u, delta, A, B, C):
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    y = np.zeros_like(u)
    for bi in range(batch):
        for di in range(dim):
            h = np.zeros(dstate, dtype=np.float32)
            for t in range(seqlen):
                a_t = np.exp(delta[bi, di, t] * A[di])
                b_t = delta[bi, di, t] * u[bi, di, t] * B[bi, :, t]
                h = a_t * h + b_t
                y[bi, di, t] = np.dot(h, C[bi, :, t])
    return y


def run_test(batch: int, dim: int, dstate: int, seqlen: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    delta = rng.uniform(0.01, 0.1, size=(batch, dim, seqlen)).astype(np.float32)
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)

    y_ref = selective_scan_ref(u, delta, A, B, C)
    y_metal = np.array(selective_scan_metal(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C)
    ))
    mx.synchronize()

    abs_err = np.max(np.abs(y_metal - y_ref))
    mask = np.abs(y_ref) > 1e-3  # exclude near-zero values for rel err
    rel_err = (np.max(np.abs(y_metal[mask] - y_ref[mask]) / np.abs(y_ref[mask]))
               if mask.any() else 0.0)
    n_chunks = (seqlen + 1023) // 1024
    print(f"  B={batch} D={dim} N={dstate} T={seqlen:>5} (chunks={n_chunks}):  "
          f"abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    print("Chunked selective scan vs numpy reference\n")
    # Within one chunk (regression vs Step 3)
    run_test(batch=1, dim=2, dstate=8, seqlen=512)
    run_test(batch=1, dim=2, dstate=8, seqlen=1024)
    # Multiple chunks
    run_test(batch=1, dim=2, dstate=8, seqlen=1025)
    run_test(batch=1, dim=4, dstate=16, seqlen=2048)
    run_test(batch=2, dim=4, dstate=16, seqlen=4096)
    run_test(batch=1, dim=2, dstate=16, seqlen=8192)
    run_test(batch=1, dim=2, dstate=16, seqlen=16384)


if __name__ == "__main__":
    main()
