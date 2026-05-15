// selective_scan_chunked — Mamba selective scan for arbitrary seqlen.
//
// Same as selective_scan.metal, but the seqlen is chunked into pieces of
// CHUNK_SIZE (1024) and the per-state running prefix (a, b) is carried
// in threadgroup memory across chunks.
//
// This mirrors Mamba CUDA's smem_running_prefix mechanism.
//
// Constraints: dstate <= 64 (MAX_DSTATE), fp32, no z gate / D / softplus.
//
// Inputs:
//   device const float* u, delta    (batch, dim, seqlen)
//   device const float* A           (dim, dstate)
//   device const float* B, C        (batch, dstate, seqlen)
//   device const uint&  batch, dim, dstate, seqlen
// Output:
//   device float* y                 (batch, dim, seqlen)
// Dispatch: grid = (1024, batch, dim), threadgroup = (1024, 1, 1)

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

// Initialise per-state carries to identity (1, 0)
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

        // SIMD pair scan
        for (uint d = 1u; d < 32u; d <<= 1) {
            float a_prev = simd_shuffle_up(a, d);
            float b_prev = simd_shuffle_up(b, d);
            if (lane >= d) {
                b = a * b_prev + b;
                a = a * a_prev;
            }
        }

        // Bridge via threadgroup memory
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
        // a * ca is the per-thread cumulative a, not needed past this point;
        // the only a we need going forward is the block-total for the carry.

        if (in_range) {
            y_t += b * C_st;
        }

        // Update carry: new_carry = block_total ∘ old_carry
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
