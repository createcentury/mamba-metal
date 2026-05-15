// selective_scan — Mamba selective scan, minimal version (seqlen <= 1024).
//
// For each (batch, dim), computes:
//   y[t] = sum_{s=0..N-1} h_t^{(s)} * C[s, t]
//   h_t^{(s)} = exp(delta[t] * A[s]) * h_{t-1}^{(s)} + delta[t] * u[t] * B[s, t]
//
// Constraints: seqlen <= 1024, fp32, no z gate, no D, no complex, no
// delta_softplus, kIsVariableB = kIsVariableC = true, kNRows = 1.
//
// Inputs:
//   device const float* u           (batch, dim, seqlen)
//   device const float* delta       (batch, dim, seqlen)
//   device const float* A           (dim, dstate)
//   device const float* B           (batch, dstate, seqlen)
//   device const float* C           (batch, dstate, seqlen)
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

bool in_range = (t < seqlen);

uint udx = batch_id * dim * seqlen + dim_id * seqlen + t;
float u_t = in_range ? u[udx] : 0.0;
float delta_t = in_range ? delta[udx] : 0.0;

float y_t = 0.0;

for (uint s = 0; s < dstate; ++s) {
    // Ensure previous iteration's reads of warp_* are done before overwriting
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float A_ds = A[dim_id * dstate + s];
    uint bcdx = batch_id * dstate * seqlen + s * seqlen + t;
    float B_st = in_range ? B[bcdx] : 0.0;
    float C_st = in_range ? C[bcdx] : 0.0;

    float a = in_range ? exp(delta_t * A_ds) : 1.0;
    float b = in_range ? (delta_t * u_t * B_st) : 0.0;

    // SIMD-level pair scan
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
        float ca = warp_a[sg - 1u];
        float cb = warp_b[sg - 1u];
        b = a * cb + b;
        a = a * ca;
    }

    // b == h_t for state s (initial state is 0)
    if (in_range) {
        y_t += b * C_st;
    }
}

if (in_range) {
    y[batch_id * dim * seqlen + dim_id * seqlen + t] = y_t;
}
