// pair_scan — threadgroup-level inclusive scan over (a, b) pairs
//
// Solves the recurrence  h_i = a_i * h_{i-1} + b_i  (with h_{-1} = 0)
// by treating each step as a pair (a_i, b_i) under the associative op
//
//   (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)
//
// simd_prefix_inclusive_sum is scalar-only, so we hand-roll a Hillis-Steele
// scan on the pair within a SIMD group, then bridge SIMD groups via
// threadgroup memory.
//
// Inputs:  device const float* a_in, device const float* b_in, device const uint& n
// Outputs: device float* a_out, device float* h_out
// Dispatch: grid = (1024,), threadgroup = (1024,)

uint i = thread_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint n_sg = simdgroups_per_threadgroup;

threadgroup float warp_a[32];
threadgroup float warp_b[32];

// Identity element for combine: (1, 0)
float a = (i < n) ? a_in[i] : 1.0;
float b = (i < n) ? b_in[i] : 0.0;

// (1) SIMD-level inclusive scan via Hillis-Steele on pairs.
// Order matters: update b first (uses old a), then a.
for (uint d = 1u; d < 32u; d <<= 1) {
    float a_prev = simd_shuffle_up(a, d);
    float b_prev = simd_shuffle_up(b, d);
    if (lane >= d) {
        b = a * b_prev + b;
        a = a * a_prev;
    }
}

// (2) Last lane writes group total
if (lane == 31u) {
    warp_a[sg] = a;
    warp_b[sg] = b;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// (3) First SIMD group scans the group totals (also pair composition)
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

// (4) Combine carry from previous groups
if (sg > 0u) {
    float ca = warp_a[sg - 1u];
    float cb = warp_b[sg - 1u];
    b = a * cb + b;
    a = a * ca;
}

if (i < n) {
    a_out[i] = a;
    h_out[i] = b;
}
