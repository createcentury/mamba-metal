// block_scan — threadgroup-level inclusive prefix sum (up to 1024 elements)
//
// Two-level scan:
//   (1) Each SIMD group scans its 32 lanes with simd_prefix_inclusive_sum
//   (2) Lane 31 of each group writes the group total to threadgroup memory
//   (3) The first SIMD group scans those 32 totals
//   (4) Each thread adds the carry from previous groups
//
// MSL analogue of cub::BlockScan with WARP_SCANS that Mamba uses.
//
// Inputs:  device const float* in_, device const uint& n
// Output:  device float* out
// Dispatch: grid = (1024,), threadgroup = (1024,)

uint i = thread_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint n_sg = simdgroups_per_threadgroup;

threadgroup float warp_totals[32];

float v = (i < n) ? in_[i] : 0.0;

// (1) SIMD-level inclusive scan
v = simd_prefix_inclusive_sum(v);

// (2) Last lane writes its group's total
if (lane == 31u) {
    warp_totals[sg] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// (3) First SIMD group scans the group totals
if (sg == 0u) {
    float t = (lane < n_sg) ? warp_totals[lane] : 0.0;
    t = simd_prefix_inclusive_sum(t);
    if (lane < n_sg) {
        warp_totals[lane] = t;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// (4) Add carry from previous groups (exclusive of self)
if (sg > 0u) {
    v += warp_totals[sg - 1u];
}

if (i < n) {
    out[i] = v;
}
