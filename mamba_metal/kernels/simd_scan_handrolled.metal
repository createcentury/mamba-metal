// simd_scan_handrolled — Hillis-Steele scan over a SIMD group via simd_shuffle_up
//
// Equivalent to simd_prefix_inclusive_sum, written explicitly so the algorithm
// is visible. After log2(32) = 5 iterations each lane holds the inclusive
// prefix sum within its SIMD group.
//
// Inputs:  device const float* in_, device const uint& n
// Output:  device float* out
// Dispatch: grid = (n,), threadgroup = (32,)

uint i = thread_position_in_grid.x;
if (i >= n) return;
uint lane = thread_index_in_simdgroup;
float v = in_[i];

float u;
u = simd_shuffle_up(v, 1u);  if (lane >= 1u)  v += u;
u = simd_shuffle_up(v, 2u);  if (lane >= 2u)  v += u;
u = simd_shuffle_up(v, 4u);  if (lane >= 4u)  v += u;
u = simd_shuffle_up(v, 8u);  if (lane >= 8u)  v += u;
u = simd_shuffle_up(v, 16u); if (lane >= 16u) v += u;

out[i] = v;
