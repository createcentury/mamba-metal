// simd_scan_builtin — SIMD-group inclusive prefix sum via MSL builtin
//
// One SIMD group (32 lanes) per threadgroup. Each lane holds one element.
//
// Inputs:  device const float* in_, device const uint& n
// Output:  device float* out
// Dispatch: grid = (n,), threadgroup = (32,)

uint i = thread_position_in_grid.x;
if (i >= n) return;
float v = in_[i];
v = simd_prefix_inclusive_sum(v);
out[i] = v;
