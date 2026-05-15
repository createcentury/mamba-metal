// vector_add — toolchain smoke test
//
// Inputs:
//   device const float* a
//   device const float* b
//   device const uint&  n
// Output:
//   device float* out
// Dispatch: grid = (n,), threadgroup = (256,)

uint i = thread_position_in_grid.x;
if (i >= n) return;
out[i] = a[i] + b[i];
