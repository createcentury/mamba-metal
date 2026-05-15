// copy_vec4 — vectorised global-memory bandwidth (4 floats per thread)
//
// Inputs:  device const float* in_,  device const uint& n4
// Output:  device float* out (aliased to float4)
// Dispatch: grid = (n4,) where n4 = n / 4

uint i = thread_position_in_grid.x;
if (i >= n4) return;
device const float4* in4 = (device const float4*)in_;
device float4* out4 = (device float4*)out;
out4[i] = in4[i];
