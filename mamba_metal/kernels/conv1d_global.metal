// conv1d_global — K-wide 1D stencil reading directly from global memory
//
// out[i] = sum over k in [0, K) of in_[i + k]
//
// {K} is substituted at load time (compile-time constant).
//
// Inputs:  device const float* in_, device const uint& n_out
// Output:  device float* out
// Dispatch: grid = ceil(n_out/256)*256, threadgroup = 256

uint i = thread_position_in_grid.x;
if (i >= n_out) return;
float acc = 0.0;
for (uint k = 0; k < {K}u; ++k) {{
    acc += in_[i + k];
}}
out[i] = acc;
