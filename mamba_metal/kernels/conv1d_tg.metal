// conv1d_tg — K-wide 1D stencil with threadgroup-memory tiling
//
// Each threadgroup loads a tile of 256 + K - 1 elements into threadgroup
// memory once, then every thread reads K elements from there.
//
// {K} is substituted at load time. tile_size = 256 + {K} - 1.
//
// Inputs: device const float* in_, device const uint& n_in, device const uint& n_out
// Output: device float* out
// Dispatch: grid = ceil(n_out/256)*256, threadgroup = 256

uint local_i = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;
uint base = tg_id * 256u;
uint i = base + local_i;

threadgroup float tile[{tile_size}];

// Each thread loads one element of the main tile.
if (base + local_i < n_in) {{
    tile[local_i] = in_[base + local_i];
}} else {{
    tile[local_i] = 0.0;
}}
// The first K-1 threads also load the halo at the end.
if (local_i < {K_minus_1}u) {{
    uint halo_idx = base + 256u + local_i;
    tile[256u + local_i] = (halo_idx < n_in) ? in_[halo_idx] : 0.0;
}}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (i >= n_out) return;
float acc = 0.0;
for (uint k = 0; k < {K}u; ++k) {{
    acc += tile[local_i + k];
}}
out[i] = acc;
