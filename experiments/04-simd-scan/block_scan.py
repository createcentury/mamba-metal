"""Step 1D-2: block (threadgroup) level prefix sum via two-level scan.

Pattern:
  1. Each SIMD-group does its own scan via simd_prefix_inclusive_sum.
  2. Lane 31 of each group writes the group total to threadgroup memory.
  3. The first SIMD-group scans those totals.
  4. Each thread adds the appropriate "carry" (sum of previous group totals).

This handles up to 32 * 32 = 1024 threads per threadgroup
(the Apple Silicon max threadgroup size).

This is the MSL analogue of cub::BlockScan with WARP_SCANS, which Mamba uses.
"""

import numpy as np
import mlx.core as mx


SIMD_W = 32
TG_SIZE = 1024  # 32 SIMD groups of 32 lanes


BLOCK_SCAN_SRC = """
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
"""


k_block_scan = mx.fast.metal_kernel(
    name="block_scan",
    input_names=["in_", "n"],
    output_names=["out"],
    source=BLOCK_SCAN_SRC,
)


def run_block_scan(a: mx.array) -> mx.array:
    """Scan a single threadgroup-sized chunk (<= 1024 elements)."""
    n = a.size
    assert n <= TG_SIZE, "this kernel handles one threadgroup only"
    (out,) = k_block_scan(
        inputs=[a, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(TG_SIZE, 1, 1),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return out


def main() -> None:
    print(f"SIMD width = {SIMD_W}, threadgroup = {TG_SIZE}")
    print()

    for n in [32, 64, 256, 1000, 1024]:
        rng = np.random.default_rng(seed=n)
        a_np = rng.standard_normal(n).astype(np.float32)
        a = mx.array(a_np)
        out = run_block_scan(a)
        mx.eval(out)

        ref = a_np.cumsum()
        out_np = np.array(out)
        err = np.max(np.abs(out_np - ref))
        # Relative because cumsum can grow large
        rel_err = np.max(np.abs(out_np - ref) / (np.abs(ref) + 1e-6))

        print(f"n = {n:>5}: max abs err = {err:.3e}   rel err = {rel_err:.3e}")

    print()
    print("Last element comparison (n=1024):")
    n = 1024
    rng = np.random.default_rng(seed=42)
    a_np = rng.standard_normal(n).astype(np.float32)
    a = mx.array(a_np)
    out = run_block_scan(a)
    mx.eval(out)
    out_np = np.array(out)
    print(f"  numpy cumsum[-1] = {a_np.cumsum()[-1]:.6f}")
    print(f"  block_scan[-1]   = {out_np[-1]:.6f}")


if __name__ == "__main__":
    main()
