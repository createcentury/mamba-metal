"""Step 8: end-to-end Mamba block (in_proj → conv1d → SSM → out_proj).

Smoke tests:
  - Random-init forward pass produces the right shape and is finite.
  - The block is "causal" in the conv path (output[..., t] depends only on input[..., :t+1])
    — verified by feeding the same prefix and checking equal outputs through that prefix.
  - Throughput at moderate seq length.
"""

import time

import numpy as np
import mlx.core as mx

from mamba_metal import MambaBlock


def smoke_test():
    print("Smoke test (random init forward pass)\n")
    block = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)

    B, L = 2, 256
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, L, 128)).astype(np.float32) * 0.5
    x = mx.array(x_np)

    y = block(x)
    mx.eval(y)
    mx.synchronize()

    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {y.shape}")
    y_np = np.array(y)
    print(f"  any NaN: {np.any(np.isnan(y_np))}")
    print(f"  any Inf: {np.any(np.isinf(y_np))}")
    print(f"  range: [{y_np.min():.3f}, {y_np.max():.3f}]   mean: {y_np.mean():.3f}   std: {y_np.std():.3f}")
    assert y.shape == x.shape
    assert not np.any(np.isnan(y_np))
    assert not np.any(np.isinf(y_np))


def causality_test():
    print("\nCausality test (output[:, :k] should not depend on input[:, k:])\n")
    block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)

    B, L = 1, 64
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal((B, L, 64)).astype(np.float32) * 0.5
    x2 = x1.copy()
    k = 32
    x2[:, k:, :] = rng.standard_normal((B, L - k, 64)).astype(np.float32) * 0.5

    y1 = np.array(block(mx.array(x1)))
    y2 = np.array(block(mx.array(x2)))
    mx.synchronize()

    diff_pre = np.max(np.abs(y1[:, :k, :] - y2[:, :k, :]))
    diff_post = np.max(np.abs(y1[:, k:, :] - y2[:, k:, :]))
    print(f"  max |y1[:, :{k}] - y2[:, :{k}]|   (should be tiny) = {diff_pre:.3e}")
    print(f"  max |y1[:, {k}:] - y2[:, {k}:]|  (expected nonzero) = {diff_post:.3e}")
    assert diff_pre < 1e-4, "Block leaks future information into past outputs"


def throughput_test():
    print("\nThroughput at L=2048, d_model=512\n")
    block = MambaBlock(d_model=512, d_state=16, d_conv=4, expand=2)
    B, L = 1, 2048
    x = mx.random.normal(shape=(B, L, 512))

    # Warmup
    for _ in range(3):
        y = block(x)
        mx.eval(y)
    mx.synchronize()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        y = block(x)
        mx.eval(y)
    mx.synchronize()
    sec = (time.perf_counter() - t0) / iters

    tokens = B * L
    print(f"  time/forward: {sec*1e3:.3f} ms")
    print(f"  throughput:   {tokens/sec/1e6:.2f} M tokens/sec")


if __name__ == "__main__":
    smoke_test()
    causality_test()
    throughput_test()
