# mamba-metal

**Language**: [English](README.md) | [日本語](README.ja.md)

A Metal Shading Language (MSL) port of [Mamba](https://arxiv.org/abs/2312.00752)'s selective scan, running on Apple Silicon.

Reference: [state-spaces/mamba](https://github.com/state-spaces/mamba) — `csrc/selective_scan/selective_scan_fwd_kernel.cuh`.

## Demo

`state-spaces/mamba-130m-hf` weights loaded straight into our MambaModel and generated through the Metal selective scan kernel (greedy, 40 new tokens, M4 Max):

```
> Mamba is a
  very popular and popularly used game in the Philippines. It is a game that is
  played by a group of people who are all very good at playing the game.

> Once upon a time
  there was a man named Billy. He was a man of great wealth, A man of great wealth,
  A man of great wealth, A

> The capital of Japan is
  Tokyo, Japan. The city is located in the northern part of the country, and is
  the capital of the Japanese state of Japan.
```

### O(1) per token via state caching

`generate_fast` carries the SSM hidden state and conv1d sliding window across
calls, so each new token costs O(1) — independent of prompt length and how much
has already been generated. Greedy output is identical to the O(L²) path.

| `max_new_tokens` | O(L²) re-forward | **O(L) `generate_fast`** | speedup |
|---:|---:|---:|---:|
| 10  | 0.24 s | **0.06 s** | 4.3× |
| 40  | 1.14 s | **0.18 s** | 6.3× |
| 100 | 3.24 s | **0.51 s** | 6.3× |
| 200 | 8.30 s | **1.38 s** | 6.0× |

`generate_fast` runs at a flat **~7 ms/token** (≈ 145 tok/s) at 130m on M4 Max; the speedup widens with longer outputs.

```python
from transformers import AutoTokenizer
from mamba_metal import load_mamba_hf, generate_fast

model, _ = load_mamba_hf("state-spaces/mamba-130m-hf")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
print(generate_fast(model, tokenizer, "The capital of Japan is", max_new_tokens=40))
```

## Status

The forward path of selective scan is functionally complete and matches the PyTorch reference (within float-precision noise) for all feature flags Mamba uses at inference time:

| Feature | Mamba CUDA | mamba-metal |
|---|---|---|
| Selective scan with variable B, C | ✓ | ✓ |
| Arbitrary `seqlen` (chunked, in-SRAM running prefix) | ✓ | ✓ |
| D skip connection | ✓ | ✓ |
| `delta_softplus` | ✓ | ✓ |
| z gate (SiLU) | ✓ | ✓ |
| fp16 inputs / fp32 scan accumulation | ✓ | ✓ |
| Complex weights | ✓ | — |
| `kNRows > 1` | △ | — |
| Backward pass | ✓ | — |

Single-kernel throughput on M4 Max (B=1, D=16, N=16): peak **~45 M tokens/s, ~190 GFLOPS** at `seqlen=32k`. Time-per-token is roughly flat above `seqlen ~ 4k` (close to the linear-time property Mamba advertises).

## Quickstart

```bash
uv sync
.venv/bin/python experiments/07-chunked/selective_scan_chunked.py
```

From Python:

```python
import mlx.core as mx
from mamba_metal import selective_scan

# u, delta: (batch, dim, seqlen)
# A:        (dim, dstate)
# B, C:     (batch, dstate, seqlen)
y = selective_scan(u, delta, A, B, C,
                   D=D, z=z, delta_softplus=True)
```

Inputs can be fp16 (Mamba style — data in half, weights in float).

## Layout

```
mamba_metal/
├── kernels/                          # MSL kernel bodies (.metal, first-class artefacts)
│   ├── selective_scan_chunked.metal  # main kernel (D / z / softplus / fp16)
│   ├── selective_scan.metal          # single-chunk minimal version
│   ├── pair_scan.metal               # block-level (a, b) pair scan
│   ├── block_scan.metal              # block-level prefix sum
│   ├── simd_scan_{builtin,handrolled}.metal
│   ├── conv1d_{global,tg}.metal      # threadgroup-memory exploration
│   ├── copy_{scalar,vec4}.metal      # bandwidth probes
│   └── vector_add.metal              # toolchain smoke test
├── _loader.py                        # reads .metal files, wraps with mx.fast.metal_kernel
├── selective_scan.py                 # high-level Python API
└── __init__.py

experiments/                          # step-by-step verification scripts
├── 01-hello-metal/                   # toolchain smoke test
├── 02-bandwidth/                     # Unified Memory bandwidth (~290 GB/s)
├── 03-threadgroup/                   # tg memory — finding: hw caches absorb data reuse
├── 04-simd-scan/                     # SIMD-group + block-level scan
├── 05-pair-scan/                     # pair-composition scan solves h_i = a_i h_{i-1} + b_i
├── 06-selective-scan/                # minimal selective scan (seqlen ≤ 1024)
├── 07-chunked/                       # arbitrary seqlen with running prefix
├── 08-benchmark/                     # throughput sweep
├── 09-features/                      # D / z / softplus combinations vs numpy reference
└── 10-fp16/                          # mixed precision (fp16 data, fp32 weights)
```

`mx.fast.metal_kernel` expects only the kernel **body**, so each `.metal` file is a snippet (no `kernel void <name>(...)` declaration); the expected signature is documented in a header comment.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1A | Hello-Metal vector add (toolchain) | ✓ |
| 1B | Unified Memory bandwidth | ✓ |
| 1C | Threadgroup memory — cache observation | ✓ |
| 1D | SIMD-group / block-level scan | ✓ |
| 2 | `(a, b)` pair scan solves recurrence | ✓ |
| 3 | Selective scan, minimal (seqlen ≤ 1024) | ✓ |
| 4 | Package as `mamba_metal`, `.metal` files | ✓ |
| 5 | Chunking + per-state running prefix | ✓ |
| 6 | D / softplus / z / fp16 | ✓ |
| 7 | Throughput benchmark | ✓ |
| 8 | Mamba block (in_proj → conv1d → SSM → out_proj) | ✓ |
| 9 | HuggingFace checkpoint inference (load → generate) | ✓ |

### Future work

- **On-device benchmark: Transformer vs Mamba on iPhone** — port both architectures to iOS (Metal or CoreML), measure speed and accuracy on equal-budget models. The selective-scan Metal kernel in this repo is the building block.

Findings from each step are written up at [createcentury.github.io/blog](https://createcentury.github.io/blog).

## Development

- Apple M4 Max / Metal 3
- Python 3.12 (uv-managed)
- MLX as the kernel host (handles JIT compilation, buffer binding, dispatch)

## References

- Albert Gu, Tri Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), arXiv:2312.00752, 2023.
- Guy E. Blelloch. [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf), CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. [Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057), arXiv:1709.04057, 2017.
- Jimmy T.H. Smith et al. [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933), arXiv:2208.04933, 2022.

## License

MIT
