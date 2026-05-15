# mamba-metal

Mamba の selective scan を **Metal Shading Language (MSL)** で書き直し、Apple Silicon 上で動かすプロジェクト。

参照実装: [state-spaces/mamba](https://github.com/state-spaces/mamba) の `csrc/selective_scan/selective_scan_fwd_kernel.cuh`

## 開発環境

- Apple M4 / Metal
- Python 3.12 (uv 管理)
- MLX（カスタム Metal カーネルのホストとして）

## 使い方

```bash
uv sync
.venv/bin/python experiments/07-chunked/selective_scan_chunked.py
```

または Python から：

```python
import mlx.core as mx
from mamba_metal import selective_scan

# u, delta: (batch, dim, seqlen)
# A:        (dim, dstate)
# B, C:     (batch, dstate, seqlen)
y = selective_scan(u, delta, A, B, C)
```

## ディレクトリ構造

```
mamba_metal/
├── kernels/                          # MSL カーネル本体（全て .metal）
│   ├── selective_scan_chunked.metal  # 任意 seqlen 対応の本命カーネル
│   ├── selective_scan.metal          # 最小版 (seqlen ≤ 1024)
│   ├── pair_scan.metal               # (a, b) ペア用 block scan
│   ├── block_scan.metal              # 汎用 block-level prefix sum
│   ├── simd_scan_{builtin,handrolled}.metal
│   ├── conv1d_{global,tg}.metal      # tg memory 実験用
│   └── copy_{scalar,vec4}.metal      # bandwidth 計測用
├── _loader.py                        # .metal を読んで mx.fast.metal_kernel に渡す
└── selective_scan.py                 # 高水準 Python エントリポイント

experiments/                          # 各段の検証スクリプト
├── 01-hello-metal/                   # toolchain smoke test
├── 02-bandwidth/                     # Unified Memory 帯域 (~290 GB/s)
├── 03-threadgroup/                   # tg memory タイル化（cache が吸収する観察）
├── 04-simd-scan/                     # SIMD-group / block-level scan
├── 05-pair-scan/                     # 漸化式を pair scan で解く
├── 06-selective-scan/                # selective scan 最小版
└── 07-chunked/                       # 任意 seqlen
```

`mx.fast.metal_kernel` はカーネル**本体**（`{}` 内）だけを受け取る API なので、`.metal` ファイルは関数シグネチャを書かないスニペット形式。冒頭にコメントで期待される signature を記載してある。

## ロードマップ

| 段 | 内容 | 状態 |
|---|---|---|
| 1A | 純 MSL で vector add | ✓ |
| 1B | Unified Memory 帯域測定 (~290 GB/s) | ✓ |
| 1C | threadgroup memory 検証（cache が吸収する観察） | ✓ |
| 1D | SIMD-group + block-level prefix scan | ✓ |
| 2 | `(a, b)` ペア合成で $h_i = a_i h_{i-1} + b_i$ を解く | ✓ |
| 3 | selective scan 最小版（seqlen ≤ 1024） | ✓ |
| 5 | チャンク跨ぎ (smem_running_prefix 相当) で任意 seqlen | ✓ |
| 4 | `.metal` ファイル分離 + パッケージ化 | ✓ |
| 6 | z ゲート / D 接続 / delta_softplus / 半精度 | TODO |
| 7 | ベンチマーク（線形スケーリング検証） | TODO |

各段の知見は [createcentury.github.io/blog](https://createcentury.github.io/blog) に記事化していく。

## 参考文献

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" Technical Report CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.

## License

MIT
