# mamba-metal

**Language**: [English](README.md) | [日本語](README.ja.md)

[Mamba](https://arxiv.org/abs/2312.00752) の selective scan を **Metal Shading Language (MSL)** で書き直し、Apple Silicon 上で動かすプロジェクト。

参照実装: [state-spaces/mamba](https://github.com/state-spaces/mamba) の `csrc/selective_scan/selective_scan_fwd_kernel.cuh`。

## デモ

`state-spaces/mamba-130m-hf` の重みをそのままロードして、自作 Metal カーネルを通じて生成（greedy、40 トークン、M4 Max）:

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

### 状態キャッシュによる O(1)/token

`generate_fast` は SSM 隠れ状態と conv1d の sliding window を呼び出し間で持ち越すため、新 1 トークンあたりのコストは O(1)（プロンプト長・生成長に依らず一定）。greedy 出力は O(L²) 版と完全一致。

| 生成トークン数 | O(L²) 再 forward | **O(L) `generate_fast`** | speedup |
|---:|---:|---:|---:|
| 10  | 0.24 s | **0.06 s** | 4.3× |
| 40  | 1.14 s | **0.18 s** | 6.3× |
| 100 | 3.24 s | **0.51 s** | 6.3× |
| 200 | 8.30 s | **1.38 s** | 6.0× |

`generate_fast` は M4 Max で 130m モデルにおいて **~7 ms/token (≈ 145 tok/s) で一定**。生成が長くなるほど O(L²) 版との差が広がる。

```python
from transformers import AutoTokenizer
from mamba_metal import load_mamba_hf, generate_fast

model, _ = load_mamba_hf("state-spaces/mamba-130m-hf")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
print(generate_fast(model, tokenizer, "The capital of Japan is", max_new_tokens=40))
```

## 進捗

selective scan の forward パスは機能完備で、PyTorch 参照実装と数値一致（float 誤差範囲内）。Mamba の推論で必要な機能フラグは全て揃っています：

| 機能 | Mamba CUDA | mamba-metal |
|---|---|---|
| selective scan + variable B, C | ✓ | ✓ |
| 任意 `seqlen`（チャンク化 + SRAM running prefix） | ✓ | ✓ |
| D 接続（skip） | ✓ | ✓ |
| `delta_softplus` | ✓ | ✓ |
| z ゲート（SiLU） | ✓ | ✓ |
| fp16 入力 / fp32 蓄積 | ✓ | ✓ |
| 複素数重み | ✓ | — |
| `kNRows > 1` | △ | — |
| 後方パス（学習用） | ✓ | — |

M4 Max でのスループット（B=1, D=16, N=16）: `seqlen=32k` で **~45 M tokens/s、~190 GFLOPS**。`seqlen ~ 4k` を超えると token あたり時間がほぼ一定（Mamba 論文の主張する**線形時間**）。

## 使い方

```bash
uv sync
.venv/bin/python experiments/07-chunked/selective_scan_chunked.py
```

Python から：

```python
import mlx.core as mx
from mamba_metal import selective_scan

# u, delta: (batch, dim, seqlen)
# A:        (dim, dstate)
# B, C:     (batch, dstate, seqlen)
y = selective_scan(u, delta, A, B, C,
                   D=D, z=z, delta_softplus=True)
```

入力は fp16 可（Mamba 流: データは half、重みは float）。

## ディレクトリ構成

```
mamba_metal/
├── kernels/                          # MSL カーネル本体（.metal を第一級資産扱い）
│   ├── selective_scan_chunked.metal  # 本命カーネル（D/z/softplus/fp16 対応）
│   ├── selective_scan.metal          # 単一チャンク最小版
│   ├── pair_scan.metal               # (a, b) ペア用 block scan
│   ├── block_scan.metal              # 汎用 block prefix sum
│   ├── simd_scan_{builtin,handrolled}.metal
│   ├── conv1d_{global,tg}.metal      # threadgroup memory 実験
│   ├── copy_{scalar,vec4}.metal      # 帯域計測
│   └── vector_add.metal              # toolchain smoke test
├── _loader.py                        # .metal 読み込み + mx.fast.metal_kernel ラップ
├── selective_scan.py                 # 高水準 Python API
└── __init__.py

experiments/                          # 各段の検証スクリプト
├── 01-hello-metal/                   # toolchain smoke test
├── 02-bandwidth/                     # Unified Memory 帯域（~290 GB/s）
├── 03-threadgroup/                   # tg memory — HW cache が再利用を吸収する観察
├── 04-simd-scan/                     # SIMD-group / block-level scan
├── 05-pair-scan/                     # 漸化式を pair scan で解く
├── 06-selective-scan/                # selective scan 最小版（seqlen ≤ 1024）
├── 07-chunked/                       # チャンク化で任意 seqlen
├── 08-benchmark/                     # スループット計測
├── 09-features/                      # D / z / softplus 機能組合せ vs numpy
└── 10-fp16/                          # 混合精度（fp16 データ / fp32 重み）
```

`mx.fast.metal_kernel` はカーネル**本体**（`{}` 内）だけを受け取るので、`.metal` ファイルは `kernel void <name>(...)` の宣言を持たないスニペット形式。期待される signature は冒頭のコメントに記載してあります。

## ロードマップ

| Step | 内容 | 状態 |
|---|---|---|
| 1A | Hello-Metal vector add（ツールチェーン確認） | ✓ |
| 1B | Unified Memory 帯域 | ✓ |
| 1C | threadgroup memory — cache 観察 | ✓ |
| 1D | SIMD-group / block-level scan | ✓ |
| 2 | `(a, b)` ペア scan で漸化式を解く | ✓ |
| 3 | selective scan 最小版（seqlen ≤ 1024） | ✓ |
| 4 | `mamba_metal` パッケージ化、`.metal` ファイル化 | ✓ |
| 5 | チャンク化 + per-state running prefix | ✓ |
| 6 | D / softplus / z / fp16 | ✓ |
| 7 | スループットベンチ | ✓ |
| 8 | Mamba ブロック（in_proj → conv1d → SSM → out_proj） | ✓ |
| 9 | HuggingFace checkpoint 推論（ロード→生成） | ✓ |

### Future work

- **iPhone 上での Transformer vs Mamba ベンチ** — 両アーキテクチャを iOS（Metal もしくは CoreML）に載せ、同等パラメータ規模で**速度と精度**を比較する。本リポジトリの selective scan Metal カーネルが Mamba 側の土台になる。

各段の知見は [createcentury.github.io/blog](https://createcentury.github.io/blog) に記事化しています。

## 開発環境

- Apple M4 Max / Metal 3
- Python 3.12（uv 管理）
- MLX をカーネルホストとして使用（JIT コンパイル・バッファバインド・ディスパッチを担当）

## 参考文献

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.
- Jimmy T.H. Smith et al. "[Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)" arXiv:2208.04933, 2022.

## License

MIT
