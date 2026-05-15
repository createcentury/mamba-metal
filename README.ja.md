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

### 長文脈プロンプト（並列 prefill）

プロンプト全体を 1 回の selective scan カーネル呼び出しで処理。カーネル末尾で最終 SSM 状態を出力（Mamba CUDA の `params.x_ptr` 相当）し、decode はその状態から O(1)/token で続行する。`prefill + 50 トークン生成`の壁時計時間（mamba-130m）：

| プロンプト長 | 旧 step-loop prefill | **並列 prefill** | speedup |
|---:|---:|---:|---:|
| 71 | 0.44 s | **0.22 s** | 2.0× |
| 351 | 1.44 s | **0.34 s** | 4.2× |
| 1,401 | 5.20 s | **0.45 s** | 11.6× |
| **5,601** | **21.80 s** | **0.88 s** | **24.8×** |

これは Mamba 公式 CUDA の推論パスと同じ設計。

### Apple 公式 mlx-lm との比較

`mlx-lm` の [`models/mamba.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py) は selective scan を Python の `for t in range(T)` ループで書いており、並列 prefix-scan カーネルを持たない。同ハードウェア（M4 Max）・同モデル（`mamba-130m-hf`）・同プロンプト、50 トークン生成：

| プロンプト長 | **mamba-metal** | mlx-lm | **speedup** |
|---:|---:|---:|---:|
| 71 | **0.21 s** | 0.26 s | 1.22× |
| 351 | **0.34 s** | 0.65 s | 1.90× |
| 1,401 | **0.30 s** | 2.41 s | 8.14× |
| **5,601** | **0.82 s** | **9.01 s** | **11.03×** |

自作 Metal scan カーネルがプロンプトが長くなるほど効く。decode 単体は両者とも 1 ステップ elementwise なので同等。

### 各モデルサイズでの実測

`state-spaces/mamba-*-hf` の 5 つ全てがロード・推論可能。プロンプト `"The capital of Japan is"`、40 トークン、greedy、M4 Max：

| model | params | load (s) | tok/s | ms/tok |
|---|---:|---:|---:|---:|
| 130m | 129 M | 1.3 | 175 | 5.7 |
| 370m | 372 M | 3.4 | 82 | 12.2 |
| 790m | 702 M | 4.8 | 42 | 23.7 |
| 1.4b | 1372 M | 11.6 | 30 | 33.2 |
| **2.8b** | **2.7 B** | **19.6** | **12** | **80.6** |

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
| 推論用の最終 SSM 状態出力 | ✓ | ✓ |
| 並列 prefill + O(L) インクリメンタル decode | ✓ | ✓ |
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

- **Swift / iOS 移植**: [createcentury/mamba-metal-swift](https://github.com/createcentury/mamba-metal-swift) — 同じ `.metal` カーネルを `MLXFast.metalKernel` (mlx-swift) 経由で公開。`pair_scan` は end-to-end 検証済み、モデル層は移植進行中。
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
