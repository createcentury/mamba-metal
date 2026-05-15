# mamba-metal

Mamba の selective scan を **Metal Shading Language (MSL)** で書き直す実験プロジェクト。Apple Silicon 上で動かす。

参照実装: [state-spaces/mamba](https://github.com/state-spaces/mamba) の `csrc/selective_scan/selective_scan_fwd_kernel.cuh`

## 開発環境

- Apple M4 / Metal
- Python 3.12 (uv 管理)
- MLX（カスタム Metal カーネルのホストとして）

## ロードマップ

| 段 | 内容 | ステータス |
|---|---|---|
| 1 | 純 MSL で block-level inclusive scan（汎用 prefix sum） | TODO |
| 2 | 結合的演算子を `(a,b)` ペアに拡張し、$h_i = a_i h_{i-1} + b_i$ を解く | TODO |
| 3 | selective scan 最小版（kIsVariableB/C, fp32, no complex, no z gate, kNRows=1） | TODO |
| 4 | MLX バインディングと参照実装との数値突合 | TODO |
| 5 | チャンク跨ぎ (running prefix) 対応 | TODO |
| 6 | 半精度 / 複素数 / z ゲート | TODO |

各段の知見は [createcentury.github.io/blog](https://createcentury.github.io/blog) に記事化していく。

## 参考文献

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" Technical Report CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.

## License

MIT
