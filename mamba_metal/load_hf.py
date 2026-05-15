"""Load `state-spaces/mamba-*-hf` weights into a MambaModel.

Transformations applied:
  - Strip "backbone." prefix from every key.
  - Transpose conv1d.weight from PyTorch order (out, in/g, k)
    to MLX order (out, k, in/g).
  - Convert numpy arrays to mx.array, preserving dtype.

Returns the loaded model and its MambaConfig.
"""

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from safetensors import safe_open

from mamba_metal.mamba_model import MambaConfig, MambaModel


def _config_from_hf(local_dir: Path) -> MambaConfig:
    cfg = json.loads((local_dir / "config.json").read_text())
    # Prefer HF transformers fields; fall back to legacy state-spaces aliases.
    # Some checkpoints (e.g. mamba-790m-hf) have a stale `d_model` field that
    # disagrees with the real model dim; `hidden_size` is the trustworthy one.
    d_model = cfg.get("hidden_size") or cfg["d_model"]
    d_inner = cfg.get("intermediate_size") or cfg.get("d_inner")
    expand = d_inner // d_model if d_inner else cfg.get("expand", 2)
    return MambaConfig(
        d_model=d_model,
        n_layer=cfg.get("num_hidden_layers") or cfg["n_layer"],
        vocab_size=cfg["vocab_size"],
        d_state=cfg.get("state_size") or cfg.get("d_state", 16),
        d_conv=cfg.get("conv_kernel") or cfg.get("d_conv", 4),
        expand=expand,
        dt_rank=cfg.get("time_step_rank") or cfg.get("dt_rank"),
        rms_norm_eps=cfg.get("layer_norm_epsilon", 1e-5),
    )


def _convert_key(key: str) -> str:
    return key.removeprefix("backbone.")


def _convert_tensor(name: str, tensor) -> mx.array:
    # conv1d.weight: PyTorch (out, in/g, k) → MLX (out, k, in/g)
    if name.endswith("conv1d.weight"):
        tensor = tensor.transpose(0, 2, 1)
    return mx.array(tensor)


def load_mamba_hf(
    repo_id: str = "state-spaces/mamba-130m-hf",
) -> tuple[MambaModel, MambaConfig]:
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "special_tokens*"],
        )
    )

    cfg = _config_from_hf(local_dir)
    model = MambaModel(cfg)

    weights: list[tuple[str, mx.array]] = []
    for st in sorted(local_dir.glob("*.safetensors")):
        with safe_open(str(st), framework="numpy") as f:
            for hf_key in f.keys():
                target = _convert_key(hf_key)
                tensor = f.get_tensor(hf_key)
                weights.append((target, _convert_tensor(target, tensor)))

    model.load_weights(weights)
    mx.eval(model.parameters())
    return model, cfg
