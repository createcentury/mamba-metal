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
    return MambaConfig(
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        vocab_size=cfg["vocab_size"],
        d_state=cfg["state_size"],
        d_conv=cfg["conv_kernel"],
        expand=cfg["expand"],
        dt_rank=cfg["time_step_rank"],
        rms_norm_eps=cfg["layer_norm_epsilon"],
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
            allow_patterns=["*.json", "*.safetensors"],
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
