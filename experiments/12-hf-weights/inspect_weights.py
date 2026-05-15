"""Step 9.2: download and inspect state-spaces/mamba-130m-hf weights.

We want to know:
  1. The exact parameter names HF uses.
  2. The shape and dtype of each parameter.
  3. Hyperparameters from config.json.

This drives the MambaModel design (9.1) and key mapping (later in 9.2).
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open  # framework-agnostic interface


REPO_ID = "state-spaces/mamba-130m-hf"


def main() -> None:
    print(f"Downloading {REPO_ID} ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )
    print(f"Local: {local_dir}\n")

    # config.json
    cfg_path = Path(local_dir) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print("=== config.json ===")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()

    # safetensors index / parameter listing
    st_files = sorted(Path(local_dir).glob("*.safetensors"))
    print(f"=== safetensors files ({len(st_files)}) ===")
    for st in st_files:
        print(f"  {st.name}  ({st.stat().st_size / 1024**2:.1f} MB)")
    print()

    # Enumerate parameters by name and shape
    print("=== parameters (name → shape, dtype) ===")
    total = 0
    layer0_seen = False
    for st in st_files:
        with safe_open(str(st), framework="numpy") as f:
            for k in f.keys():
                t = f.get_slice(k)
                shape = tuple(t.get_shape())
                dtype = str(t.get_dtype())
                # Print every key in the first layer and any non-layer-indexed parameter
                if "layers.0." in k or "layers" not in k:
                    print(f"  {k:<60}  {str(shape):<24} {dtype}")
                    layer0_seen = layer0_seen or "layers.0." in k
                total += 1
    print(f"\n  (showing layer 0 only; total parameter tensors across model: {total})")


if __name__ == "__main__":
    main()
