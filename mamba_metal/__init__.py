"""mamba-metal: Mamba selective scan in Metal Shading Language."""

from mamba_metal._loader import load_kernel
from mamba_metal.selective_scan import selective_scan

__all__ = ["load_kernel", "selective_scan"]
