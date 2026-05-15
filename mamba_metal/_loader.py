"""Load .metal kernel bodies and wrap them with mx.fast.metal_kernel.

`mx.fast.metal_kernel` takes only the kernel function *body* — it generates the
`kernel void <name>(...)` signature automatically from input/output names.
So each .metal file in `kernels/` contains just the body MSL code.
"""

from pathlib import Path

import mlx.core as mx


KERNELS_DIR = Path(__file__).parent / "kernels"


def read_kernel_source(name: str, params: dict | None = None) -> str:
    """Read `kernels/<name>.metal` and optionally format with `params`.

    Use ``{KEY}`` placeholders in the .metal file when parametrising
    (e.g. window sizes that need to be compile-time constants in MSL).
    """
    path = KERNELS_DIR / f"{name}.metal"
    src = path.read_text()
    if params:
        src = src.format(**params)
    return src


def load_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    params: dict | None = None,
):
    """Compile a Metal kernel from the corresponding .metal file."""
    source = read_kernel_source(name, params=params)
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
    )
