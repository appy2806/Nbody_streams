"""CUDA source for the direct-sum kernels.

The kernels live in ``nbody_streams/cuda/``: ``softening.cuh`` (the softening
formulas) and ``direct.cu`` (forces and potentials). The softening kernel is
chosen at compile time with ``-DKERNEL_ID``.
"""

from functools import lru_cache
from pathlib import Path

CUDA_DIR = Path(__file__).with_name("cuda")

__all__ = ["CUDA_DIR", "direct_source"]


@lru_cache(maxsize=None)
def direct_source() -> str:
    """Return ``direct.cu`` with ``softening.cuh`` inlined.

    Inlining keeps CuPy's compile cache keyed on both files.
    """
    src = (CUDA_DIR / "direct.cu").read_text()
    hdr = (CUDA_DIR / "softening.cuh").read_text()
    return src.replace('#include "softening.cuh"', hdr)
