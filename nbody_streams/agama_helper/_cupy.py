"""
_cupy.py
~~~~~~~~
Optional-CuPy shim for ``agama_helper``.

CuPy is an *optional* dependency (``pip install nbody_streams[cuda]``).  Most of
this subpackage --- coefficient readers, HDF5 archives, Agama potential loaders,
FIRE helpers --- is pure NumPy and must stay usable on a CPU-only machine.  Only
the ``*PotentialGPU`` classes need CuPy.

Importing this module never fails:

* CuPy present  -> ``cp`` is the real module and ``CUPY_AVAILABLE`` is True.
* CuPy missing  -> ``cp`` is a stub.  Every attribute access raises a clear
  :class:`ImportError`, *except* the kernel constructors (``ElementwiseKernel``,
  ``RawKernel``, ``RawModule``, ``ReductionKernel``, ``fuse``), which return a
  placeholder that raises the same error only when the kernel is finally called.

The exception keeps module-level kernel definitions (e.g. the ~35
``cp.ElementwiseKernel`` calls in ``_analytic_potentials.py``) importable, while
still failing loudly the moment a GPU code path is actually exercised.

Usage
-----
>>> from ._cupy import CUPY_AVAILABLE, cp, require_cupy
>>> if CUPY_AVAILABLE:
...     ...
>>> require_cupy("PotentialGPU")   # raise now, with a helpful message
"""

from __future__ import annotations

__all__ = ["cp", "CUPY_AVAILABLE", "require_cupy"]


_INSTALL_HINT = (
    "CuPy is required for the GPU potential classes of "
    "nbody_streams.agama_helper, but it is not installed.\n"
    "Install with:  pip install cupy-cuda12x        (adjust to your CUDA version)\n"
    "           or:  pip install 'nbody_streams[cuda]'\n"
    "CPU-only features (read_coefs, load_agama_potential, write_*_h5, FIRE "
    "helpers) do not need CuPy."
)


def _error(what: str | None = None) -> ImportError:
    prefix = f"{what} requires CuPy.\n" if what else ""
    return ImportError(prefix + _INSTALL_HINT)


class _MissingKernel:
    """Stand-in for a CuPy kernel constructed while CuPy is unavailable."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "CUDA kernel") -> None:
        self._name = name

    def __call__(self, *args, **kwargs):
        raise _error(self._name)

    def get_function(self, name: str) -> "_MissingKernel":
        return _MissingKernel(name)

    def __repr__(self) -> str:
        return f"<_MissingKernel {self._name!r} (CuPy not installed)>"


# Attributes that are *called at import time* to build kernels.  These return a
# placeholder instead of raising, so module import stays cheap and side-effect
# free on CPU-only installs.
_KERNEL_FACTORIES = frozenset(
    {"ElementwiseKernel", "RawKernel", "RawModule", "ReductionKernel", "fuse"}
)


class _MissingCuPy:
    """Stub standing in for the ``cupy`` module when it is not installed."""

    __slots__ = ()

    def __getattr__(self, name: str):
        if name.startswith("__"):           # keep introspection/pickling sane
            raise AttributeError(name)
        if name in _KERNEL_FACTORIES:
            def _deferred(*args, **kwargs):
                # cp.ElementwiseKernel(..., 'nfw_phi') -> name is the last arg
                label = next(
                    (a for a in reversed(args) if isinstance(a, str)), name
                )
                return _MissingKernel(label)
            return _deferred
        raise _error(f"cupy.{name}")

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<cupy stub (not installed)>"


try:
    import cupy as cp                       # type: ignore[import-not-found]
    CUPY_AVAILABLE = True
except ImportError:
    cp = _MissingCuPy()                     # type: ignore[assignment]
    CUPY_AVAILABLE = False


def require_cupy(what: str | None = None) -> None:
    """Raise a descriptive :class:`ImportError` if CuPy is unavailable.

    Parameters
    ----------
    what : str, optional
        Name of the feature being requested, used to prefix the message.
    """
    if not CUPY_AVAILABLE:
        raise _error(what)
