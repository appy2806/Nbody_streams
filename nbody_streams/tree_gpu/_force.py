"""
_force.py  --  ctypes interface to the GPU tree-code gravity shared library.

Supports single-species and multi-species simulations via per-particle softening.

Usage
-----
Single species (scalar eps):
    from nbody_streams.tree_gpu import tree_gravity_gpu
    acc, phi = tree_gravity_gpu(pos, mass, eps=0.05)

Multi-species (per-particle eps array):
    eps_arr = cp.where(is_dark_matter, 0.05, 0.1)
    acc, phi = tree_gravity_gpu(pos, mass, eps=eps_arr)

Pre-allocated handle for time-stepping loops (avoids ~27 ms malloc per step):
    from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU
    tree = TreeGPU(n, eps=0.05, theta=0.75)   # eps here is a fallback default
    for step in range(nsteps):
        acc, phi = tree_gravity_gpu(pos, mass, eps=eps_arr, tree=tree)
    tree.close()

Softening convention  (max convention — matches nbody_streams)
--------------------
Direct particle-particle:  eps2_ij = max(eps_i^2, eps_j^2)
Cell approximation:        eps2_ij = max(eps_i^2, eps_cell_max^2)
Both are Plummer: r^2 -> r^2 + eps2.

Performance note
----------------
Passing a scalar eps is equivalent to passing an array of repeated values.
The Python side broadcasts with cp.full() (a fast GPU memset).  Inside the
CUDA kernel all source-particle eps reads will cache-hit perfectly — the
per-particle path has negligible overhead over the old single-eps path.
"""

import ctypes
import ctypes.util
import os
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError("CuPy is required for tree_gpu.py")

__all__ = ["tree_gravity_gpu", "TreeGPU", "cuda_alive"]

G_DEFAULT = float(4.3009173e-06)  # kpc (km/s)^2 / Msun

# ---------------------------------------------------------------------------
# cuda_alive() — lightweight CUDA context health check via ctypes
# ---------------------------------------------------------------------------
# CuPy does not bind cudaGetLastError() in all versions, so we call libcudart
# directly.  cudaGetLastError() is O(1) with no GPU synchronisation — it just
# reads a thread-local error variable.  Safe to call every few hundred steps.
def _find_cudart() -> ctypes.CDLL | None:
    """Load libcudart; try versioned names as fallback."""
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0",
                 "libcudart.so.11", "cudart64_12.dll", "cudart64_110.dll"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaGetLastError.restype = ctypes.c_int
            return lib
        except OSError:
            pass
    return None

_libcudart = _find_cudart()

def cuda_alive() -> bool:
    """Return True if the CUDA context has no pending errors.

    Uses cudaGetLastError() via ctypes — zero GPU overhead, no synchronisation.
    Recommended call frequency: every e_every steps (alongside energy checks),
    NOT every single step.  The NaN sentinel after each force call is the
    primary per-step health indicator.

    Returns True (assume OK) if libcudart cannot be loaded.
    """
    if _libcudart is None:
        return True           # can't check; assume healthy
    return _libcudart.cudaGetLastError() == 0   # 0 == cudaSuccess

_BYTES_PER_PARTICLE = 223  # ~215 base + 8 (two float eps arrays: d_ptclEpsTree + d_ptclEpsGrp)
_IDX_CAP = 0x0FFFFFFF

def _max_n() -> int:
    free, _ = cp.cuda.runtime.memGetInfo()
    return min(int(free * 0.80) // _BYTES_PER_PARTICLE, _IDX_CAP)

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "libtreeGPU.so"))

# ---------------------------------------------------------------------------
# ctypes declarations
# ---------------------------------------------------------------------------
_ptr = ctypes.c_void_p
_int = ctypes.c_int
_flt = ctypes.c_float

class _ResInteractions(ctypes.Structure):
    _fields_ = [
        ("direct_avg", ctypes.c_double),
        ("direct_max", ctypes.c_double),
        ("approx_avg", ctypes.c_double),
        ("approx_max", ctypes.c_double),
    ]

_lib.tree_new.restype   = _ptr
_lib.tree_new.argtypes  = [_flt, _flt]
_lib.tree_delete.argtypes = [_ptr]
_lib.tree_alloc.argtypes  = [_ptr, _int]
_lib.tree_set_verbose.argtypes = [_ptr, _int]

# Primary load: pos + mass + per-particle eps (float* GPU pointer)
_lib.tree_set_pos_mass_eps_device.argtypes = [
    _ptr, _int,
    _ptr, _ptr, _ptr,   # x, y, z
    _ptr, _ptr,         # mass, eps_arr
]

# Legacy load: pos + mass only (uses constructor eps as uniform)
_lib.tree_set_pos_mass_device.argtypes = [_ptr, _int, _ptr, _ptr, _ptr, _ptr]

# Pipeline
_lib.tree_build.argtypes              = [_ptr, _int]
_lib.tree_compute_multipoles.argtypes = [_ptr]
_lib.tree_make_groups.argtypes        = [_ptr, _int, _int]
_lib.tree_compute_forces.restype      = _ResInteractions
_lib.tree_compute_forces.argtypes     = [_ptr]

# Output
_lib.tree_get_acc_device.argtypes = [_ptr, _int, _ptr, _ptr, _ptr, _ptr]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _gpu_ptr(arr: cp.ndarray):
    assert isinstance(arr, cp.ndarray)
    assert arr.flags.c_contiguous
    assert arr.dtype == cp.float32
    return ctypes.c_void_p(arr.data.ptr)

def _coerce_eps(eps, n: int) -> cp.ndarray:
    """Return a contiguous float32 CuPy array of length n for eps.

    Accepts:
      - float / int  → cp.full(n, eps)  (fast GPU broadcast, perfect cache)
      - np.ndarray   → transfer to GPU
      - cp.ndarray   → contiguous float32 view (zero-copy if already correct)
    """
    if isinstance(eps, (int, float)):
        return cp.full(n, float(eps), dtype=cp.float32)
    if isinstance(eps, np.ndarray):
        return cp.asarray(eps, dtype=cp.float32)
    # cp.ndarray
    return cp.ascontiguousarray(eps, dtype=cp.float32)

# ---------------------------------------------------------------------------
# Reusable tree handle
# ---------------------------------------------------------------------------

class TreeGPU:
    """Pre-allocated GPU tree handle for time-stepping loops.

    Parameters
    ----------
    n : int
        Number of particles.  Must not exceed available VRAM.
    eps : float
        Default (fallback) softening length.  Used only when
        ``tree_set_pos_mass_device`` is called; ignored when
        ``tree_gravity_gpu`` is called with an explicit eps.
    theta : float
        Barnes-Hut opening angle.
    verbose : bool
        Print timing/interaction counts from the C++ layer.

    Usage
    -----
        tree = TreeGPU(n, eps=0.05, theta=0.75)
        for step in range(nsteps):
            acc, phi = tree_gravity_gpu(pos, mass, eps=0.05, tree=tree)
        tree.close()   # or use as context manager
    """

    def __init__(self, n: int, eps: float = 0.0, theta: float = 0.6,
                 verbose: bool = False):
        cap = _max_n()
        assert n <= cap, (
            f"N={n:,} exceeds GPU cap ({cap:,}); "
            f"limited by {'VRAM (80% free)' if cap < _IDX_CAP else '28-bit Particle4 index'}."
        )
        self._ptr = _lib.tree_new(_flt(eps), _flt(theta))
        _lib.tree_set_verbose(self._ptr, int(verbose))
        _lib.tree_alloc(self._ptr, n)
        self.n = n
        self.eps = eps
        self.theta = theta

    def close(self):
        if self._ptr is not None:
            _lib.tree_delete(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tree_gravity_gpu(
    pos: cp.ndarray,
    mass,
    eps,
    G: float            = G_DEFAULT,
    *,
    theta: float        = 0.6,
    nleaf: int          = 64,
    ncrit: int          = 64,
    level_split: int    = 5,
    verbose: bool       = False,
    tree: TreeGPU | None = None,
):
    """
    Compute gravitational accelerations and potential using the GPU tree code
    (Barnes-Hut with monopole + quadrupole moments).

    Parameters
    ----------
    pos : cp.ndarray, shape (N, 3)
        Particle positions (float32 on GPU).
    mass : float or cp.ndarray, shape (N,)
        Particle masses.
    eps : float or cp.ndarray, shape (N,)
        Plummer softening length(s).
        - scalar  → uniform softening, broadcast to all N particles.
        - array   → per-particle softening for multi-species simulations.
        Kernel uses: direct:  eps2_ij = max(eps_i^2, eps_j^2)
                     approx:  max(eps_i^2, eps_cell_max^2)  (max convention)
    G : float
        Gravitational constant.  Default: kpc / (km/s)^2 / Msun.
    theta : float
        Opening angle (smaller = more accurate).  Typical: 0.75 (fast),
        0.5 (accurate), 0.3 (very accurate).
    nleaf : int
        Maximum particles per leaf cell.  Must be in {16, 24, 32, 48, 64}.
    ncrit : int
        Group size for the tree walk.  64 is a good default.
    level_split : int
        Tree level for spatial grouping.
    verbose : bool
        Print timing / interaction counts from the C++ layer.
    tree : TreeGPU, optional
        Pre-allocated tree handle.  Skips ~27 ms malloc/free per call.

    Returns
    -------
    acc : cp.ndarray, shape (N, 3)
        Gravitational accelerations in original particle order.
    phi : cp.ndarray, shape (N,)
        Gravitational potential per particle in original particle order.
    """
    # --- coerce pos to float32 F-order (column-major: x/y/z are contiguous views) ---
    if isinstance(pos, cp.ndarray) and pos.dtype == cp.float32:
        pos_f = cp.asfortranarray(pos)
    else:
        pos_f = cp.array(pos, dtype=cp.float32, order='F')

    n = pos_f.shape[0]

    # --- coerce mass ---
    if isinstance(mass, (np.ndarray, cp.ndarray)):
        if not isinstance(mass, cp.ndarray):
            mass = cp.asarray(mass, dtype=cp.float32)
        mass = cp.ascontiguousarray(mass, dtype=cp.float32)
    else:
        mass = cp.full(n, mass, dtype=cp.float32)

    # --- coerce eps (scalar or array) ---
    eps_arr = _coerce_eps(eps, n)

    assert pos_f.shape == (n, 3), f"pos must be (N,3), got {pos_f.shape}"
    assert mass.shape == (n,),    f"mass must be (N,), got {mass.shape}"
    assert eps_arr.shape == (n,), f"eps must be scalar or (N,), got {eps_arr.shape}"
    assert nleaf in (16, 24, 32, 48, 64), f"nleaf must be in {{16,24,32,48,64}}"

    # SoA views — F-order columns are contiguous, zero allocation
    x = pos_f[:, 0]
    y = pos_f[:, 1]
    z = pos_f[:, 2]

    owns_tree = tree is None
    if owns_tree:
        cap = _max_n()
        assert n <= cap, (
            f"N={n:,} exceeds GPU cap ({cap:,}); "
            f"limited by {'VRAM (80% free)' if cap < _IDX_CAP else '28-bit Particle4 index'}."
        )
        # Use 0.0 as dummy global eps (not used; per-particle eps drives everything)
        _tree = _lib.tree_new(_flt(0.0), _flt(theta))
        _lib.tree_set_verbose(_tree, int(verbose))
        _lib.tree_alloc(_tree, n)
    else:
        assert n == tree.n, f"pos has {n} particles but tree was allocated for {tree.n}"
        _tree = tree._ptr

    try:
        _lib.tree_set_pos_mass_eps_device(
            _tree, n,
            _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z),
            _gpu_ptr(mass), _gpu_ptr(eps_arr),
        )

        _lib.tree_build(_tree, nleaf)
        _lib.tree_compute_multipoles(_tree)
        _lib.tree_make_groups(_tree, level_split, ncrit)
        _lib.tree_compute_forces(_tree)

        ax  = cp.empty(n, dtype=cp.float32)
        ay  = cp.empty(n, dtype=cp.float32)
        az  = cp.empty(n, dtype=cp.float32)
        phi = cp.empty(n, dtype=cp.float32)

        _lib.tree_get_acc_device(
            _tree, n,
            _gpu_ptr(ax), _gpu_ptr(ay), _gpu_ptr(az), _gpu_ptr(phi),
        )
        # cp.cuda.runtime.deviceSynchronize()

    finally:
        if owns_tree:
            _lib.tree_delete(_tree)

    acc = cp.stack([ax, ay, az], axis=1)

    if G != 1.0:
        acc *= G
        phi *= G

    return acc, phi
