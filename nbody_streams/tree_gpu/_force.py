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

from .._recentre import to_float32_relative_soa

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

# GPU memory model of one TreeGPU handle, measured with cudaMemGetInfo (see
# docs/tree_gpu.md, "GPU memory layout"):
#   tree buffers after the first force call: 151 B/particle at allocation
#   (particle arrays 76, node pool 48, cell list 24, work queues 4) plus the
#   per-cell multipole arrays (124 B/cell), walk groups (8) and sort scratch;
#   216 covers trees of up to 0.4 N cells (measured <= 0.37 N, a cubic
#   lattice at nleaf=16: 208 B/particle).
_TREE_BYTES_PER_PARTICLE = 216
#   per tree_gravity_gpu call (CuPy pool): float32 SoA positions (12), SoA
#   acc + phi (16), the returned (N, 3) acc (12).
_CALL_BYTES_PER_PARTICLE = 40
#   N-independent: minimum buffer sizes, Thrust scratch, CuPy kernels ...
_FIXED_BYTES = 64 << 20
#   ... and the tree-walk pool, 2 MiB per persistent block x 6 blocks per SM
#   (capped at 1/4 of the free memory when it is allocated).
_WALK_POOL_BYTES_PER_SM = 12 << 20
# Index-packing limit of the tree build: the particle index has 29 bits in
# Particle4 (= _lib.tree_max_particles()).
_IDX_CAP = 1 << 29

_n_sm = None
_MEM_FRACTION = 0.80


def set_memory_fraction(fraction: float) -> float:
    """Set the fraction of free GPU memory the GPU tree may plan for.

    ``TreeGPU`` and ``run_nbody_gpu_tree`` refuse an N whose modelled
    footprint exceeds this fraction of the free memory (default 0.80; the
    rest absorbs allocator rounding, fragmentation and CuPy pool growth).
    The model is sized for the worst distribution measured (a cubic lattice
    at ``nleaf=16``); smooth distributions use ~10-20% less, so e.g. 0.95
    admits ~1.2x more particles on a dedicated GPU.  Above the default an
    unusual distribution can run out of memory inside the build, which ends
    the process.

    Returns the previous value.
    """
    global _MEM_FRACTION
    f = float(fraction)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    old, _MEM_FRACTION = _MEM_FRACTION, f
    return old


def get_memory_fraction() -> float:
    """Fraction of free GPU memory the GPU tree may plan for (see
    :func:`set_memory_fraction`)."""
    return _MEM_FRACTION


def _max_n(extra_bytes_per_particle: int = 0) -> int:
    """Largest N whose tree (plus ``extra_bytes_per_particle`` that the caller
    will allocate later) fits in ``get_memory_fraction()`` (default 80%) of
    the free GPU memory; the rest absorbs allocator rounding, fragmentation
    and CuPy pool growth."""
    global _n_sm
    if _n_sm is None:
        _n_sm = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    free, _ = cp.cuda.runtime.memGetInfo()
    walk = min(free // 4, _WALK_POOL_BYTES_PER_SM * _n_sm)
    usable = int(free * _MEM_FRACTION) - _FIXED_BYTES - walk
    per = _TREE_BYTES_PER_PARTICLE + _CALL_BYTES_PER_PARTICLE + int(extra_bytes_per_particle)
    return max(0, min(usable // per, _IDX_CAP))

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
_lib.tree_set_theta.argtypes   = [_ptr, _flt]

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
_lib.tree_build.restype               = _int
_lib.tree_compute_multipoles.argtypes = [_ptr]
_lib.tree_make_groups.argtypes        = [_ptr, _int, _int]
_lib.tree_compute_forces.restype      = _ResInteractions
_lib.tree_compute_forces.argtypes     = [_ptr]

# Output
_lib.tree_get_acc_device.argtypes = [_ptr, _int, _ptr, _ptr, _ptr, _ptr]

# Diagnostics: last build's usage / capacities, index limits, packing self-test
_lib.tree_get_build_stats.argtypes = [_ptr, _ptr]
_lib.tree_max_particles.restype    = _int
_lib.tree_max_cells.restype        = _int
_lib.tree_selftest_packing.restype = _int

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
    if np.isscalar(eps):
        if not float(eps) > 0.0:
            raise ValueError(f"softening must be > 0, got {eps}")
        return cp.full(n, float(eps), dtype=cp.float32)
    if isinstance(eps, np.ndarray):
        if not np.all(eps > 0.0):
            raise ValueError("softening must be > 0 for every particle")
        return cp.asarray(eps, dtype=cp.float32)
    # cp.ndarray: the check costs one reduction + a host sync
    eps_arr = cp.ascontiguousarray(eps, dtype=cp.float32)
    if not bool(cp.all(eps_arr > 0.0)):
        raise ValueError("softening must be > 0 for every particle")
    return eps_arr

# ---------------------------------------------------------------------------
# Reusable tree handle
# ---------------------------------------------------------------------------

class TreeGPU:
    """Pre-allocated GPU tree handle for time-stepping loops.

    Parameters
    ----------
    n : int
        Number of particles (>= 0; 0 gives empty results).  Must not exceed
        available VRAM.
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
        self._ptr = None   # so close()/__del__ are safe if the checks below fail
        if int(n) != n or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        n = int(n)
        if not 0.0 < theta < 1.15:
            raise ValueError(f"theta must be in (0, 1.15), got {theta}")
        cap = _max_n()
        if n > cap:
            raise ValueError(
                f"N={n:,} exceeds the GPU tree limit ({cap:,}), set by "
                + (f"{_MEM_FRACTION:.0%} of free GPU memory (raise it with "
                   "nbody_streams.tree_gpu.set_memory_fraction)" if cap < _IDX_CAP
                   else "the 29-bit particle index (2^29 particles)") + ".")
        self._ptr = _lib.tree_new(_flt(eps), _flt(theta))
        _lib.tree_set_verbose(self._ptr, int(verbose))
        if n > 0:   # n = 0: no buffers; tree_gravity_gpu returns empty arrays
            _lib.tree_alloc(self._ptr, n)
        self.n = n
        self.eps = eps
        self._theta = float(theta)
        # Softening array validated by the last tree_gravity_gpu call; the
        # same array object is not re-checked (the check costs a host sync).
        self._eps_checked = None

    def build_stats(self) -> dict:
        """Usage of the last tree build and the current build capacities.

        ``cells``, ``nodes`` (cells that were split), ``leaves``, ``levels``,
        ``work_items`` (largest number of cells split at one level), and
        ``cell_capacity``, ``node_capacity``, ``work_capacity``: the buffer
        sizes, which grow (and stay grown) when a build overflows them.
        """
        out = (ctypes.c_longlong * 8)()
        if self._ptr is not None and self.n > 0:
            _lib.tree_get_build_stats(self._ptr, out)
        keys = ("cells", "nodes", "leaves", "levels", "work_items",
                "cell_capacity", "node_capacity", "work_capacity")
        return dict(zip(keys, (int(v) for v in out)))

    @property
    def theta(self) -> float:
        """Barnes-Hut opening angle used by the next force evaluation."""
        return self._theta

    @theta.setter
    def theta(self, value: float):
        if not 0.0 < value < 1.15:
            raise ValueError(f"theta must be in (0, 1.15), got {value}")
        _lib.tree_set_theta(self._ptr, _flt(value))
        self._theta = float(value)

    def close(self):
        if getattr(self, "_ptr", None) is not None:
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

# One pass after the tree walk: SoA (3, N) -> (N, 3), remove the self term
# from phi, scale by G.  Same float32 operations, in the same order, as the
# separate stack / += / *= passes it replaces.
_finish_kernel = cp.ElementwiseKernel(
    "raw float32 a, float32 m, float32 e, float32 G, int64 n",
    "raw float32 acc, float32 phi",
    """
    acc[3*i]     = a[i] * G;
    acc[3*i + 1] = a[n + i] * G;
    acc[3*i + 2] = a[2*n + i] * G;
    phi = (phi + m / e) * G;
    """, "tree_finish")

def tree_gravity_gpu(
    pos: cp.ndarray,
    mass,
    eps,
    G: float            = G_DEFAULT,
    *,
    theta: float | None = None,
    nleaf: int          = 64,
    ncrit: int          = 64,
    level_split: int    = 5,
    verbose: bool       = False,
    tree: TreeGPU | None = None,
    center=None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute gravitational accelerations and potential using the GPU tree code
    (Barnes-Hut with monopole + quadrupole moments).

    Parameters
    ----------
    pos : cp.ndarray or np.ndarray, shape (N, 3)
        Particle positions.  float64 input is converted to float32 relative
        to ``center`` (subtracted in float64), so compact structure far from
        the origin keeps float32 precision; float32 input is used as given.
        Forces and potential are translation invariant, so the output does
        not depend on the centre beyond precision.
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
    theta : float, optional
        Opening angle (smaller = more accurate).  Typical: 0.75 (fast),
        0.5 (accurate), 0.3 (very accurate).  Default: ``tree.theta`` when a
        handle is passed, else 0.6.  Passing a value together with ``tree``
        updates the handle's theta.
    nleaf : int
        Maximum particles per leaf cell.  Must be in {16, 24, 32, 48, 64}.
    ncrit : int
        Group size for the tree walk.  64 is a good default.
    level_split : int
        Unused; accepted for backward compatibility (walk groups are
        cut from tree cells).
    verbose : bool
        Print timing / interaction counts from the C++ layer.
    tree : TreeGPU, optional
        Pre-allocated tree handle.  Skips ~27 ms malloc/free per call.
    center : array_like, shape (3,), optional
        Centre subtracted from float64 ``pos`` before the float32 cast.
        Default: mass-weighted mean of a ~4096-particle strided subsample
        (adds ~0.2 ms at N = 1e6).  Pass the progenitor position when most of
        the mass has been stripped and the mean lies far from the bound core.
        Ignored for float32 ``pos``.

    With a ``tree`` handle, an ``eps`` array is checked for positive values
    only the first time that array object is passed: modify it in place
    between calls only with positive values.

    Returns
    -------
    acc : cp.ndarray, shape (N, 3)
        Gravitational accelerations in original particle order.
    phi : cp.ndarray, shape (N,)
        Gravitational potential per particle in original particle order,
        excluding the particle's interaction with itself (same convention as
        the direct-sum backend).
    """
    # --- float32 positions (relative to a centre for float64 input) as
    #     (3, N): the x/y/z rows are the contiguous arrays the tree reads ---
    if not isinstance(pos, cp.ndarray):
        pos = np.asarray(pos)
        pos = cp.asarray(pos, dtype=cp.float32 if pos.dtype == np.float32 else cp.float64)
    assert pos.ndim == 2 and pos.shape[1] == 3, f"pos must be (N,3), got {pos.shape}"
    if pos.shape[0] == 0:
        if tree is not None:
            assert tree.n == 0, f"pos has 0 particles but tree was allocated for {tree.n}"
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32)
    pos_soa, _ = to_float32_relative_soa(pos, mass, center)

    n = pos_soa.shape[1]

    # --- coerce mass ---
    if isinstance(mass, (np.ndarray, cp.ndarray)):
        if not isinstance(mass, cp.ndarray):
            mass = cp.asarray(mass, dtype=cp.float32)
        mass = cp.ascontiguousarray(mass, dtype=cp.float32)
    else:
        mass = cp.full(n, mass, dtype=cp.float32)

    # --- coerce eps (scalar or array) ---
    if tree is not None and eps is tree._eps_checked:
        eps_arr = eps
    else:
        eps_arr = _coerce_eps(eps, n)
        if tree is not None:
            tree._eps_checked = eps if eps_arr is eps else None

    assert mass.shape == (n,),    f"mass must be (N,), got {mass.shape}"
    assert eps_arr.shape == (n,), f"eps must be scalar or (N,), got {eps_arr.shape}"
    assert nleaf in (16, 24, 32, 48, 64), f"nleaf must be in {{16,24,32,48,64}}"

    x, y, z = pos_soa       # contiguous rows, zero allocation

    owns_tree = tree is None
    if owns_tree:
        # eps=0.0 is a dummy global value; per-particle eps drives everything
        tree = TreeGPU(n, eps=0.0, theta=0.6 if theta is None else theta,
                       verbose=verbose)
    else:
        assert n == tree.n, f"pos has {n} particles but tree was allocated for {tree.n}"
        if theta is not None and theta != tree.theta:
            tree.theta = theta
    _tree = tree._ptr

    try:
        _lib.tree_set_pos_mass_eps_device(
            _tree, n,
            _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z),
            _gpu_ptr(mass), _gpu_ptr(eps_arr),
        )

        if _lib.tree_build(_tree, nleaf) != 0:
            raise RuntimeError(
                "GPU tree build overflowed and its buffers could not grow "
                "(see stderr); too many (nearly) coincident particles?")
        _lib.tree_compute_multipoles(_tree)
        _lib.tree_make_groups(_tree, level_split, ncrit)
        _lib.tree_compute_forces(_tree)

        acc_soa = cp.empty((3, n), dtype=cp.float32)
        phi     = cp.empty(n, dtype=cp.float32)

        _lib.tree_get_acc_device(
            _tree, n,
            _gpu_ptr(acc_soa[0]), _gpu_ptr(acc_soa[1]), _gpu_ptr(acc_soa[2]),
            _gpu_ptr(phi),
        )
        # cp.cuda.runtime.deviceSynchronize()

    finally:
        if owns_tree:
            tree.close()

    # The leaf direct sum includes i == j (r = 0, so no force but a potential
    # term -m_i/eps_i under the max-eps convention).  Remove it so phi matches
    # the direct-sum backend; bound-particle criteria (phi + v^2/2 < 0) rely on it.
    acc = cp.empty((n, 3), dtype=cp.float32)
    if n:
        _finish_kernel(acc_soa, mass, eps_arr, np.float32(G), np.int64(n), acc, phi)

    return acc, phi
