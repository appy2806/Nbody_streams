"""
sph_kernels.py
==============
SPH smoothing-length computation and 2-D surface-density rendering for the
Nbody_streams package.

Public API
----------
render_surface_density(x, y, mass, ...)
    High-level entry point -- auto-selects GPU or CPU backend.
get_smoothing_lengths(pos, k_neighbors, ...)
    Compute per-particle SPH smoothing lengths (GPU -> CPU fallback).
render_cpu(x, y, mass, h, ...)
    CPU-only Numba-parallel SPH splatting.
render_gpu(x, y, mass, h, ...)
    GPU-only CUDA SPH splatting (requires CuPy).

GPU path  : Numba CUDA kernel (splatting) + CuPy KDTree (smoothing lengths).
CPU path  : Numba njit(parallel=True) / prange + SciPy KDTree.

``cKDTree`` is replaced throughout by ``KDTree`` (merged in SciPy 1.12;
identical interface, same ``balanced_tree`` / ``workers`` kwargs).

Performance notes
-----------------
GPU splatting uses a *scatter* approach: one CUDA thread per particle performs
atomic adds to the shared grid buffer.  This is efficient for moderate
smoothing lengths.  For repeated renders on the same particle set, pass
pre-computed ``h`` to avoid re-running the KNN query.

CPU splatting uses Numba ``prange`` with an array-reduction pattern: Numba
allocates per-thread private copies of the grid and sums them after the loop,
so there are no race conditions.  The first call incurs JIT compilation
overhead (~1-5 s); subsequent calls use the on-disk cache.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Tuple

import numpy as np
from numba import cuda, njit, prange
from scipy.spatial import KDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional CuPy imports -- graceful degradation when CUDA is absent
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.spatial import KDTree as CupyKDTree
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ===========================================================================
# SECTION 0 -- Morton Z-curve helper
# ===========================================================================

def _argsort_morton_2d(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: float,
) -> np.ndarray:
    """Return argsort indices that sort particles into 2-D Morton Z-order.

    Maps each particle to a (uint16, uint16) grid cell, interleaves the
    bits into a uint32 Morton code, and returns ``np.argsort`` of those
    codes.  Sorting the four input arrays (x, y, mass, h) by this index
    before the GPU scatter kernel groups spatially adjacent particles
    contiguously in memory, reducing atomic-add contention on the
    accumulation grid and improving L1/L2 cache hit rates.

    The sort itself runs in pure NumPy on the CPU and typically takes
    50-150 ms for N=5M particles; the permutation copies add ~80 MB of
    temporary host memory.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Particle positions (same units as *gridsize*).
    gridsize : float
        Total size of the rendered region (grid spans [-gridsize/2, gridsize/2]).

    Returns
    -------
    order : np.ndarray, shape (N,), dtype int64
        Index array such that ``x[order], y[order]`` is in Morton order.
    """
    half = float(gridsize) / 2.0
    scale = np.float32(65535.0 / float(gridsize))
    xi = np.clip((x + half) * scale, 0, 65535).astype(np.uint32)
    yi = np.clip((y + half) * scale, 0, 65535).astype(np.uint32)

    # Spread 16-bit value into even bit positions of a 32-bit word
    # (standard "magic bits" interleaving for 16-bit inputs -> 32-bit Morton)
    xi = (xi | (xi << np.uint32(8))) & np.uint32(0x00FF00FF)
    xi = (xi | (xi << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    xi = (xi | (xi << np.uint32(2))) & np.uint32(0x33333333)
    xi = (xi | (xi << np.uint32(1))) & np.uint32(0x55555555)

    yi = (yi | (yi << np.uint32(8))) & np.uint32(0x00FF00FF)
    yi = (yi | (yi << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    yi = (yi | (yi << np.uint32(2))) & np.uint32(0x33333333)
    yi = (yi | (yi << np.uint32(1))) & np.uint32(0x55555555)

    codes = xi | (yi << np.uint32(1))
    return np.argsort(codes)


# ===========================================================================
# SECTION 1 -- Smoothing-length computation
# ===========================================================================

def get_smoothing_lengths(
    pos: np.ndarray,
    k_neighbors: int = 32,
    safety_factor: float = 0.6,
    gpu_vram_threshold_gb: float = 10.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute SPH smoothing lengths as the distance to the *k_neighbors*-th
    nearest neighbour, with automatic GPU -> CPU fallback.

    The GPU path (CuPy KDTree) is attempted first when sufficient VRAM is
    available.  On consumer cards (RTX 3070, 8 GB) the tree structure alone
    can consume most of the device memory for N > ~20 M, so the threshold
    and safety factor are deliberately conservative.  On datacenter cards
    (H200, 80 GB) the GPU path will handle 70-100 M particles comfortably.

    If the GPU attempt fails for any reason (OOM, fragmentation, driver
    error), the function transparently retries on the CPU using a
    multi-threaded SciPy KDTree (``workers=-1``).

    Parameters
    ----------
    pos : np.ndarray, shape (N, D), dtype float32 or float64
        Particle positions.  *D* is typically 2 (projected) or 3.
    k_neighbors : int, optional
        Number of neighbours (including the particle itself).  The
        smoothing length is set to ``dist[:, k_neighbors-1]``.
        Default ``32``.
    safety_factor : float, optional
        Fraction of post-build free VRAM used for query result buffers.
        Kept at 0.6 to leave headroom for internal CuPy tree-traversal
        allocations that are not reflected in ``mem_info``.  Default ``0.6``.
    gpu_vram_threshold_gb : float, optional
        Minimum free VRAM in GiB required to attempt the GPU path.
        Default ``10.0``.
    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    h : np.ndarray, shape (N,), dtype float32
        Smoothing lengths.  Always a plain NumPy array.

    Raises
    ------
    ValueError
        If *pos* is not a 2-D array.
    """
    pos = np.asarray(pos)
    if pos.ndim != 2:
        raise ValueError(f"pos must be 2-D (N, D); got shape {pos.shape}")

    if not CUPY_AVAILABLE:
        return _get_smoothing_lengths_cpu(pos, k=k_neighbors, verbose=verbose)

    free_mem, _ = cp.cuda.Device().mem_info
    free_gb = free_mem / (1024 ** 3)

    if free_gb < gpu_vram_threshold_gb:
        warnings.warn(
            f"[SPH] Free VRAM {free_gb:.1f} GiB < threshold "
            f"{gpu_vram_threshold_gb:.1f} GiB -- using CPU KDTree.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _get_smoothing_lengths_cpu(pos, k=k_neighbors, verbose=verbose)

    try:
        return _get_smoothing_lengths_gpu(
            pos, k=k_neighbors, safety_factor=safety_factor, verbose=verbose
        )
    except Exception as exc:
        warnings.warn(
            f"[SPH] GPU KDTree failed ({type(exc).__name__}: {exc}). "
            "Falling back to CPU KDTree.",
            RuntimeWarning,
            stacklevel=2,
        )
        cp.get_default_memory_pool().free_all_blocks()
        return _get_smoothing_lengths_cpu(pos, k=k_neighbors, verbose=verbose)


def _get_smoothing_lengths_gpu(
    pos: np.ndarray,
    k: int = 32,
    safety_factor: float = 0.6,
    verbose: bool = False,
) -> np.ndarray:
    """GPU smoothing-length computation via CuPy KDTree (internal)."""
    n_points = pos.shape[0]
    pos_cp = cp.asarray(pos, dtype=cp.float32)

    if verbose:
        print("[SPH] Building GPU KDTree...")
    tree = CupyKDTree(pos_cp)

    free_mem, _ = cp.cuda.Device().mem_info
    # Each query point returns k distances (float32 = 4 B) + k indices (int32 = 4 B)
    bytes_per_query_point = k * 8
    usable_mem = free_mem * safety_factor
    chunk_size = int(usable_mem // bytes_per_query_point)

    if chunk_size < 1_000:
        raise MemoryError(
            f"Only {free_mem / 1e9:.2f} GB VRAM free after tree build; "
            "chunk size would be < 1 000."
        )

    if verbose:
        print(f"[SPH] GPU querying {n_points:,} points in chunks of {chunk_size:,}...")
    h_gpu = cp.zeros(n_points, dtype=cp.float32)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        dist, _ = tree.query(pos_cp[start:end], k=k)
        h_gpu[start:end] = dist[:, -1]
        # Eagerly release query buffers to avoid fragmentation
        del dist
        cp.get_default_memory_pool().free_all_blocks()

    return h_gpu.get()


def _get_smoothing_lengths_cpu(
    pos: np.ndarray,
    k: int = 32,
    verbose: bool = False,
) -> np.ndarray:
    """CPU smoothing-length computation via SciPy KDTree (internal).

    Uses ``balanced_tree=False`` for faster build times on large datasets
    and ``workers=-1`` to saturate all available CPU threads during the
    query phase.
    """
    # Accept CuPy arrays transparently
    if CUPY_AVAILABLE and isinstance(pos, cp.ndarray):
        pos_np = pos.get()
    else:
        pos_np = np.asarray(pos, dtype=np.float32)

    if verbose:
        print(f"[SPH] Building CPU KDTree for {pos_np.shape[0]:,} points...")
    # balanced_tree=False skips the median-finding pass -> ~2x faster build
    tree = KDTree(pos_np, balanced_tree=False)

    if verbose:
        print(f"[SPH] CPU querying {k} neighbours (workers=-1)...")
    dist, _ = tree.query(pos_np, k=k, workers=-1)

    return dist[:, -1].astype(np.float32)


# ===========================================================================
# SECTION 2 -- SPH kernel (CPU, Numba JIT)
# ===========================================================================

@njit(cache=True, fastmath=True)
def _cubic_spline_2d_scalar(r: float, h: float) -> float:
    """2-D cubic-spline SPH kernel for a single (r, h) pair.

    Parameters
    ----------
    r : float
        Distance between particle and evaluation point.
    h : float
        Smoothing length of the particle.

    Returns
    -------
    float
        Kernel value W(r, h).
    """
    q = r / h
    norm = 1.818913635332045 / (h * h)   # 40 / (7*pi*h^2)
    if q <= 0.5:
        return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q)
    elif q <= 1.0:
        return norm * (2.0 * (1.0 - q) ** 3)
    return 0.0


# ===========================================================================
# SECTION 3 -- CPU SPH splatting (Numba parallel)
# ===========================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _sph_splat_cpu_kernel(
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    h: np.ndarray,
    grid: np.ndarray,
    x_min: float,
    y_min: float,
    pixel_size: float,
) -> None:
    """Numba-parallel CPU SPH splatting kernel (in-place, no return value).

    Each particle is processed by a separate thread via ``prange``.  Numba
    detects the ``grid[i, j] += value`` pattern as an array reduction and
    allocates per-thread private copies of the grid, summed at the end.
    This avoids race conditions and is faster than explicit atomic ops when
    the average particle footprint spans many pixels.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Particle positions.
    mass : np.ndarray, shape (N,)
        Particle masses.
    h : np.ndarray, shape (N,)
        Smoothing lengths.
    grid : np.ndarray, shape (Nx, Ny), dtype float32
        Output density grid (modified in-place).
    x_min, y_min : float
        Lower-left corner of the grid in data coordinates.
    pixel_size : float
        Physical size of one pixel.
    """
    n = x.shape[0]
    nx = grid.shape[0]
    ny = grid.shape[1]
    inv_pixel = 1.0 / pixel_size

    for idx in prange(n):
        px = x[idx]
        py = y[idx]
        hi = h[idx]
        mi = mass[idx]

        inv_h = 1.0 / hi
        inv_h2 = inv_h * inv_h
        norm = 1.818913635332045 * inv_h2

        i_start = max(0,      int((px - x_min - hi) * inv_pixel))
        i_end   = min(nx - 1, int((px - x_min + hi) * inv_pixel))
        j_start = max(0,      int((py - y_min - hi) * inv_pixel))
        j_end   = min(ny - 1, int((py - y_min + hi) * inv_pixel))

        hi2 = hi * hi

        for i in range(i_start, i_end + 1):
            dx = (i * pixel_size + x_min) - px
            dx2 = dx * dx
            if dx2 >= hi2:
                continue
            for j in range(j_start, j_end + 1):
                dy = (j * pixel_size + y_min) - py
                dist2 = dx2 + dy * dy
                if dist2 >= hi2:
                    continue
                q = math.sqrt(dist2) * inv_h
                if q <= 0.5:
                    w = 1.0 - 6.0 * q * q + 6.0 * q * q * q
                elif q < 1.0:
                    tmp = 1.0 - q
                    w = 2.0 * tmp * tmp * tmp
                else:
                    continue
                grid[i, j] += mi * norm * w


def render_cpu(
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    h: np.ndarray,
    resolution: int = 512,
    gridsize: float = 200.0,
    sort_by_morton: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Render SPH surface density on the CPU using Numba parallel splatting.

    All particles are processed in a single Numba ``prange`` call.  This
    is the recommended path when a GPU is not available or when the
    particle count is small enough that PCIe transfer overhead dominates.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Particle positions (projected 2-D coordinates, host arrays).
    mass : np.ndarray, shape (N,)
        Particle masses.
    h : np.ndarray, shape (N,)
        Smoothing lengths.
    resolution : int, optional
        Grid resolution (number of pixels per side).  Default ``512``.
    gridsize : float, optional
        Total size of the rendered region in data units.  The grid spans
        ``[-gridsize/2, gridsize/2]`` in both axes.  Default ``200.0``.
    sort_by_morton : bool, optional
        If ``True``, reorder particles into 2-D Morton Z-order before
        splatting.  This groups spatially adjacent particles together,
        improving cache locality in the Numba reduction.
        Default ``False``.
    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    grid : np.ndarray, shape (resolution, resolution), dtype float32
        Surface density map (M_sun / kpc^2 in standard nbody_streams units).
    bounds : tuple of float
        ``(x_min, x_max, y_min, y_max)`` suitable for ``imshow(extent=...)``.
    """
    half = float(gridsize) / 2.0
    if verbose:
        print(f"[SPH] CPU render: {len(x):,} particles, "
              f"resolution={resolution}, gridsize={gridsize}")

    if sort_by_morton:
        order = _argsort_morton_2d(x, y, gridsize)
        x, y, mass, h = x[order], y[order], mass[order], h[order]

    x_min = y_min = -half
    pixel_size = float(gridsize) / resolution

    # Must be C-contiguous float32 for Numba
    grid = np.zeros((resolution, resolution), dtype=np.float32)
    _sph_splat_cpu_kernel(
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
        np.ascontiguousarray(mass, dtype=np.float32),
        np.ascontiguousarray(h, dtype=np.float32),
        grid,
        x_min, y_min, pixel_size,
    )
    bounds = (-half, half, -half, half)
    return grid, bounds


# ===========================================================================
# SECTION 4 -- GPU SPH splatting (Numba CUDA)
# ===========================================================================

@cuda.jit(device=True, inline=True)
def _cubic_spline_2d_gpu(r: float, h: float) -> float:
    """2-D cubic-spline SPH kernel evaluated on the GPU (device function)."""
    q = r / h
    norm = 1.818913635332045 / (h * h)
    if q <= 0.5:
        return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q)
    elif q <= 1.0:
        return norm * (2.0 * (1.0 - q) ** 3)
    return 0.0


@cuda.jit
def _sph_splat_gpu_kernel(
    x, y, mass, h, grid, x_min, y_min, pixel_size
):
    """CUDA kernel: splat one particle per thread onto ``grid``.

    One CUDA thread per particle (scatter approach).  Atomic float32 adds
    are used to accumulate contributions to shared grid cells; contention
    is low except in extremely dense regions.

    Parameters
    ----------
    x, y : device array, shape (N,)
        Particle positions (float32).
    mass : device array, shape (N,)
        Particle masses (float32).
    h : device array, shape (N,)
        Smoothing lengths (float32).
    grid : device array, shape (Nx, Ny), float32
        Accumulation buffer (atomic adds).
    x_min, y_min : float
        Lower-left corner coordinates.
    pixel_size : float
        Physical size of one pixel.
    """
    idx = cuda.grid(1)
    if idx >= x.shape[0]:
        return

    px, py = x[idx], y[idx]
    hi = h[idx]
    mi = mass[idx]

    inv_h  = 1.0 / hi
    inv_h2 = inv_h * inv_h
    norm   = 1.818913635332045 * inv_h2
    hi2    = hi * hi

    i_start = max(0,                 int((px - x_min - hi) / pixel_size))
    i_end   = min(grid.shape[0] - 1, int((px - x_min + hi) / pixel_size))
    j_start = max(0,                 int((py - y_min - hi) / pixel_size))
    j_end   = min(grid.shape[1] - 1, int((py - y_min + hi) / pixel_size))

    for i in range(i_start, i_end + 1):
        dx  = (i * pixel_size + x_min) - px
        dx2 = dx * dx
        if dx2 >= hi2:
            continue
        for j in range(j_start, j_end + 1):
            dy    = (j * pixel_size + y_min) - py
            dist2 = dx2 + dy * dy
            if dist2 >= hi2:
                continue
            q = math.sqrt(dist2) * inv_h
            if q <= 0.5:
                w = 1.0 - 6.0 * q * q + 6.0 * q * q * q
            elif q < 1.0:
                tmp = 1.0 - q
                w = 2.0 * tmp * tmp * tmp
            else:
                continue
            cuda.atomic.add(grid, (i, j), mi * norm * w)


def render_gpu(
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    h: np.ndarray,
    resolution: int = 512,
    gridsize: float = 200.0,
    chunk_size: int = 5_000_000,
    sort_by_morton: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Render SPH surface density on the GPU, processing particles in tiles.

    Tiling prevents GPU command-queue timeouts (Windows TDR) and OOM
    errors by transferring and processing ``chunk_size`` particles at a
    time.  The accumulation buffer lives on-device for the full duration
    to avoid round-trips.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Particle positions (host arrays; float32 preferred).
    mass : np.ndarray, shape (N,)
        Particle masses (host arrays).
    h : np.ndarray, shape (N,)
        Smoothing lengths (host arrays).
    resolution : int, optional
        Grid resolution (pixels per side).  Default ``512``.
    gridsize : float, optional
        Total size of the rendered region in data units.  Default ``200.0``.
    chunk_size : int, optional
        Number of particles transferred to the GPU per tile.
        Default ``5_000_000``.
    sort_by_morton : bool, optional
        If ``True``, reorder particles into 2-D Morton Z-order before
        tiling.  Consecutive threads in the CUDA scatter kernel then
        target nearby grid cells, reducing atomic-add contention and
        improving L2 cache hit rates.  The sort runs on the CPU
        (NumPy, ~50-150 ms for 5M particles) and adds one permutation
        copy (~80 MB for 5M particles at float32).  Default ``False``.
    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    grid : np.ndarray, shape (resolution, resolution), dtype float32
        Surface density map (host array).
    bounds : tuple of float
        ``(x_min, x_max, y_min, y_max)``.

    Raises
    ------
    RuntimeError
        If CuPy is not installed (``CUPY_AVAILABLE`` is ``False``).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError(
            "CuPy is not installed.  Use render_cpu() instead."
        )

    half      = float(gridsize) / 2.0
    n_points  = len(x)
    x_min = y_min = -half
    pixel_size = float(gridsize) / resolution

    if sort_by_morton:
        if verbose:
            print("[SPH] Sorting particles by Morton Z-order (CPU)...")
        order = _argsort_morton_2d(x, y, gridsize)
        x, y, mass, h = x[order], y[order], mass[order], h[order]

    grid_gpu  = cp.zeros((resolution, resolution), dtype=cp.float32)
    num_tiles = math.ceil(n_points / chunk_size)

    if verbose:
        print(f"[SPH] GPU rendering {n_points:,} particles in {num_tiles} tile(s)...")

    threads = 256
    for start in tqdm(range(0, n_points, chunk_size), disable=not verbose):
        end = min(start + chunk_size, n_points)

        x_tile = cp.asarray(x[start:end],    dtype=cp.float32)
        y_tile = cp.asarray(y[start:end],    dtype=cp.float32)
        m_tile = cp.asarray(mass[start:end], dtype=cp.float32)
        h_tile = cp.asarray(h[start:end],    dtype=cp.float32)

        blocks = math.ceil(x_tile.shape[0] / threads)
        _sph_splat_gpu_kernel[blocks, threads](
            x_tile, y_tile, m_tile, h_tile,
            grid_gpu, x_min, y_min, pixel_size,
        )

        # Release tile buffers immediately to minimise fragmentation
        del x_tile, y_tile, m_tile, h_tile
        cp.get_default_memory_pool().free_all_blocks()

    result = grid_gpu.get()
    bounds = (-half, half, -half, half)
    return result, bounds


# ===========================================================================
# SECTION 5 -- Unified entry-point (auto-selects GPU / CPU via arch flag)
# ===========================================================================

def render_surface_density(
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    h: Optional[np.ndarray] = None,
    resolution: int = 512,
    gridsize: float = 200.0,
    chunk_size: int = 10_000_000,
    k_neighbors: int = 32,
    arch: Literal["auto", "gpu", "cpu"] = "auto",
    sort_by_morton: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    High-level SPH surface-density renderer with GPU/CPU dispatch.

    Computes smoothing lengths if not supplied, then renders the 2-D
    surface-density map.  Dispatch is controlled by ``arch``:

    * ``"auto"`` -- GPU if CuPy is available, CPU Numba otherwise.  If the
      GPU kernel raises at runtime (OOM, driver error) the CPU path is
      tried automatically.
    * ``"gpu"``  -- GPU only; raises ``RuntimeError`` if CuPy is absent.
    * ``"cpu"``  -- CPU Numba only; never touches the GPU.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Particle positions (projected 2-D coordinates in data units).
    mass : np.ndarray, shape (N,)
        Particle masses.
    h : np.ndarray, shape (N,), optional
        Pre-computed smoothing lengths.  When ``None``, they are computed
        from the 2-D projected positions ``(x, y)``, which is the correct
        choice for a 2-D kernel density estimate of surface density.  If
        the particles are actual SPH fluid particles with physical
        smoothing lengths (e.g., from a gas simulation), pass those
        ``h`` values here together with the matching projected 3-D kernel;
        note that a 3-D ``h`` is inconsistent with the 2-D cubic-spline
        kernel implemented here.
    resolution : int, optional
        Grid resolution (pixels per side).  Default ``512``.
    gridsize : float, optional
        Total size of the rendered region in data units.  The grid spans
        ``[-gridsize/2, gridsize/2]`` along both projected axes.
        Default ``200.0``.
    chunk_size : int, optional
        Particles transferred to the GPU per tile.  ``10_000_000`` is safe
        on an RTX 3070 (8 GB); push to ``30_000_000`` on a 24 GB card or
        ``50_000_000`` on an H200.  Default ``10_000_000``.
    k_neighbors : int, optional
        Neighbours used when computing smoothing lengths (ignored when
        ``h`` is supplied).  Default ``32``.
    arch : {"auto", "gpu", "cpu"}, optional
        Compute backend.  Default ``"auto"``.
    sort_by_morton : bool, optional
        If ``True``, sort particles into Morton Z-order before rendering
        (via :func:`_argsort_morton_2d`).  Passed through to
        ``render_gpu`` / ``render_cpu``.  Default ``False``.
    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    grid : np.ndarray, shape (resolution, resolution), dtype float32
        Surface density map (M_sun / kpc^2 in standard nbody_streams units).
    bounds : tuple of float
        ``(x_min, x_max, y_min, y_max)`` for ``imshow(extent=...)``.

    Raises
    ------
    RuntimeError
        When ``arch="gpu"`` but CuPy is not installed.
    ValueError
        When ``arch`` is not one of the accepted literals, or when input
        arrays have mismatched lengths.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x, y = rng.normal(0, 20, 100_000), rng.normal(0, 20, 100_000)
    >>> mass  = np.ones(100_000, dtype=np.float32)
    >>> grid, bounds = render_surface_density(x, y, mass,
    ...                                       resolution=512, gridsize=120.0)
    """
    if arch not in ("auto", "gpu", "cpu"):
        raise ValueError(f"arch must be 'auto', 'gpu', or 'cpu'; got {arch!r}")

    if arch == "gpu" and not CUPY_AVAILABLE:
        raise RuntimeError("arch='gpu' requested but CuPy is not installed.")

    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    mass = np.asarray(mass, dtype=np.float32).ravel()

    if not (x.shape == y.shape == mass.shape):
        raise ValueError(
            f"x, y, mass must have the same length; "
            f"got {x.shape}, {y.shape}, {mass.shape}"
        )

    pos = np.column_stack([x, y])
    if h is None:
        h = get_smoothing_lengths(
            pos, k_neighbors=k_neighbors, verbose=verbose
        )
    else:
        h = np.asarray(h, dtype=np.float32).ravel()
        if h.shape[0] != x.shape[0]:
            raise ValueError(
                f"h length {h.shape[0]} must match x length {x.shape[0]}"
            )

    # --- GPU path ---
    if arch in ("auto", "gpu") and CUPY_AVAILABLE:
        try:
            return render_gpu(
                x, y, mass, h,
                resolution=resolution, gridsize=gridsize,
                chunk_size=chunk_size, sort_by_morton=sort_by_morton,
                verbose=verbose,
            )
        except Exception as exc:
            if arch == "gpu":
                raise
            warnings.warn(
                f"[SPH] GPU render failed ({type(exc).__name__}: {exc}). "
                "Falling back to CPU render.",
                RuntimeWarning,
                stacklevel=2,
            )
            cp.get_default_memory_pool().free_all_blocks()

    # --- CPU path ---
    return render_cpu(
        x, y, mass, h, resolution=resolution, gridsize=gridsize,
        sort_by_morton=sort_by_morton, verbose=verbose,
    )
