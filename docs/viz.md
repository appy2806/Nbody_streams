# `nbody_streams.viz` — Visualization

Projected density maps, Mollweide sky projections, stream diagnostic plots,
and an SPH surface-density renderer.

```python
from nbody_streams import viz
```

Optional dependencies:

- **CuPy** — GPU-accelerated SPH splatting and KDTree (`render_surface_density`, `get_smoothing_lengths`).
- **Numba** — CPU-parallel SPH splatting (always available as fallback).
- **healpy** — required for `plot_mollweide`.
- **agama** — required for `plot_stream_sky` (observed sky coordinates).
- **cmasher** — optional; used for default colormaps in `plot_density`.

---

## `plot_density`

```python
plot_density(
    pos=None,
    mass=None,
    snap=None,
    spec="dark",
    gridsize=200.0,
    resolution=512,
    xval="x",
    yval="z",
    density_kind="surface",
    method="sph",
    slice_width=0.0,
    slice_axis=None,
    return_dens=False,
    ax=None,
    colorbar_ax=None,
    scale_size=0,
    cmap=None,
    vmin=None,
    vmax=None,
    smooth_sigma=1.0,
    **kwargs,
)
```

Generate a projected density image using `imshow`.

Accepts particle arrays directly, or a `ParticleReader` snapshot from which
positions and masses are extracted by species name.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | ndarray `(N, 3)` or None | Particle positions in kpc. Required when `snap` is None. |
| `mass` | ndarray `(N,)` or None | Particle masses in M_sun. Defaults to uniform unit mass. |
| `snap` | ParticleReader snapshot or None | Object from `ParticleReader.read_snapshot()`. When provided, `pos` and `mass` are extracted from `snap[spec]`. |
| `spec` | str | Species key — `'dark'`, `'star'`, `'gas'`, etc. Used for default colormap and snap extraction. |
| `gridsize` | float | Total rendered region in kpc. Grid spans `[-gridsize/2, gridsize/2]`. Default 200.0. |
| `resolution` | int | Grid resolution (pixels per axis). Default 512. |
| `xval`, `yval` | str | Projection axes: `'x'`, `'y'`, or `'z'`. |
| `density_kind` | `'surface'` or `'volume'` | `'surface'` returns M/kpc^2; `'volume'` divides by 2*`slice_width` to give M/kpc^3. |
| `method` | `'sph'`, `'histogram'`, or `'gauss_smooth'` | Density estimation method. `'sph'` (default): SPH kernel splatting via `render_surface_density`. `'histogram'`: raw 2-D mass histogram. `'gauss_smooth'`: histogram + Gaussian filter. |
| `slice_width` | float | Retain only particles within +/- `slice_width` kpc along `slice_axis`. Default 0 (no slicing). |
| `slice_axis` | str or None | Axis along which slicing is applied. Inferred as the remaining axis when None. |
| `return_dens` | bool | If True, return `(im_obj, density_array)` where `density_array` is the raw (pre-log10) density map. |
| `ax` | Axes or None | Existing axes. A new 3x3 in figure is created when None. |
| `colorbar_ax` | Axes, True, or None | Axes for colorbar, or True for auto-inset horizontal bar. |
| `scale_size` | float | If > 0, draw a physical scale bar of this length in kpc. |
| `cmap` | colormap or str or None | Defaults to cmasher palettes when available, else `'cubehelix'`. |
| `vmin`, `vmax` | float or None | Colour limits in log10 density units. |
| `smooth_sigma` | float | Gaussian smoothing width in pixels. Only used when `method='gauss_smooth'`. Default 1.0. |
| `**kwargs` | | SPH options: `arch` (`'auto'`/`'gpu'`/`'cpu'`), `k_neighbors` (int, default 32), `chunk_size` (int, default 10_000_000). |

**Returns**

`None`, or `(AxesImage, ndarray)` when `return_dens=True`.

**Example**

```python
from nbody_streams import viz
import numpy as np

pos  = np.random.randn(100_000, 3) * 20
mass = np.ones(100_000)

# SPH density map
viz.plot_density(pos=pos, mass=mass, gridsize=100.0)

# Histogram method, return raw density
im, dens = viz.plot_density(
    pos=pos, mass=mass,
    method="histogram",
    gridsize=100.0,
    return_dens=True,
)

# From a ParticleReader snapshot
from nbody_streams.nbody_io import ParticleReader
reader = ParticleReader("output/snapshot.h5")
snap = reader.read_snapshot(50)
viz.plot_density(snap=snap, spec="dark", gridsize=150.0)
```

---

## `render_surface_density`

```python
render_surface_density(
    x,
    y,
    mass,
    h=None,
    resolution=512,
    gridsize=200.0,
    chunk_size=10_000_000,
    k_neighbors=32,
    arch="auto",
    sort_by_morton=False,
    verbose=False,
)
```

High-level SPH surface-density renderer with automatic GPU/CPU dispatch.

Computes SPH smoothing lengths from the projected 2-D positions if not
supplied, then renders the surface-density map using the 2-D cubic-spline
kernel.

**Dispatch logic**

- `arch="auto"` (default): GPU if CuPy is available; CPU Numba otherwise.
  Falls back to CPU automatically if the GPU kernel raises at runtime.
- `arch="gpu"`: GPU only; raises `RuntimeError` if CuPy is absent.
- `arch="cpu"`: CPU Numba only; never touches the GPU.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `x`, `y` | ndarray `(N,)` | Projected 2-D particle positions in data units. |
| `mass` | ndarray `(N,)` | Particle masses. |
| `h` | ndarray `(N,)` or None | Pre-computed smoothing lengths. When None, computed from 2-D projected positions using `get_smoothing_lengths`. |
| `resolution` | int | Grid resolution (pixels per side). Default 512. |
| `gridsize` | float | Total rendered region in data units. Grid spans `[-gridsize/2, gridsize/2]`. Default 200.0. |
| `chunk_size` | int | Particles transferred to GPU per tile. Default 10_000_000. |
| `k_neighbors` | int | Neighbours used when computing smoothing lengths (ignored when `h` is supplied). Default 32. |
| `arch` | `"auto"`, `"gpu"`, or `"cpu"` | Compute backend. Default `"auto"`. |
| `sort_by_morton` | bool | Sort particles into 2-D Morton Z-order before rendering (see note below). Default False. |
| `verbose` | bool | Print progress messages. Default False. |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `grid` | ndarray `(resolution, resolution)`, float32 | Surface density map (M_sun / kpc^2 in standard nbody_streams units). |
| `bounds` | tuple of float | `(x_min, x_max, y_min, y_max)` for `imshow(extent=...)`. |

**Raises**

- `RuntimeError` when `arch="gpu"` but CuPy is not installed.
- `ValueError` when `arch` is not one of the accepted literals, or input arrays have mismatched lengths.

**Example**

```python
import numpy as np
from nbody_streams.viz import render_surface_density

rng = np.random.default_rng(0)
x, y = rng.normal(0, 20, 100_000), rng.normal(0, 20, 100_000)
mass = np.ones(100_000, dtype=np.float32)

grid, bounds = render_surface_density(x, y, mass, resolution=512, gridsize=120.0)

import matplotlib.pyplot as plt
plt.imshow(grid.T, origin="lower", extent=bounds, norm="log")
plt.colorbar(label="M_sun / kpc^2")
```

---

## `get_smoothing_lengths`

```python
get_smoothing_lengths(
    pos,
    k_neighbors=32,
    safety_factor=0.6,
    gpu_vram_threshold_gb=10.0,
    verbose=False,
)
```

Compute SPH smoothing lengths as the distance to the `k_neighbors`-th
nearest neighbour, with automatic GPU -> CPU fallback.

The GPU path (CuPy KDTree) is attempted first when sufficient VRAM is
available.  The safety factor and VRAM threshold are conservative to
handle consumer cards (RTX 3070, 8 GB).  On datacenter cards (H200, 80 GB)
the GPU path handles 70-100 M particles comfortably.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | ndarray `(N, D)` | Particle positions. `D` is typically 2 (projected) or 3. |
| `k_neighbors` | int | Number of neighbours (including the particle itself). Default 32. |
| `safety_factor` | float | Fraction of post-build free VRAM used for query result buffers. Default 0.6. |
| `gpu_vram_threshold_gb` | float | Minimum free VRAM in GiB required to attempt the GPU path. Default 10.0. |
| `verbose` | bool | Print progress messages. Default False. |

**Returns**

`ndarray, shape (N,), dtype float32` — smoothing lengths. Always a plain NumPy array.

**Raises**

`ValueError` if `pos` is not a 2-D array.

**Example**

```python
import numpy as np
from nbody_streams.viz import get_smoothing_lengths

pos_2d = np.random.randn(50_000, 2).astype(np.float32)
h = get_smoothing_lengths(pos_2d, k_neighbors=32)
```

---

## `render_cpu` and `render_gpu`

Lower-level entry points for the SPH splatting kernels.  `render_cpu` uses
Numba `prange` with a per-thread array reduction (no race conditions).
`render_gpu` uses a CUDA scatter kernel (one thread per particle) with
atomic float32 adds.

Both share the same signature:

```python
render_cpu(x, y, mass, h, resolution=512, gridsize=200.0,
           sort_by_morton=False, verbose=False)

render_gpu(x, y, mass, h, resolution=512, gridsize=200.0,
           chunk_size=5_000_000, sort_by_morton=False, verbose=False)
```

**Returns** `(grid, bounds)` — same as `render_surface_density`.

`render_cpu` incurs Numba JIT compilation overhead (~1-5 s) on first call.
Subsequent calls use the on-disk cache.

---

## Morton Z-curve sort note

When `sort_by_morton=True` is passed to `render_surface_density`,
`render_cpu`, or `render_gpu`, particles are reordered into 2-D Morton
Z-order (via `_argsort_morton_2d`) before splatting.  This groups spatially
adjacent particles contiguously in memory, reducing atomic-add contention on
the GPU and improving L1/L2 cache hit rates on both backends.

The sort runs in pure NumPy on the CPU and takes approximately 50-150 ms for
N=5M particles.  The permutation copies add ~80 MB of temporary host memory.

**Important**: Morton sorting changes float32 rounding in Numba `fastmath`
mode at the 1-5% level (SIMD reordering of operations).  Grids rendered with
and without `sort_by_morton=True` are **not** numerically identical.

---

## `plot_mollweide`

```python
plot_mollweide(
    pos,
    weights=None,
    initial_nside=60,
    normalize=False,
    log_scale=True,
    filter_radius=(0, 0),
    return_map=False,
    smooth_fwhm_deg=None,
    verbose=False,
    add_traj=None,
    add_end_pt=False,
    density_threshold=1e5,
    cmap="bone",
    **kwargs,
)
```

Mollweide projection of a 3-D particle field using Healpix.

Requires **healpy**.  For large datasets (> 1M particles), **vaex** is
recommended for fast pixel aggregation; otherwise pandas with multiprocessing
is used.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | ndarray `(N, 3)` | Cartesian positions. |
| `weights` | ndarray or None | Per-particle weights (e.g. masses). Default: uniform. |
| `initial_nside` | int | Starting Healpix resolution. Auto-upscaled for large N. Default 60. |
| `normalize` | bool | Compute fractional variation relative to median density. |
| `log_scale` | bool | Convert density to log10 before plotting. Default True. |
| `filter_radius` | tuple `(float, float)` | Radial filter: `(rmin, rmax)` to keep only particles in that range. |
| `return_map` | bool | Return the smoothed sky map array. |
| `smooth_fwhm_deg` | float or None | Smoothing FWHM in degrees. Auto-set from nside when None. |
| `density_threshold` | float | Particle count that triggers nside upscaling. Default 1e5. |
| `cmap` | str | Colormap name. Default `'bone'`. |

**Returns**

`ndarray` or None — smoothed sky map when `return_map=True`.

---

## `plot_stream_sky`

```python
plot_stream_sky(
    xv_stream,
    color="ro",
    ax=None,
    xv_prog=None,
    alpha_lim=(None, None),
    delta_lim=(None, None),
    phi1_lim=(None, None),
    phi2_lim=(None, None),
    ms=0.5,
    mew=0,
)
```

2x3 diagnostic panel: (RA, Dec), (RA, v_los), (phi1, phi2),
(X, Y), (X, Z), (Y, Z).

Requires **agama** for the galactocentric-to-ICRS transformation.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `xv_stream` | ndarray `(N, 6)` | Stream particles in galactocentric coordinates. |
| `color` | str | Matplotlib format string (e.g. `'ro'`, `'b.'`). |
| `ax` | ndarray of Axes `(2, 3)` or None | Existing axes. |
| `xv_prog` | ndarray `(6,)` or None | Progenitor phase-space vector for observed-frame computation. |
| `alpha_lim`, `delta_lim` | tuple | RA / Dec axis limits. |
| `phi1_lim`, `phi2_lim` | tuple | Stream coordinate axis limits. |
| `ms`, `mew` | float | Marker size / edge width. |

**Returns**

`(fig, ax)` where `fig` is Figure or None and `ax` is ndarray of Axes `(2, 3)`.

---

## `plot_stream_evolution`

```python
plot_stream_evolution(
    prog_xv,
    times=None,
    part_xv=None,
    bound_mass=None,
    time_step=-1,
    x_axis=0,
    y_axis=2,
    LMC_traj=None,
    three_d_plot=False,
    interactive=False,
    dpi=200,
    figsize=(12, 3),
)
```

Three-panel evolution plot: galactocentric distance, bound fraction (or
3-D trajectory), and projected particle positions at a chosen snapshot.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `prog_xv` | ndarray `(T, 6)` or dict | Progenitor trajectory. If a dict, treated as a `fast_sims` result dict with keys `'prog_xv'`, `'times'`, `'part_xv'`, and optionally `'bound_mass'`. |
| `times` | ndarray `(T,)` or None | Time array (required when `prog_xv` is an array). |
| `part_xv` | ndarray or None | Particle positions. Shape `(N, T, 6)` for multi-snapshot or `(N, 3)` / `(N, 6)` for a single snapshot. |
| `bound_mass` | ndarray `(T,)` or None | Bound mass history for the middle panel. |
| `time_step` | int | Snapshot index for the right panel. Default -1 (last). |
| `x_axis`, `y_axis` | int | Coordinate indices for the right panel (0=X, 1=Y, 2=Z). |
| `LMC_traj` | ndarray `(M, 4)` or None | `[t, x, y, z]` LMC trajectory to overplot. |
| `three_d_plot` | bool | Use a 3-D middle panel instead of bound-fraction. |
| `interactive` | bool | Enable IPython widget backend (requires `three_d_plot=True`). |

**Returns**

`(fig, ax)` — Figure and list of 3 Axes.

**Example**

```python
from nbody_streams import viz, fast_sims

result = fast_sims.create_particle_spray_stream(
    pot_host, initmass=5e8, sat_cen_present=[20,0,5,-30,200,10],
    scaleradius=0.3, time_total=5.0, save_rate=50,
)

fig, axes = viz.plot_stream_evolution(result)
```
