# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-03-07

### Added

- **SPH surface-density renderer** -- new `nbody_streams/viz/sph_kernels.py` module.
  - `render_surface_density(x, y, mass, ...)` -- unified entry point with automatic GPU
    (Numba CUDA + CuPy KDTree) -> CPU (Numba `prange` + SciPy KDTree) fallback.
    Exposed at `nbody_streams.viz.render_surface_density`.
  - `get_smoothing_lengths(pos, k_neighbors, ...)` -- per-particle smoothing lengths
    via k-NN; GPU-accelerated with CuPy KDTree and transparent CPU fallback.
    Exposed at `nbody_streams.viz.get_smoothing_lengths`.
  - `render_cpu` / `render_gpu` -- low-level Numba-parallel / CUDA splatting kernels
    (direct use optional; `render_surface_density` is the recommended entry point).
  - 2-D cubic-spline SPH kernel (40 / (7*pi*h^2) normalisation) on both CPU and GPU.
  - `verbose=False` on all public SPH functions; GPU fallback events raise
    `RuntimeWarning` regardless of verbosity.
- **`examples/density_methods_comparison.ipynb`** -- notebook comparing `'histogram'`,
  `'gauss_smooth'`, and `'sph'` rendering on the included example dark-matter stream
  data (`nbody_streams/data/example_nbody_dm_stream.npz`).

### Changed

- **`plot_density` refactored** -- cosmological and Gizmo-style dependencies removed;
  API is now purely nbody_streams-native.
  - Removed parameters: `part`, `host_props`, `spec_ind`, `cosmo_box`.
  - `pos=(N,3)` and `mass=(N,)` are now explicit keyword arguments.
  - New `snap` parameter accepts a `ParticleReader` snapshot directly; positions and
    masses are extracted from `snap[spec]`.
  - `no_bins` renamed to `resolution` (pixels per axis).
  - New `gridsize` parameter (total size in data coordinates; grid spans
    `[-gridsize/2, gridsize/2]`) replaces the old `grid_len`.  Default ``200.0`` kpc.
  - `gauss_convol: bool` replaced by `method: str` with three choices:
    - `'sph'` *(new default)* -- physics-motivated SPH kernel splatting.
    - `'gauss_smooth'` -- 2-D histogram + Gaussian filter (`smooth_sigma` pixels).
    - `'histogram'` -- raw 2-D mass histogram divided by pixel area.
  - `arch`, `k_neighbors`, `chunk_size` moved to `**kwargs` (advanced options, still
    documented; `smooth_sigma` remains an explicit parameter).
  - `return_dens=True` now returns the method-specific density before log10 is applied.
  - Scale bar: `scale_size` is directly in kpc data units; cosmo correction removed.
- **`sph_kernels.py` API aligned** with `plot_density`:
  - `res` renamed to `resolution`; `grid_len` replaced by `gridsize` (total size in
    data coordinates; grid spans `[-gridsize/2, gridsize/2]`).  Default ``200.0``.
  - `k` (public) renamed to `k_neighbors` in `get_smoothing_lengths` and
    `render_surface_density`.
  - All public functions have `verbose: bool = False`.
  - GPU fallback and OOM events now raise `RuntimeWarning` (were silent prints).
  - All non-ASCII characters removed from source.
- `nbody_streams.viz` now exports `render_surface_density` and `get_smoothing_lengths`
  alongside the existing plot functions.
- Version bumped to **2.1.0**.

### Removed

- `plot_density`: `part`, `host_props`, `spec_ind`, `cosmo_box`, `gauss_convol`,
  `no_bins`, `grid_len` parameters (replaced by `gridsize`, `resolution`, `method`).

## [2.0.0] - 2026-02-28

### Added

- **GPU Barnes-Hut tree-code** — new `nbody_streams.tree_gpu` subpackage.
  - C++/CUDA shared library (`libtreeGPU.so`) implementing a GPU Barnes-Hut tree
    with monopole + quadrupole moments, per-particle softening (max convention),
    and auto-detected GPU architecture.  Build with `make -j$(nproc)` inside
    `nbody_streams/tree_gpu/`.
  - `tree_gravity_gpu(pos, mass, eps, G, theta, ...)` — one-shot force + potential
    computation; accepts scalar or per-particle softening.
  - `TreeGPU(N, eps, theta)` — pre-allocated tree handle for time-stepping loops
    (saves ~27 ms of GPU malloc/free overhead per step).
  - `cuda_alive()` — lightweight CUDA context health check via `cudaGetLastError()`;
    zero GPU overhead, no synchronisation.
  - `run_nbody_gpu_tree(phase_space, masses, ...)` — KDK leapfrog integrator using
    the GPU tree code; same call signature as `run_nbody_gpu`.
    - `_StepWatchdog` background thread: fires `KeyboardInterrupt` in the main
      thread if any integration step exceeds `step_timeout_s` seconds, protecting
      against deadlocked CUDA kernels.
    - Restart/snapshot I/O is fully compatible with the existing nbody_streams
      HDF5 format.
    - Supports `external_potential` (Agama) and multi-species via `species=` kwarg.
  - `nbody_streams/tree_gpu/tests/` — self-contained test suite (accuracy, API
    timing, comprehensive validation, cross-comparison with direct-sum).
- `run_simulation(..., architecture='gpu', method='tree')` now dispatches to
  `run_nbody_gpu_tree` (previously raised `NotImplementedError`).
- `examples/mw_stability.ipynb` — end-to-end Milky Way stability test (2M
  particles, 5 Gyr, GPU tree, multi-species IC from Agama).
- `examples/plummer_stability.ipynb` — Plummer sphere energy conservation test.
- `setup.cfg`: added `gpu` extra (`cupy-cuda12x`); `tree_gpu` package data
  (`*.cu`, `*.h`, `Makefile`, `libtreeGPU.so`).

### Changed

- `sim.py`: `architecture='gpu', method='tree'` route now works (dispatches to
  `run_nbody_gpu_tree`).  `ImportError` is raised if `libtreeGPU.so` is not built
  yet, with build instructions in the message.
- `nbody_streams/__init__.py`: tree_gpu symbols exposed at top level when the
  shared library is built (`tree_gravity_gpu`, `TreeGPU`, `cuda_alive`,
  `run_nbody_gpu_tree`, `_TREE_GPU_AVAILABLE`).
- README: GPU tree section (build, API, watchdog); updated package table and
  "Under the hood" implementation table.

## [1.3.0] - 2026-02-24

### Added

- **Multi-species simulation support** — arbitrary number of particle types (dark matter, stars, gas tracers, black holes, …) in a single run.
  - `Species` dataclass (`name`, `N`, `mass`, `softening`) with `Species.dark()` and `Species.star()` convenience constructors; scalar or per-particle mass/softening.
  - `run_simulation(phase_space, species, ..., architecture='cpu'|'gpu', method='direct'|'tree')` — unified high-level entry point; returns `dict[str, ndarray]` keyed by species name.
  - `PerformanceWarning` emitted automatically when particle counts exceed recommended thresholds (CPU direct >20k, GPU direct >500k, any >2M).
  - `nbody_streams.species` module exposing `Species`, `PerformanceWarning`, and internal helpers.
  - `nbody_streams.sim` module containing `run_simulation`.

- **Smart HDF5 snapshot storage** — uniform mass/softening stored as a scalar dataset; variable (per-particle) stored as a compressed `m_array`/`eps_array` dataset.

- **New HDF5 schema** for multi-species snapshots — `properties.attrs['n_species']` and `properties.attrs['species_names']`; each species in its own HDF5 group.

- **Enriched restart files** — species names, per-species N, combined mass and softening arrays stored alongside phase-space data; old restart files load cleanly (missing keys return `None`).

- **Physics test suite** (`tests/test_physics.py`, 24 tests):
  - Energy and momentum conservation for old API (`run_nbody_cpu`, devel-equivalent) and new API (`run_simulation`).
  - Multi-species energy and momentum conservation (2-species and 3-species).
  - Regression suite: devel-style trajectory vs feat-multi_spec trajectory agree to `rtol=1e-5`.
  - IO round-trip: final returned array matches last written snapshot.
  - GPU conservation tests (auto-skipped without CuPy).

- **`examples/run_simulation.ipynb`** — end-to-end notebook demonstrating single-species (Globular Cluster) and multi-species (Dwarf Galaxy: DM + Stars + Gas) workflows with visualisation.

### Changed

- **`ParticleReader` (backward-compatible rewrite)**:
  - Detects HDF5 format automatically: presence of `n_species` attribute → new multi-species schema; absence → legacy dark/star schema.
  - Populates `reader.species_list: list[Species]` for any format.
  - `read_snapshot()` returns `part.species: dict[str, dict]` as the primary API.
  - Legacy attributes (`part.dark`, `part.star`, `reader.num_dark`, `reader.mass_dark`, …) remain fully functional — no code changes required for existing workflows.
  - `extract_orbits` returns `orbits.species` dict plus `orbits.dark`/`orbits.star` aliases.

- **`make_plummer_sphere` overhaul**:
  - Default units changed to physical (kpc/km/s/Msun): `M_total=10_000` Msun, `a=0.01` kpc.
  - Added `G` parameter (defaults to `G_DEFAULT`) so the virial velocity scale is computed in physical units.
  - Rejection sampling now uses the correct theoretical envelope `h_max = 0.09375`; velocity direction sampling switched to the cosine form for isotropy.
  - Centre-of-mass and centre-of-momentum correction applied before returning.
  - `masses` returned as `float64` (was `float32`).
  - Improved docstring (references Aarseth, Henon & Wielen 1974).

- **`run_nbody_gpu` / `run_nbody_cpu`** — added optional `species: list[Species] | None = None` parameter (default `None` preserves full backward compatibility); snapshot and restart kwargs now built once from species context.

- **README** — multi-species overview, `Species` parameter table, `run_simulation` signature, single-species (Globular Cluster) and multi-species (Dwarf Galaxy) worked examples; package overview table updated.

### Fixed

- `_load_restart` now returns an 8-tuple; callers unpack via `[:4]` so both old (4-element) and new (8-element) restart files load without error.

## [1.2.0] - 2026-02-14

### Added
- **`nbody_streams.fast_sims` subpackage** — fast stream-generation methods as lightweight alternatives to full N-body integration (requires AGAMA).
  - `create_particle_spray_stream` — particle-spray method with Chen+2025 (default) or Fardal+2015 initial conditions.
  - `run_restricted_nbody` — restricted (collisionless) N-body with an evolving progenitor potential rebuilt from bound particles at each step.
  - `create_ic_particle_spray_chen2025` / `create_ic_particle_spray_fardal2015` — standalone IC generators for custom workflows.
- Support for **custom stripping times** (`time_stripping`) in particle spray, enabling episodic / pericenter-weighted particle release with automatic strict-monotonicity guard rails at floating-point precision.
- Support for **subhalo perturbers** on a self-consistent orbit in the host potential (NFW profile), available in both particle spray and restricted N-body methods.
- Optional **dynamical friction** on the progenitor orbit (Chandrasekhar formula with core-stalling suppression).
- `fit_dehnen_profile` and `fit_plummer_profile` utilities with input validation.
- `nbody_streams.coords` subpackage — coordinate transforms and stream-to-observable conversions.
- `nbody_streams.viz` subpackage — mollweide projections, surface density, stream evolution plots.

## [1.1.0] - 2026-02-08

### Added
- Float4 vectorized memory loads for float32 kernels (3x performance improvement)
- Comprehensive Newton's 3rd Law test suite (`tests/test_newtons_third_law.py`)
- Kahan summation variants with float4 support for all kernels (forces and potential)
- Branch-free kernel implementations using switch statements
- Detailed documentation on float32 precision limitations at small scales
- Performance benchmarking framework in `fields.py`
- Best practices guide for unit scaling with AGAMA Plummer spheres

### Changed
- **TILE_SIZE reduced from 256 to 128** for better GPU occupancy on modern GPUs
- Kernel selection now uses switch statements instead of if/else chains (eliminates branch divergence)
- Self-interaction masking now branch-free (multiplication by 0/1 instead of continue)
- Optimized mathematical operations (use rsqrt where possible, reduced divisions)
- Updated all potential kernels to match force kernel optimizations

### Fixed
- **Critical bug**: Strided array views in `skip_validation` path now properly converted to contiguous arrays
- Float64 force calculation accuracy restored (was affected by CuPy disk cache)
- Memory layout issues when converting NumPy arrays to CuPy
- Jupyter kernel caching issues documented with workarounds

### Performance
- Float32 force computation: **8.0ms → 2.3ms** (3.5x speedup, N=10,240 particles, RTX 3080)
- Float32_kahan: **8.1ms → 2.4ms** (3.4x speedup)
- Memory bandwidth utilization: **99% of theoretical peak** (760 GB/s)
- Throughput: **45.7 Ginteractions/s** for float32 (was 12.9)
- Energy conservation: **<0.001% drift** over 20 dynamical times (all precision modes)

### Documentation
- Added comprehensive analysis of float32 precision vs scale
- Documented three workflow options (scaling, as-is, float64)
- Added comparison: direct N-body vs tree methods for energy conservation
- Included production-ready code examples with AGAMA integration

## [1.0.0] - 2025-XX-XX

### Added
- Initial GPU N-body force and potential computation with CUDA/CuPy
- CPU fallback implementation using Numba (multithreaded)
- Support for multiple softening kernels:
  - Newtonian (regularized 1/r³)
  - Plummer
  - Dehnen k=1 and k=2 (C2/C4 corrections)
  - Spline (Monaghan 1992, compact support)
- Three precision modes: float32, float32_kahan, float64
- Leapfrog integrator for symplectic time evolution
- AGAMA external potential integration
- Snapshot and restart file management (HDF5)
- Particle reader utilities for analysis
- Basic test suite

### Features
- Direct O(N²) computation with tiled shared memory optimization
- Validation and error handling for all input arrays
- Automatic GPU detection and fallback to CPU
- Configurable time-stepping and snapshot intervals
- Support for both CPU and GPU arrays (NumPy/CuPy)

---

## Version History

- **2.0.0** (2026-02-28): GPU Barnes-Hut tree-code (`nbody_streams.tree_gpu`), `run_nbody_gpu_tree`, watchdog, `run_simulation` gpu+tree path
- **1.3.0** (2026-02-24): Multi-species simulation support, `run_simulation` API, ParticleReader rewrite, `make_plummer_sphere` overhaul
- **1.2.0** (2026-02-14): Fast stream-generation methods (particle spray, restricted N-body)
- **1.1.0** (2026-02-08): Float4 vectorization and major performance improvements
- **1.0.0** (2025-XX-XX): Initial release

---

## Upgrade Notes

### From 1.3.0 to 2.0.0

**No breaking changes — fully backwards compatible!**

**What you get:**
- `architecture='gpu', method='tree'` in `run_simulation` now works (previously raised `NotImplementedError`).
- Build `libtreeGPU.so` once to unlock the GPU tree backend:
  `cd nbody_streams/tree_gpu && make -j$(nproc)`
- `tree_gravity_gpu`, `TreeGPU`, `cuda_alive`, and `run_nbody_gpu_tree` available at the top level when the library is built.
- `_TREE_GPU_AVAILABLE` flag in `nbody_streams` indicates whether the library is loaded.
- Old `run_nbody_gpu` / `run_nbody_cpu` / `run_simulation` calls work unchanged.

### From 1.2.0 to 1.3.0

**No breaking changes** — fully backwards compatible!

**What you get:**
- Call `run_simulation(xv, [Species.dark(N, mass, softening)], ...)` instead of `run_nbody_gpu/cpu` for the new cleaner API.
- Old `run_nbody_gpu` / `run_nbody_cpu` calls work unchanged — nothing needs updating.
- Old HDF5 snapshot files and restart files are still read correctly.
- Multi-species systems (dark matter + stars + gas + …) are now first-class citizens.
- `make_plummer_sphere` now returns physically-scaled ICs by default (kpc/km/s/Msun); pass `M_total`, `a`, and `G` explicitly if you were relying on the old dimensionless defaults.

### From 1.1.0 to 1.2.0

**No breaking changes** — fully backwards compatible!

**New dependency:** AGAMA is required for `fast_sims` (optional extra: `pip install nbody_streams[agama]`).

**What you get:**
- Particle spray streams in seconds instead of hours
- Restricted N-body with automatic bound-mass tracking
- Subhalo perturbers on self-consistent orbits
- Custom episodic stripping with monotonicity guard rails

### From 1.0.0 to 1.1.0

**No breaking changes** - fully backwards compatible!

**Action items:**
1. Clear CuPy kernel cache: `rm -rf ~/.cupy/kernel_cache/*`
2. Reinstall: `pip install -e .` (if using editable install)
3. Restart Jupyter kernels if using notebooks

**What you get:**
- Automatic 3x speedup on float32 (no code changes needed)
- Better energy conservation at all scales
- More accurate Newton's 3rd Law

**Recommended for new projects:**
- Use unit scaling for AGAMA Plummer spheres (see documentation)
- Consider float32 instead of float32_kahan (marginal difference now)

---

## Future Roadmap

### Planned for 1.4.0
- [ ] GPU+Tree solver (`feat-cuda_kernel` + GPU tree code)
- [ ] Adaptive time-stepping
- [ ] Additional integrators (RK4, Hermite)
- [ ] GPU-accelerated analysis tools
