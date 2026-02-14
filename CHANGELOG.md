# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **1.2.0** (2026-02-14): Fast stream-generation methods (particle spray, restricted N-body)
- **1.1.0** (2026-02-08): Float4 vectorization and major performance improvements
- **1.0.0** (2025-XX-XX): Initial release

---

## Upgrade Notes

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

### Planned for 1.3.0
- [ ] Adaptive time-stepping
- [ ] Additional integrators (RK4, Hermite)
- [ ] Parallel snapshot I/O
- [ ] GPU-accelerated analysis tools
