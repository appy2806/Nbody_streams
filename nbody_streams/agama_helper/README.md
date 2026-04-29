# AGAMA_GPU

GPU-accelerated drop-in for [Agama](https://github.com/GalacticDynamics-Oxford/Agama) potential evaluation. Targets N >= 50k particles where GPU throughput gives a 5–10× speedup over Agama CPU.

## Overview

Three files comprise the package:

| File | Purpose |
|------|---------|
| `_potential.py`           | Python wrapper, quintic spline builder, unified factory `PotentialGPU` |
| `_*.cu`                   | CUDA for cylspl and multipole kernels: potential, force, density, hessian |
| `_analytic_potentials.py` | Fused CuPy `ElementwiseKernel` analytic types |

The CUDA module is compiled once at first import via `cp.RawModule` + nvcc.

---

## Supported potential types

### Multipole BFE — `MultipolePotentialGPU`

Quintic C2 splines with Agama log-scaling (replicates `MultipoleInterp1d` from Agama's `potential_multipole.cpp`). Requires coefficient files with dPhi/dr data.

- lmax up to 32. PREFACT[33] ,  COEF[33]  in kernel; Python raises if exceeded.
- Inner power-law extrapolation + outer Keplerian extrapolation (all four kernels)
- Non-uniform radial grid: auto-resampled to log-uniform via `CubicHermiteSpline` when max spacing error > 0.1%

### CylSpl BFE - `CylSplinePotentialGPU`

Agama's CylSpline::evalCyl with 2D bicubic Hermite splines. Requires coefficient files with Phi(R, Z) data.

### Analytic — `_analytic_potentials.py`

| Class | Agama type |
|-------|-----------|
| `NFWPotentialGPU` | NFW |
| `PlummerPotentialGPU` | Plummer |
| `HernquistPotentialGPU` | Hernquist |
| `IsochronePotentialGPU` | Isochrone |
| `MiyamotoNagaiPotentialGPU` | MiyamotoNagai |
| `LogHaloPotentialGPU` | Logarithmic (triaxial) |
| `DehnenSphericalPotentialGPU` | Dehnen spherical (gamma in [0,2)) |
| `DiskAnsatzPotentialGPU` | DiskAnsatz |
| `UniformAccelerationGPU` | UniformAcceleration |

Via Agama CPU export (in _potential.py ):
`Disk`     →  _build_disk_gpu : DiskAnsatz (from input kwargs) + Multipole (from Agama export) →  CompositePotentialGPU
`Spheroid` →  _build_spheroid_gpu : Agama export →  MultipolePotentialGPU
`King`     →  _build_king_gpu : Agama export →  MultipolePotentialGPU
`Dehnen`   →  triaxial or gamma=2 → Spheroid(alpha=1, beta=4) export

### Modifiers

- `ShiftedPotentialGPU`: static offset `(3,)`, cubic-spline center trajectory `(T,4)`, or Hermite-spline `(T,7)`. Linear extrapolation outside time range.
- `ScaledPotentialGPU`: static float, or time-dependent `(T,2)` / `(T,3)` tables.

### Composite types

- `CompositePotentialGPU`: sum of arbitrary GPU components, built automatically when multiple components are passed.
- `EvolvingPotentialGPU`: linear lerp between BFE snapshots at fixed timestamps (matches Agama `Evolving` with `interpLinear=True`).

---

##  Unified factory  — `PotentialGPU`

Mirrors `agama.Potential` API:

`PotentialGPU(type='NFW', ...)` — analytic dispatch via `_ANALYTIC_TYPE_MAP`  
`PotentialGPU('file.ini')` or `PotentialGPU(file='file.ini')` — INI parsing via `_load_potential_ini`  
`PotentialGPU('snap.coef_mul')` — coef file → `MultipolePotentialGPU`  
`PotentialGPU(dict1, dict2)` — Agama-style component dicts  
`PotentialGPU(pot1, pot2)` — variadic GPU objects → composite  
`center=`, `scale=`, `ampl=` → wraps in Shifted/ScaledPotentialGPU  
`+` operator via `_GPUPotBase` mixin

`_load_potential_ini` handles:
- `[Potential]`, `[Potential halo]`, `[Potential disk1]` — arbitrary section headers (case-insensitive)
- `type=Multipole` (inline or `file=`), `type=CylSpline` (inline or `file=`)
- `type=Evolving` (Timestamps block) → `EvolvingPotentialGPU`
- `type=DiskAnsatz` silently skipped (no stored params in Agama export)
- All param keys normalized to canonical camelCase (case-insensitive: `ScaleRadius` == `scaleRadius`)

## `load_agama_potential` / `load_agama_evolving_potential` with `gpu=True`

The existing loader functions in `_load.py` accept a `gpu=False` flag:

```python
from nbody_streams.agama_helper import load_agama_potential, load_agama_evolving_potential

# returns agama.Potential (default)
pot_cpu = load_agama_potential("snap.coef_mult")

# returns PotentialGPU / CylSplinePotentialGPU / etc.
pot_gpu = load_agama_potential("snap.coef_mult", gpu=True)
pot_gpu = load_agama_potential("snap.coef_cylsp", gpu=True)
pot_gpu = load_agama_potential(mc_filtered, gpu=True)            # from MultipoleCoefs
pot_gpu = load_agama_potential("archive.h5", group_name="snap_090", gpu=True)

# time-evolving on GPU
ev_gpu = load_agama_evolving_potential("archive.h5", times, gpu=True)
```

All filtering (`keep_lm_mult`, `keep_m_cylspl`) is applied before building the GPU object.
`center=` is forwarded to `ShiftedPotentialGPU` as a `(3,)` or `(T,4)` / `(T,7)` array.

---

## API

All methods accept CuPy or NumPy arrays, shape `(N,3)` or `(3,)` (scalar squeezed):

```python
pot.potential(xyz, t=0.)     # -> (N,)    [km/s]^2
pot.force(xyz, t=0.)         # -> (N,3)   [km/s]^2/kpc  (= -grad Phi)
pot.density(xyz, t=0.)       # -> (N,)    [Msol/kpc^3]
pot.forceDeriv(xyz, t=0.)    # -> (force (N,3), deriv (N,6))
pot.evalDeriv(xyz, t=0.)     # -> (phi (N,), force (N,3), deriv (N,6))
```

`forceDeriv` returns `deriv = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]`, matching `agama.Potential.forceDeriv` exactly.

Units follow Agama convention: mass = Msol, length = kpc, velocity = km/s.

---

## Accuracy

| Component | phi rel err | force rel err |
|-----------|-------------|---------------|
| Multipole l=0 (monopole) | ~1e-12 | ~1e-12 |
| Multipole l>0 harmonics | ~1e-7 | ~1e-5 |
| Disk composite (DiskAnsatz + Multipole) | ~5e-5 | ~3e-3 |
| Analytic (NFW, Hernquist, etc.) | ~1e-12 | ~1e-12 |

l>0 errors are a numerical floor from log-scaling derivative cancellation — both GPU and Agama CPU hit the same floor. BFE fitting error for N-body data is typically >1%, so this is not a practical issue.

---

## Requirements

- CUDA GPU (tested on NVIDIA L40)
- `cupy >= 10.0` (matching CUDA version)
- `nvcc` accessible on PATH
- `scipy` (quintic spline construction; falls back gracefully if missing)
- `agama` (for `Disk`/`Spheroid`/`King`/`Dehnen` types that use CPU export to build BFE)

---

## File layout

```
agama_helper/
  _potential.py                     <- main wrapper: PotentialGPU factory + GPU classes
  _*.cu                             <- CUDA kernels for multipole and cylspl (potential, force, density, hessian)
  _analytic_potentials.py           <- analytic GPU potentials
  _load.py                          <- load_agama_potential / load_agama_evolving_potential (cpu + gpu= flag)
  tests/
    test_phase1_multipole.py        <- MultipolePotentialGPU correctness + benchmarks vs Agama CPU
    test_phase2_analytic.py         <- analytic GPU potential tests
    test_phase3_cylspline.py        <- CylSplinePotentialGPU correctness + benchmarks
    test_zero_pruning.py            <- zero-coefficient pruning correctness + speedup
  tech_err.md                       <- architecture decisions and precision notes
```

---

## Known gotchas

- **`Disk` type requires Agama**: `PotentialGPU(type='Disk', ...)` calls `agama.Potential` internally to export a Multipole coefficient file, then wraps it as `DiskAnsatz + MultipolePotentialGPU`.
- **`from_agama()` raises for pure analytic types**: Agama does not export NFW/Plummer/etc. parameters programmatically; these must be constructed directly by keyword.
- **EvolvingPotential interpolation**: GPU `interpolate=True` is linear lerp. Agama default (`interpLinear=False`) is nearest-neighbor. The INI parser maps `interpLinear=True` → GPU linear lerp.
- **lmax limit**: Kernel supports lmax <= 32. Python raises `ValueError` if exceeded.
