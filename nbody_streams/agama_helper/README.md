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
| `UniformAccelerationGPU` | UniformAcceleration (constant `ax/ay/az`, or time-dependent `file=`) |

`UniformAccelerationGPU` — `Phi(x,t) = -a(t)·x`; force `= a(t)`, Hessian and density
identically zero.  `file=` accepts a `(T,4)` `[t, ax, ay, az]` table (regularized
**natural** cubic spline, matching `agama.Spline(..., reg=True)` — *not* SciPy's
default not-a-knot) or a `(T,7)` `[t, a, da/dt]` table (cubic Hermite), as an array
or a path.  Linear extrapolation outside the range, as in Agama.  Interpolation is
a CPU-side O(log T) lookup per call; kernels see three floats.
`from_agama()` is not supported — Agama does not export the table.

Via Agama CPU export (in _potential.py ):
`Disk`     →  _build_disk_gpu : DiskAnsatz (from input kwargs) + Multipole (from Agama export) →  CompositePotentialGPU
`Spheroid` →  _build_spheroid_gpu : Agama export →  MultipolePotentialGPU
`King`     →  _build_king_gpu : Agama export →  MultipolePotentialGPU
`Dehnen`   →  triaxial or gamma=2 → Spheroid(alpha=1, beta=4) export

### Modifiers

- `ShiftedPotentialGPU`: static offset `(3,)`, center trajectory `(T,4)`, or Hermite-spline `(T,7)`. Linear extrapolation outside time range.
- `ScaledPotentialGPU`: static float, or time-dependent `(T,2)` / `(T,3)` tables.

All three time-dependent inputs — `center=`, `scale=` and the `UniformAcceleration`
table — share one interpolator (`_AgamaTimeSpline` in `_potential.py`), because Agama
reads all three through the same `readTimeDependentArray`: a **regularized natural**
cubic spline (values only) or a **Hermite** spline (values + derivatives), linearly
extrapolated beyond the endpoints.  Note `scale=` in Agama always carries two values
`[t, ampl, scale]`; the `(T,2)` `[t, scale]` form here is a GPU-side convenience that
takes `ampl` from the separate keyword.

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

## Coefficient objects — one snapshot or a whole time series

`MultipoleCoefs` and `CylSplineCoefs` hold either a single snapshot or a time
series, in the *same* class. Time is always the **last** array axis and is
always optional. There is no separate series reader and no separate series class.

| field | no time axis | with time axis |
|---|---|---|
| `MultipoleCoefs.phi` | `(nR, n_lm)` | `(nR, n_lm, nt)` |
| `MultipoleCoefs.dphi_dr` | `(nR, n_lm)` or `None` | `(nR, n_lm, nt)` or `None` |
| `CylSplineCoefs.phi[m]` | `(nR, nz)` | `(nR, nz, nt)` |
| `CylSplineCoefs.dphi_dR[m]`, `.dphi_dz[m]` | `(nR, nz)` or `None` | `(nR, nz, nt)` or `None` |
| `times` | `None` | `(nt,)` |

### Reading

```python
import numpy as np
from nbody_streams import agama_helper as ah

# single snapshot -- unchanged, times is None
mc = ah.read_coefs("potential/600.dark.none_8.coef_mul_DR")

# whole archive, numerically sorted; times from the .h5 root "times" dataset
ser = ah.read_coefs("MW_mult.h5", group_name="all")

# specific groups, in the given order
ser = ah.read_coefs("MW_mult.h5", group_name=["snap_090", "snap_095"], times=[6.0, 6.5])

# an Agama Evolving .ini (times come from the file), or a plain list of sources
ser = ah.read_coefs("potential/MW_mult.ini")
ser = ah.read_coefs(sorted(glob("potential/*.coef_mult")), times=t_gyr)

# or stack snapshots you already have
ser = ah.stack_coefs([ah.read_coefs(p) for p in paths], times=t_gyr)
```

`times` resolves in the order: explicit argument, then the `.h5` root `times`
dataset, then the `.ini` timestamps. If a time axis is requested and none of
those yield times, this **raises** — an index-based axis is never invented.

### Manipulating

```python
ser.has_time_axis, ser.n_times      # True, 11
ser.snapshot(3)                     # time-less view at index 3; same as ser[3]
ser.column(2, 0)                    # Multipole: index of (l=2, m=0) in lm_labels
ser.radial_power(2).shape           # (nR, nt)   -- (nR,) without a time axis
ser.total_power(2).shape            # (nt,)      -- a float without a time axis
ser.zeroed([0, 2])                  # works with or without a time axis

# Direct field assignment is legal -- do whatever surgery you like.
ser.phi[:, ser.column(2, 2), :] *= 1.5
ser.validate()                      # names the offending field and both shapes
```

Nothing is ever interpolated or resampled here. Compute your own time
resampling and attach it with `with_times` — the number of times may change
freely:

```python
new_phi = my_spline(ser.phi, ser.times, t_new)     # your code, any nt
ser2 = ser.with_times(t_new, phi=new_phi, dphi_dr=new_dphi)
```

Grids and labels are never reconciled: stacking snapshots whose `R_grid`,
`z_grid`, `lm_labels`, `m_values` or header `metadata` differ is a hard error,
as is mixing Multipole and CylSpline.

### Materialising and writing

```python
pot = ser.materialize_potential()                  # -> agama.Potential (Evolving)
pot = ser.materialize_potential(gpu=True)          # -> EvolvingPotentialGPU
pot = mc.materialize_potential()                   # no time axis -> single snapshot

ser.to_coef_string(t=3)          # one snapshot; t is required with a time axis
ser.to_coef_strings()            # every time; 1-element list if time-less
ser.to_coef_files("out/")        # one plain-text file per time
ser.to_h5("series.h5")           # round-trips via read_coefs(group_name="all")
ser.to_evolving_ini("series.ini")
```

`materialize_potential` serialises from the *live* arrays, so manual surgery is
picked up automatically. `validate()` runs at the top of every write and
materialise entry point.

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

Always required (CPU-only stack):

- `numpy`, `h5py`
- `scipy` (quintic spline construction; falls back gracefully if missing)
- `agama` (for `Disk`/`Spheroid`/`King`/`Dehnen` types that use CPU export to build BFE)

Required **only** for the `*PotentialGPU` classes:

- CUDA GPU (tested on NVIDIA L40)
- `cupy >= 10.0` (matching CUDA version) — `pip install 'nbody_streams[cuda]'`
- `nvcc` accessible on PATH

CuPy is optional.  `import nbody_streams.agama_helper` succeeds without it, and
everything that does not touch the GPU — `read_coefs`, `load_agama_potential`,
`load_agama_evolving_potential`, the HDF5 writers, the FIRE helpers, the coef
dataclasses — works unchanged.  Constructing or evaluating a GPU potential on
such an install raises an `ImportError` naming the missing package:

```python
from nbody_streams import agama_helper as ah

ah.CUPY_AVAILABLE          # False on a CPU-only install
pot = ah.load_agama_potential("MW_mult.h5", group_name="snap_090")   # fine
pot = ah.PotentialGPU(...)  # ImportError: ... pip install 'nbody_streams[cuda]'
```

---

## File layout

```
agama_helper/
  _potential.py                     <- main wrapper: PotentialGPU factory + GPU classes
  _*.cu                             <- CUDA kernels for multipole and cylspl (potential, force, density, hessian)
  _analytic_potentials.py           <- analytic GPU potentials
  _load.py                          <- load_agama_potential / load_agama_evolving_potential (cpu + gpu= flag)
  _coefs.py                         <- MultipoleCoefs / CylSplineCoefs (optional time axis), readers, stack_coefs
  _io.py                            <- HDF5 archive I/O, temp-file helpers, source resolution
  _cupy.py                          <- optional-CuPy shim: real cupy, or a stub that raises on GPU use
  _fit.py                           <- BFE fitting from an N-body snapshot
  _fire.py                          <- FIRE-specific loaders, Evolving .ini generation, comoving-host centre acceleration
  tests/
    test_phase1_multipole.py        <- MultipolePotentialGPU correctness + benchmarks vs Agama CPU
    test_phase2_analytic.py         <- analytic GPU potential tests
    test_phase3_cylspline.py        <- CylSplinePotentialGPU correctness + benchmarks
    test_zero_pruning.py            <- zero-coefficient pruning correctness + speedup
    test_cylspl_sections.py         <- section-aware CylSpline parsing (#Phi / #dPhi/dR / #dPhi/dz)
    test_series.py                  <- coefficient time axis: stacking, round-trips, validation
    test_comoving_host.py           <- FIRE rotation, centre splines, centre-acceleration table (CPU == GPU == .ini)
  tech_err.md                       <- architecture decisions and precision notes
```

---

## Known gotchas

- **`Disk` type requires Agama**: `PotentialGPU(type='Disk', ...)` calls `agama.Potential` internally to export a Multipole coefficient file, then wraps it as `DiskAnsatz + MultipolePotentialGPU`.
- **`from_agama()` raises for pure analytic types**: Agama does not export NFW/Plummer/etc. parameters programmatically; these must be constructed directly by keyword.
- **EvolvingPotential interpolation**: GPU `interpolate=True` is linear lerp. Agama default (`interpLinear=False`) is nearest-neighbor. The INI parser maps `interpLinear=True` → GPU linear lerp.
- **lmax limit**: Kernel supports lmax <= 32. Python raises `ValueError` if exceeded.
- **CylSpline files carry three sections**: a real `Potential.export()` writes `#Phi`, `#dPhi/dR` and `#dPhi/dz`, each repeating the full set of `\t#m` blocks. `read_cylspl_coefs` splits on the section markers before scanning `#m` blocks; before that fix it was section-blind and returned `dPhi/dz` as `phi` with a duplicated `m_values`. Anything cached from an older run should be re-read.
- **The coef text format truncates float64**: `to_coef_string` writes `%.13g` (Multipole) / `%.14g` (CylSpline), so a value -> text -> value round-trip loses roughly the last digit (~1e-13 relative). `materialize_potential(gpu=True)` on a coef object feeds the arrays straight to `_build_multipole_data` / `_build_cylspline_data` and skips the text round-trip, which is why it is *more* precise than the string path, not merely faster.
- **A Multipole file needs `#dPhi/dr`**: Agama fails with `RuntimeError: Error loading Multipole potential` on a `#Phi`-only Multipole file, so `MultipoleCoefs.to_coef_string()` raises when `dphi_dr is None` rather than writing an unloadable file. A `#Phi`-only *CylSpline* file does load (Agama reconstructs the derivatives, at ~7e-5 relative error on `Phi`), so that stays a legal fallback.
