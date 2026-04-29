# AGAMA_GPU — Development Notes

## Project overview
GPU-accelerated Agama multipole BFE potential evaluation.
- `gpu_potential.py` — Python wrapper + kernel compiler + factory `PotentialGPU`
- `_multipole_potential_kernel.cu` — CUDA kernels (potential, force, density, hessian)
- `_analytic_potentials.py` — fused CuPy kernels for analytic types

## Numerical precision

Comparing GPU to Agama CPU at the same points:
- **l=0 monopole**: ~1e-11 to 1e-13 relative error (machine precision)
- **l>0 harmonics**: ~1e-7 to 1e-9 for potential; ~1e-5 to 1e-6 for force

This is not a GPU bug. Both GPU and Agama CPU hit the same floor from log-scaling derivative cancellation:  
`l>0` coefficient stored as ratio `C_lm/C_00`; derivative involves cancellation `dC_lm - ratio*dC_00` that loses ~4-7 digits when both terms track each other.

Force errors are ~100× larger than potential errors (derivative amplification). The BFE fitting error for N-body data is typically >1%, so this precision floor has no practical impact.

**Hessian**: self-consistent with GPU's own force to ~1e-8 (FD truncation limited).

**Disk composite** (DiskAnsatz + Multipole lmax=32): ~5e-5 on phi, ~3e-3 on force vs Agama CPU — inherent to the composite approximation, not a numerical error.

---

## Architecture decisions

### Kernel design
- **On-the-fly `pfact_cur` recurrence** (replaces `NORM_LM[289]` table): `pfact_cur *= sqrt((2l+1)/(2l-1)*(l-m)/(l+m))` per l-step, starting from `PREFACT[absm]`. Eliminates 2.3 KB constant memory.
- **Rolling trig state** (replaces `cos_mf[33]/sin_mf[33]`): `(cm, sm)` pair advanced per m-group. Eliminates stack arrays; needed for lmax>16 support.
- **lmax limit**: 32 (PREFACT/COEF tables have 33 entries; Python raises ValueError if exceeded).
- **lm sort order**: lm pairs sorted by `(|m|, l, cos-before-sin)` in Python before upload. Kernels assume this order for the on-the-fly recurrence.
- **d2Plm** (hessian kernel): computed from Legendre ODE: `d²P/dθ² = -cot(θ)·dP/dθ - [l(l+1) - m²/sin²(θ)]·P`. Near-pole guard: `sin_theta < 1e-10 → d2Plm = 0`.

### Outer extrapolation
Beyond r_max: `Phi(r) = W*(r/r_max)^(-1) + U*(r/r_max)^s` — Keplerian + subleading.  
`outer_s, outer_U, outer_W` computed from last two grid points in `_build_multipole_data`.  
Density kernel returns 0 beyond r_max.

### Non-uniform grid resampling
Only triggered when max log-r spacing error > 0.1%. FIRE files skip this path.  
Uses `CubicHermiteSpline` to `max(nR, 1000)` log-uniform points (preserves King tidal cutoff).

### Performance
No radius sorting (`_SORT_THRESHOLD = 999_999_999`): L40 96 MB L2 holds any realistic poly array; argsort+scatter costs more than it saves.  
No shared memory for lm_l/lm_m: L1-broadcast-cached, __syncthreads() overhead net-negative.

---

## File layout
```
agama_helper/
  _potential.py                     ← PotentialGPU factory + all GPU classes
  _multipole_potential_kernel.cu    ← CUDA kernels (potential, force, density, hessian)
  _cylspl_potential_kernel.cu       ← CUDA kernels for CylSpline (potential, force, density, hessian)
  _analytic_potentials.py           ← analytic GPU potentials (NFW, Plummer, Disk, etc.)
  _load.py                          ← load_agama_potential / load_agama_evolving_potential (cpu + gpu= flag)
  tests/
    test_phase1_multipole.py        ← MultipolePotentialGPU accuracy vs Agama CPU + benchmarks
    test_phase2_analytic.py         ← analytic GPU potential correctness
    test_phase3_cylspline.py        ← CylSplinePotentialGPU accuracy + mmax sweep + benchmarks
    test_zero_pruning.py            ← zero-coefficient pruning correctness + speedup
```

## INI loading (multi-section, case-insensitive)

`_load_potential_ini` and `_build_single` support:
- Arbitrary `[Potential xxx]` section headers (regex `^\s*\[Potential` is case-insensitive)
- Case-insensitive `type=` key detection (looks for any key whose lowercase == "type")
- `_normalize_params` maps mixed-case INI keys to canonical camelCase before dispatching to GPU constructors
- `_pop_ci` helper for case-insensitive key extraction (center, scale, ampl, type)
