"""
test_phase2_analytic.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests each Phase 2 analytic GPU potential against Agama CPU.

For each model:
  - .potential()  vs agama.Potential
  - .force()      vs agama.Potential
  - .density()    vs agama.Potential
  - .forceDeriv() vs agama.Potential

Category A (direct GPU kernels): NFW, Plummer, Hernquist, Isochrone, MiyamotoNagai,
  LogHalo, DehnenSpherical, DiskAnsatz, UniformAcceleration
Category B (via Agama CPU export → MultipolePotentialGPU): King, Spheroid,
  Dehnen triaxial, Dehnen gamma=2
"""

import time
import numpy as np

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp
from nbody_streams.agama_helper._analytic_potentials import (
    NFWPotentialGPU, PlummerPotentialGPU, HernquistPotentialGPU,
    IsochronePotentialGPU, MiyamotoNagaiPotentialGPU,
    LogHaloPotentialGPU, DehnenSphericalPotentialGPU,
    DiskAnsatzPotentialGPU, UniformAccelerationGPU,
)
from nbody_streams.agama_helper import PotentialGPU

# ---------------------------------------------------------------------------
# Test grids
# ---------------------------------------------------------------------------

def _pts(N, seed=42, lo=-20, hi=20):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lo, hi, (N, 3)).astype(np.float64)
    xyz[:, :2] += 0.1   # avoid exact pole
    return xyz

def _pts_king(N, seed=42):
    """Points within King tidal radius (scaleRadius=0.01 kpc, trunc=2 → r_t ~0.02 kpc)."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-0.015, 0.015, (N, 3)).astype(np.float64)
    xyz[:, :2] += 1e-4
    return xyz


def _rel(gpu, cpu):
    diff = np.abs(cp.asnumpy(gpu) - cpu)
    return np.max(diff / (np.abs(cpu) + 1e-30))


def _frel(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1)))
    return np.max(np.abs(cp.asnumpy(gpu_f) - cpu_f)) / (fmag + 1e-30)


def _time_fn(fn, warmup=5, reps=20, is_gpu=True):
    for _ in range(warmup):
        fn()
    if is_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if is_gpu:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / reps


# ---------------------------------------------------------------------------
# Model registry
# MODELS entries: (name, gpu_fn, agama_kw_or_None, xyz_fn_or_None)
#   agama_kw=None  → GPU-only (no Agama comparison)
#   xyz_fn=None    → use default _pts(N)
# ---------------------------------------------------------------------------

# Tolerances: (phi_tol, f_tol, hess_tol)
_TOL_A  = (1e-4, 1e-4, 1e-3)   # Category A — direct GPU kernels (typically 1e-15)
_TOL_B  = (1e-4, 1e-4, 1e-3)   # Category B — Multipole-routed smooth potentials
_TOL_BK = (1e-3, 1e-3, 0.1)    # King — lmax=0 truncated profile; Hessian limited

MODELS = [
    # ------------------------------------------------------------------
    # Category A — direct GPU kernels
    # (name, gpu_fn, agama_kw, xyz_fn, tol)
    # ------------------------------------------------------------------
    ("NFW",
     lambda: NFWPotentialGPU(mass=1e12, scaleRadius=20.0),
     dict(type='NFW', mass=1e12, scaleRadius=20.0),
     None, _TOL_A),

    ("Plummer",
     lambda: PlummerPotentialGPU(mass=1e11, scaleRadius=5.0),
     dict(type='Plummer', mass=1e11, scaleRadius=5.0),
     None, _TOL_A),

    # Agama has no 'Hernquist' type — it's Dehnen(gamma=1)
    ("Hernquist (=Dehnen γ=1)",
     lambda: HernquistPotentialGPU(mass=5e11, scaleRadius=10.0),
     dict(type='Dehnen', mass=5e11, scaleRadius=10.0, gamma=1.0),
     None, _TOL_A),

    ("Isochrone",
     lambda: IsochronePotentialGPU(mass=1e11, scaleRadius=2.0),
     dict(type='Isochrone', mass=1e11, scaleRadius=2.0),
     None, _TOL_A),

    ("MiyamotoNagai",
     lambda: MiyamotoNagaiPotentialGPU(mass=5e10, scaleRadius=3.0, scaleHeight=0.3),
     dict(type='MiyamotoNagai', mass=5e10, scaleRadius=3.0, scaleHeight=0.3),
     None, _TOL_A),

    ("LogHalo",
     lambda: LogHaloPotentialGPU(velocity=200.0, coreRadius=0.1),
     dict(type='Logarithmic', v0=200.0, scaleRadius=0.1),
     None, _TOL_A),

    ("LogHalo (triaxial)",
     lambda: LogHaloPotentialGPU(velocity=200.0, coreRadius=0.1,
                                  axisRatioY=0.9, axisRatioZ=0.8),
     dict(type='Logarithmic', v0=200.0, scaleRadius=0.1,
          axisRatioY=0.9, axisRatioZ=0.8),
     None, _TOL_A),

    ("Dehnen (gamma=0)",
     lambda: DehnenSphericalPotentialGPU(mass=1e12, scaleRadius=5.0, gamma=0.0),
     dict(type='Dehnen', mass=1e12, scaleRadius=5.0, gamma=0.0),
     None, _TOL_A),

    ("Dehnen (gamma=1)",
     lambda: DehnenSphericalPotentialGPU(mass=1e12, scaleRadius=5.0, gamma=1.0),
     dict(type='Dehnen', mass=1e12, scaleRadius=5.0, gamma=1.0),
     None, _TOL_A),

    ("Dehnen (gamma=1.5)",
     lambda: DehnenSphericalPotentialGPU(mass=1e12, scaleRadius=5.0, gamma=1.5),
     dict(type='Dehnen', mass=1e12, scaleRadius=5.0, gamma=1.5),
     None, _TOL_A),

    # DiskAnsatz and UniformAcceleration don't exist as standalone Agama types
    ("DiskAnsatz (exp)  [GPU only]",
     lambda: DiskAnsatzPotentialGPU(surfaceDensity=1e8, scaleRadius=3.0, scaleHeight=0.3),
     None, None, _TOL_A),

    ("DiskAnsatz (sech2) [GPU only]",
     lambda: DiskAnsatzPotentialGPU(surfaceDensity=1e8, scaleRadius=3.0, scaleHeight=-0.3),
     None, None, _TOL_A),

    ("UniformAccel [GPU only]",
     lambda: UniformAccelerationGPU(ax=0.01, ay=-0.02, az=0.005),
     None, None, _TOL_A),

    # ------------------------------------------------------------------
    # Category B — Agama CPU → Multipole export → MultipolePotentialGPU
    # ------------------------------------------------------------------

    # King: scaleRadius=0.01 kpc → test points must be within tidal radius.
    # lmax=0 (spherical); Hessian limited (~3%) for truncated profile near tidal edge.
    ("King",
     lambda: PotentialGPU(type='King', mass=1e5, scaleRadius=0.01, W0=7.0, trunc=2.0),
     dict(type='King', mass=1e5, scaleRadius=0.01, W0=7.0, trunc=2.0),
     _pts_king, _TOL_BK),

    # Spheroid: Hernquist-profile equivalent (alpha=1,beta=4,gamma=1)
    ("Spheroid (spherical)",
     lambda: PotentialGPU(type='Spheroid', mass=1e12, scaleRadius=20.,
                          alpha=1., beta=4., gamma=1.),
     dict(type='Spheroid', mass=1e12, scaleRadius=20.,
          alpha=1., beta=4., gamma=1.),
     None, _TOL_B),

    # Spheroid triaxial
    ("Spheroid (triaxial y=0.9 z=0.8)",
     lambda: PotentialGPU(type='Spheroid', mass=1e12, scaleRadius=20.,
                          alpha=1., beta=4., gamma=1.,
                          axisRatioY=0.9, axisRatioZ=0.8),
     dict(type='Spheroid', mass=1e12, scaleRadius=20.,
          alpha=1., beta=4., gamma=1.,
          axisRatioY=0.9, axisRatioZ=0.8),
     None, _TOL_B),

    # Spheroid with outer cutoff
    ("Spheroid (outerCutoff)",
     lambda: PotentialGPU(type='Spheroid', mass=1e12, scaleRadius=20., gamma=1.,
                          outerCutoffRadius=200., cutoffStrength=2.),
     dict(type='Spheroid', mass=1e12, scaleRadius=20., gamma=1.,
          outerCutoffRadius=200., cutoffStrength=2.),
     None, _TOL_B),

    # Dehnen gamma=2 (can't do in GPU kernel; routed through Agama)
    ("Dehnen (gamma=2, Agama route)",
     lambda: PotentialGPU(type='Dehnen', mass=1e12, scaleRadius=5., gamma=2.),
     dict(type='Dehnen', mass=1e12, scaleRadius=5., gamma=2.),
     None, _TOL_B),

    # Dehnen triaxial (routed through Agama as Spheroid alpha=1,beta=4)
    ("Dehnen triaxial (y=0.9 z=0.8)",
     lambda: PotentialGPU(type='Dehnen', mass=1e12, scaleRadius=5., gamma=1.,
                          axisRatioY=0.9, axisRatioZ=0.8),
     dict(type='Spheroid', mass=1e12, scaleRadius=5., gamma=1.,
          alpha=1, beta=4, axisRatioY=0.9, axisRatioZ=0.8),
     None, _TOL_B),
]


# ---------------------------------------------------------------------------
# Hessian FD check
# ---------------------------------------------------------------------------

def _hess_fd_check(gpu_pot, xyz_cp, h=1e-5):
    xyz_np = cp.asnumpy(xyz_cp)
    N = xyz_np.shape[0]
    pairs = [(0,0),(1,1),(2,2),(0,1),(1,2),(0,2)]
    fd = np.zeros((N, 6))
    for slot, (fi, xj) in enumerate(pairs):
        xp = xyz_np.copy(); xp[:, xj] += h
        xm = xyz_np.copy(); xm[:, xj] -= h
        fp = cp.asnumpy(gpu_pot.force(cp.asarray(xp)))[:, fi]
        fm = cp.asnumpy(gpu_pot.force(cp.asarray(xm)))[:, fi]
        fd[:, slot] = (fp - fm) / (2*h)
    _, dF_gpu = gpu_pot.forceDeriv(xyz_cp)
    dF_arr = cp.asnumpy(dF_gpu)
    diff = np.abs(dF_arr - fd)
    scale = np.abs(fd) + 1e-30
    return float(np.max(diff / scale))


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_accuracy():
    print("=" * 105)
    print(f"{'PHASE 2 ACCURACY: GPU vs Agama CPU (N=2000)':^105}")
    print("=" * 105)
    print(f"  {'Model':>30} | {'Phi Rel':>10} | {'F Rel':>10} | {'Rho Rel':>10} | {'Hess Rel':>10} | Status")
    print("-" * 105)

    N = 2000

    for name, gpu_fn, agama_kw, xyz_fn, tol in MODELS:
        xyz_np = (xyz_fn(N) if xyz_fn is not None else _pts(N)).astype(np.float64)
        xyz_cp = cp.asarray(xyz_np)
        try:
            gpu_pot = gpu_fn()

            if agama_kw is None:
                phi_gpu = gpu_pot.potential(xyz_cp)
                f_gpu, _ = gpu_pot.forceDeriv(xyz_cp)
                hfd = _hess_fd_check(gpu_pot, xyz_cp[:20])
                is_finite = cp.all(cp.isfinite(phi_gpu)) and cp.all(cp.isfinite(f_gpu))
                status = "OK" if (is_finite and hfd < 1e-4) else "CHECK"
                print(f"  {name:>30} | {'GPU-only':>10} | {'GPU-only':>10} | {'GPU-only':>10} | {hfd:>10.2e} | {status}")
                continue

            phi_tol, f_tol, hess_tol = tol
            agama_pot = agama.Potential(**agama_kw)
            phi_cpu, acc_cpu, dF_cpu = agama_pot.eval(xyz_np, pot=True, acc=True, der=True)

            phi_gpu          = cp.asnumpy(gpu_pot.potential(xyz_cp))
            f_gpu, dF_gpu_cp = gpu_pot.forceDeriv(xyz_cp)
            f_gpu            = cp.asnumpy(f_gpu)
            dF_gpu           = cp.asnumpy(dF_gpu_cp)

            phi_rel  = _rel(phi_gpu, phi_cpu)
            f_rel    = _frel(f_gpu, acc_cpu)

            rho_cpu = agama_pot.density(xyz_np)
            rho_gpu = cp.asnumpy(gpu_pot.density(xyz_cp))
            mask    = np.abs(rho_cpu) > 1e-20
            rho_rel = float(np.max(np.abs(rho_gpu[mask] - rho_cpu[mask])
                                   / (np.abs(rho_cpu[mask]) + 1e-30))) if mask.any() else 0.0

            dF_scale = np.max(np.abs(dF_cpu)) + 1e-30
            hess_rel = np.max(np.abs(dF_gpu - dF_cpu)) / dF_scale

            is_ok  = phi_rel < phi_tol and f_rel < f_tol and hess_rel < hess_tol
            status = "PASS" if is_ok else "FAIL"
            print(f"  {name:>30} | {phi_rel:>10.2e} | {f_rel:>10.2e} | {rho_rel:>10.2e} | {hess_rel:>10.2e} | {status}")

        except Exception as e:
            print(f"  {name:>30} | CRITICAL ERROR: {e}")

    print("=" * 105)


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

class GPUTimer:
    @staticmethod
    def benchmark(fn, is_gpu=True, warmup=5, reps=20):
        for _ in range(warmup):
            fn()
        if is_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if is_gpu:
            cp.cuda.Stream.null.synchronize()
        return (time.perf_counter() - t0) / reps


def run_comprehensive_benchmark():
    print("=" * 100)
    print(f"{'ANALYTIC POTENTIAL PERFORMANCE BENCHMARK':^100}")
    print("=" * 100)
    print(f"{'Model':<30} | {'N':<8} | {'CPU (ms)':<10} | {'GPU (ms)':<10} | {'Speedup':<10} | {'M-eval/s'}")
    print("-" * 100)

    N_list = [10**3, 10**4, 10**5, 10**6, 10**7]

    for name, gpu_fn, agama_kw, xyz_fn, *_ in MODELS:
        if agama_kw is None:
            continue
        try:
            gpu_pot  = gpu_fn()
            agama_pot = agama.Potential(**agama_kw)
        except Exception as e:
            print(f"{name:<30} | SKIP: {e}")
            continue

        for N in N_list:
            xyz_np = (xyz_fn(N, seed=99) if xyz_fn is not None
                      else np.random.default_rng(99).uniform(-20, 20, (N, 3)).astype(np.float64))
            xyz_cp = cp.asarray(xyz_np)

            t_cpu = GPUTimer.benchmark(lambda: agama_pot.force(xyz_np), is_gpu=False)
            t_gpu = GPUTimer.benchmark(lambda: gpu_pot.force(xyz_cp),   is_gpu=True)

            # Release pooled benchmark output arrays before next N step.
            # Each benchmark call creates a new output array; without this the
            # CuPy memory pool accumulates O(warmup+reps) × N × 24 bytes and
            # can exhaust GPU memory on large N (e.g. N=10^7 on H-200).
            del xyz_cp
            cp.get_default_memory_pool().free_all_blocks()

            speedup    = t_cpu / t_gpu
            throughput = (N / t_gpu) / 1e6
            cpu_ms     = t_cpu * 1000
            gpu_ms     = t_gpu * 1000
            note       = "<<" if speedup > 10 else "  "
            print(f"{name:<30} | {N:<8} | {cpu_ms:>10.3f} | {gpu_ms:>10.3f} | {speedup:>9.1f}x | {throughput:>8.2f} {note}")

        print("-" * 100)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_accuracy()
    run_comprehensive_benchmark()
