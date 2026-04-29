"""
test_phase3_cylspline.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests CylSplinePotentialGPU against Agama CPU reference for the FIRE bar
potential (600.bar.none_8.coef_cylsp_DR).

Tests
-----
1. Accuracy   — Phi, force, forceDeriv vs Agama CPU at N=5000 interior points
2. mmax sweep — accuracy as a function of azimuthal truncation mmax=0..8
3. Speedup    — GPU vs Agama CPU at N = 1e3, 1e4, 1e5 particles

File required: 600.bar.none_8.coef_cylsp_DR  (same directory as this script)
"""

import os, time
import numpy as np
import cupy as cp

import agama
agama.setUnits(mass=1, length=1, velocity=1)

from nbody_streams.agama_helper import PotentialGPU, read_coefs, load_agama_potential
from nbody_streams.agama_helper._potential import CylSplinePotentialGPU

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_HERE     = os.path.dirname(__file__)
_COEF_BAR = os.path.join(_HERE, "600.bar.none_8.coef_cylsp_DR")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(N, seed=42, rmin=1.0, rmax=80.0):
    """Random Cartesian points inside a sphere, avoiding z-axis."""
    rng  = np.random.default_rng(seed)
    r    = rng.uniform(rmin, rmax, N)
    cost = rng.uniform(-1, 1, N)
    phi  = rng.uniform(0, 2*np.pi, N)
    sint = np.sqrt(1 - cost**2)
    xyz  = np.column_stack([r*sint*np.cos(phi),
                            r*sint*np.sin(phi),
                            r*cost]).astype(np.float64)
    xyz[:, :2] += 0.1   # avoid exact z-axis
    return xyz


def _rel_phi(gpu, cpu):
    g, c = np.asarray(gpu), np.asarray(cpu)
    return float(np.max(np.abs(g - c) / (np.abs(c) + 1e-30)))


def _rel_force(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1))) + 1e-30
    return float(np.max(np.sqrt(np.sum((np.asarray(gpu_f) - cpu_f)**2, axis=1))) / fmag)


def _rel_deriv(gpu_d, cpu_d):
    scale = np.max(np.abs(cpu_d)) + 1e-30
    return float(np.max(np.abs(np.asarray(gpu_d) - cpu_d)) / scale)


def _time_fn(fn, warmup=3, reps=10, is_gpu=True):
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


# ---------------------------------------------------------------------------
# 1. Accuracy test
# ---------------------------------------------------------------------------

def test_accuracy():
    if not os.path.exists(_COEF_BAR):
        print(f"SKIP test_accuracy: {_COEF_BAR} not found")
        return

    N     = 5000
    xyz   = _pts(N)
    xyz_cp = cp.asarray(xyz)

    pot_cpu = load_agama_potential(_COEF_BAR)
    pot_gpu = CylSplinePotentialGPU.from_file(_COEF_BAR)

    phi_cpu = pot_cpu.potential(xyz)
    f_cpu   = pot_cpu.force(xyz)
    fd_cpu  = pot_cpu.forceDeriv(xyz)[1]   # (N,6) force derivatives

    phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
    f_gpu   = cp.asnumpy(pot_gpu.force(xyz_cp))
    _, dF   = pot_gpu.forceDeriv(xyz_cp)
    fd_gpu  = cp.asnumpy(dF)

    e_phi = _rel_phi(phi_gpu, phi_cpu)
    e_f   = _rel_force(f_gpu, f_cpu)
    e_fd  = _rel_deriv(fd_gpu, fd_cpu)

    W = 72
    print()
    print("=" * W)
    print(f"{'PHASE 3 ACCURACY:  CylSplineGPU  vs  Agama CPU  (N=5000)':^{W}}")
    print("=" * W)
    print(f"  {'Quantity':<20}  {'Max rel error':>15}  {'Pass?':>8}")
    print("-" * W)

    rows = [
        ("Potential Phi",   e_phi, 5e-3),
        ("Force |dF|",      e_f,   5e-3),
        ("Force deriv",     e_fd,  3e-2),
    ]
    all_pass = True
    for name, err, tol in rows:
        ok = err < tol
        all_pass &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:<20}  {err:>15.2e}  {flag:>8}")

    print("=" * W)
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    assert all_pass, "Accuracy test failed — see table above"


# ---------------------------------------------------------------------------
# 2. mmax sweep
# ---------------------------------------------------------------------------

def test_mmax_sweep():
    if not os.path.exists(_COEF_BAR):
        print(f"SKIP test_mmax_sweep: {_COEF_BAR} not found")
        return

    N      = 3000
    xyz    = _pts(N)
    xyz_cp = cp.asarray(xyz)

    coefs_full = read_coefs(_COEF_BAR)
    mmax_full  = max(abs(m) for m in coefs_full.m_values)

    # Reference: full mmax CPU potential
    pot_cpu_full = load_agama_potential(_COEF_BAR)
    phi_ref = pot_cpu_full.potential(xyz)
    f_ref   = pot_cpu_full.force(xyz)

    W = 72
    print()
    print("=" * W)
    print(f"{'PHASE 3 mmax SWEEP:  GPU(mmax=k) vs CPU(mmax=8 full)':^{W}}")
    print("=" * W)
    print(f"  Note: errors here are the *harmonic contribution*, not GPU inaccuracy.")
    print(f"  At mmax=8 (bottom row) the error is pure GPU numerical error.")
    print("-" * W)
    print(f"  {'mmax':<6}  {'Phi err':>12}  {'Force err':>12}  {'n_harm':>8}")
    print("-" * W)

    for mmax_test in range(0, mmax_full + 1):
        keep_m = list(range(-mmax_test, mmax_test + 1))
        coefs_trunc = coefs_full.zeroed(keep_m=keep_m, include_negative=False)
        pot_gpu = CylSplinePotentialGPU(coefs_trunc)
        n_harm  = len([m for m in coefs_trunc.m_values if m in keep_m or m == 0])

        phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
        f_gpu   = cp.asnumpy(pot_gpu.force(xyz_cp))

        e_phi = _rel_phi(phi_gpu, phi_ref)
        e_f   = _rel_force(f_gpu, f_ref)

        print(f"  {mmax_test:<6}  {e_phi:>12.2e}  {e_f:>12.2e}  {2*mmax_test+1:>8}")

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# 3. Speedup benchmark
# ---------------------------------------------------------------------------

def test_speedup():
    if not os.path.exists(_COEF_BAR):
        print(f"SKIP test_speedup: {_COEF_BAR} not found")
        return

    pot_cpu = load_agama_potential(_COEF_BAR)
    pot_gpu = CylSplinePotentialGPU.from_file(_COEF_BAR)

    W = 72
    print()
    print("=" * W)
    print(f"{'PHASE 3 SPEEDUP:  CylSplineGPU  vs  Agama CPU  (force)':^{W}}")
    print("=" * W)
    print(f"  {'N':>10}  {'CPU (ms)':>12}  {'GPU (ms)':>12}  {'Speedup':>10}")
    print("-" * W)

    for N in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        xyz    = _pts(N)
        xyz_cp = cp.asarray(xyz)

        t_cpu = _time_fn(lambda: pot_cpu.force(xyz), is_gpu=False)
        t_gpu = _time_fn(lambda: pot_gpu.force(xyz_cp))

        speedup = t_cpu / t_gpu
        print(f"  {N:>10,}  {t_cpu*1e3:>12.2f}  {t_gpu*1e3:>12.2f}  {speedup:>10.1f}x")

        del xyz_cp
        cp.get_default_memory_pool().free_all_blocks()

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_accuracy()
    test_mmax_sweep()
    test_speedup()
