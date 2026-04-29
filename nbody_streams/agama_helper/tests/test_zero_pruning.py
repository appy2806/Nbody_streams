"""
test_zero_pruning.py
~~~~~~~~~~~~~~~~~~~~
Benchmarks zero-coefficient pruning (Phase 1a) against Agama CPU.

Three-way comparison:
  (A) Agama CPU                  — reference
  (B) GPU  WITHOUT pruning       — prune_threshold=0  (iterates all n_lm pairs)
  (C) GPU  WITH    pruning       — prune_threshold=1e-16 (drops zero columns)

Two potential configs:
  1. FIRE halo lmax=8, all (l,m)       → n_lm=81, none pruned   → B≈C
  2. Axisymmetric halo lmax=8, m=0 only → n_lm_raw=81, pruned→5 → B>>C for GPU

Run with: python test_zero_pruning.py
"""

import os, time
import numpy as np

import agama
agama.setUnits(mass=1, length=1, velocity=1)
import cupy as cp
from nbody_streams.agama_helper._potential import MultipolePotentialGPU
from nbody_streams.agama_helper import read_coefs, load_agama_potential

_HERE = os.path.dirname(__file__)
COEF_FILE_8 = os.path.join(_HERE, "600.dark.none_8.coef_mul_DR")

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def make_halo(lmax=8):
    """Full FIRE halo: all (l,m) up to lmax, non-zero for most."""
    mc = read_coefs(COEF_FILE_8)
    keep = [(l, m) for l, m in mc.lm_labels if l <= lmax]
    mc2 = mc.zeroed(keep_lm=keep)
    pot_cpu     = load_agama_potential(COEF_FILE_8, keep_lm_mult=keep)
    pot_no_prune = MultipolePotentialGPU(mc2, prune_threshold=0)
    pot_pruned   = MultipolePotentialGPU(mc2, prune_threshold=1e-16)
    return pot_cpu, pot_no_prune, pot_pruned


def make_axisym(lmax=8):
    """Axisymmetric (m=0, even l only): maximum pruning benefit."""
    mc = read_coefs(COEF_FILE_8)
    keep = [(l, 0) for l in range(0, lmax + 1, 2)]  # (0,0),(2,0),...,(8,0)
    mc2 = mc.zeroed(keep_lm=keep)
    pot_cpu      = load_agama_potential(COEF_FILE_8, keep_lm_mult=keep)
    pot_no_prune = MultipolePotentialGPU(mc2, prune_threshold=0)
    pot_pruned   = MultipolePotentialGPU(mc2, prune_threshold=1e-16)
    return pot_cpu, pot_no_prune, pot_pruned


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timeit(fn, is_gpu=True, warmup=5, reps=20):
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
    return (t1 - t0) / reps * 1e3  # ms


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_correctness():
    if not os.path.exists(COEF_FILE_8):
        import pytest; pytest.skip(f"Coef file not found: {COEF_FILE_8}")
    print("=" * 72)
    print("  CORRECTNESS: GPU (pruned) vs Agama CPU — lmax=8, N=5000")
    print("=" * 72)

    rng = np.random.default_rng(42)
    N = 5000
    xyz_np = rng.uniform(-50, 50, (N, 3)).astype(np.float64)
    xyz_np[:, :2] += 0.1
    xyz_cp = cp.asarray(xyz_np)
    n_lm_raw = (8 + 1) ** 2

    for label, make_fn in [("halo (non-axisym, all l,m)", make_halo),
                            ("axisym (m=0, even l only)", make_axisym)]:
        pot_cpu, pot_no, pot_pr = make_fn(lmax=8)

        phi_cpu = pot_cpu.potential(xyz_np)
        f_cpu   = pot_cpu.force(xyz_np)
        fmag    = np.sqrt(np.sum(f_cpu**2, axis=1))

        for gpu_label, pot_gpu in [("no prune", pot_no), ("pruned  ", pot_pr)]:
            phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
            f_gpu   = cp.asnumpy(pot_gpu.force(xyz_cp))
            phi_err = np.median(np.abs(phi_gpu - phi_cpu) / (np.abs(phi_cpu) + 1e-30))
            f_err   = np.median(np.abs(f_gpu - f_cpu).max(axis=1) / (fmag + 1e-30))
            ok = "OK" if phi_err < 1e-4 and f_err < 1e-4 else "FAIL <<<"
            print(f"\n  {label}  [{gpu_label}  n_lm_gpu={pot_gpu._n_lm:3d}/{n_lm_raw}]")
            print(f"    Phi median rel err: {phi_err:.2e}   F median rel err: {f_err:.2e}   {ok}")


# ---------------------------------------------------------------------------
# Speedup benchmark
# ---------------------------------------------------------------------------

def test_speedup():
    print("\n" + "=" * 72)
    print("  SPEEDUP: Agama CPU vs GPU-no-prune vs GPU-pruned  (.force())")
    print("=" * 72)

    N_list = [1_000, 10_000, 100_000, 1_000_000]

    configs = [
        ("FIRE halo  lmax=8  (no pruning expected, n_lm=81)", make_halo,   8),
        ("Axisym     lmax=8  (m=0 only, prunes 81→5)       ", make_axisym, 8),
    ]

    for label, make_fn, lmax in configs:
        pot_cpu, pot_no, pot_pr = make_fn(lmax=lmax)
        n_lm_raw = (lmax + 1) ** 2
        print(f"\n  {label}")
        print(f"  n_lm_raw={n_lm_raw}, no-prune n_lm={pot_no._n_lm}, pruned n_lm={pot_pr._n_lm}")
        hdr = (f"  {'N':>9}  {'CPU(ms)':>9}  {'GPU-noP(ms)':>12}  "
               f"{'GPU-prune(ms)':>14}  {'noP/CPU':>8}  {'prune/CPU':>10}  {'prune/noP':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for N in N_list:
            rng = np.random.default_rng(7)
            xyz_np = rng.uniform(-50, 50, (N, 3)).astype(np.float64)
            xyz_cp = cp.asarray(xyz_np)

            t_cpu = timeit(lambda: pot_cpu.force(xyz_np), is_gpu=False)
            t_no  = timeit(lambda: pot_no.force(xyz_cp),  is_gpu=True)
            t_pr  = timeit(lambda: pot_pr.force(xyz_cp),  is_gpu=True)

            print(f"  {N:>9,}  {t_cpu:>9.2f}  {t_no:>12.2f}  {t_pr:>14.2f}"
                  f"  {t_cpu/t_no:>7.1f}x  {t_cpu/t_pr:>9.1f}x  {t_no/t_pr:>9.1f}x")
            del xyz_cp
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_correctness()
    test_speedup()
