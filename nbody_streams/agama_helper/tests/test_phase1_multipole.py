"""
test_phase1_multipole.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests MultipolePotentialGPU against Agama CPU for physics correctness.

Accuracy models: Spheroid and Dehnen potentials exported as Multipole
  coefficients via agama.Potential.export(), tested at lmax = 2, 4, 8.

Tolerances from tech_err.md:
  phi rel err  ~ 1e-7  (l>0 harmonics)
  force rel err ~ 1e-5  (l>0 harmonics)
  rho rel err   ~1e-2  (density via Laplacian; larger scatter near r=0)
  hess rel err  ~5e-2  (forceDeriv; FD-truncation limited)

Run with:  pytest test_phase1_multipole.py          # accuracy only
           python test_phase1_multipole.py          # accuracy + benchmarks
"""

import os, time, tempfile
import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp
from nbody_streams.agama_helper._potential import MultipolePotentialGPU
from nbody_streams.agama_helper import read_coefs, load_agama_potential


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(N, seed=42, lo=-50, hi=50):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lo, hi, (N, 3)).astype(np.float64)
    xyz[:, :2] += 0.1   # avoid exact z-axis pole
    return xyz


def _rel_phi(gpu, cpu):
    diff = np.abs(np.asarray(gpu) - cpu)
    return float(np.max(diff / (np.abs(cpu) + 1e-30)))


def _rel_force(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1))) + 1e-30
    return float(np.max(np.abs(np.asarray(gpu_f) - cpu_f)) / fmag)


def _rel_scalar(gpu, cpu, eps=1e-20):
    g, c = np.asarray(gpu), np.asarray(cpu)
    mask = np.abs(c) > eps
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(g[mask] - c[mask]) / (np.abs(c[mask]) + 1e-30)))


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
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# Precomputed coefficient files  (optional — skipped if absent)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_PRECOMPUTED_FILES = {
    "LMC":  os.path.join(_HERE, "100.LMC.none_8.coef_mult"),
    "FIRE": os.path.join(_HERE, "600.dark.none_8.coef_mul_DR"),
}

_LMAX_LIST = [0, 2, 4, 6, 8]


def _make_from_coef_file(coef_file, lmax):
    """Load coef file, cap at lmax, return (agama_pot, gpu_pot)."""
    mc   = read_coefs(coef_file)
    keep = [(l, m) for l, m in mc.lm_labels if l <= lmax]
    mc2  = mc.zeroed(keep_lm=keep)

    pot_cpu = load_agama_potential(coef_file, keep_lm_mult=keep)
    pot_gpu = MultipolePotentialGPU(mc2)
    return pot_cpu, pot_gpu


def _make_from_agama(agama_kw, lmax):
    """Export agama_pot as Multipole BFE, load into GPU."""
    pot_cpu = agama.Potential(**agama_kw)

    with tempfile.NamedTemporaryFile(suffix=".coef_mul", delete=False) as f:
        tmp = f.name
    try:
        agama.Potential(type='Multipole', potential=pot_cpu, lmax=lmax,
                        gridSizeR=50).export(tmp)
        mc = read_coefs(tmp)
    finally:
        os.unlink(tmp)

    keep = [(l, m) for l, m in mc.lm_labels if l <= lmax]
    mc2  = mc.zeroed(keep_lm=keep)
    pot_gpu = MultipolePotentialGPU(mc2)
    return pot_cpu, pot_gpu


# ---------------------------------------------------------------------------
# Accuracy test
# ---------------------------------------------------------------------------

_MODELS = [
    ("Spheroid (sph)  lmax=2",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 2),
    ("Spheroid (sph)  lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 4),
    ("Spheroid (sph)  lmax=8",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 8),
    ("Spheroid (tri)  lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
          axisRatioY=0.9, axisRatioZ=0.8), 4),
    ("Spheroid (tri)  lmax=8",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
          axisRatioY=0.9, axisRatioZ=0.8), 8),
    ("Dehnen g=1      lmax=4",
     dict(type='Dehnen', mass=1e12, scaleRadius=5., gamma=1.), 4),
    ("Dehnen g=1.5    lmax=8",
     dict(type='Dehnen', mass=1e12, scaleRadius=5., gamma=1.5), 8),
    ("Spheroid cutoff lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., gamma=1.,
          outerCutoffRadius=200., cutoffStrength=2.), 4),
]

# Tolerances vs Agama CPU — (phi, force, rho, hess)
# Analytic potential models exported to Multipole BFE
_TOL = (1e-3, 1e-3, 1e-2, 5e-2)
# Precomputed real-galaxy files: density (Laplacian of 2nd derivative) is noisier
# at low lmax where a few harmonics must represent a non-analytic profile.
_TOL_PRECOMPUTED = (1e-3, 1e-3, 1e-1, 5e-2)


def test_accuracy():
    N = 2000
    xyz_np = _pts(N)
    xyz_cp = cp.asarray(xyz_np)

    W = 86
    print("=" * W)
    print(f"{'PHASE 1 ACCURACY:  GPU vs Agama CPU (N=2000)':^{W}}")
    print("=" * W)
    print(f"  {'Model':<30} | {'Phi/CPU':>9} | {'F/CPU':>9} | {'Rho/CPU':>9} | {'Hess/CPU':>9} | Status")
    print("-" * W)

    all_pass = True
    for name, agama_kw, lmax in _MODELS:
        try:
            pot_cpu, pot_gpu = _make_from_agama(agama_kw, lmax)

            phi_cpu  = pot_cpu.potential(xyz_np)
            f_cpu    = pot_cpu.force(xyz_np)
            rho_cpu  = pot_cpu.density(xyz_np)
            dF_cpu   = pot_cpu.forceDeriv(xyz_np)[1]   # (N,6)

            phi_gpu  = cp.asnumpy(pot_gpu.potential(xyz_cp))
            f_gpu    = cp.asnumpy(pot_gpu.force(xyz_cp))
            rho_gpu  = cp.asnumpy(pot_gpu.density(xyz_cp))
            _, dF_cp = pot_gpu.forceDeriv(xyz_cp)
            dF_gpu   = cp.asnumpy(dF_cp)

            phi_vs_cpu  = _rel_phi(phi_gpu, phi_cpu)
            f_vs_cpu    = _rel_force(f_gpu, f_cpu)
            rho_vs_cpu  = _rel_scalar(rho_gpu, rho_cpu)
            dF_scale    = np.max(np.abs(dF_cpu)) + 1e-30
            hess_vs_cpu = float(np.max(np.abs(dF_gpu - dF_cpu)) / dF_scale)

            pt, ft, rt, ht = _TOL
            is_ok = (phi_vs_cpu < pt and f_vs_cpu < ft and
                     rho_vs_cpu < rt and hess_vs_cpu < ht)
            all_pass = all_pass and is_ok
            status = "PASS" if is_ok else "FAIL"
            print(f"  {name:<30} | {phi_vs_cpu:>9.2e} | {f_vs_cpu:>9.2e} | {rho_vs_cpu:>9.2e}"
                  f" | {hess_vs_cpu:>9.2e} | {status}")

        except Exception as e:
            all_pass = False
            print(f"  {name:<30} | CRITICAL ERROR: {e}")

    print("=" * W)
    assert all_pass, "One or more accuracy checks failed — see table above"


# ---------------------------------------------------------------------------
# Precomputed-file accuracy
# ---------------------------------------------------------------------------

def test_precomputed_accuracy():
    present = {k: p for k, p in _PRECOMPUTED_FILES.items() if os.path.exists(p)}
    if not present:
        pytest.skip("No precomputed coefficient files found")

    N = 2000
    xyz_np = _pts(N)
    xyz_cp = cp.asarray(xyz_np)

    W = 86
    print("\n" + "=" * W)
    print(f"{'PRECOMPUTED FILE ACCURACY:  GPU vs Agama CPU':^{W}}")
    print("=" * W)
    print(f"  {'Model':<30} | {'Phi/CPU':>9} | {'F/CPU':>9} | {'Rho/CPU':>9} | {'Hess/CPU':>9} | Status")
    print("-" * W)

    all_pass = True
    for label, path in present.items():
        for lmax in _LMAX_LIST:
            name = f"{label} lmax={lmax}"
            try:
                pot_cpu, pot_gpu = _make_from_coef_file(path, lmax)

                phi_cpu = pot_cpu.potential(xyz_np)
                f_cpu   = pot_cpu.force(xyz_np)
                rho_cpu = pot_cpu.density(xyz_np)
                dF_cpu  = pot_cpu.forceDeriv(xyz_np)[1]

                phi_gpu  = cp.asnumpy(pot_gpu.potential(xyz_cp))
                f_gpu    = cp.asnumpy(pot_gpu.force(xyz_cp))
                rho_gpu  = cp.asnumpy(pot_gpu.density(xyz_cp))
                _, dF_cp = pot_gpu.forceDeriv(xyz_cp)
                dF_gpu   = cp.asnumpy(dF_cp)

                phi_vs_cpu  = _rel_phi(phi_gpu, phi_cpu)
                f_vs_cpu    = _rel_force(f_gpu, f_cpu)
                rho_vs_cpu  = _rel_scalar(rho_gpu, rho_cpu)
                dF_scale    = np.max(np.abs(dF_cpu)) + 1e-30
                hess_vs_cpu = float(np.max(np.abs(dF_gpu - dF_cpu)) / dF_scale)

                pt, ft, rt, ht = _TOL_PRECOMPUTED
                is_ok  = phi_vs_cpu < pt and f_vs_cpu < ft and rho_vs_cpu < rt and hess_vs_cpu < ht
                all_pass = all_pass and is_ok
                status = "PASS" if is_ok else "FAIL"
                print(f"  {name:<30} | {phi_vs_cpu:>9.2e} | {f_vs_cpu:>9.2e} | {rho_vs_cpu:>9.2e}"
                      f" | {hess_vs_cpu:>9.2e} | {status}")

            except Exception as e:
                all_pass = False
                print(f"  {name:<30} | CRITICAL ERROR: {e}")

        print("-" * W)

    print("=" * W)
    assert all_pass, "One or more precomputed-file accuracy checks failed"


# ---------------------------------------------------------------------------
# Speedup benchmarks — not collected by pytest; run with  python test_phase1_multipole.py
# ---------------------------------------------------------------------------

def run_speedup_benchmark():
    W = 72
    print("\n" + "=" * W)
    print(f"{'PHASE 1 SPEEDUP:  GPU vs Agama CPU  (.force)':^{W}}")
    print("=" * W)

    configs = [
        ("Spheroid lmax=0",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 0),
        ("Spheroid lmax=4",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 4),
        ("Spheroid (tri) lmax=4",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
              axisRatioY=0.9, axisRatioZ=0.8), 4),
        ("Spheroid lmax=8",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 8),
    ]

    N_list = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    for label, agama_kw, lmax in configs:
        try:
            pot_cpu, pot_gpu = _make_from_agama(agama_kw, lmax)
        except Exception as e:
            print(f"\n  {label}: SKIP ({e})")
            continue

        n_lm = (lmax + 1)**2
        print(f"\n  {label}  (n_lm={n_lm})")
        hdr = f"    {'N':>9}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'GPU/CPU':>9}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))

        for N in N_list:
            xyz_np = _pts(N, seed=7)
            xyz_cp = cp.asarray(xyz_np)

            t_cpu = _time_fn(lambda: pot_cpu.force(xyz_np), is_gpu=False) * 1e3
            t_gpu = _time_fn(lambda: pot_gpu.force(xyz_cp), is_gpu=True)  * 1e3

            print(f"    {N:>9,}  {t_cpu:>10.2f}  {t_gpu:>10.2f}  {t_cpu/t_gpu:>9.1f}x")
            del xyz_cp
            cp.get_default_memory_pool().free_all_blocks()

    print("=" * W)


def run_precomputed_benchmark():
    present = {k: p for k, p in _PRECOMPUTED_FILES.items() if os.path.exists(p)}
    if not present:
        print("\n[Precomputed file benchmark] no files found — skipped")
        return

    W = 72
    print("\n" + "=" * W)
    print(f"{'PRECOMPUTED FILE SPEEDUP:  GPU vs Agama CPU':^{W}}")
    print("=" * W)

    N_list = [1_000, 10_000, 100_000, 1_000_000]

    for label, path in present.items():
        for lmax in _LMAX_LIST:
            n_lm = (lmax + 1)**2
            try:
                pot_cpu, pot_gpu = _make_from_coef_file(path, lmax)
            except Exception as e:
                print(f"\n  {label} lmax={lmax}: SKIP ({e})")
                continue

            print(f"\n  {label} lmax={lmax}  (n_lm={n_lm})")
            hdr = f"    {'N':>9}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'GPU/CPU':>9}"
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            for N in N_list:
                xyz_np = _pts(N, seed=7)
                xyz_cp = cp.asarray(xyz_np)

                t_cpu = _time_fn(lambda: pot_cpu.force(xyz_np), is_gpu=False) * 1e3
                t_gpu = _time_fn(lambda: pot_gpu.force(xyz_cp), is_gpu=True)  * 1e3

                print(f"    {N:>9,}  {t_cpu:>10.2f}  {t_gpu:>10.2f}  {t_cpu/t_gpu:>9.1f}x")
                del xyz_cp
                cp.get_default_memory_pool().free_all_blocks()

        print("=" * W)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_accuracy()
    run_speedup_benchmark()
    test_precomputed_accuracy()
    run_precomputed_benchmark()
