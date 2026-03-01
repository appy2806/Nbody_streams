#!/usr/bin/env python
"""
test_tree.py  --  Accuracy and performance benchmarks for tree_gravity_gpu.

Compares the GPU tree code against direct-sum N-body from
``nbody_streams`` and measures timing across different N and theta values.
"""

import time
import numpy as np
import cupy as cp

from nbody_streams.tree_gpu import tree_gravity_gpu
from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu


# =========================================================================
# Plummer sphere generator (N-body units: G=1, M_total=1)
# =========================================================================
def generate_plummer(n, seed=19810614):
    """Plummer model ICs matching the C++ binary (double precision + R<100 rejection)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scale = 3.0 * np.pi / 16.0

    pos = np.empty((n, 3), dtype=np.float64)
    i = 0
    while i < n:
        batch = min(n - i, max(n // 10, 1024))
        X1 = rng.random(batch)
        X2 = rng.random(batch)
        X3 = rng.random(batch)
        R = 1.0 / np.sqrt(np.maximum(X1 ** (-2.0/3.0) - 1.0, 0.0))
        valid = R < 100.0
        R, X2, X3 = R[valid], X2[valid], X3[valid]
        Z = (1.0 - 2.0*X2)*R
        X = np.sqrt(np.maximum(R*R - Z*Z, 0.0)) * np.cos(2.0*np.pi*X3)
        Y = np.sqrt(np.maximum(R*R - Z*Z, 0.0)) * np.sin(2.0*np.pi*X3)
        take = min(len(R), n - i)
        pos[i:i+take, 0] = X[:take] * scale
        pos[i:i+take, 1] = Y[:take] * scale
        pos[i:i+take, 2] = Z[:take] * scale
        i += take

    M = cp.full(n, 1.0/n, dtype=cp.float32)
    return cp.asarray(pos, dtype=cp.float32), M

# =========================================================================
# Error metrics
# =========================================================================
def error_report(label, ref, test):
    """Print relative-error statistics between ref and test arrays."""
    ref_np  = cp.asnumpy(ref)
    test_np = cp.asnumpy(test)

    # Per-particle relative error in acceleration magnitude
    if ref_np.ndim == 2:
        ref_mag  = np.linalg.norm(ref_np, axis=1)
        diff_mag = np.linalg.norm(test_np - ref_np, axis=1)
    else:
        ref_mag  = np.abs(ref_np)
        diff_mag = np.abs(test_np - ref_np)

    mask = ref_mag > 0
    rel = diff_mag[mask] / ref_mag[mask]

    p50 = np.median(rel)
    p95 = np.percentile(rel, 95)
    p99 = np.percentile(rel, 99)
    mean = np.mean(rel)
    mx   = np.max(rel)

    print(f"  {label:30s}  mean={mean:.4e}  med={p50:.4e}  "
          f"p95={p95:.4e}  p99={p99:.4e}  max={mx:.4e}")


# =========================================================================
# 1.  Accuracy test
# =========================================================================
def test_accuracy():
    print("=" * 72)
    print("ACCURACY TEST:  GPU tree-code  vs  direct-sum (nbody_streams)")
    print("=" * 72)

    n   = 1_000_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=123)
    h_gpu = cp.full(n, 0.05, dtype=cp.float32)

    print(f"\nPlummer sphere: N={n},  eps={eps},  G=1\n")

    # ---------- reference: direct-sum ----------
    print("Computing direct-sum reference (float32) ...")
    
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        acc_ref = compute_nbody_forces_gpu(
            pos, mass, softening=h_gpu,
            precision='float32', kernel='plummer', G=1.0,
            return_cupy=True, skip_validation=True
        )
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    t_ref = np.median(times)
    print(f"  direct-sum time: {t_ref*1000:.3f} ms")

    phi_ref = compute_nbody_potential_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=1.0,
        return_cupy=True, skip_validation=True
    )
    cp.cuda.runtime.deviceSynchronize()

    # ---------- tree-code with several theta values ----------
    for theta in [0.75]:
    # for theta in [0.3, 0.5, 0.75, 1.0]:
        print(f"\n--- theta = {theta} ---")
        times = []
        for _ in range(1):
            t0 = time.perf_counter()
            acc_tree, phi_tree = tree_gravity_gpu(
                pos, mass, eps=eps, theta=theta,
                nleaf=64, ncrit=64, level_split=5, G=1.0,
                verbose=False, # Set to True to print timing and interaction counts from the tree library
            )
            times.append(time.perf_counter() - t0)
    
        t_tree = np.median(times)
        print(f"  GPU time: {t_tree*1000:.3f} ms   (speedup vs direct: {t_ref/t_tree:.1f}x)")
        error_report("acceleration |a|", acc_ref, acc_tree)
        error_report("potential phi",    phi_ref, phi_tree)


# =========================================================================
# 2.  Performance benchmark
# =========================================================================
def test_performance():
    print("\n" + "=" * 72)
    print("PERFORMANCE BENCHMARK:  GPU tree-code pipeline timing")
    print("=" * 72)

    eps   = 0.05
    theta = 0.75

    for n in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000]:
        pos, mass = generate_plummer(n, seed=42)
        cp.cuda.runtime.deviceSynchronize()

        # warmup
        _ = tree_gravity_gpu(pos, mass, eps=eps, theta=theta)
        cp.cuda.runtime.deviceSynchronize()

        # timed runs
        n_runs = 3
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            acc, phi = tree_gravity_gpu(pos, mass, eps=eps, theta=theta, verbose=False)
            cp.cuda.runtime.deviceSynchronize()
            times.append(time.perf_counter() - t0)

        avg = np.mean(times)
        print(f"  N={n:>10,d}   avg={avg:.4f} s   ({n/avg:.2e} ptcl/s)")


# =========================================================================
# 3.  Quick sanity check
# =========================================================================
def test_sanity():
    """Run with a small N and check that forces point inward."""
    print("\n" + "=" * 72)
    print("SANITY CHECK:  forces point inward for a Plummer sphere")
    print("=" * 72)

    n   = 10_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=99)

    acc, phi = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    # Radial acceleration should be negative (pointing inward)
    r_hat = pos / cp.linalg.norm(pos, axis=1, keepdims=True).clip(1e-10)
    a_rad = cp.sum(acc * r_hat, axis=1)

    frac_inward = float((a_rad < 0).sum()) / n
    mean_phi    = float(phi.mean())

    print(f"  Fraction with a_r < 0 (inward): {frac_inward:.4f}  (expect ~1.0)")
    print(f"  Mean potential:                  {mean_phi:.4f}  (expect < 0)")

    assert frac_inward > 0.95, f"Most forces should point inward, got {frac_inward}"
    assert mean_phi < 0,       f"Potential should be negative, got {mean_phi}"
    print("  PASSED")


# =========================================================================
if __name__ == "__main__":
    test_sanity()
    test_accuracy()
    test_performance()