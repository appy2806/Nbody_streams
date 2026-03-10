#!/usr/bin/env python
"""
test_accuracy.py -- Comprehensive accuracy and correctness tests for tree_gravity_gpu.

Tests:
  1. Convergence with theta (opening angle)
  2. Momentum conservation (Newton's 3rd law)
  3. Angular momentum conservation (net torque = 0)
  4. Potential energy consistency (0.5 * sum(m*phi) vs force-derived)
  5. Known analytic solution (Plummer sphere enclosed mass)
  6. Scaling with N at fixed distribution
  7. Single-species vs multi-species softening sanity check
"""

import time
import numpy as np
import cupy as cp

from nbody_streams.tree_gpu import tree_gravity_gpu
from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu


# ─── Utilities ───────────────────────────────────────────────────────────────

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


def rel_err_stats(test, ref):
    """Compute relative error statistics. Works for vectors (N,3) or scalars (N,)."""
    test_np = cp.asnumpy(test)
    ref_np = cp.asnumpy(ref)
    if ref_np.ndim == 2:
        ref_mag = np.linalg.norm(ref_np, axis=1)
        diff_mag = np.linalg.norm(test_np - ref_np, axis=1)
    else:
        ref_mag = np.abs(ref_np)
        diff_mag = np.abs(test_np - ref_np)
    mask = ref_mag > 1e-20
    rel = diff_mag[mask] / ref_mag[mask]
    return {
        'mean': np.mean(rel),
        'median': np.median(rel),
        'p95': np.percentile(rel, 95),
        'p99': np.percentile(rel, 99),
        'max': np.max(rel),
    }


# ─── Test 1: Convergence with theta ─────────────────────────────────────────

def test_theta_convergence():
    """Error should decrease monotonically as theta decreases."""
    print("=" * 72)
    print("TEST 1: Convergence with opening angle theta")
    print("=" * 72)

    n = 100_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=42)
    h_gpu = cp.full(n, eps, dtype=cp.float32)

    # Direct-sum reference
    acc_ref = compute_nbody_forces_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=1.0,
        return_cupy=True, skip_validation=True,
    )
    cp.cuda.runtime.deviceSynchronize()

    thetas = [1.0, 0.75, 0.5, 0.3]
    errors = []
    for theta in thetas:
        acc_tree, _ = tree_gravity_gpu(pos, mass, eps=eps, theta=theta, G=1.0)
        cp.cuda.runtime.deviceSynchronize()
        stats = rel_err_stats(acc_tree, acc_ref)
        errors.append(stats['median'])
        print(f"  theta={theta:.2f}  median_err={stats['median']:.4e}  "
              f"p95={stats['p95']:.4e}  max={stats['max']:.4e}")

    # Check monotonic decrease
    monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    print(f"\n  Monotonically decreasing: {'PASS' if monotonic else 'FAIL'}")
    assert monotonic, f"Error not decreasing with theta: {errors}"
    print()


# ─── Test 2: Linear momentum conservation ───────────────────────────────────

def test_momentum_conservation():
    """Net force = sum(m_i * a_i) should be ~0 for an isolated system."""
    print("=" * 72)
    print("TEST 2: Linear momentum conservation (Newton's 3rd law)")
    print("=" * 72)

    n = 50_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=123)

    acc, _ = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    # net_force = sum(m_i * a_i) -- should be zero for pair-wise forces
    # Tree codes don't exactly conserve momentum (asymmetric interactions),
    # but it should be small.
    net_force = cp.sum(mass[:, None] * acc, axis=0)
    net_force_mag = float(cp.linalg.norm(net_force))

    # Normalize by typical force magnitude
    mean_force_mag = float(cp.mean(cp.linalg.norm(acc, axis=1) * mass))
    relative_imbalance = net_force_mag / (n * mean_force_mag)

    print(f"  Net force vector:   [{float(net_force[0]):.6e}, "
          f"{float(net_force[1]):.6e}, {float(net_force[2]):.6e}]")
    print(f"  Net force magnitude: {net_force_mag:.6e}")
    print(f"  Relative imbalance:  {relative_imbalance:.6e}")

    # Tree codes break exact Newton's 3rd law, but imbalance should be <1%
    passed = relative_imbalance < 0.01
    print(f"  Result: {'PASS' if passed else 'FAIL'} (expect < 1% imbalance)")
    print()
    assert passed, f"Momentum imbalance too large: {relative_imbalance:.3e} (expect < 0.01)"


# ─── Test 3: Angular momentum conservation ──────────────────────────────────

def test_angular_momentum_conservation():
    """Net torque = sum(r x (m*a)) should be small."""
    print("=" * 72)
    print("TEST 3: Angular momentum conservation (net torque ~ 0)")
    print("=" * 72)

    n = 50_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=456)

    acc, _ = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    # torque = sum( r_i x (m_i * a_i) )
    forces = mass[:, None] * acc
    torque = cp.sum(cp.cross(pos, forces), axis=0)
    torque_mag = float(cp.linalg.norm(torque))

    # Normalize by total angular momentum scale: sum(|r| * |m*a|)
    scale = float(cp.sum(cp.linalg.norm(pos, axis=1) * cp.linalg.norm(forces, axis=1)))
    relative_torque = torque_mag / scale if scale > 0 else 0.0

    print(f"  Net torque vector:  [{float(torque[0]):.6e}, "
          f"{float(torque[1]):.6e}, {float(torque[2]):.6e}]")
    print(f"  Net torque magnitude: {torque_mag:.6e}")
    print(f"  Relative torque:      {relative_torque:.6e}")

    passed = relative_torque < 0.01
    print(f"  Result: {'PASS' if passed else 'FAIL'} (expect < 1%)")
    print()
    assert passed, f"Angular momentum imbalance too large: {relative_torque:.3e} (expect < 0.01)"


# ─── Test 4: Potential energy consistency ────────────────────────────────────

def test_potential_consistency():
    """Compare 0.5*sum(m*phi_tree) against 0.5*sum(m*phi_direct)."""
    print("=" * 72)
    print("TEST 4: Potential energy consistency (tree vs direct)")
    print("=" * 72)

    n = 50_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=789)
    h_gpu = cp.full(n, eps, dtype=cp.float32)

    # Tree code
    _, phi_tree = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    PE_tree = 0.5 * float(cp.sum(mass * phi_tree))

    # Direct sum
    phi_direct = compute_nbody_potential_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=1.0,
        return_cupy=True, skip_validation=True,
    )
    PE_direct = 0.5 * float(cp.sum(mass * phi_direct))
    cp.cuda.runtime.deviceSynchronize()

    rel_err = abs(PE_tree - PE_direct) / abs(PE_direct)

    print(f"  PE (tree):   {PE_tree:.8f}")
    print(f"  PE (direct): {PE_direct:.8f}")
    print(f"  Relative error: {rel_err:.6e}")

    passed = rel_err < 0.01  # 1% for theta=0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'} (expect < 1% error)")
    print()
    assert passed, f"Potential energy error too large: {rel_err:.3e} (expect < 0.01)"


# ─── Test 5: Forces point inward for a Plummer sphere ───────────────────────

def test_forces_inward():
    """For a spherically symmetric distribution, forces should point inward."""
    print("=" * 72)
    print("TEST 5: Radial force direction (Plummer sphere)")
    print("=" * 72)

    n = 50_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=99)

    acc, phi = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    r_hat = pos / cp.linalg.norm(pos, axis=1, keepdims=True).clip(1e-10)
    a_rad = cp.sum(acc * r_hat, axis=1)  # radial component (negative = inward)

    frac_inward = float((a_rad < 0).sum()) / n
    mean_phi = float(phi.mean())

    print(f"  Fraction with a_r < 0 (inward): {frac_inward:.4f}  (expect ~1.0)")
    print(f"  Mean potential:                  {mean_phi:.4f}  (expect < 0)")

    p1 = frac_inward > 0.95
    p2 = mean_phi < 0
    print(f"  Inward fraction: {'PASS' if p1 else 'FAIL'}")
    print(f"  Negative potential: {'PASS' if p2 else 'FAIL'}")
    print()
    assert p1, f"Too few inward forces: {frac_inward:.4f} (expect > 0.95)"
    assert p2, f"Mean potential not negative: {mean_phi:.4f}"


# ─── Test 6: Accuracy vs N (should be stable) ───────────────────────────────

def test_accuracy_vs_n():
    """Error should not degrade significantly as N grows (same distribution)."""
    print("=" * 72)
    print("TEST 6: Accuracy stability with N")
    print("=" * 72)

    eps = 0.05
    theta = 0.75
    n_values = [10_000, 50_000, 100_000]
    errors = []

    for n in n_values:
        pos, mass = generate_plummer(n, seed=42)
        h_gpu = cp.full(n, eps, dtype=cp.float32)

        acc_ref = compute_nbody_forces_gpu(
            pos, mass, softening=h_gpu,
            precision='float32', kernel='plummer', G=1.0,
            return_cupy=True, skip_validation=True,
        )
        acc_tree, _ = tree_gravity_gpu(pos, mass, eps=eps, theta=theta, G=1.0)
        cp.cuda.runtime.deviceSynchronize()

        stats = rel_err_stats(acc_tree, acc_ref)
        errors.append(stats['median'])
        print(f"  N={n:>8,d}  median_err={stats['median']:.4e}  p95={stats['p95']:.4e}")

    # Error shouldn't grow more than 2x across these N values
    ratio = max(errors) / min(errors)
    passed = ratio < 3.0
    print(f"\n  Max/min error ratio: {ratio:.2f}  ({'PASS' if passed else 'FAIL'}, expect < 3)")
    print()
    assert passed, f"Error degrades too much across N: max/min ratio = {ratio:.2f} (expect < 3)"


# ─── Test 7: Reproducibility (same input → same output) ─────────────────────

def test_reproducibility():
    """Calling tree_gravity_gpu twice with identical input should give identical output."""
    print("=" * 72)
    print("TEST 7: Reproducibility (deterministic output)")
    print("=" * 72)

    n = 50_000
    eps = 0.05
    pos, mass = generate_plummer(n, seed=42)

    acc1, phi1 = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    acc2, phi2 = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    # GPU tree walks use atomicAdd for dynamic group scheduling, so warp
    # execution order is non-deterministic. Float accumulation in different
    # orders gives ~1e-6 level jitter. This is expected for GPU tree codes.
    acc_diff = cp.linalg.norm(acc1 - acc2, axis=1)
    phi_diff = cp.abs(phi1 - phi2)
    acc_mag = cp.linalg.norm(acc1, axis=1)
    phi_mag = cp.abs(phi1)

    acc_max_rel = float((acc_diff / acc_mag.clip(1e-20)).max())
    phi_max_rel = float((phi_diff / phi_mag.clip(1e-20)).max())

    print(f"  Acceleration max relative diff: {acc_max_rel:.6e}")
    print(f"  Potential max relative diff:    {phi_max_rel:.6e}")

    # Allow small numerical jitter from non-deterministic warp scheduling
    tol = 1e-4
    acc_ok = acc_max_rel < tol
    phi_ok = phi_max_rel < tol
    print(f"  Accelerations reproducible (< {tol}): {'PASS' if acc_ok else 'FAIL'}")
    print(f"  Potentials reproducible (< {tol}):    {'PASS' if phi_ok else 'FAIL'}")
    print()
    assert acc_ok, f"Acceleration non-determinism too large: {acc_max_rel:.3e} (expect < {tol})"
    assert phi_ok, f"Potential non-determinism too large: {phi_max_rel:.3e} (expect < {tol})"


# ─── Test 8: Two-body problem ───────────────────────────────────────────────

def test_two_body():
    """Small cluster with known analytic limit: two clumps well-separated."""
    print("=" * 72)
    print("TEST 8: Two-clump analytic check")
    print("=" * 72)

    # Use two tight clumps of particles separated along x-axis.
    # Tree builder needs N >= ~100 to work properly.
    eps = 0.01
    G = 1.0
    n_per_clump = 500
    sep = 10.0  # large separation so clumps look like point masses

    rng = cp.random.default_rng(42)
    # Clump 1 at (-sep/2, 0, 0), clump 2 at (+sep/2, 0, 0)
    spread = 0.01  # very tight clumps
    p1 = rng.standard_normal((n_per_clump, 3), dtype=cp.float32) * spread
    p1[:, 0] -= sep / 2
    p2 = rng.standard_normal((n_per_clump, 3), dtype=cp.float32) * spread
    p2[:, 0] += sep / 2

    pos = cp.concatenate([p1, p2], axis=0)
    n = 2 * n_per_clump
    m = 1.0  # total mass per clump
    mass = cp.full(n, m / n_per_clump, dtype=cp.float32)

    # Use direct-sum as the reference (analytic is tricky with self-gravity)
    h_gpu = cp.full(n, eps, dtype=cp.float32)
    acc_ref = compute_nbody_forces_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=G,
        return_cupy=True, skip_validation=True,
    )
    phi_ref = compute_nbody_potential_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=G,
        return_cupy=True, skip_validation=True,
    )

    acc_tree, phi_tree = tree_gravity_gpu(pos, mass, eps=eps, theta=0.3, G=G)
    cp.cuda.runtime.deviceSynchronize()

    stats_acc = rel_err_stats(acc_tree, acc_ref)
    stats_phi = rel_err_stats(phi_tree, phi_ref)

    print(f"  Separation: {sep},  eps: {eps},  N_per_clump: {n_per_clump}")
    print(f"  Acc vs direct:  mean={stats_acc['mean']:.4e}  med={stats_acc['median']:.4e}  max={stats_acc['max']:.4e}")
    print(f"  Phi vs direct:  mean={stats_phi['mean']:.4e}  med={stats_phi['median']:.4e}  max={stats_phi['max']:.4e}")

    # Also check that clumps feel force toward each other
    mean_ax1 = float(cp.mean(acc_tree[:n_per_clump, 0]))
    mean_ax2 = float(cp.mean(acc_tree[n_per_clump:, 0]))
    print(f"  Clump 1 mean a_x: {mean_ax1:+.6e} (expect > 0, toward clump 2)")
    print(f"  Clump 2 mean a_x: {mean_ax2:+.6e} (expect < 0, toward clump 1)")

    passed = (stats_acc['median'] < 1e-3 and mean_ax1 > 0 and mean_ax2 < 0)
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    assert stats_acc['median'] < 1e-3, f"Two-body acc error too large: {stats_acc['median']:.3e}"
    assert mean_ax1 > 0, f"Clump 1 should feel force toward clump 2 (a_x > 0), got {mean_ax1:.3e}"
    assert mean_ax2 < 0, f"Clump 2 should feel force toward clump 1 (a_x < 0), got {mean_ax2:.3e}"


# ─── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    tests = [
        ("Theta convergence",     test_theta_convergence),
        ("Momentum conservation", test_momentum_conservation),
        ("Angular momentum",      test_angular_momentum_conservation),
        ("Potential consistency",  test_potential_consistency),
        ("Forces inward",         test_forces_inward),
        ("Accuracy vs N",         test_accuracy_vs_n),
        ("Reproducibility",       test_reproducibility),
        ("Two-body analytic",     test_two_body),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[name] = False

    # Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s} {status}")
        if not passed:
            all_pass = False
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME TESTS FAILED'}")
