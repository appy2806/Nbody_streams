#!/usr/bin/env python
"""
test_api.py -- Time the tree_gravity_gpu() Python helper for 1M particles
and compare against the low-level ctypes pipeline (treecode.py style).

Goal: see if the helper overhead is now minimal after removing redundant syncs.
"""

import time
import numpy as np
import cupy as cp
import ctypes

from nbody_streams.tree_gpu import tree_gravity_gpu
from nbody_streams.tree_gpu._force import _lib, _flt, _gpu_ptr
from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu


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


def time_low_level(pos, mass, n, eps, theta, nleaf=64, ncrit=64, level_split=5, n_runs=5):
    """Time using raw ctypes calls, no extraction."""
    x   = cp.ascontiguousarray(pos[:, 0])
    y   = cp.ascontiguousarray(pos[:, 1])
    z   = cp.ascontiguousarray(pos[:, 2])
    eps_arr = cp.full(n, float(eps), dtype=cp.float32)

    start = cp.cuda.Event()
    stop  = cp.cuda.Event()

    # Warmup
    tree = _lib.tree_new(_flt(eps), _flt(theta))
    _lib.tree_set_verbose(tree, 0)
    _lib.tree_alloc(tree, n)
    _lib.tree_set_pos_mass_eps_device(tree, n,
        _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z), _gpu_ptr(mass), _gpu_ptr(eps_arr))
    _lib.tree_build(tree, nleaf)
    _lib.tree_compute_multipoles(tree)
    _lib.tree_make_groups(tree, level_split, ncrit)
    _lib.tree_compute_forces(tree)
    cp.cuda.runtime.deviceSynchronize()
    _lib.tree_delete(tree)

    # Timed runs (CUDA events, no Python-side syncs between stages)
    times = []
    for _ in range(n_runs):
        tree = _lib.tree_new(_flt(eps), _flt(theta))
        _lib.tree_set_verbose(tree, 0)
        _lib.tree_alloc(tree, n)
        _lib.tree_set_pos_mass_eps_device(tree, n,
            _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z), _gpu_ptr(mass), _gpu_ptr(eps_arr))

        start.record()
        _lib.tree_build(tree, nleaf)
        _lib.tree_compute_multipoles(tree)
        _lib.tree_make_groups(tree, level_split, ncrit)
        _lib.tree_compute_forces(tree)
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
        _lib.tree_delete(tree)

    return times


def time_low_level_with_extract(pos, mass, n, eps, theta, nleaf=64, ncrit=64, level_split=5, n_runs=5):
    """Time using raw ctypes calls INCLUDING extraction."""
    x   = cp.ascontiguousarray(pos[:, 0])
    y   = cp.ascontiguousarray(pos[:, 1])
    z   = cp.ascontiguousarray(pos[:, 2])
    eps_arr = cp.full(n, float(eps), dtype=cp.float32)
    ax  = cp.empty(n, dtype=cp.float32)
    ay  = cp.empty(n, dtype=cp.float32)
    az  = cp.empty(n, dtype=cp.float32)
    phi = cp.empty(n, dtype=cp.float32)

    start = cp.cuda.Event()
    stop  = cp.cuda.Event()

    # Warmup
    tree = _lib.tree_new(_flt(eps), _flt(theta))
    _lib.tree_set_verbose(tree, 0)
    _lib.tree_alloc(tree, n)
    _lib.tree_set_pos_mass_eps_device(tree, n,
        _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z), _gpu_ptr(mass), _gpu_ptr(eps_arr))
    _lib.tree_build(tree, nleaf)
    _lib.tree_compute_multipoles(tree)
    _lib.tree_make_groups(tree, level_split, ncrit)
    _lib.tree_compute_forces(tree)
    _lib.tree_get_acc_device(tree, n, _gpu_ptr(ax), _gpu_ptr(ay), _gpu_ptr(az), _gpu_ptr(phi))
    cp.cuda.runtime.deviceSynchronize()
    _lib.tree_delete(tree)

    # Timed runs
    times = []
    for _ in range(n_runs):
        tree = _lib.tree_new(_flt(eps), _flt(theta))
        _lib.tree_set_verbose(tree, 0)
        _lib.tree_alloc(tree, n)
        _lib.tree_set_pos_mass_eps_device(tree, n,
            _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z), _gpu_ptr(mass), _gpu_ptr(eps_arr))

        start.record()
        _lib.tree_build(tree, nleaf)
        _lib.tree_compute_multipoles(tree)
        _lib.tree_make_groups(tree, level_split, ncrit)
        _lib.tree_compute_forces(tree)
        _lib.tree_get_acc_device(tree, n, _gpu_ptr(ax), _gpu_ptr(ay), _gpu_ptr(az), _gpu_ptr(phi))
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
        _lib.tree_delete(tree)

    return times


def time_helper_api(pos, mass, eps, theta, n_runs=5):
    """Time the tree_gravity_gpu() Python helper function."""
    # Warmup
    tree_gravity_gpu(pos, mass, eps=eps, theta=theta, G=1.0)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        acc, phi = tree_gravity_gpu(pos, mass, eps=eps, theta=theta, G=1.0)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return times


if __name__ == "__main__":
    n = 1_000_000
    eps = 0.05
    theta = 0.75
    n_runs = 5

    pos, mass = generate_plummer(n, seed=42)
    cp.cuda.runtime.deviceSynchronize()

    print("=" * 72)
    print(f"API TIMING COMPARISON  (N = {n:,}, theta = {theta}, eps = {eps})")
    print("=" * 72)

    # 1. Low-level without extraction (like treecode.py)
    t_ll = time_low_level(pos, mass, n, eps, theta, n_runs=n_runs)
    med_ll = np.median(t_ll)
    print(f"\n  Low-level (no extract):   {med_ll:.2f} ms  (median of {n_runs})")

    # 2. Low-level with extraction
    t_lle = time_low_level_with_extract(pos, mass, n, eps, theta, n_runs=n_runs)
    med_lle = np.median(t_lle)
    print(f"  Low-level (+ extract):    {med_lle:.2f} ms  (median of {n_runs})")
    print(f"  Extraction overhead:      {med_lle - med_ll:.2f} ms")

    # 3. Python helper API
    t_api = time_helper_api(pos, mass, eps, theta, n_runs=n_runs)
    med_api = np.median(t_api)
    print(f"  Python helper API:        {med_api:.2f} ms  (median of {n_runs})")
    print(f"  API overhead vs low+ext:  {med_api - med_lle:.2f} ms")

    # 4. Quick accuracy check at 1M
    print(f"\n{'=' * 72}")
    print(f"ACCURACY CHECK  (N = {n:,}, theta = {theta})")
    print(f"{'=' * 72}")

    h_gpu = cp.full(n, eps, dtype=cp.float32)

    print("\n  Computing direct-sum reference...")
    t0 = time.perf_counter()
    acc_ref = compute_nbody_forces_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=1.0,
        return_cupy=True, skip_validation=True,
    )
    phi_ref = compute_nbody_potential_gpu(
        pos, mass, softening=h_gpu,
        precision='float32', kernel='plummer', G=1.0,
        return_cupy=True, skip_validation=True,
    )
    cp.cuda.runtime.deviceSynchronize()
    t_direct = (time.perf_counter() - t0) * 1000
    print(f"  Direct-sum time: {t_direct:.1f} ms")

    print("  Computing tree-code...")
    acc_tree, phi_tree = tree_gravity_gpu(pos, mass, eps=eps, theta=theta, G=1.0)
    cp.cuda.runtime.deviceSynchronize()

    # Acceleration error
    diff = cp.linalg.norm(acc_tree - acc_ref, axis=1)
    amag = cp.linalg.norm(acc_ref, axis=1)
    rel_err = diff / amag.clip(1e-20)

    print(f"\n  Acceleration relative error:")
    print(f"    mean:   {float(rel_err.mean()):.4e}")
    print(f"    median: {float(cp.median(rel_err)):.4e}")
    print(f"    p95:    {float(cp.percentile(rel_err, 95)):.4e}")
    print(f"    max:    {float(rel_err.max()):.4e}")

    # Potential error
    phi_err = cp.abs(phi_tree - phi_ref) / cp.abs(phi_ref).clip(1e-20)
    print(f"\n  Potential relative error:")
    print(f"    mean:   {float(phi_err.mean()):.4e}")
    print(f"    median: {float(cp.median(phi_err)):.4e}")
    print(f"    max:    {float(phi_err.max()):.4e}")

    # Speedup
    print(f"\n  Speedup: {t_direct / med_api:.1f}x  (tree vs direct-sum)")
