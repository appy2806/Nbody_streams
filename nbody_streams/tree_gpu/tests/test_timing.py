#!/usr/bin/env python
"""
test_timing.py -- Component-level timing for the GPU tree-code pipeline.

Times each stage: data load, tree build, multipoles, grouping, force eval, extraction.
"""

import time
import ctypes
import numpy as np
import cupy as cp

# Re-use the low-level ctypes bindings from _force
from nbody_streams.tree_gpu._force import _lib, _flt, _gpu_ptr, _ResInteractions

# ─── Plummer sphere generator ───────────────────────────────────────────────
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

def time_pipeline(n, eps=0.05, theta=0.75, nleaf=64, ncrit=64, level_split=5, n_runs=3):
    """Time each component of the GPU tree-code pipeline individually."""
    pos, mass = generate_plummer(n, seed=42)
    pos = cp.ascontiguousarray(pos, dtype=cp.float32)
    mass = cp.ascontiguousarray(mass, dtype=cp.float32)

    x   = cp.ascontiguousarray(pos[:, 0])
    y   = cp.ascontiguousarray(pos[:, 1])
    z   = cp.ascontiguousarray(pos[:, 2])
    eps_arr = cp.full(n, float(eps), dtype=cp.float32)

    # Output buffers (reuse across runs)
    ax  = cp.empty(n, dtype=cp.float32)
    ay  = cp.empty(n, dtype=cp.float32)
    az  = cp.empty(n, dtype=cp.float32)
    phi = cp.empty(n, dtype=cp.float32)

    timings = {
        'set_data':    [],
        'build_tree':  [],
        'multipoles':  [],
        'make_groups': [],
        'forces':      [],
        'extract':     [],
        'total':       [],
    }

    # Warmup run (includes alloc)
    tree = _lib.tree_new(_flt(eps), _flt(theta))
    _lib.tree_set_verbose(tree, 0)
    _lib.tree_alloc(tree, n)
    _lib.tree_set_pos_mass_eps_device(
        tree, n,
        _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z),
        _gpu_ptr(mass), _gpu_ptr(eps_arr),
    )
    _lib.tree_build(tree, nleaf)
    _lib.tree_compute_multipoles(tree)
    _lib.tree_make_groups(tree, level_split, ncrit)
    _lib.tree_compute_forces(tree)
    _lib.tree_get_acc_device(tree, n, _gpu_ptr(ax), _gpu_ptr(ay), _gpu_ptr(az), _gpu_ptr(phi))
    cp.cuda.runtime.deviceSynchronize()
    _lib.tree_delete(tree)

    # Timed runs
    for _ in range(n_runs):
        tree = _lib.tree_new(_flt(eps), _flt(theta))
        _lib.tree_set_verbose(tree, 1)
        _lib.tree_alloc(tree, n)
        cp.cuda.runtime.deviceSynchronize()

        t_total_start = time.perf_counter()

        # 1. Set data (SoA → AoS copy on GPU, eps packed into ptclVel.x)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_set_pos_mass_eps_device(
            tree, n,
            _gpu_ptr(x), _gpu_ptr(y), _gpu_ptr(z),
            _gpu_ptr(mass), _gpu_ptr(eps_arr),
        )
        cp.cuda.runtime.deviceSynchronize()
        timings['set_data'].append(time.perf_counter() - t0)

        # 2. Build tree (sort + octree construction)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_build(tree, nleaf)
        cp.cuda.runtime.deviceSynchronize()
        timings['build_tree'].append(time.perf_counter() - t0)

        # 3. Compute multipoles (monopole + quadrupole moments)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_compute_multipoles(tree)
        cp.cuda.runtime.deviceSynchronize()
        timings['multipoles'].append(time.perf_counter() - t0)

        # 4. Make groups (Hilbert-curve sort + group formation)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_make_groups(tree, level_split, ncrit)
        cp.cuda.runtime.deviceSynchronize()
        timings['make_groups'].append(time.perf_counter() - t0)

        # 5. Compute forces (tree walk + P-P/P-C evaluation)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_compute_forces(tree)
        cp.cuda.runtime.deviceSynchronize()
        timings['forces'].append(time.perf_counter() - t0)

        # 6. Extract results (unsort to original order)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        _lib.tree_get_acc_device(
            tree, n,
            _gpu_ptr(ax), _gpu_ptr(ay), _gpu_ptr(az), _gpu_ptr(phi),
        )
        cp.cuda.runtime.deviceSynchronize()
        timings['extract'].append(time.perf_counter() - t0)

        timings['total'].append(time.perf_counter() - t_total_start)

        _lib.tree_delete(tree)

    return timings


def print_timings(n, timings):
    print(f"\n{'='*72}")
    print(f"COMPONENT TIMING BREAKDOWN  (N = {n:,})")
    print(f"{'='*72}")

    total_med = np.median(timings['total']) #* 1000
    for stage in ['set_data', 'build_tree', 'multipoles', 'make_groups', 'forces', 'extract']:
        t = np.array(timings[stage]) #* 1000  # ms
        med = np.median(t)
        pct = med / total_med * 100
        label = {
            'set_data':    'Data load (SoA→AoS)',
            'build_tree':  'Build tree (sort+octree)',
            'multipoles':  'Compute multipoles',
            'make_groups': 'Make groups (Hilbert sort)',
            'forces':      'Compute forces (tree walk)',
            'extract':     'Extract results (unsort)',
        }[stage]
        print(f"  {label:35s} {med:8.6f} s  ({pct:5.1f}%)")

    print(f"  {'─'*55}")
    print(f"  {'Total':35s} {total_med:8.6f} s")
    print(f"  {'Throughput':35s} {n/total_med:.2e} ptcl/s")
    print()


if __name__ == "__main__":
    for n in [100_000, 500_000, 1_000_000, 5_000_000, 7_500_000]:
    # for n in [128, 512, 1024, 5096, 10240, 25600, 49920, 100096, 256000, 512000, 1000064, 2000128, 4000256, 7_500_000]:
        timings = time_pipeline(n, n_runs=1)
        print_timings(n, timings)
