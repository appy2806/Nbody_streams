"""
test_limits.py -- Index-packing limits, build capacities and the memory guard.

  1. Particle4 (29-bit index, 3-bit octant, float and double) and CellData
     (32-bit parent, 29-bit first child, 1-8 children) round-trip values up
     to their limits (2^29 - 1 particles, 2^29 - 1 cells): a device self-test
     that counts failures (no device assert), so it runs in process.
  2. The Python limits match the library's.
  3. Realistic trees fit the initial build capacities (no grow-and-rebuild).
  4. The memory guard rejects N above the limit with ValueError, and its
     per-particle model is not below the measured tree footprint.
"""

import numpy as np
import cupy as cp
import pytest

from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU
from nbody_streams.tree_gpu import _force


def test_packing_roundtrip_selftest():
    assert _force._lib.tree_selftest_packing() == 0


def test_limits_match_library():
    assert _force._IDX_CAP == _force._lib.tree_max_particles() == 1 << 29
    assert _force._lib.tree_max_cells() == (1 << 29) - 1


def _positions(kind, n, rng):
    if kind == "plummer":
        r = 1.0 / np.sqrt(rng.uniform(1e-4, 1.0, n) ** (-2.0 / 3.0) - 1.0)
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None] * r[:, None]
    if kind == "cube":
        return rng.uniform(-1.0, 1.0, (n, 3))
    if kind == "filament":           # straight, width 1e-4 of its length
        t = rng.uniform(-1.0, 1.0, n)
        return np.stack([t, 0.3 * t, -0.2 * t], 1) + 1e-4 * rng.normal(size=(n, 3))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["plummer", "cube", "filament"])
def test_realistic_trees_fit_initial_capacity(kind):
    n = 100_000
    rng = np.random.default_rng(5)
    x = cp.asarray(_positions(kind, n, rng))
    m = cp.full(n, 1.0 / n, dtype=cp.float32)
    with TreeGPU(n) as t:
        cap0 = t.build_stats()
        for nleaf in (16, 64):
            tree_gravity_gpu(x, m, 0.01, G=1.0, nleaf=nleaf, tree=t)
            s = t.build_stats()
            for k in ("cell_capacity", "node_capacity", "work_capacity"):
                assert s[k] == cap0[k], (kind, nleaf, s, cap0)   # nothing grew
            # and with the margin the sizes were chosen for (>= 1.5x)
            assert 1.5 * s["cells"] <= s["cell_capacity"], s
            assert 1.5 * s["nodes"] <= s["node_capacity"], s
            assert 1.5 * s["work_items"] <= s["work_capacity"], s
            assert 0 < s["leaves"] <= s["cells"] and s["levels"] > 1


def test_guard_rejects_too_many_particles():
    with pytest.raises(ValueError, match="exceeds the GPU tree limit"):
        TreeGPU(_force._IDX_CAP + 1)
    assert 0 < _force._max_n() <= _force._IDX_CAP
    assert _force._max_n(1000) < _force._max_n()


def _free():
    cp.cuda.Device().synchronize()
    return cp.cuda.runtime.memGetInfo()[0]


def test_guard_model_covers_measured_footprint():
    # Tree bytes per particle from the slope between two N (cancels the
    # fixed walk pool and minimum sizes), nleaf=16 (the most cells).
    rng = np.random.default_rng(2)
    pool = cp.get_default_memory_pool()
    per = []
    for n in (500_000, 1_500_000):
        x = cp.asarray(_positions("plummer", n, rng))
        m = cp.full(n, 1.0 / n, dtype=cp.float32)
        best = None
        for _ in range(2):
            pool.free_all_blocks()
            f0 = _free()
            p0 = pool.total_bytes()
            t = TreeGPU(n)
            tree_gravity_gpu(x, m, 0.01, G=1.0, nleaf=16, tree=t)
            used = f0 - _free() - (pool.total_bytes() - p0)   # tree only
            t.close()
            best = used if best is None else min(best, used)
        per.append(best)
        del x, m
    slope = (per[1] - per[0]) / 1_000_000
    assert 120 < slope <= _force._TREE_BYTES_PER_PARTICLE, slope
