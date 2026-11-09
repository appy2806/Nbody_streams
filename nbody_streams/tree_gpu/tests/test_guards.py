"""
test_guards.py -- Regression tests for GPU-tree input guards and conventions.

  1. Massless tracer particles give finite forces (no NaN from zero-mass cells).
  2. phi excludes the self-interaction (matches the direct-sum backend).
  3. theta passed with a pre-built TreeGPU handle is honoured.
  4. Non-positive softening raises ValueError.
  5. Results are deterministic across repeated calls (leaf-list race).
  7. Coincident particles (more than nleaf at one point) give correct forces.
  6. run_nbody_gpu_tree evolves an offset, moving cluster like the same
     cluster at rest at the origin (float64 state + recentred tree).
"""

import tempfile

import numpy as np
import cupy as cp
import pytest

from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU, run_nbody_gpu_tree
from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu


def _hernquist(n, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.999, n)
    r = np.minimum(np.sqrt(u) / (1.0 - np.sqrt(u)), 200.0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return r[:, None] * v


def _rel_err(a, b):
    a = cp.asnumpy(a).astype(np.float64)
    b = cp.asnumpy(b).astype(np.float64)
    if a.ndim == 2:
        return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    return np.abs(a - b) / np.abs(b)


def test_massless_tracers_finite_and_accurate():
    n = 20_000
    x = _hernquist(n, seed=3)
    for m in (np.where(np.arange(n) % 2 == 0, 2.0 / n, 0.0),   # interleaved
              np.where(x[:, 0] > 0, 2.0 / n, 0.0)):              # clumped
        eps = np.where(m > 0, 0.01, 0.005)
        acc, phi = tree_gravity_gpu(cp.asarray(x, dtype=cp.float32),
                                    cp.asarray(m, dtype=cp.float32),
                                    cp.asarray(eps, dtype=cp.float32), G=1.0, theta=0.5)
        assert bool(cp.isfinite(acc).all()) and bool(cp.isfinite(phi).all())
        ref = compute_nbody_forces_gpu(x, m, softening=eps, G=1.0, precision="float64",
                                       kernel="plummer", return_cupy=True)
        assert np.percentile(_rel_err(acc, ref), 99) < 5e-3


def test_phi_excludes_self_term():
    n = 20_000
    x = _hernquist(n)
    m = np.full(n, 1.0 / n)
    eps = 0.01                       # self term m/eps = 5e-3, ~1% of |phi|
    _, phi = tree_gravity_gpu(cp.asarray(x, dtype=cp.float32), cp.asarray(m, dtype=cp.float32),
                              eps, G=1.0, theta=0.5)
    ref = compute_nbody_potential_gpu(x, m, softening=eps, G=1.0, precision="float64",
                                      kernel="plummer", return_cupy=True)
    err = (cp.asnumpy(phi) - cp.asnumpy(ref)) / np.abs(cp.asnumpy(ref))
    assert abs(np.median(err)) < 1e-4


def test_theta_with_prebuilt_handle_is_honoured():
    n = 20_000
    pos = cp.asarray(_hernquist(n), dtype=cp.float32)
    mass = cp.full(n, 1.0 / n, dtype=cp.float32)
    ref = compute_nbody_forces_gpu(cp.asnumpy(pos), cp.asnumpy(mass), softening=0.01, G=1.0,
                                   precision="float64", kernel="plummer", return_cupy=True)
    with TreeGPU(n, theta=0.9) as tree:
        a_coarse, _ = tree_gravity_gpu(pos, mass, 0.01, G=1.0, tree=tree)
        a_fine, _ = tree_gravity_gpu(pos, mass, 0.01, G=1.0, theta=0.3, tree=tree)
        assert tree.theta == 0.3
    with TreeGPU(n, theta=0.3) as tree:
        a_fresh, _ = tree_gravity_gpu(pos, mass, 0.01, G=1.0, tree=tree)
    e_coarse = np.median(_rel_err(a_coarse, ref))
    e_fine = np.median(_rel_err(a_fine, ref))
    assert e_fine < 0.5 * e_coarse
    assert np.median(_rel_err(a_fine, a_fresh)) < 1e-5


@pytest.mark.parametrize("eps", [0.0, -1.0, np.zeros(16), cp.zeros(16, dtype=cp.float32)])
def test_nonpositive_softening_raises(eps):
    pos = cp.asarray(np.random.default_rng(0).normal(size=(16, 3)), dtype=cp.float32)
    with pytest.raises(ValueError):
        tree_gravity_gpu(pos, 1.0, eps)


def test_repeated_calls_deterministic():
    n = 50_000
    x = _hernquist(n, seed=3)
    m = np.where(np.arange(n) % 2 == 0, 2.0 / n, 0.0)
    pos = cp.asarray(x, dtype=cp.float32)
    mass = cp.asarray(m, dtype=cp.float32)
    with TreeGPU(n, theta=0.6) as tree:
        accs = [tree_gravity_gpu(pos, mass, 0.01, G=1.0, tree=tree)[0] for _ in range(4)]
    for a in accs[1:]:
        assert np.max(_rel_err(a, accs[0])) < 1e-4


def test_integrator_galilean_invariance():
    """Cluster (M=1e5 Msun, a=1 pc) at 40 kpc moving at 200 km/s evolves like at rest."""
    G = 4.3009173e-06
    rng = np.random.default_rng(2)
    n, a, M = 5000, 1e-3, 1e5
    r = a / np.sqrt(rng.uniform(1e-6, 0.999, n) ** (-2.0 / 3.0) - 1.0)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    x = np.minimum(r, 30 * a)[:, None] * u
    sig = np.sqrt(G * M / (6 * np.sqrt(r ** 2 + a ** 2)))
    v = rng.normal(size=(n, 3)) * np.minimum(sig, np.sqrt(2 * G * M / np.sqrt(r ** 2 + a ** 2)) / 2)[:, None]
    x -= x.mean(0); v -= v.mean(0)
    xv0 = np.hstack([x, v]); m = np.full(n, M / n)
    dt = 5e-7
    off = np.array([40.0, 0.0, 0.0, 0.0, 200.0, 0.0])
    out = []
    for o in (0 * off, off):
        with tempfile.TemporaryDirectory() as d:
            xv = run_nbody_gpu_tree(xv0 + o, m, 0.0, 400 * dt, dt, 5e-5, theta=0.5,
                                    output_dir=d, save_snapshots=False, verbose=False,
                                    restart_interval=10 ** 9)
        out.append(xv - o)
    # bulk drift: the offset run moved by v*t along y
    out[1][:, 1] -= 200.0 * 400 * dt
    dx = np.linalg.norm(out[0][:, :3] - out[1][:, :3], axis=1)
    # float32 absolute coordinates at 40 kpc would round each drift by ~2e-6 kpc
    assert np.median(dx) < 1e-6


@pytest.mark.parametrize("k", [65, 1000])
def test_coincident_particles(k):
    """More than nleaf particles at one point: the depth cap turns them into a
    single oversized leaf instead of splitting past the 31-level limit (which
    left cells unwritten and corrupted every force)."""
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(10_000, 3)), np.tile([0.3, 0.2, 0.1], (k, 1))])
    m = np.full(len(x), 1.0 / len(x))
    acc, phi = tree_gravity_gpu(cp.asarray(x, dtype=cp.float32), cp.asarray(m, dtype=cp.float32),
                                0.01, G=1.0, theta=0.6)
    assert bool(cp.isfinite(acc).all()) and bool(cp.isfinite(phi).all())
    ref = compute_nbody_forces_gpu(x, m, softening=0.01, G=1.0, precision="float64",
                                   kernel="plummer", return_cupy=True)
    assert np.percentile(_rel_err(acc, ref), 99) < 1e-2
