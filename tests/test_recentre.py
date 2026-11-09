"""float64 positions far from the origin keep float32 precision.

A 1-pc Hernquist cluster at 40 kpc: float32 rounds positions to ~4e-3 pc,
a few percent of close separations.  The float32 force paths convert float64
input relative to a nearby centre (translation invariant), so they match
their at-origin accuracy.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu
from nbody_streams._recentre import to_float32_relative, estimate_centre


def _cluster(n=20_000, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.999, n)
    r = np.minimum(np.sqrt(u) / (1 - np.sqrt(u)), 200.0) * 1e-3     # a = 1 pc in kpc
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return r[:, None] * v + np.array([40.0, 0.0, 0.0]), np.full(n, 1.0 / n), 1e-5


def _p99(a, ref):
    a = cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
    ref = cp.asnumpy(ref) if isinstance(ref, cp.ndarray) else ref
    if a.ndim == 2:
        return np.percentile(np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1), 99)
    return np.percentile(np.abs(a - ref) / np.abs(ref), 99)


def test_to_float32_relative():
    x, m, _ = _cluster(1000)
    for arr in (x, cp.asarray(x)):
        rel, c = to_float32_relative(arr, m)
        assert rel.dtype == np.float32 and rel.flags.c_contiguous
        back = (cp.asnumpy(rel) if isinstance(rel, cp.ndarray) else rel) + (cp.asnumpy(c) if isinstance(c, cp.ndarray) else c)
        # float32 rounding relative to the recentred extent (~1e-8 kpc here),
        # not to 40 kpc (~2e-6 kpc without recentring)
        extent = float(np.abs(cp.asnumpy(rel) if isinstance(rel, cp.ndarray) else rel).max())
        assert np.max(np.abs(back - x)) < 1e-7 * extent
    rel32, c32 = to_float32_relative(x.astype(np.float32))
    assert c32 is None
    c = estimate_centre(cp.asarray(x), cp.zeros(len(x)))  # all-massless: plain mean, finite
    assert bool(cp.isfinite(c).all())


@pytest.mark.parametrize("prec", ["float32", "float32_kahan"])
def test_direct_gpu_float64_input_far_from_origin(prec):
    x, m, eps = _cluster()
    ref = compute_nbody_forces_gpu(x, m, softening=eps, G=1.0, precision="float64", kernel="plummer")
    for pos in (x, cp.asarray(x)):
        acc = compute_nbody_forces_gpu(pos, m, softening=eps, G=1.0, precision=prec, kernel="plummer")
        assert _p99(acc, ref) < 1e-4                      # was ~1e-2 before recentring
    phi_ref = compute_nbody_potential_gpu(x, m, softening=eps, G=1.0, precision="float64", kernel="plummer")
    phi = compute_nbody_potential_gpu(x, m, softening=eps, G=1.0, precision=prec, kernel="plummer")
    assert _p99(phi, phi_ref) < 1e-5


def test_tree_float64_input_far_from_origin():
    from nbody_streams.tree_gpu import tree_gravity_gpu
    x, m, eps = _cluster()
    ref = compute_nbody_forces_gpu(x, m, softening=eps, G=1.0, precision="float64", kernel="plummer")
    acc_abs, _ = tree_gravity_gpu(cp.asarray(x, dtype=cp.float32), m, eps, G=1.0)
    acc, _ = tree_gravity_gpu(x, m, eps, G=1.0)
    acc_c, _ = tree_gravity_gpu(cp.asarray(x), m, eps, G=1.0, center=[40.0, 0.0, 0.0])
    assert _p99(acc, ref) < 2e-3 and _p99(acc_c, ref) < 2e-3
    assert _p99(acc, ref) < 0.2 * _p99(acc_abs, ref)
