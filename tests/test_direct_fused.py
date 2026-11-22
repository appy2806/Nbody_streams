"""
Fused GPU direct-sum paths.

1. ``_direct_forces_gpu_soa`` (the integrator's hot path: inputs prepared
   once, one packing kernel, SoA output) equals ``compute_nbody_forces_gpu``
   bit for bit, for every precision and for float64 / float32 positions.
2. ``run_nbody_gpu`` runs in every precision (float64 used to break when the
   split kernel had no sized output) and agrees across precisions, with and
   without an external potential and with per-particle softening.
"""

import tempfile

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from nbody_streams.fields import (
    compute_nbody_forces_gpu, _direct_forces_gpu_soa, KERNEL_MAP, _PRECISION_MAP,
)
from nbody_streams.run import run_nbody_gpu


def _cluster(N=2000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(N, 3)) + [40.0, 3.0, -2.0]
    v = rng.normal(scale=2.0, size=(N, 3)) + [0.0, 180.0, 0.0]
    m = rng.uniform(5.0, 15.0, N)
    return x, v, m


@pytest.mark.parametrize("precision", ["float32", "float32_kahan", "float64"])
@pytest.mark.parametrize("pos_dtype", [np.float64, np.float32])
def test_soa_hot_path_matches_public(precision, pos_dtype):
    x, _, m = _cluster()
    if pos_dtype == np.float32:
        x = x - x.mean(axis=0)                       # float32 input is used as given
    dt = _PRECISION_MAP[precision][0]
    pos = cp.asarray(x, dtype=pos_dtype)
    h = np.linspace(5e-5, 2e-4, len(m))
    ref = compute_nbody_forces_gpu(pos, cp.asarray(m, dtype=dt), cp.asarray(h, dtype=dt),
                                   precision=precision, kernel='spline', return_cupy=True)
    ax, ay, az = _direct_forces_gpu_soa(pos, cp.asarray(m, dtype=dt), cp.asarray(h, dtype=dt),
                                        4.300917270069976e-06, precision, KERNEL_MAP['spline'])
    got = cp.stack([ax, ay, az], axis=1)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(ref))


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("soft", ["scalar", "array"])
def test_run_nbody_gpu_precisions_agree(external, soft):
    x, v, m = _cluster(N=1500)
    ps = np.c_[x, v]
    h = 1e-4 if soft == "scalar" else np.linspace(5e-5, 2e-4, len(m))
    host = None
    if external:
        from nbody_streams.agama_helper import PotentialGPU
        host = PotentialGPU(type='NFW', mass=1e12, scaleRadius=15)
    out = {}
    for precision in ("float64", "float32", "float32_kahan"):
        with tempfile.TemporaryDirectory() as d:
            out[precision] = run_nbody_gpu(ps, m, 0.0, 5e-5, 1e-5, h, precision=precision,
                                           external_potential=host, output_dir=d,
                                           save_snapshots=False, verbose=False, snapshots=1)
    for precision in ("float32", "float32_kahan"):
        dv = np.abs(out[precision][:, 3:] - out["float64"][:, 3:]).max()
        assert dv < 1e-4 * np.abs(ps[:, 3:] - ps[:, 3:].mean(axis=0)).max()
    assert np.all(np.isfinite(out["float64"]))
