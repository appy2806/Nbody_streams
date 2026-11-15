"""
test_cylspline_precision.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ``CylSplinePotentialGPU(precision='float32')``: the mixed-precision
kernel (m != 0 harmonics in float, everything else in double).

Covers
------
1. float32 vs float64 for the FIRE m12i baryons (mmax = 8), in and near the
   disk plane and off the grid.
2. Differences of the force over 10 pc (tidal field) against float64.
3. Hessian and density stay float64 in float32 mode.
4. Routing: INI components and ``PotentialGPU(type='CylSpline', ...)``.
"""

import os

import numpy as np
import pytest

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._potential import CylSplinePotentialGPU

_HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(_HERE, "600.bar.none_8.coef_cylsp_DR")


@pytest.fixture(scope="module")
def pots():
    return (CylSplinePotentialGPU.from_file(FILE),
            CylSplinePotentialGPU.from_file(FILE, precision='float32'))


def _pts(n, seed, lo=-2, hi=2.8, flatten=0.1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    x = v * 10 ** rng.uniform(lo, hi, n)[:, None]
    x[: n // 2, 2] *= flatten              # half of them close to the disk plane
    return x


def _rel(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    if a.ndim == 1:
        return np.abs(a / b - 1)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def test_pointwise(pots):
    p64, p32 = pots
    assert (p64.precision, p32.precision) == ('float64', 'float32')
    x = cp.asarray(_pts(20000, 0))
    f32, f64 = p32.force(x), p64.force(x)
    assert f32.dtype == cp.float64
    e = _rel(f32, f64)
    # the max (~3e-6) sits within 10 pc of the centre, where the force goes
    # through zero and dPhi/dphi / R amplifies the float rounding
    assert np.median(e) < 5e-8 and e.max() < 5e-6
    assert _rel(p32.potential(x), p64.potential(x)).max() < 1e-7


def test_tidal_difference(pots):
    p64, p32 = pots
    n = 20000
    c = _pts(n, 1, lo=np.log10(8), hi=np.log10(40), flatten=0.1)
    d = 0.01 * _pts(n, 2, lo=0, hi=0, flatten=1.0)
    a = cp.asarray(np.vstack([c, c + d]))
    f64, f32 = cp.asnumpy(p64.force(a)), cp.asnumpy(p32.force(a))
    e = _rel(f32[n:] - f32[:n], f64[n:] - f64[:n])
    assert np.median(e) < 1e-4


def test_hessian_and_density_stay_float64(pots):
    p64, p32 = pots
    x = cp.asarray(_pts(200, 3))
    np.testing.assert_array_equal(cp.asnumpy(p32.forceDeriv(x)[1]),
                                  cp.asnumpy(p64.forceDeriv(x)[1]))
    np.testing.assert_array_equal(cp.asnumpy(p32.density(x)), cp.asnumpy(p64.density(x)))


def test_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        CylSplinePotentialGPU.from_file(FILE, precision='float16')


def test_routing(tmp_path):
    assert PotentialGPU(type='CylSpline', file=FILE, precision='float32').precision == 'float32'
    ini = tmp_path / "host.ini"
    ini.write_text(f"[Potential disk]\ntype=CylSpline\nfile={FILE}\n\n"
                   "[Potential halo]\ntype=NFW\nmass=1e12\nscaleRadius=15\n")
    comps = PotentialGPU(file=str(ini), precision='float32')._components
    assert [c.precision for c in comps] == ['float32', 'float32']
