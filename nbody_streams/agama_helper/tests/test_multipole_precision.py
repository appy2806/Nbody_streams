"""
test_multipole_precision.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ``MultipolePotentialGPU(precision='float32')``: the mixed-precision
kernel (l > 0 harmonic sums in float, everything else in double).

Covers
------
1. float32 vs float64 over 0.1-300 kpc for the FIRE m12i halo (lmax = 8) and
   the LMC (lmax = 8, no ``_DR``): force ~1e-8, potential ~1e-8.
2. Differences of the force over 10 pc (tidal field) against float64.
3. Hessian and density stay float64 in float32 mode.
4. The force on the z axis, where Y_l1 / sin(theta) is replaced by its limit
   (the south pole had the wrong sign), against Agama just off the axis.
5. Routing: ``PotentialGPU(file=..., precision='float32')``.
"""

import os

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._potential import MultipolePotentialGPU

_HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "FIRE": os.path.join(_HERE, "600.dark.none_8.coef_mul_DR"),
    "LMC":  os.path.join(_HERE, "100.LMC.none_8.coef_mult"),
}


@pytest.fixture(scope="module")
def pots():
    return {k: (MultipolePotentialGPU.from_file(f),
                MultipolePotentialGPU.from_file(f, precision='float32'))
            for k, f in FILES.items()}


def _dirs(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _rel(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    if a.ndim == 1:
        return np.abs(a / b - 1)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


# ---------------------------------------------------------------------------
# 1-2. float32 vs float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FILES))
def test_pointwise(name, pots):
    p64, p32 = pots[name]
    assert (p64.precision, p32.precision) == ('float64', 'float32')
    rng = np.random.default_rng(0)
    x = cp.asarray(_dirs(20000, 1) * 10 ** rng.uniform(-1, np.log10(300), 20000)[:, None])
    f32, f64 = p32.force(x), p64.force(x)
    assert f32.dtype == cp.float64
    e = _rel(f32, f64)
    assert np.median(e) < 1e-7 and e.max() < 2e-6
    assert _rel(p32.potential(x), p64.potential(x)).max() < 2e-7


def test_tidal_difference(pots):
    """dF over 10 pc at 40 kpc: ~1e-7 f_ns R/d, below the pure-float32 4e-4."""
    p64, p32 = pots["FIRE"]
    n = 20000
    c = 40.0 * _dirs(n, 2)
    a = cp.asarray(np.vstack([c, c + 0.01 * _dirs(n, 3)]))
    f64, f32 = cp.asnumpy(p64.force(a)), cp.asnumpy(p32.force(a))
    e = _rel(f32[n:] - f32[:n], f64[n:] - f64[:n])
    assert np.median(e) < 2e-4


def test_hessian_and_density_stay_float64(pots):
    p64, p32 = pots["FIRE"]
    x = cp.asarray(20.0 * _dirs(100, 4))
    np.testing.assert_array_equal(cp.asnumpy(p32.forceDeriv(x)[1]),
                                  cp.asnumpy(p64.forceDeriv(x)[1]))
    np.testing.assert_array_equal(cp.asnumpy(p32.density(x)), cp.asnumpy(p64.density(x)))


def test_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        MultipolePotentialGPU.from_file(FILES["FIRE"], precision='float16')


# ---------------------------------------------------------------------------
# 4. On the z axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ['float64', 'float32'])
@pytest.mark.parametrize("z", [30.0, -30.0, -3.0])
def test_force_on_axis(precision, z):
    pot = MultipolePotentialGPU.from_file(FILES["FIRE"], precision=precision)
    ref = agama.Potential(FILES["FIRE"])
    on = cp.asnumpy(pot.force(cp.asarray([[0.0, 0.0, z]])))[0]
    off = ref.force(np.array([[1e-7, 0.0, z]]))[0]        # Agama's own axis value is not the limit
    np.testing.assert_allclose(on, off, rtol=0, atol=2e-5 * np.abs(off).max())
    hess_force = cp.asnumpy(pot.forceDeriv(cp.asarray([[0.0, 0.0, z]]))[0])[0]
    np.testing.assert_allclose(hess_force, off, rtol=0, atol=2e-5 * np.abs(off).max())


# ---------------------------------------------------------------------------
# 5. Routing
# ---------------------------------------------------------------------------

def test_potentialgpu_file_precision():
    pot = PotentialGPU(file=FILES["FIRE"], precision='float32')
    assert pot.precision == 'float32'
    assert PotentialGPU(file=FILES["FIRE"]).precision == 'float64'


def test_ini_components_get_precision(tmp_path):
    ini = tmp_path / "host.ini"
    ini.write_text(f"[Potential halo]\ntype=Multipole\nfile={FILES['FIRE']}\n\n"
                   "[Potential bulge]\ntype=Hernquist\nmass=1e10\nscaleRadius=1\n")
    pot = PotentialGPU(file=str(ini), precision='float32')
    assert [c.precision for c in pot._components] == ['float32', 'float32']
    assert [c.precision for c in PotentialGPU(file=str(ini))._components] == ['float64'] * 2


def test_load_agama_potential_precision():
    from nbody_streams.agama_helper import load_agama_potential
    assert load_agama_potential(FILES["FIRE"], gpu=True, precision='float32').precision == 'float32'
