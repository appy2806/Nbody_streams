"""
test_analytic_precision.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the ``precision=`` option of the analytic GPU potentials.

Covers
------
1. float32 vs float64 for every model: potential, force, force derivatives
   and density agree to float32 rounding; the interface stays float64.
2. Cancellation-free forms, checked against arbitrary precision (mpmath):
   NFW force/Hessian at r << rs, the DiskAnsatz H(z) near the midplane,
   Dehnen potential at r >> a, isochrone density at r >> b.
3. Force = -grad(Phi) and deriv = -Hessian (finite differences), so the
   sign folded into the kernels is right.
4. ``PotentialGPU`` routing: precision reaches the analytic classes; types
   built as Agama Multipole expansions reject float32.
"""

import numpy as np
import pytest

import cupy as cp
mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._analytic_potentials import (
    _G, NFWPotentialGPU, PlummerPotentialGPU, HernquistPotentialGPU,
    DehnenSphericalPotentialGPU, IsochronePotentialGPU, MiyamotoNagaiPotentialGPU,
    LogHaloPotentialGPU, DiskAnsatzPotentialGPU, UniformAccelerationGPU,
)


MODELS = {
    'nfw':       (NFWPotentialGPU, dict(mass=1e12, scaleRadius=15.)),
    'plummer':   (PlummerPotentialGPU, dict(mass=1e5, scaleRadius=0.01)),
    'hernquist': (HernquistPotentialGPU, dict(mass=5e10, scaleRadius=1.)),
    'dehnen05':  (DehnenSphericalPotentialGPU, dict(mass=1e10, scaleRadius=2., gamma=0.5)),
    'dehnen15':  (DehnenSphericalPotentialGPU, dict(mass=1e10, scaleRadius=2., gamma=1.5)),
    'isochrone': (IsochronePotentialGPU, dict(mass=1e11, scaleRadius=3.)),
    'mn':        (MiyamotoNagaiPotentialGPU, dict(mass=6e10, scaleRadius=3., scaleHeight=0.28)),
    'log':       (LogHaloPotentialGPU, dict(velocity=200., coreRadius=1., axisRatioY=0.9,
                                            axisRatioZ=0.8)),
    'disk':      (DiskAnsatzPotentialGPU, dict(surfaceDensity=1e9, scaleRadius=3.,
                                               scaleHeight=0.3, innerCutoffRadius=0.5)),
}


@pytest.fixture(scope="module")
def xyz():
    """Points over 0.05-200 kpc, a tenth of them close to the z=0 plane."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=(4000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    x = v * 10 ** rng.uniform(np.log10(0.05), np.log10(200), 4000)[:, None]
    x[:400, 2] *= 1e-3
    return cp.asarray(x)


def _rel(a, b):
    """Max error relative to each row's largest component (or |value|)."""
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    return float((np.abs(a - b).max(1) / np.abs(b).max(1)).max())


# ---------------------------------------------------------------------------
# 1. float32 vs float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODELS))
def test_float32_close_to_float64(name, xyz):
    cls, kw = MODELS[name]
    p64, p32 = cls(**kw), cls(**kw, precision='float32')
    assert (p64.precision, p32.precision) == ('float64', 'float32')
    phi64, f64, d64 = p64.evalDeriv(xyz)
    phi32, f32, d32 = p32.evalDeriv(xyz)
    rho64, rho32 = p64.density(xyz), p32.density(xyz)
    for arr in (phi32, f32, d32, rho32):
        assert arr.dtype == cp.float64                # interface stays float64
    # LogHalo's potential crosses zero at m = 1: compare against v0^2 instead
    phi_scale = cp.maximum(cp.abs(phi64), 1.0 if name != 'log' else 4e4)
    # float32 exp(-r/hr) carries ~eps r/hr, so the DiskAnsatz far out is looser
    assert float((cp.abs(phi32 - phi64) / phi_scale).max()) < (1e-5 if name == 'disk' else 5e-7)
    loose = 10 if name == 'disk' else 1
    assert _rel(f32, f64) < 2e-6 * loose
    assert _rel(d32, d64) < 5e-6 * loose
    if name == 'disk':
        # the ansatz part's density changes sign; measure against its peak there
        rho_scale = cp.maximum(cp.abs(rho64), 1e-5 * float(cp.abs(rho64).max()))
        assert float((cp.abs(rho32 - rho64) / rho_scale).max()) < 1e-5 * loose
    else:
        assert _rel(rho32, rho64) < 1e-5


def test_uniform_acceleration_float32():
    tab = np.array([[0., 1., -2., .5], [1., 2., -1., 0.], [2., 0., 0., 1.]])
    p64 = UniformAccelerationGPU(file=tab)
    p32 = UniformAccelerationGPU(file=tab, precision='float32')
    x = cp.asarray([[1., 2., 3.], [-4., 5., 6.]])
    for t in (0.3, 1.7, 0.3):                       # also revisits a cached a(t)
        np.testing.assert_allclose(cp.asnumpy(p32.force(x, t)),
                                   cp.asnumpy(p64.force(x, t)), rtol=1e-7)
        np.testing.assert_allclose(cp.asnumpy(p32.potential(x, t)),
                                   cp.asnumpy(p64.potential(x, t)), rtol=1e-6)
    assert float(cp.abs(p32.forceDeriv(x, 1.)[1]).max()) == 0.0


def test_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        NFWPotentialGPU(precision='float16')


# ---------------------------------------------------------------------------
# 2. Cancellation-free forms vs arbitrary precision
# ---------------------------------------------------------------------------

def _nfw_exact(r, GM, rs):
    """|force| and the radial Hessian d2Phi/dr2 of the NFW potential."""
    r = mp.mpf(r)
    phi = lambda q: -GM * mp.log1p(q / rs) / q
    return float(mp.diff(phi, r)), float(mp.diff(phi, r, 2))


@pytest.mark.parametrize("precision, tol", [('float64', 1e-14), ('float32', 1e-6)])
def test_nfw_small_radius(precision, tol):
    rs = 15.0
    r = np.array([1e-6, 1e-4, 1e-2, 0.5, 3.7, 3.8, 14.9, 15.1, 30.])  # s0 = 0.25, 1
    pot = NFWPotentialGPU(mass=1e12, scaleRadius=rs, precision=precision)
    f, d = pot.forceDeriv(cp.asarray(np.c_[r, 0 * r, 0 * r]))
    ex = np.array([_nfw_exact(ri, _G * 1e12, rs) for ri in r])
    np.testing.assert_allclose(-cp.asnumpy(f)[:, 0], ex[:, 0], rtol=tol)
    np.testing.assert_allclose(-cp.asnumpy(d)[:, 0], ex[:, 1], rtol=4 * tol)


@pytest.mark.parametrize("precision, tol", [('float64', 1e-12), ('float32', 3e-6)])
def test_disk_near_midplane(precision, tol):
    S, hr, hz, hin = 1e9, 3.0, 0.3, 0.5
    pot = DiskAnsatzPotentialGPU(S, hr, hz, hin, precision=precision)
    z = np.array([1e-7, 1e-5, 1e-3, 0.1, 0.29, 0.31, 1.0])        # u = 1 at z = 0.3
    x = np.c_[8.0 + 0 * z, 0 * z, z]

    def exact(q):
        r = mp.sqrt(64 + q * q)
        u = abs(q) / hz
        return 4 * mp.pi * _G * S * mp.exp(-r / hr - hin / r) * hz / 2 * (mp.exp(-u) - 1 + u)

    phi = cp.asnumpy(pot.potential(cp.asarray(x)))
    fz = cp.asnumpy(pot.force(cp.asarray(x)))[:, 2]
    np.testing.assert_allclose(phi, [float(exact(mp.mpf(q))) for q in z], rtol=tol)
    np.testing.assert_allclose(fz, [-float(mp.diff(exact, mp.mpf(q))) for q in z], rtol=tol)


@pytest.mark.parametrize("precision, tol", [('float64', 1e-14), ('float32', 1e-6)])
def test_dehnen_and_isochrone_large_radius(precision, tol):
    r = np.array([1e2, 1e4, 1e6])
    x = cp.asarray(np.c_[r, 0 * r, 0 * r])
    GM, a, g = _G * 1e10, 2.0, 0.5
    deh = DehnenSphericalPotentialGPU(mass=1e10, scaleRadius=a, gamma=g, precision=precision)
    ex = [float(-(GM / a) * (1 - (mp.mpf(q) / (q + a)) ** (2 - g)) / (2 - g)) for q in r]
    np.testing.assert_allclose(cp.asnumpy(deh.potential(x)), ex, rtol=tol)

    iso = IsochronePotentialGPU(mass=1e11, scaleRadius=3.0, precision=precision)
    b = mp.mpf(3)
    lap = lambda q: (lambda s: 1e11 * b * (3 * b * b + 3 * s * b + 2 * q * q)
                     / (4 * mp.pi * (b + s) ** 3 * s ** 3))(mp.sqrt(q * q + b * b))
    np.testing.assert_allclose(cp.asnumpy(iso.density(x)),
                               [float(lap(mp.mpf(q))) for q in r], rtol=tol)


# ---------------------------------------------------------------------------
# 3. Signs: force = -grad(Phi), deriv = -Hessian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODELS))
def test_force_and_deriv_signs(name):
    cls, kw = MODELS[name]
    pot = cls(**kw)
    x0 = np.array([[2.3, -1.1, 0.7]])
    h = 1e-5
    fd_f, fd_d = np.zeros(3), np.zeros((3, 3))
    for k in range(3):
        e = np.zeros((1, 3)); e[0, k] = h
        xp, xm = cp.asarray(x0 + e), cp.asarray(x0 - e)
        fd_f[k] = -(float(pot.potential(xp)[0]) - float(pot.potential(xm)[0])) / (2 * h)
        fd_d[:, k] = cp.asnumpy(pot.force(xp)[0] - pot.force(xm)[0]) / (2 * h)
    f, d = pot.forceDeriv(cp.asarray(x0))
    np.testing.assert_allclose(cp.asnumpy(f)[0], fd_f, rtol=1e-6, atol=1e-8 * np.abs(fd_f).max())
    d = cp.asnumpy(d)[0]
    sym = np.array([d[0], d[1], d[2], d[3], d[4], d[5]])
    want = np.array([fd_d[0, 0], fd_d[1, 1], fd_d[2, 2], fd_d[0, 1], fd_d[1, 2], fd_d[0, 2]])
    np.testing.assert_allclose(sym, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())


# ---------------------------------------------------------------------------
# 4. PotentialGPU routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(type='NFW', mass=1e12, scaleRadius=15),
    dict(type='Dehnen', mass=1e10, scaleRadius=2, gamma=0.5),
    dict(type='Logarithmic', v0=200, scaleRadius=1),
    dict(type='UniformAcceleration', ax=1., ay=2., az=3.),
])
def test_precision_via_potentialgpu(kw):
    assert PotentialGPU(**kw, precision='float32').precision == 'float32'
    assert PotentialGPU(**kw).precision == 'float64'


@pytest.mark.parametrize("kw", [
    dict(type='Spheroid', mass=1e10),
    dict(type='Dehnen', mass=1e10, axisRatioZ=0.8),
])
def test_multipole_backed_types_reject_float32(kw):
    with pytest.raises(ValueError, match="precision"):
        PotentialGPU(**kw, precision='float32')
    PotentialGPU(**kw, precision='float64')
