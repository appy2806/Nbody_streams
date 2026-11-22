"""
test_spheroid_shape.py
~~~~~~~~~~~~~~~~~~~~~~
Tests for ``SpheroidPotentialGPU``: the triaxial double-power-law potential
evaluated directly from its density, with a time-dependent shape.

Covers
------
1. Spherical NFW limit vs the closed-form ``NFWPotentialGPU``.
2. Triaxial Dehnen vs Agama's exact ``Dehnen(axisRatioY, axisRatioZ)``.
3. Triaxial Spheroid with a cutoff vs Agama's Multipole at high lmax.
4. Self-consistency: Hessian vs finite-differenced force, and Poisson
   (trace of the Hessian = 4 pi G rho), on both kernel paths.
5. Time dependence: at every t the potential equals the static one with the
   interpolated axis ratios; ``mass`` keeps the mass in every ellipsoid fixed.
6. ``PotentialGPU`` routing: only time tables leave the Agama route.
"""

import math

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._analytic_potentials import NFWPotentialGPU, _G
from nbody_streams.agama_helper._potential import (
    MultipolePotentialGPU, RotatingPotentialGPU,
)
from nbody_streams.agama_helper._spheroid import SpheroidPotentialGPU


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xyz():
    """Points log-uniform in r from 0.01 to 600 kpc (40 r_s), random directions."""
    rng = np.random.default_rng(42)
    r = 10 ** rng.uniform(-2, np.log10(600), 3000)
    d = rng.normal(size=(3000, 3))
    return r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]


def _rel(gpu, ref):
    """Max relative error; vectors are normalised by their row norm."""
    a, b = cp.asnumpy(gpu), np.asarray(ref)
    if b.ndim == 1:
        return np.max(np.abs(a - b) / np.abs(b))
    return np.max(np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1))


# General (pow/exp) path and integer-slope fast path.
_PROFILES = {
    "general": dict(alpha=2.0, beta=3.5, gamma=0.7),
    "fast":    dict(alpha=1.0, beta=3.0, gamma=1.0, outerCutoffRadius=300.0),
}


# ---------------------------------------------------------------------------
# 1-3. Against references
# ---------------------------------------------------------------------------

def test_spherical_nfw_matches_closed_form(xyz):
    g = SpheroidPotentialGPU(mass=1e12, scaleRadius=15, alpha=1, beta=3, gamma=1)
    n = NFWPotentialGPU(mass=1e12, scaleRadius=15)
    assert _rel(g.potential(xyz), cp.asnumpy(n.potential(xyz))) < 1e-11
    assert _rel(g.force(xyz), cp.asnumpy(n.force(xyz))) < 1e-11
    assert _rel(g.forceDeriv(xyz)[1], cp.asnumpy(n.forceDeriv(xyz)[1])) < 1e-10
    assert _rel(g.density(xyz), cp.asnumpy(n.density(xyz))) < 1e-13


def test_triaxial_dehnen_matches_agama(xyz):
    kw = dict(mass=1e12, scaleRadius=15, gamma=1.0, axisRatioY=0.8, axisRatioZ=0.55)
    a = agama.Potential(type='Dehnen', **kw)
    g = SpheroidPotentialGPU(alpha=1, beta=4, **kw)
    phi, acc, der = a.eval(xyz, pot=True, acc=True, der=True)
    assert _rel(g.potential(xyz), phi) < 1e-10
    assert _rel(g.force(xyz), acc) < 1e-12
    assert _rel(g.forceDeriv(xyz)[1], der) < 1e-7      # 32 nodes; 1e-12 with 64
    assert _rel(g.density(xyz), a.density(xyz)) < 1e-13


def test_triaxial_spheroid_converges_to_agama_multipole(xyz):
    """Agama's Multipole has truncation error that shrinks with lmax; ours does not."""
    kw = dict(densityNorm=1e7, scaleRadius=15, alpha=1, beta=3, gamma=1,
              outerCutoffRadius=300, axisRatioY=0.8, axisRatioZ=0.6)
    g = SpheroidPotentialGPU(**kw)
    errs = []
    for lmax in (6, 12, 24):
        a = agama.Potential(type='Multipole', density=agama.Density(type='Spheroid', **kw),
                            lmax=lmax, gridSizeR=60, rmin=0.01, rmax=3000)
        errs.append(_rel(g.force(xyz), a.force(xyz)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 5e-6


@pytest.mark.parametrize("mass", [1e12, 3e11])
def test_mass_matches_agama_spheroid(mass, xyz):
    """Finite-mass profile: ``mass=`` sets rho0 exactly as Agama does."""
    kw = dict(mass=mass, scaleRadius=10, alpha=1, beta=4, gamma=1.2,
              axisRatioY=0.9, axisRatioZ=0.6)
    a = agama.Density(type='Spheroid', **kw)
    g = SpheroidPotentialGPU(**kw)
    assert _rel(g.density(xyz), a.density(xyz)) < 1e-12


# ---------------------------------------------------------------------------
# 4. Self-consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(_PROFILES))
def test_hessian_matches_finite_difference(path, xyz):
    g = SpheroidPotentialGPU(densityNorm=1e7, scaleRadius=15, axisRatioY=0.8,
                             axisRatioZ=0.5, **_PROFILES[path])
    pts = xyz[np.linalg.norm(xyz, axis=1) > 0.1][:300]
    _, der = g.forceDeriv(pts)
    der = cp.asnumpy(der)
    h = 1e-5 * np.linalg.norm(pts, axis=1)[:, None]
    fd = np.empty((len(pts), 3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = 1.0
        fd[:, :, j] = cp.asnumpy(g.force(pts + h * e) - g.force(pts - h * e)) / (2 * h)
    # deriv layout [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
    fd6 = np.stack([fd[:, 0, 0], fd[:, 1, 1], fd[:, 2, 2],
                    fd[:, 0, 1], fd[:, 1, 2], fd[:, 2, 0]], axis=1)
    assert np.max(np.abs(der - fd6).max(1) / np.abs(der).max(1)) < 1e-6


@pytest.mark.parametrize("path", sorted(_PROFILES))
def test_poisson(path, xyz):
    g = SpheroidPotentialGPU(densityNorm=1e7, scaleRadius=15, axisRatioY=0.8,
                             axisRatioZ=0.5, **_PROFILES[path])
    _, der = g.forceDeriv(xyz)
    der = cp.asnumpy(der)
    lap = -der[:, :3].sum(axis=1)
    rho = cp.asnumpy(g.density(xyz))
    # relative to the Hessian scale: near the cutoff rho is a small difference
    # of large Hessian terms, so rho itself is the wrong yardstick
    scale = np.abs(der[:, :3]).max(axis=1)
    assert np.max(np.abs(lap - 4 * math.pi * _G * rho) / scale) < 1e-6


# ---------------------------------------------------------------------------
# 5. Time dependence
# ---------------------------------------------------------------------------

_QT = np.array([[0.0, 0.5], [2.0, 0.6], [4.0, 0.9], [6.0, 0.95]])   # [t, q]


def test_time_dependent_equals_static_snapshot(xyz):
    g = SpheroidPotentialGPU(mass=1e12, scaleRadius=15, beta=3,
                             axisRatioY=0.85, axisRatioZ=_QT)
    assert g.is_time_dependent
    for t in (0.0, 1.3, 4.0, 5.7, 7.0):          # knots, between, extrapolated
        p, q = g.shape(t)
        s = SpheroidPotentialGPU(mass=1e12, scaleRadius=15, beta=3,
                                 axisRatioY=p, axisRatioZ=q)
        assert p == 0.85
        assert _rel(g.force(xyz, t), cp.asnumpy(s.force(xyz))) < 1e-15
        assert _rel(g.potential(xyz, t), cp.asnumpy(s.potential(xyz))) < 1e-15


def test_shape_follows_agama_regularized_spline():
    g = SpheroidPotentialGPU(axisRatioZ=_QT)
    t = np.linspace(0, 6, 25)
    ref = agama.Spline(_QT[:, 0], _QT[:, 1], reg=True)(t)
    assert np.allclose([g.shape(ti)[1] for ti in t], ref, rtol=0, atol=1e-14)


def test_mass_conserved_in_every_ellipsoid():
    """rho0 * p * q is constant, so M(<m) does not change with the shape."""
    for prof in (dict(beta=3), dict(beta=4)):     # infinite and finite mass
        g = SpheroidPotentialGPU(mass=1e12, scaleRadius=15, axisRatioZ=_QT, **prof)
        vals = [g.rho0(t) * math.prod(g.shape(t)) for t in (0.0, 1.0, 3.3, 6.0)]
        assert np.allclose(vals, vals[0], rtol=1e-14)


def test_density_norm_table():
    tab = [[0.0, 1e7], [5.0, 2e7]]
    g = SpheroidPotentialGPU(densityNorm=tab, scaleRadius=15, beta=3)
    assert g.rho0(0.0) == 1e7 and g.rho0(5.0) == 2e7


def test_force_is_smooth_through_a_knot(xyz):
    g = SpheroidPotentialGPU(mass=1e12, scaleRadius=15, beta=3, axisRatioZ=_QT)
    eps = 1e-6
    f = [cp.asnumpy(g.force(xyz, 2.0 + d)) for d in (-eps, 0.0, eps)]
    # central second difference ~ eps^2 F'' when F is C1 in t
    assert np.max(np.abs(f[0] - 2 * f[1] + f[2]).max(1) / np.abs(f[1]).max(1)) < 1e-9


# ---------------------------------------------------------------------------
# 6. Construction and routing
# ---------------------------------------------------------------------------

def test_routing_only_time_tables_leave_agama():
    static = PotentialGPU(type='Spheroid', densityNorm=1e7, scaleRadius=15,
                          axisRatioZ=0.6, beta=3, outerCutoffRadius=300)
    assert isinstance(static, MultipolePotentialGPU)

    for typ, extra in (("Spheroid", dict(beta=3)), ("NFW", {}), ("Dehnen", dict(gamma=1.5))):
        pot = PotentialGPU(type=typ, mass=1e12, scaleRadius=15, axisRatioZ=_QT, **extra)
        assert isinstance(pot, SpheroidPotentialGPU), typ
    nfw = PotentialGPU(type='NFW', mass=1e12, scaleRadius=15, axisRatioZ=_QT)
    assert (nfw._alpha, nfw._beta, nfw._gamma) == (1.0, 3.0, 1.0)


def test_routing_rejects_inapplicable_keywords():
    with pytest.raises(TypeError, match="lmax"):
        PotentialGPU(type='Spheroid', axisRatioZ=_QT, lmax=12)
    with pytest.raises(TypeError, match="fixes"):
        PotentialGPU(type='NFW', axisRatioZ=_QT, beta=4)


def test_composes_with_rotation(xyz):
    """A pattern-rotating halo whose flattening also evolves."""
    pot = PotentialGPU(type='NFW', mass=1e12, scaleRadius=15, axisRatioY=0.8,
                       axisRatioZ=_QT, rotation=[[0.0, 0.0], [6.0, 1.2]])
    assert isinstance(pot, RotatingPotentialGPU)
    assert isinstance(pot._base, SpheroidPotentialGPU)
    t = 3.0
    R = pot.matrix(t)                     # Phi(x, t) = Phi_base(R^T x, t)
    ref = cp.asnumpy(pot._base.force(xyz @ R, t)) @ R.T
    assert _rel(pot.force(xyz, t), ref) < 1e-13


@pytest.mark.parametrize("bad", [
    dict(axisRatioZ=[[0, 0.5], [1, -0.1]]),
    dict(mass=1.0, densityNorm=1.0),
    dict(beta=2.0),
    dict(gamma=3.0),
    dict(axisRatioZ=np.ones((3, 4))),
])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        SpheroidPotentialGPU(**bad)


# ---------------------------------------------------------------------------
# 7. Single precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(_PROFILES))
def test_float32_close_to_float64(path, xyz):
    kw = dict(densityNorm=1e7, scaleRadius=15, axisRatioY=0.8, axisRatioZ=_QT,
              **_PROFILES[path])
    f64 = SpheroidPotentialGPU(**kw)
    f32 = SpheroidPotentialGPU(precision='float32', **kw)
    t = 2.7
    out = f32.force(xyz, t)
    assert out.dtype == cp.float64                   # interface stays float64
    assert _rel(out, cp.asnumpy(f64.force(xyz, t))) < 2e-6
    assert _rel(f32.forceDeriv(xyz, t)[1], cp.asnumpy(f64.forceDeriv(xyz, t)[1])) < 5e-6
    assert _rel(f32.potential(xyz, t), cp.asnumpy(f64.potential(xyz, t))) < 5e-6
    assert _rel(f32.density(xyz, t), cp.asnumpy(f64.density(xyz, t))) < 2e-5


def test_float32_via_potentialgpu():
    pot = PotentialGPU(type='NFW', mass=1e12, scaleRadius=15, axisRatioZ=_QT,
                       precision='float32')
    assert pot._dt is np.float32
    with pytest.raises(ValueError, match="precision"):
        SpheroidPotentialGPU(precision='float16')
