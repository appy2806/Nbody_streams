"""
test_rotation_modifiers.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
GPU ``orientation=`` (TiltedPotentialGPU) and ``rotation=`` (RotatingPotentialGPU)
modifiers against Agama CPU.

Agama conventions, checked numerically:

* ``orientation=[a, b, g]``: ``Phi(x) = Phi_in(R^T x)``, ``R = Rz(a) Rx(b) Rz(g)``.
* ``rotation=angle`` (scalar): a *constant* angle about z, not a pattern speed.
* ``rotation=[[t, angle], ...]``: regularized natural cubic spline, linear
  extrapolation (same reader as ``center=`` / ``scale=``).
* several modifiers in one call nest as scale -> rotation -> orientation -> center
  (innermost first).

The inner potentials are off-centre so that a sign or order error cannot hide
behind a reflection symmetry.
"""

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._potential import (
    MultipolePotentialGPU,
    RotatingPotentialGPU,
    TiltedPotentialGPU,
    euler_matrix,
)

_TOL = 1e-12
_PL = dict(type='Plummer', mass=1e11, scaleRadius=4.0, center=[3.0, -5.0, 7.0])
_EULER = [0.4, 1.1, -0.7]
_TAB = np.array([[-2.0, 0.1], [-1.2, 0.4], [-0.3, 1.6], [0.5, 1.7], [1.6, 0.3]])


def _pts(N=3000, seed=7):
    return np.random.default_rng(seed).uniform(-60.0, 60.0, (N, 3))


def _rel(got, ref):
    got, ref = cp.asnumpy(cp.asarray(got)), cp.asnumpy(cp.asarray(ref))
    return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))


def _pair(**mod):
    """(agama, GPU) Plummer, off-centre, wrapped in the same modifiers."""
    ap = agama.Potential(potential=agama.Potential(**_PL), **mod)
    gp = PotentialGPU(PotentialGPU(**_PL), **mod)
    return ap, gp


def test_euler_matrix_matches_agama():
    X = _pts(200)
    inner = agama.Potential(**_PL)
    R = euler_matrix(*_EULER)
    ap = agama.Potential(potential=inner, orientation=_EULER)
    assert np.allclose(ap.potential(X), inner.potential(X @ R), rtol=_TOL, atol=0)
    assert np.allclose(euler_matrix(0, 0.3, 0), [[1, 0, 0], [0, np.cos(.3), -np.sin(.3)],
                                                 [0, np.sin(.3), np.cos(.3)]])


@pytest.mark.parametrize("mod", [dict(orientation=_EULER), dict(rotation=0.8),
                                 dict(rotation=_TAB)])
def test_potential_and_force_vs_agama(mod):
    ap, gp = _pair(**mod)
    X = _pts()
    for t in (-3.0, -1.2, -0.77, 0.0, 0.31, 2.5):      # nodes, between nodes, both wings
        assert _rel(gp.potential(X, t=t), ap.potential(X, t=t)) < _TOL
        assert _rel(gp.force(X, t=t), ap.force(X, t=t)) < _TOL


def test_types_and_angle():
    _, gp = _pair(rotation=_TAB)
    assert isinstance(gp, RotatingPotentialGPU)
    spl = agama.Spline(_TAB[:, 0], _TAB[:, 1], reg=True)
    for t in (-1.0, 0.0, 1.0):
        assert abs(gp.angle(t) - spl(t)) < 1e-13
    _, gp = _pair(orientation=_EULER)
    assert isinstance(gp, TiltedPotentialGPU)


def test_all_modifiers_one_call_vs_agama():
    """scale + rotation + orientation + center in one call: Agama's nesting order."""
    kw = dict(type='Plummer', mass=1e11, scaleRadius=4.0)
    traj = np.column_stack([np.linspace(-2, 2, 9), np.linspace(-5, 5, 9),
                            np.linspace(2, -2, 9), np.full(9, 1.5)])
    mods = dict(scale="1 1.7", orientation=_EULER, rotation=_TAB, center=traj)
    ap = agama.Potential(**kw, **mods)
    gp = PotentialGPU(**kw, **mods)
    X = _pts()
    for t in (-1.5, 0.2, 1.0):
        assert _rel(gp.potential(X, t=t), ap.potential(X, t=t)) < _TOL
        assert _rel(gp.force(X, t=t), ap.force(X, t=t)) < _TOL


def test_dict_component():
    mods = dict(orientation=_EULER, rotation=_TAB)
    ap = agama.Potential(dict(**_PL, **mods))
    gp = PotentialGPU(dict(**_PL, **mods))
    X = _pts(500)
    assert _rel(gp.force(X, t=0.4), ap.force(X, t=0.4)) < _TOL


def test_forcederiv_vs_agama():
    """Hessian rotation (R D R^T) and evalDeriv, on the off-centre Plummer."""
    ap, gp = _pair(orientation=_EULER, rotation=_TAB)
    X = _pts(2000)
    for t in (-0.5, 0.9):
        phi_a, Fa, Da = ap.eval(X, pot=True, acc=True, der=True, t=t)
        Fg, Dg = gp.forceDeriv(X, t=t)
        assert _rel(Fg, Fa) < _TOL
        assert _rel(Dg, Da) < _TOL
        phi, F, D = gp.evalDeriv(X, t=t)
        assert _rel(phi, phi_a) < _TOL and _rel(D, Da) < _TOL
        p2, F2, D2 = gp.eval(X, pot=True, acc=True, der=True, t=t)
        assert _rel(D2, Da) < _TOL


def test_multipole_inner():
    """Triaxial Multipole halo (the realistic case): rotated GPU == rotated GPU-inner by hand."""
    dens = agama.Density(type='Spheroid', densityNorm=1e7, scaleRadius=15, gamma=1, beta=3,
                         alpha=1, axisRatioY=0.9, axisRatioZ=0.75)
    inner = MultipolePotentialGPU.from_agama(
        agama.Potential(type='Multipole', density=dens, lmax=6, mmax=6, rmin=0.05, rmax=500))
    gp = PotentialGPU(inner, orientation=_EULER, rotation=_TAB)
    X = _pts(2000)
    t = -0.5
    R = gp.matrix(t)
    assert np.allclose(R, euler_matrix(*_EULER) @ euler_matrix(gp._inner_rot.angle(t), 0, 0), atol=1e-14)
    assert _rel(gp.force(X, t=t), cp.asnumpy(inner.force(cp.asarray(X @ R), t)) @ R.T) < _TOL


def test_time_dependent_tilt_about_x():
    """Outer(-90,-90,0) . Rz(theta(t)) . Inner(0,90,90) == R_x(theta(t)); nesting collapses."""
    inner = PotentialGPU(**_PL)
    tab = np.column_stack([np.linspace(-3, 1, 401), 0.2 + 0.5 * np.sin(np.linspace(-3, 1, 401))])
    tilt = TiltedPotentialGPU(
        RotatingPotentialGPU(TiltedPotentialGPU(inner, [0, np.pi / 2, np.pi / 2]), tab),
        [-np.pi / 2, -np.pi / 2, 0])
    X = _pts(1000)
    for t in (-2.5, -0.123, 0.7):
        th = tilt._inner_rot.angle(t)
        ref = TiltedPotentialGPU(inner, [0, th, 0])
        assert np.allclose(tilt.matrix(t), euler_matrix(0, th, 0), atol=1e-14)
        assert _rel(tilt.force(X, t=t), ref.force(X, t=t)) < _TOL
    assert tilt._base is inner


def test_single_point_shapes():
    _, gp = _pair(orientation=_EULER, rotation=_TAB)
    x = np.array([10.0, -3.0, 4.0])
    assert gp.potential(x, t=0.0).shape == ()
    assert gp.force(x, t=0.0).shape == (3,)
    F, D = gp.forceDeriv(x, t=0.0)
    assert F.shape == (3,) and D.shape == (6,)


def test_bad_inputs():
    inner = PotentialGPU(**_PL)
    with pytest.raises(ValueError):
        TiltedPotentialGPU(inner, [0.1, 0.2])
    with pytest.raises(ValueError):
        RotatingPotentialGPU(inner, np.zeros((4, 5)))
