"""
test_time_modifiers.py
~~~~~~~~~~~~~~~~~~~~~~
Numerical tests for the time-dependent ``center=`` and ``scale=`` modifiers
against Agama CPU.

Agama reads all three time-dependent inputs --- ``center=``, ``scale=`` and the
``UniformAcceleration`` table --- through the same
``potential_factory.cpp::readTimeDependentArray``, so they must all use the same
interpolation: a **natural** cubic spline with the Hyman (1983) regularization
filter (``agama.Spline(..., reg=True)``), linearly extrapolated beyond the
endpoints.  SciPy's default ``not-a-knot`` spline is a different curve.

The pre-existing tests only asserted that ``ShiftedPotentialGPU`` came back from
the factory; nothing compared trajectory *values* against Agama, so a
boundary-condition mismatch went unnoticed.  These tests close that gap.
"""

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._potential import (
    ShiftedPotentialGPU,
    ScaledPotentialGPU,
    _AgamaTimeSpline,
)

_TOL = 1e-12

_HALO = dict(type='Plummer', mass=1e11, scaleRadius=5.0)


def _traj(T=40, t0=-6.0, t1=0.0):
    """A coarsely sampled LMC-like infall --- the regime where the spline choice bites."""
    t = np.linspace(t0, t1, T)
    return np.column_stack([
        t,
        50.0 * np.cos(0.9 * t) * np.exp(0.25 * t),
        50.0 * np.sin(0.9 * t) * np.exp(0.25 * t),
        -30.0 * np.exp(0.2 * t),
    ])


def _pts(N=2000, seed=3):
    return np.random.default_rng(seed).uniform(-60.0, 60.0, (N, 3)).astype(np.float64)


def _rel(gpu, cpu):
    cpu = np.asarray(cpu)
    return float(np.max(np.abs(np.asarray(gpu) - cpu)) / (np.max(np.abs(cpu)) + 1e-30))


# ---------------------------------------------------------------------------
# center= trajectory
# ---------------------------------------------------------------------------

def test_center_spline_matches_agama_spline():
    """The interpolated center itself, vs agama.Spline(reg=True) per component."""
    tr = _traj()
    spl = _AgamaTimeSpline(tr[:, 0], tr[:, 1:4])
    tq = np.linspace(-8.0, 2.0, 601)          # includes both extrapolation wings
    for k in range(3):
        ref = agama.Spline(tr[:, 0], tr[:, k + 1], reg=True)(tq)
        mine = np.array([spl(float(q))[k] for q in tq])
        assert _rel(mine, ref) < _TOL, f"component {k}"


def test_center_trajectory_vs_agama_potential():
    tr = _traj()
    ap = agama.Potential(dict(center=tr, **_HALO))
    gp = PotentialGPU(type='Plummer', mass=_HALO['mass'],
                      scaleRadius=_HALO['scaleRadius'], center=tr)
    assert isinstance(gp, ShiftedPotentialGPU)

    xyz = _pts()
    xg = cp.asarray(xyz)
    for t in np.linspace(-8.0, 2.0, 41):
        t = float(t)
        assert _rel(cp.asnumpy(gp.potential(xg, t=t)), ap.potential(xyz, t=t)) < 1e-10
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-10


def test_center_hermite_seven_column():
    tr = _traj()
    vel = np.gradient(tr[:, 1:4], tr[:, 0], axis=0)
    tr7 = np.column_stack([tr, vel])

    ap = agama.Potential(dict(center=tr7, **_HALO))
    gp = PotentialGPU(type='Plummer', mass=_HALO['mass'],
                      scaleRadius=_HALO['scaleRadius'], center=tr7)

    xyz = _pts(512)
    xg = cp.asarray(xyz)
    for t in np.linspace(-8.0, 2.0, 31):
        t = float(t)
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-10


def test_center_static_unchanged():
    c = np.array([3.0, -4.0, 5.0])
    ap = agama.Potential(dict(center=c, **_HALO))
    gp = PotentialGPU(type='Plummer', mass=_HALO['mass'],
                      scaleRadius=_HALO['scaleRadius'], center=c)
    xyz = _pts(512)
    xg = cp.asarray(xyz)
    assert _rel(cp.asnumpy(gp.force(xg)), ap.force(xyz)) < 1e-10


def test_center_regression_not_a_knot_would_fail():
    """Guard: the fixture is coarse enough that not-a-knot is measurably wrong.

    If this stops holding, the other tests here lose their teeth.
    """
    from scipy.interpolate import CubicSpline
    tr = _traj()
    tq = np.linspace(-6.0, 0.0, 2001)
    ref = agama.Spline(tr[:, 0], tr[:, 1], reg=True)(tq)
    nak = CubicSpline(tr[:, 0], tr[:, 1], bc_type='not-a-knot')(tq)
    assert np.max(np.abs(ref - nak)) > 1e-3      # kpc


# ---------------------------------------------------------------------------
# scale= / ampl=
# ---------------------------------------------------------------------------

# Agama's scale= always carries *two* values A(t), S(t), so its time table is
# (T,3) = [t, ampl, scale].  Our (T,2) = [t, scale] form is a GPU-only
# convenience in which ampl comes from the separate ampl= kwarg; it is compared
# below against the equivalent Agama (T,3) table with A == 1.

def _scale_table(T=25):
    t = np.linspace(-6.0, 0.0, T)
    return np.column_stack([t, 1.0 + 0.4 * np.tanh(1.5 * (t + 3.0))])


def _scale_ampl_table(T=25):
    t = np.linspace(-6.0, 0.0, T)
    return np.column_stack([t,
                            0.5 + 0.5 * np.exp(0.3 * t),         # ampl(t)
                            1.0 + 0.4 * np.tanh(1.5 * (t + 3.0))])  # scale(t)


def _as_agama_scale(tab):
    """Our (T,2) [t, scale] -> Agama's (T,3) [t, ampl=1, scale]."""
    if tab.shape[1] == 3:
        return tab
    return np.column_stack([tab[:, 0], np.ones(tab.shape[0]), tab[:, 1]])


def test_scale_spline_matches_agama_spline():
    tab = _scale_table()
    spl = _AgamaTimeSpline(tab[:, 0], tab[:, 1:])
    tq = np.linspace(-8.0, 2.0, 401)
    ref = agama.Spline(tab[:, 0], tab[:, 1], reg=True)(tq)
    mine = np.array([spl(float(q))[0] for q in tq])
    assert _rel(mine, ref) < _TOL


def test_scale_ampl_splines_match_agama_spline():
    tab = _scale_ampl_table()
    spl = _AgamaTimeSpline(tab[:, 0], tab[:, 1:])
    tq = np.linspace(-8.0, 2.0, 401)
    for k in (0, 1):
        ref = agama.Spline(tab[:, 0], tab[:, k + 1], reg=True)(tq)
        mine = np.array([spl(float(q))[k] for q in tq])
        assert _rel(mine, ref) < _TOL, f"column {k}"


@pytest.mark.parametrize("table_fn", [_scale_table, _scale_ampl_table],
                         ids=["scale-only", "ampl+scale"])
def test_scaled_potential_vs_agama(table_fn):
    tab = table_fn()
    ap = agama.Potential(dict(scale=_as_agama_scale(tab), **_HALO))
    gp = ScaledPotentialGPU(PotentialGPU(**_HALO), scale=tab)

    xyz = _pts(1000)
    xg = cp.asarray(xyz)
    for t in np.linspace(-8.0, 2.0, 31):
        t = float(t)
        assert _rel(cp.asnumpy(gp.potential(xg, t=t)), ap.potential(xyz, t=t)) < 1e-9
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-9


def test_scaled_static_unchanged():
    # Agama's static form is the two-value string "A S"
    gp = ScaledPotentialGPU(PotentialGPU(**_HALO), scale=2.0, ampl=0.5)
    ap = agama.Potential(dict(scale="0.5 2.0", **_HALO))
    xyz = _pts(512)
    xg = cp.asarray(xyz)
    assert _rel(cp.asnumpy(gp.potential(xg)), ap.potential(xyz)) < 1e-10
    assert _rel(cp.asnumpy(gp.force(xg)), ap.force(xyz)) < 1e-10


# ---------------------------------------------------------------------------
# combined: shifted + scaled + uniform acceleration, the full MW/LMC shape
# ---------------------------------------------------------------------------

def test_full_stack_vs_agama():
    tr = _traj()
    t = np.linspace(-6.0, 0.0, 200)
    accMW = np.column_stack([t, 30 * np.sin(0.7 * t),
                             120 * np.exp(0.3 * t), 50 * np.cos(0.4 * t)])

    mw = dict(type='NFW', mass=1e12, scaleRadius=20.0)
    ap = agama.Potential(
        dict(**mw),
        dict(center=tr, **_HALO),
        dict(type='UniformAcceleration', file=accMW),
    )
    gp = (PotentialGPU(**mw)
          + PotentialGPU(type='Plummer', mass=_HALO['mass'],
                         scaleRadius=_HALO['scaleRadius'], center=tr)
          + PotentialGPU(type='UniformAcceleration', file=accMW))

    xyz = _pts(5000)
    xg = cp.asarray(xyz)
    for t_ in np.linspace(-7.0, 1.0, 25):
        t_ = float(t_)
        assert _rel(cp.asnumpy(gp.force(xg, t=t_)), ap.force(xyz, t=t_)) < 1e-9
        assert _rel(cp.asnumpy(gp.potential(xg, t=t_)), ap.potential(xyz, t=t_)) < 1e-9
