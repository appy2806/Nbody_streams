"""
test_uniform_acceleration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the time-dependent ``UniformAcceleration`` GPU potential against
Agama CPU.

``Phi(x, t) = -a(t)·x`` --- the non-inertial-frame term used when integrating in
the MW disc frame while the LMC pulls the halo around.

Covers
------
1. The time interpolator itself vs ``agama.Spline(t, y, reg=True)`` --- Agama
   builds a *natural* cubic spline with the Hyman regularization filter, which
   is **not** SciPy's default ``not-a-knot`` spline.
2. potential / force / density / forceDeriv vs Agama CPU, inside and outside
   the tabulated time range (Agama extrapolates linearly).
3. The 7-column Hermite form ``[t, a, da/dt]``.
4. Constant (no-file) construction, unchanged from before.
5. Construction routes: ``PotentialGPU(type=..., file=array)``,
   ``file=<path>``, an INI section, and inside a composite.
6. Regression: ``file=`` used to be silently dropped, yielding a=0.
"""

import os
import tempfile

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

from nbody_streams.agama_helper import PotentialGPU
from nbody_streams.agama_helper._analytic_potentials import (
    UniformAccelerationGPU,
    _AgamaTimeSpline,
    _read_accel_table,
)

# Machine-precision agreement is expected: both sides evaluate the same cubic
# Hermite polynomial and the same linear Phi = -a.x.
_TOL = 1e-12


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _acc_table(T=200, t0=-14.0, t1=0.0):
    """A smooth-ish (T,4) acceleration table [t, ax, ay, az]."""
    t = np.linspace(t0, t1, T)
    return np.column_stack([
        t,
        30.0 * np.sin(0.7 * t),
        120.0 * np.exp(0.3 * t),
        50.0 * np.cos(0.4 * t) + 0.5 * t ** 2,
    ])


def _acc_table_jumpy(T=40):
    """A table with a sharp jump --- where the regularization filter actually bites."""
    rng = np.random.default_rng(7)
    t = np.linspace(-10.0, 0.0, T)
    return np.column_stack([
        t,
        np.where(t < -5, 0.0, 80.0) + rng.normal(scale=0.5, size=T),
        np.cumsum(np.abs(rng.normal(scale=5.0, size=T))),
        rng.normal(scale=20.0, size=T),
    ])


def _pts(N=2000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30.0, 30.0, (N, 3)).astype(np.float64)


def _rel(gpu, cpu):
    cpu = np.asarray(cpu)
    return float(np.max(np.abs(np.asarray(gpu) - cpu)) / (np.max(np.abs(cpu)) + 1e-30))


# ---------------------------------------------------------------------------
# 1. The time interpolator vs agama.Spline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table_fn", [_acc_table, _acc_table_jumpy],
                         ids=["smooth", "sharp-jump"])
def test_spline_matches_agama_spline(table_fn):
    """The regularized natural cubic spline must reproduce agama.Spline(reg=True)."""
    tab = table_fn()
    spl = _read_accel_table(tab)

    # sample well outside the range on both sides to exercise extrapolation
    span = tab[-1, 0] - tab[0, 0]
    tq = np.linspace(tab[0, 0] - 0.3 * span, tab[-1, 0] + 0.3 * span, 501)

    for k in range(3):
        ref = agama.Spline(tab[:, 0], tab[:, k + 1], reg=True)(tq)
        mine = np.array([spl(float(q))[k] for q in tq])
        assert _rel(mine, ref) < _TOL, f"component {k}"


def test_spline_differs_from_not_a_knot():
    """Guard: SciPy's default spline is *not* what Agama uses.

    If this ever stops differing, the fixture is too smooth to be a meaningful
    regression test for the boundary-condition choice.
    """
    from scipy.interpolate import CubicSpline

    tab = _acc_table_jumpy()
    spl = _read_accel_table(tab)
    tq = np.linspace(tab[0, 0], tab[-1, 0], 301)

    nak = CubicSpline(tab[:, 0], tab[:, 1], bc_type='not-a-knot')(tq)
    mine = np.array([spl(float(q))[0] for q in tq])
    assert _rel(mine, nak) > 1e-3


def test_unsorted_and_single_row():
    tab = _acc_table(T=20)
    shuffled = tab[np.random.default_rng(0).permutation(tab.shape[0])]
    a_sorted = _read_accel_table(tab)
    a_shuf = _read_accel_table(shuffled)
    for q in np.linspace(-14, 0, 37):
        assert np.allclose(a_sorted(float(q)), a_shuf(float(q)), rtol=0, atol=1e-12)

    # a single row is a constant in time
    one = _read_accel_table(np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert np.allclose(one(-99.0), (1.0, 2.0, 3.0))
    assert np.allclose(one(+99.0), (1.0, 2.0, 3.0))


def test_duplicate_timestamps_rejected():
    tab = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        _read_accel_table(tab)


def test_bad_shape_rejected():
    with pytest.raises(ValueError, match=r"\(T,4\)"):
        _read_accel_table(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# 2. Field evaluation vs Agama CPU
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table_fn", [_acc_table, _acc_table_jumpy],
                         ids=["smooth", "sharp-jump"])
def test_fields_match_agama(table_fn):
    tab = table_fn()
    ap = agama.Potential(type='UniformAcceleration', file=tab)
    gp = PotentialGPU(type='UniformAcceleration', file=tab)

    xyz = _pts()
    xg = cp.asarray(xyz)

    span = tab[-1, 0] - tab[0, 0]
    times = np.concatenate([
        np.linspace(tab[0, 0] - 0.2 * span, tab[-1, 0] + 0.2 * span, 41),
        tab[:5, 0], tab[-5:, 0],          # exactly on knots
    ])

    for t in times:
        t = float(t)
        phi_cpu, acc_cpu, dF_cpu = ap.eval(xyz, pot=True, acc=True, der=True, t=t)

        assert _rel(cp.asnumpy(gp.potential(xg, t=t)), phi_cpu) < _TOL, f"phi at t={t}"
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), acc_cpu) < _TOL, f"force at t={t}"

        f_gpu, dF_gpu = gp.forceDeriv(xg, t=t)
        assert _rel(cp.asnumpy(f_gpu), acc_cpu) < _TOL
        # both are identically zero for a potential linear in x
        assert np.max(np.abs(cp.asnumpy(dF_gpu))) == 0.0
        assert np.max(np.abs(dF_cpu)) == 0.0

        # Agama's BasePotential::densityCar returns Laplacian/(4 pi) = 0 here
        rho_cpu = ap.density(xyz, t=t)
        rho_gpu = cp.asnumpy(gp.density(xg, t=t))
        assert np.max(np.abs(rho_cpu)) == 0.0
        assert np.max(np.abs(rho_gpu)) == 0.0


def test_force_is_uniform_in_space():
    """The whole point: the force must not depend on position."""
    gp = PotentialGPU(type='UniformAcceleration', file=_acc_table())
    f = cp.asnumpy(gp.force(cp.asarray(_pts(500)), t=-4.2))
    assert np.allclose(f, f[0], rtol=0, atol=1e-14)


def test_eval_and_evaldriv_agree_with_force():
    gp = PotentialGPU(type='UniformAcceleration', file=_acc_table())
    xg = cp.asarray(_pts(256))
    for t in (-13.0, -6.5, -0.1, 3.0):
        phi, f, d = gp.evalDeriv(xg, t=t)
        assert cp.allclose(f, gp.force(xg, t=t))
        assert cp.allclose(phi, gp.potential(xg, t=t))
        assert cp.allclose(gp.eval(xg, acc=True, t=t), gp.force(xg, t=t))
        assert float(cp.max(cp.abs(d))) == 0.0


def test_single_point_input():
    gp = PotentialGPU(type='UniformAcceleration', file=_acc_table())
    ap = agama.Potential(type='UniformAcceleration', file=_acc_table())
    p = np.array([3.0, -4.0, 5.0])
    for t in (-9.0, -1.0):
        assert cp.asnumpy(gp.force(cp.asarray(p), t=t)).shape == (3,)
        assert _rel(cp.asnumpy(gp.force(cp.asarray(p), t=t)), ap.force(p, t=t)) < _TOL
        assert _rel(float(cp.asnumpy(gp.potential(cp.asarray(p), t=t))),
                    ap.potential(p, t=t)) < _TOL


# ---------------------------------------------------------------------------
# 3. Seven-column Hermite form
# ---------------------------------------------------------------------------

def test_hermite_seven_column_form():
    tab = _acc_table()
    der = np.gradient(tab[:, 1:4], tab[:, 0], axis=0)
    tab7 = np.column_stack([tab, der])

    ap = agama.Potential(type='UniformAcceleration', file=tab7)
    gp = PotentialGPU(type='UniformAcceleration', file=tab7)

    xyz = _pts(512)
    xg = cp.asarray(xyz)
    for t in np.linspace(-16.0, 2.0, 41):
        t = float(t)
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < _TOL
        assert _rel(cp.asnumpy(gp.potential(xg, t=t)), ap.potential(xyz, t=t)) < _TOL


# ---------------------------------------------------------------------------
# 4. Constant form (no file) --- unchanged behaviour
# ---------------------------------------------------------------------------

def test_constant_form():
    gp = UniformAccelerationGPU(ax=0.01, ay=-0.02, az=0.005)
    assert not gp.is_time_dependent
    xg = cp.asarray(_pts(128))
    f = cp.asnumpy(gp.force(xg))
    assert np.allclose(f, [0.01, -0.02, 0.005])
    # time argument must not change anything
    assert np.allclose(cp.asnumpy(gp.force(xg, t=-7.0)), f)
    # Phi = -a.x
    xyz = cp.asnumpy(xg)
    assert np.allclose(cp.asnumpy(gp.potential(xg)),
                       -(xyz @ np.array([0.01, -0.02, 0.005])))


def test_from_agama_raises_informatively():
    ap = agama.Potential(type='UniformAcceleration', file=_acc_table())
    with pytest.raises(TypeError, match="does not export"):
        UniformAccelerationGPU.from_agama(ap)


# ---------------------------------------------------------------------------
# 5. Construction routes
# ---------------------------------------------------------------------------

def test_file_path_route(tmp_path):
    tab = _acc_table(T=60)
    path = tmp_path / "accMW"
    np.savetxt(path, tab)

    ap = agama.Potential(type='UniformAcceleration', file=str(path))
    gp = PotentialGPU(type='UniformAcceleration', file=str(path))

    xyz = _pts(256)
    xg = cp.asarray(xyz)
    for t in (-13.0, -5.5, -0.2, 1.0):
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-9


def test_ini_route(tmp_path):
    """An INI [Potential] section with type=UniformAcceleration and a relative file=."""
    tab = _acc_table(T=60)
    np.savetxt(tmp_path / "acc.txt", tab)
    ini = tmp_path / "pot.ini"
    ini.write_text(
        "[Potential accel]\n"
        "type=UniformAcceleration\n"
        "file=acc.txt\n"
    )

    gp = PotentialGPU(file=str(ini))
    ap = agama.Potential(type='UniformAcceleration', file=str(tmp_path / "acc.txt"))

    xyz = _pts(256)
    xg = cp.asarray(xyz)
    for t in (-12.0, -3.0, -0.5):
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-9


def test_composite_with_nfw():
    """UniformAcceleration summed with a static halo, vs the same Agama composite."""
    tab = _acc_table(T=80)
    gp = (PotentialGPU(type='NFW', mass=1e12, scaleRadius=20.0)
          + PotentialGPU(type='UniformAcceleration', file=tab))
    ap = agama.Potential(
        dict(type='NFW', mass=1e12, scaleRadius=20.0),
        dict(type='UniformAcceleration', file=tab),
    )

    xyz = _pts(1000)
    xg = cp.asarray(xyz)
    for t in (-13.0, -7.0, -1.0, 0.5):
        assert _rel(cp.asnumpy(gp.potential(xg, t=t)), ap.potential(xyz, t=t)) < 1e-10
        assert _rel(cp.asnumpy(gp.force(xg, t=t)), ap.force(xyz, t=t)) < 1e-10


# ---------------------------------------------------------------------------
# 6. Regression --- file= must not be silently dropped
# ---------------------------------------------------------------------------

def test_file_is_not_silently_dropped():
    """Regression: PotentialGPU(type='UniformAcceleration', file=...) once ignored
    file= and built a zero-acceleration no-op instead of erroring."""
    tab = _acc_table()
    gp = PotentialGPU(type='UniformAcceleration', file=tab)
    assert gp.is_time_dependent
    f = cp.asnumpy(gp.force(cp.asarray(_pts(64)), t=-7.0))
    assert np.max(np.abs(f)) > 1.0


def test_agama_object_conversion_rejected():
    """Converting an agama UniformAcceleration object must fail loudly, not fit a
    Multipole to a potential that is linear in x."""
    ap = agama.Potential(type='UniformAcceleration', file=_acc_table())
    with pytest.raises(TypeError, match="does not expose"):
        PotentialGPU(ap)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
