"""
tests/test_cosmology.py
=======================

Tests for ``nbody_streams.utils._cosmology`` -- the flat LCDM background and
the comoving/peculiar frame transforms.

Covers
------
1. ``time(a=...)`` is the exact analytic inverse of ``scale_factor(t)``.
2. ``H``, ``a''`` and ``z`` against finite differences of ``a(t)`` -- the
   closed forms, not a spline, so they must match to quadrature accuracy.
3. ``FlatLCDM.static()`` is the degenerate a=1, H=0 background, and both
   transforms then reduce to the identity.
4. ``comoving_to_physical`` / ``physical_to_comoving`` round-trip, for a single
   vector and for a batch, with scalar and per-particle times.
5. ``FlatLCDM.fit`` recovers the parameters it was generated with.
6. Argument validation ("give exactly one of ...").

Run with::

    pytest tests/test_cosmology.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from nbody_streams.utils import (
    KPC_PER_GYR_PER_KMS,
    FlatLCDM,
    comoving_to_physical,
    physical_to_comoving,
)

TIMES = np.array([1.0, 3.0, 7.5, 10.0, 13.8])


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def test_time_inverts_scale_factor():
    """t(a(t)) == t to machine precision -- both branches are closed form."""
    cosmo = FlatLCDM()
    assert np.allclose(cosmo.time(a=cosmo.scale_factor(TIMES)), TIMES, rtol=0, atol=1e-12)


def test_redshift_matches_scale_factor():
    cosmo = FlatLCDM()
    a = cosmo.scale_factor(TIMES)
    assert np.allclose(cosmo.redshift(t=TIMES), 1.0 / a - 1.0)
    assert np.allclose(cosmo.redshift(a=a), 1.0 / a - 1.0)
    assert np.allclose(cosmo.scale_factor(z=cosmo.redshift(t=TIMES)), a)


def test_hubble_parameter_is_adot_over_a():
    """H = a'/a, checked against a central difference of a(t)."""
    cosmo = FlatLCDM()
    h = 1e-5
    adot = (cosmo.scale_factor(TIMES + h) - cosmo.scale_factor(TIMES - h)) / (2 * h)
    # a is in Gyr^-1 here; H is in (km/s)/kpc.
    H_fd = adot / cosmo.scale_factor(TIMES) / KPC_PER_GYR_PER_KMS
    assert np.allclose(cosmo.hubble_parameter(t=TIMES), H_fd, rtol=1e-8)


def test_a_double_dot_matches_finite_difference():
    """The closed-form a'' is what a spline through a(t) cannot give reliably."""
    cosmo = FlatLCDM()
    h = 1e-3
    addot_fd = (cosmo.scale_factor(TIMES + h) - 2 * cosmo.scale_factor(TIMES)
                + cosmo.scale_factor(TIMES - h)) / h ** 2
    addot_fd /= KPC_PER_GYR_PER_KMS ** 2          # Gyr^-2 -> ((km/s)/kpc)^2
    # A second difference is only O(h^2) accurate; 1e-5 is its floor, not the
    # closed form's.
    assert np.allclose(cosmo.a_double_dot(t=TIMES), addot_fd, rtol=1e-5)


def test_present_day_age_is_sensible():
    cosmo = FlatLCDM()                     # FIRE m12i parameters
    assert 13.0 < float(cosmo.time(a=1.0)) < 14.5
    assert float(cosmo.time(z=0.0)) == pytest.approx(float(cosmo.time(a=1.0)))


def test_fit_recovers_parameters():
    truth = FlatLCDM(hubble=0.68, omega_matter=0.31)
    t = np.linspace(0.5, 13.8, 200)
    fitted = FlatLCDM.fit(t, truth.scale_factor(t))
    assert fitted.hubble == pytest.approx(truth.hubble, rel=1e-6)
    assert fitted.omega_matter == pytest.approx(truth.omega_matter, rel=1e-6)
    assert fitted.omega_lambda == pytest.approx(1.0 - fitted.omega_matter)


def test_repr_round_trips():
    assert repr(FlatLCDM.static()) == "FlatLCDM.static()"
    assert "hubble=0.702" in repr(FlatLCDM())


# ---------------------------------------------------------------------------
# Static (non-cosmological) background
# ---------------------------------------------------------------------------

def test_static_background_is_degenerate():
    s = FlatLCDM.static()
    assert np.all(s.scale_factor(TIMES) == 1.0)
    assert np.all(s.hubble_parameter(t=TIMES) == 0.0)
    assert np.all(s.a_double_dot(t=TIMES) == 0.0)
    assert np.all(s.redshift(t=TIMES) == 0.0)


def test_static_transforms_are_the_identity():
    s = FlatLCDM.static()
    rng = np.random.default_rng(0)
    x, v = rng.normal(size=(11, 3)) * 40.0, rng.normal(size=(11, 3)) * 150.0
    r, vv = comoving_to_physical(x, v, 9.0, s)
    assert np.array_equal(r, x) and np.array_equal(vv, v)
    x2, v2 = physical_to_comoving(x, v, 9.0, s)
    assert np.array_equal(x2, x) and np.array_equal(v2, v)


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3,), (11, 3), (4, 5, 3)])
def test_transform_round_trip(shape):
    cosmo = FlatLCDM()
    rng = np.random.default_rng(1)
    x, v_pec = rng.normal(size=shape) * 40.0, rng.normal(size=shape) * 150.0
    r, v = comoving_to_physical(x, v_pec, 8.5, cosmo)
    x2, v2 = physical_to_comoving(r, v, 8.5, cosmo)
    assert np.allclose(x2, x, rtol=1e-13, atol=1e-11)
    assert np.allclose(v2, v_pec, rtol=1e-13, atol=1e-11)


def test_transform_with_per_particle_times():
    """t broadcasts against the leading axes, not the trailing axis of 3."""
    cosmo = FlatLCDM()
    rng = np.random.default_rng(2)
    x, v_pec = rng.normal(size=(9, 3)) * 40.0, rng.normal(size=(9, 3)) * 150.0
    t = np.linspace(6.0, 13.0, 9)
    r, v = comoving_to_physical(x, v_pec, t, cosmo)
    assert np.allclose(r, x * cosmo.scale_factor(t)[:, None])
    x2, v2 = physical_to_comoving(r, v, t, cosmo)
    assert np.allclose(x2, x) and np.allclose(v2, v_pec)


def test_hubble_flow_sign():
    """A particle at rest in the comoving frame recedes at v = H r."""
    cosmo = FlatLCDM()
    x = np.array([100.0, 0.0, 0.0])
    r, v = comoving_to_physical(x, np.zeros(3), 13.0, cosmo)
    assert np.allclose(v, cosmo.hubble_parameter(t=13.0) * r)
    assert v[0] > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_exactly_one_argument_is_enforced():
    cosmo = FlatLCDM()
    with pytest.raises(ValueError, match="exactly one"):
        cosmo.scale_factor(1.0, z=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        cosmo.scale_factor()
    with pytest.raises(ValueError, match="exactly one"):
        cosmo.time(a=1.0, z=0.0)
    with pytest.raises(ValueError, match="exactly one"):
        cosmo.hubble_parameter()
    with pytest.raises(ValueError, match="exactly one"):
        cosmo.a_double_dot(t=1.0, a=0.5)


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        FlatLCDM.fit([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="nothing to fit"):
        FlatLCDM.fit([0.1, 0.2], [0.05, 0.1], t_min=0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
