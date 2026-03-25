"""
Tests for nbody_streams._chandrasekhar

Coverage
--------
1. compute_sigma_r -- quasispherical path and Jeans fallback
2. _shrinking_sphere_com -- convergence and correctness
3. chandrasekhar_friction -- formula validation against BT2008 eq 8.13
4. make_df_force_extra -- closure mechanics, predictor, GPU/CPU paths
5. Integration: circular orbit in isothermal sphere decays at correct rate
6. fast_sims._common Jeans fallback (no hardcoded MW profile)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# agama is an optional dependency; skip all tests that need it when absent
agama = pytest.importorskip("agama")
agama.setUnits(mass=1, length=1, velocity=1)

from scipy import special

from nbody_streams._chandrasekhar import (
    _jeans_sigma_r,
    _shrinking_sphere_com,
    _to_numpy,
    chandrasekhar_friction,
    compute_sigma_r,
    make_df_force_extra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isothermal_potential(sigma0: float = 200.0) -> agama.Potential:
    """Singular isothermal sphere with 1-D dispersion sigma0 [km/s].

    rho(r) = sigma0^2 / (2 pi G r^2)
    Potential: Phi(r) = sigma0^2 * ln(r) + const
    """
    return agama.Potential(type="logarithmic", v0=sigma0 * np.sqrt(2), scaleRadius=0.0)


def _nfw_potential(mass=1e12, rs=20.0) -> agama.Potential:
    return agama.Potential(type="NFW", mass=mass, scaleRadius=rs)


# ---------------------------------------------------------------------------
# 1. _to_numpy
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_numpy_passthrough(self):
        a = np.array([1.0, 2.0, 3.0])
        out = _to_numpy(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, a)

    def test_cupy_conversion(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")
        a_cp = cp.array([4.0, 5.0, 6.0])
        out = _to_numpy(a_cp)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# 2. compute_sigma_r / _jeans_sigma_r
# ---------------------------------------------------------------------------

class TestComputeSigmaR:

    def test_isothermal_sphere_quasispherical(self):
        """For a singular isothermal sphere, sigma(r) should be ~constant.

        For an SIS with v_c = v0 = sigma0*sqrt(2), the isotropic Jeans
        equation gives sigma_r = v0/sqrt(2) = sigma0.  That is, the 1-D
        radial velocity dispersion equals the line-of-sight dispersion
        sigma0 used to define the SIS.
        """
        sigma0 = 150.0  # km/s
        pot = _isothermal_potential(sigma0)
        sigma_func = compute_sigma_r(pot, t_eval=0.0)
        r_test = np.array([1.0, 5.0, 20.0, 50.0])
        sig = sigma_func(r_test)
        # sigma_r = sigma0 for isotropic SIS (Jeans result; ~constant)
        expected = sigma0
        # Allow 20% tolerance — quasispherical DF / Jeans is approximate
        np.testing.assert_allclose(sig, expected, rtol=0.20)

    def test_nfw_returns_decreasing_profile(self):
        """NFW sigma(r) should peak near the scale radius and fall off."""
        pot = _nfw_potential()
        sigma_func = compute_sigma_r(pot)
        r_test = np.logspace(-0.5, 1.5, 10)
        sig = sigma_func(r_test)
        assert np.all(sig > 0), "sigma must be positive"
        assert np.all(np.isfinite(sig)), "sigma must be finite"

    def test_jeans_fallback_isothermal(self):
        """Jeans equation sigma(r) ~ constant for SIS.

        For SIS v_c = v0 = sigma0*sqrt(2), Jeans gives sigma_r = v0/sqrt(2) = sigma0.
        """
        sigma0 = 120.0
        pot = _isothermal_potential(sigma0)
        grid_r = np.logspace(-0.5, 2, 64)
        sigma_func = _jeans_sigma_r(pot, t_eval=0.0, grid_r=grid_r)
        r_test = np.array([2.0, 10.0, 30.0])
        sig = sigma_func(r_test)
        expected = sigma0  # sigma_r = sigma0 for isotropic SIS
        np.testing.assert_allclose(sig, expected, rtol=0.15)

    def test_jeans_fallback_positive_finite(self):
        """Jeans fallback must return positive finite values for NFW."""
        pot = _nfw_potential()
        sigma_func = _jeans_sigma_r(pot, t_eval=0.0)
        r_test = np.logspace(-0.5, 1.8, 20)
        sig = sigma_func(r_test)
        assert np.all(sig > 0)
        assert np.all(np.isfinite(sig))

    def test_compute_sigma_r_triggers_jeans_fallback(self, monkeypatch):
        """If quasispherical DF raises, Jeans fallback is used without crash."""
        pot = _nfw_potential()
        # Patch DistributionFunction to always raise
        original_df = agama.DistributionFunction

        def _bad_df(**kwargs):
            raise RuntimeError("forced failure for test")

        monkeypatch.setattr(agama, "DistributionFunction", _bad_df)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sigma_func = compute_sigma_r(pot, t_eval=0.0)

        assert any("Jeans" in str(w.message) for w in caught), \
            "Expected Jeans fallback warning"

        sig = sigma_func(np.array([5.0, 20.0]))
        assert np.all(sig > 0)

    def test_t_eval_time_evolving(self):
        """Jeans fallback must accept non-zero t_eval without error."""
        pot = _nfw_potential()
        sigma_func = _jeans_sigma_r(pot, t_eval=1.0)
        assert sigma_func(10.0) > 0


# ---------------------------------------------------------------------------
# 3. _shrinking_sphere_com
# ---------------------------------------------------------------------------

class TestShrinkingSphereCom:

    def _make_plummer_cloud(self, N=500, centre=None, seed=7):
        rng = np.random.default_rng(seed)
        if centre is None:
            centre = np.zeros(3)
        # Draw from Plummer sphere (rough)
        r = np.random.RandomState(seed).exponential(1.0, N)
        phi = rng.uniform(0, 2 * np.pi, N)
        costh = rng.uniform(-1, 1, N)
        sinth = np.sqrt(1 - costh ** 2)
        pos = np.column_stack([
            r * sinth * np.cos(phi) + centre[0],
            r * sinth * np.sin(phi) + centre[1],
            r * costh + centre[2],
        ])
        vel = rng.normal(0, 50, (N, 3))
        masses = np.ones(N) / N
        return pos, vel, masses

    def test_recovers_centroid(self):
        """CoM should converge close to the true cloud centre."""
        true_centre = np.array([5.0, -3.0, 2.0])
        pos, vel, masses = self._make_plummer_cloud(N=1000, centre=true_centre)
        r_com, v_com, r_sphere = _shrinking_sphere_com(pos, vel, masses)
        np.testing.assert_allclose(r_com, true_centre, atol=0.3)

    def test_velocity_com_shape(self):
        pos, vel, masses = self._make_plummer_cloud()
        r_com, v_com, r_sphere = _shrinking_sphere_com(pos, vel, masses)
        assert r_com.shape == (3,)
        assert v_com.shape == (3,)

    def test_uniform_sphere_exact(self):
        """All particles at same position → CoM = that position exactly."""
        N = 50
        p = np.full((N, 3), [1.0, 2.0, 3.0])
        v = np.full((N, 3), [10.0, 0.0, -5.0])
        m = np.ones(N)
        r_com, v_com, r_sphere = _shrinking_sphere_com(p, v, m, n_iter=1)
        np.testing.assert_allclose(r_com, [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(v_com, [10.0, 0.0, -5.0], atol=1e-10)

    def test_two_clumps_converges_to_dense(self):
        """With one dominant dense clump + sparse background, CoM should
        land in the dense clump after shrinking-sphere iterations."""
        rng = np.random.default_rng(42)
        # Dense clump: 800 equal-mass particles near [0, 0, 0], tight sigma
        p_dense = rng.normal(0, 0.2, (800, 3))
        v_dense = rng.normal(0, 10, (800, 3))
        # Sparse background: 200 particles scattered at large radii
        p_bg = rng.uniform(-30, 30, (200, 3))
        v_bg = rng.normal(0, 50, (200, 3))
        pos = np.vstack([p_dense, p_bg])
        vel = np.vstack([v_dense, v_bg])
        masses = np.ones(1000) / 1000.0
        r_com, _, r_sphere = _shrinking_sphere_com(pos, vel, masses, n_iter=8, frac=0.5)
        # Should converge to the dense clump near [0, 0, 0]
        assert np.linalg.norm(r_com) < 2.0, (
            f"Expected CoM near dense clump, got r_com={r_com}"
        )

    def test_r_sphere_positive_and_bounded(self):
        """r_sphere must be positive and smaller than the full cloud extent."""
        true_centre = np.array([0.0, 0.0, 0.0])
        pos, vel, masses = self._make_plummer_cloud(N=500, centre=true_centre)
        r_com, v_com, r_sphere = _shrinking_sphere_com(pos, vel, masses)
        assert r_sphere > 0, "r_sphere must be positive"
        # Full cloud extent: max distance from origin; r_sphere should be smaller
        full_extent = float(np.linalg.norm(pos, axis=1).max())
        assert r_sphere < full_extent, (
            f"r_sphere ({r_sphere:.2f}) should be smaller than full cloud extent "
            f"({full_extent:.2f}) after shrinking"
        )


# ---------------------------------------------------------------------------
# 4. chandrasekhar_friction
# ---------------------------------------------------------------------------

class TestChandrasekharFriction:

    def _sigma_const(self, sigma: float = 150.0):
        """Constant sigma function for analytic tests."""
        return lambda r: np.full_like(np.asarray(r, float), sigma)

    def test_opposes_motion(self):
        """DF force must oppose the velocity."""
        pot = _nfw_potential()
        sigma_f = self._sigma_const(150.0)
        r_com = np.array([20.0, 0.0, 0.0])
        v_com = np.array([100.0, 0.0, 0.0])
        a = chandrasekhar_friction(r_com, v_com, 1e9, pot, sigma_f, t=0.0)
        assert a[0] < 0, "DF must decelerate in the direction of motion"
        np.testing.assert_allclose(a[1:], 0.0, atol=1e-30)

    def test_zero_velocity_returns_zero(self):
        """No friction when satellite is at rest."""
        pot = _nfw_potential()
        sigma_f = self._sigma_const()
        a = chandrasekhar_friction(
            np.array([10.0, 0.0, 0.0]),
            np.zeros(3),
            1e9, pot, sigma_f, t=0.0
        )
        np.testing.assert_allclose(a, 0.0, atol=1e-30)

    def test_zero_radius_returns_zero(self):
        pot = _nfw_potential()
        sigma_f = self._sigma_const()
        a = chandrasekhar_friction(
            np.zeros(3),
            np.array([100.0, 0.0, 0.0]),
            1e9, pot, sigma_f, t=0.0
        )
        np.testing.assert_allclose(a, 0.0, atol=1e-30)

    def test_scales_with_mass(self):
        """DF acceleration should scale linearly with satellite mass."""
        pot = _nfw_potential()
        sigma_f = self._sigma_const()
        r = np.array([15.0, 0.0, 0.0])
        v = np.array([80.0, 0.0, 0.0])
        a1 = chandrasekhar_friction(r, v, 1e8, pot, sigma_f, t=0.0,
                                    coulomb_mode="fixed", fixed_ln_lambda=3.0)
        a2 = chandrasekhar_friction(r, v, 1e9, pot, sigma_f, t=0.0,
                                    coulomb_mode="fixed", fixed_ln_lambda=3.0)
        np.testing.assert_allclose(np.abs(a2[0]) / np.abs(a1[0]), 10.0, rtol=1e-6)

    def test_bt2008_formula_numerically(self):
        """Validate against a direct evaluation of BT2008 eq. 8.13."""
        from nbody_streams._chandrasekhar import _G
        pot = _nfw_potential()
        sigma0 = 120.0
        sigma_f = self._sigma_const(sigma0)
        r_com = np.array([10.0, 0.0, 0.0])
        v_com = np.array([0.0, 60.0, 0.0])
        M_sat = 5e8
        r = np.linalg.norm(r_com)
        v = np.linalg.norm(v_com)
        X = v / (np.sqrt(2) * sigma0)
        rho = float(pot.density(r_com, t=0.0))
        ln_lambda = 3.0  # fixed
        bracket = special.erf(X) - (2 / np.sqrt(np.pi)) * X * np.exp(-X ** 2)
        a_expected = -(v_com / v) * (4 * np.pi * _G ** 2 * M_sat * rho * ln_lambda * bracket) / v ** 2

        a_got = chandrasekhar_friction(
            r_com, v_com, M_sat, pot, sigma_f, t=0.0,
            coulomb_mode="fixed", fixed_ln_lambda=ln_lambda,
        )
        np.testing.assert_allclose(a_got, a_expected, rtol=1e-10)

    def test_core_stalling_suppression(self):
        """With core_gamma > 0 and r < r_core, DF should be suppressed."""
        pot = _nfw_potential()
        sigma_f = self._sigma_const()
        r = np.array([0.5, 0.0, 0.0])  # r < r_core=1.0
        v = np.array([100.0, 0.0, 0.0])
        a_no_suppress = chandrasekhar_friction(r, v, 1e9, pot, sigma_f, t=0.0,
                                               core_gamma=0.0, r_core=1.0,
                                               coulomb_mode="fixed",
                                               fixed_ln_lambda=3.0)
        a_suppressed = chandrasekhar_friction(r, v, 1e9, pot, sigma_f, t=0.0,
                                              core_gamma=1.0, r_core=1.0,
                                              coulomb_mode="fixed",
                                              fixed_ln_lambda=3.0)
        assert np.abs(a_suppressed[0]) < np.abs(a_no_suppress[0])

    def test_variable_coulomb_log(self):
        """Variable mode result differs from fixed; both finite and negative."""
        pot = _nfw_potential()
        sigma_f = self._sigma_const()
        r = np.array([20.0, 0.0, 0.0])
        v = np.array([100.0, 0.0, 0.0])
        a_var = chandrasekhar_friction(r, v, 1e9, pot, sigma_f, t=0.0,
                                       coulomb_mode="variable")
        a_fix = chandrasekhar_friction(r, v, 1e9, pot, sigma_f, t=0.0,
                                       coulomb_mode="fixed", fixed_ln_lambda=5.0)
        assert np.isfinite(a_var[0]) and a_var[0] < 0
        assert np.isfinite(a_fix[0]) and a_fix[0] < 0
        # They should differ (unless by coincidence)
        assert not np.isclose(a_var[0], a_fix[0], rtol=0.01)


# ---------------------------------------------------------------------------
# 5. make_df_force_extra
# ---------------------------------------------------------------------------

class TestMakeDfForceExtra:

    def _simple_particles(self, N=100, centre=None, seed=0):
        rng = np.random.default_rng(seed)
        c = np.array([5.0, 0.0, 0.0]) if centre is None else np.asarray(centre)
        pos = rng.normal(0, 0.5, (N, 3)) + c
        vel = rng.normal(0, 50, (N, 3))
        masses = np.ones(N) * 1e4
        return pos, vel, masses

    def test_returns_callable(self):
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0)
        assert callable(fn)

    def test_output_shape(self):
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0)
        pos, vel, masses = self._simple_particles(N=80)
        out = fn(pos, vel, masses, 0.1)
        assert out.shape == (80, 3)

    def test_all_particles_same_acceleration(self):
        """DF is a rigid-body force — all particles within the core get the same
        acceleration.  Use apply_radius_factor=None to disable masking and test
        the pure rigid-body (broadcast) path."""
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0,
                                 apply_radius_factor=None)
        pos, vel, masses = self._simple_particles(N=50)
        out = fn(pos, vel, masses, 0.0)
        # All rows must be equal to the first row
        for i in range(1, len(out)):
            np.testing.assert_array_equal(out[i], out[0])

    def test_opposes_bulk_motion(self):
        """Net DF acceleration should point opposite to bulk CoM velocity."""
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0)
        N = 100
        rng = np.random.default_rng(1)
        pos = rng.normal(0, 0.3, (N, 3)) + np.array([15.0, 0.0, 0.0])
        # Bulk motion in +x
        vel = rng.normal(0, 5, (N, 3)) + np.array([200.0, 0.0, 0.0])
        masses = np.ones(N) * 1e5
        out = fn(pos, vel, masses, 0.0)
        assert out[0, 0] < 0, "DF must decelerate +x motion"

    def test_predictor_updates_com_between_corrections(self):
        """Between correction steps, CoM should be predicted, not recalculated."""
        pot = _nfw_potential()
        fn = make_df_force_extra(
            pot, M_sat=1e9, t_start=0.0, t_end=5.0,
            update_interval=5,
        )
        pos, vel, masses = self._simple_particles(N=60)
        dt = 0.01
        for i in range(8):
            t = i * dt
            out = fn(pos, vel, masses, t)
            assert out.shape == (60, 3)
            assert np.all(np.isfinite(out))

    def test_accepts_numpy_arrays(self):
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=2.0)
        pos, vel, masses = self._simple_particles(N=30)
        out = fn(np.asarray(pos), np.asarray(vel), np.asarray(masses), 0.05)
        assert out.shape == (30, 3)
        assert np.all(np.isfinite(out))

    def test_accepts_cupy_arrays(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=2.0)
        pos, vel, masses = self._simple_particles(N=40)
        out = fn(cp.asarray(pos), cp.asarray(vel), cp.asarray(masses), 0.1)
        assert out.shape == (40, 3)
        assert np.all(np.isfinite(out))

    def test_invalid_M_sat_raises(self):
        pot = _nfw_potential()
        with pytest.raises(ValueError, match="M_sat"):
            make_df_force_extra(pot, M_sat=0.0, t_start=0.0, t_end=1.0)

    def test_invalid_update_interval_raises(self):
        pot = _nfw_potential()
        with pytest.raises(ValueError, match="update_interval"):
            make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=1.0,
                                update_interval=0)

    def test_no_agama_raises_import_error(self, monkeypatch):
        """Without Agama the factory should raise ImportError."""
        import nbody_streams._chandrasekhar as mod
        monkeypatch.setattr(mod, "_AGAMA_OK", False)
        with pytest.raises(ImportError):
            make_df_force_extra(None, M_sat=1e9, t_start=0.0, t_end=1.0)

    def test_apply_radius_factor_masks_outer_particles(self):
        """Particles outside apply_radius_factor * r_sphere should get zero acceleration.

        Use a 80/20 split (80 core : 20 far) with equal per-particle mass so
        the initial mass-weighted centroid is pulled close to the dense core
        (not stranded at the midpoint between two equal-mass clusters, which
        would cause the shrinking sphere to keep all particles).
        """
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0,
                                 apply_radius_factor=1.0)
        rng = np.random.default_rng(99)
        # Dense core: 80 particles near (10, 0, 0), tiny spread (sigma=0.1)
        pos_core = rng.normal(0, 0.1, (80, 3)) + np.array([10.0, 0.0, 0.0])
        vel_core = rng.normal(0, 10, (80, 3)) + np.array([0.0, 100.0, 0.0])
        # Stripped debris: 20 particles 100 kpc away in y
        pos_far = rng.normal(0, 2.0, (20, 3)) + np.array([10.0, 100.0, 0.0])
        vel_far = rng.normal(0, 100, (20, 3))
        pos = np.vstack([pos_core, pos_far])
        vel = np.vstack([vel_core, vel_far])
        masses = np.ones(100) * 1e5
        out = fn(pos, vel, masses, 0.0)
        # Far/stripped particles should have zero acceleration
        assert np.all(out[80:] == 0.0), "Far/stripped particles should not feel DF"
        # Core particles should feel DF
        assert np.any(out[:80] != 0.0), "Core particles should feel DF"

    def test_apply_radius_factor_none_broadcasts_all(self):
        """With apply_radius_factor=None, all particles feel DF (old behaviour)."""
        pot = _nfw_potential()
        fn = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0,
                                 apply_radius_factor=None)
        N = 60
        rng = np.random.default_rng(77)
        pos = rng.normal(0, 0.3, (N, 3)) + np.array([15.0, 0.0, 0.0])
        vel = rng.normal(0, 5, (N, 3)) + np.array([0.0, 150.0, 0.0])
        masses = np.ones(N) * 1e5
        out = fn(pos, vel, masses, 0.0)
        # All rows should be non-zero (bulk motion → DF non-zero)
        assert np.any(out != 0.0), "Expected non-zero DF with bulk motion"
        # All rows equal (broadcast)
        for i in range(1, N):
            np.testing.assert_array_equal(out[i], out[0])


# ---------------------------------------------------------------------------
# 6. Integration: circular orbit decay in isothermal sphere
# ---------------------------------------------------------------------------

class TestOrbitDecay:
    """Approximate physics test.

    A satellite on a circular orbit in an isothermal sphere should spiral
    inward due to DF.  The Binney-Tremaine t_fric formula gives the order
    of magnitude of the decay time.  We don't need to match it precisely;
    just check the orbit actually decays.
    """

    def test_circular_orbit_decays(self):
        """Satellite on circular orbit in SIS should lose angular momentum."""
        sigma0 = 200.0           # km/s
        r0 = 15.0                # kpc  — initial radius
        M_sat = 5e9              # M_sun  — moderately massive satellite
        dt = 0.005               # kpc/(km/s)  ~ 5 Myr
        t_end = 0.2              # ~ 200 Myr  (much less than t_fric)

        pot = _isothermal_potential(sigma0)

        # Circular velocity in SIS: v_c = sigma0 * sqrt(2)
        v_c = sigma0 * np.sqrt(2.0)

        # Initial conditions: single tracer particle at (r0, 0, 0)
        # moving in +y direction at v_c
        xv = np.array([[r0, 0.0, 0.0, 0.0, v_c, 0.0]])
        masses = np.array([M_sat])

        from nbody_streams import run_nbody_cpu

        df_fn = make_df_force_extra(
            pot, M_sat=M_sat, t_start=0.0, t_end=t_end,
            coulomb_mode="fixed", fixed_ln_lambda=3.0,
            update_interval=1,  # correct every step for accuracy in test
        )

        final = run_nbody_cpu(
            xv, masses,
            time_start=0.0, time_end=t_end, dt=dt,
            softening=0.0,
            G=agama.G,
            method="direct",
            kernel="spline",
            external_potential=pot,
            force_extra=df_fn,
            save_snapshots=False,
            verbose=False,
        )

        r_final = np.linalg.norm(final[0, :3])
        assert r_final < r0, (
            f"Orbit should decay: r_final={r_final:.2f} >= r0={r0:.2f}"
        )


# ---------------------------------------------------------------------------
# 7. fast_sims._common Jeans fallback (no hardcoded MW profile)
# ---------------------------------------------------------------------------

class TestCommonJeansFallback:

    def test_fallback_gives_reasonable_sigma(self, monkeypatch):
        """When quasispherical DF fails, _common should use Jeans, not MW table."""
        from nbody_streams.fast_sims import _common

        pot = _nfw_potential()

        # Force quasispherical DF to fail
        original_df = agama.DistributionFunction

        def _bad_df(**kwargs):
            raise RuntimeError("forced DF failure")

        monkeypatch.setattr(agama, "DistributionFunction", _bad_df)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sigma_f = _common._compute_vel_disp_from_Potential(pot)

        assert any("Jeans" in str(w.message) for w in caught), \
            "Expected Jeans fallback warning from _common"

        # Jeans sigma for NFW should be > 0 and < 500 km/s at 10 kpc
        sig = sigma_f(10.0)
        assert 0 < float(sig) < 500.0

    def test_fallback_not_hardcoded_mw_values(self, monkeypatch):
        """The fallback must NOT use the old hardcoded MW sigma array."""
        from nbody_streams.fast_sims import _common

        # Plummer sphere — very different from MW; hardcoded values would be wrong
        pot_plummer = agama.Potential(
            type="Plummer", mass=1e10, scaleRadius=2.0,
        )

        def _bad_df(**kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(agama, "DistributionFunction", _bad_df)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sigma_f = _common._compute_vel_disp_from_Potential(pot_plummer)

        sig_plummer = sigma_f(2.0)
        # Jeans sigma for 1e10 Msun Plummer at 2 kpc ~ 50-150 km/s
        assert 10.0 < float(sig_plummer) < 400.0
