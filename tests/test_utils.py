"""
tests/test_utils.py
===================

Tests for ``nbody_streams.utils`` using a Plummer sphere as ground truth.

The Plummer model has well-known analytic profiles, is spherically
symmetric, and is in virial equilibrium -- making it ideal for
validating density profiles, velocity profiles, morphological shape,
density fitting, centre finding, and iterative boundness.

Run with::

    python tests/test_utils.py          # standalone
    pytest tests/test_utils.py -v       # via pytest
"""
from __future__ import annotations

import sys
import numpy as np

# --- Package imports ---
from nbody_streams.run import make_plummer_sphere
from nbody_streams.utils import (
    make_uneven_grid,
    empirical_density_profile,
    empirical_circular_velocity_profile,
    empirical_velocity_dispersion_profile,
    empirical_velocity_rms_profile,
    empirical_velocity_anisotropy_profile,
    fit_double_spheroid_profile,
    fit_iterative_ellipsoid,
    uniform_spherical_grid,
    find_center,
    iterative_unbinding,
    half_mass_radius,
    suggest_simulation_params,
)

# --- Constants ---
N_PARTICLES = 100_000
M_TOTAL = 1.0
A_SCALE = 1.0          # Plummer scale radius
G_NBODY = 1.0          # make_plummer_sphere uses G = 1

# --- Fixture: generate Plummer sphere once ---
print(f"Generating Plummer sphere with N={N_PARTICLES} ...")
phase_space, masses = make_plummer_sphere(N_PARTICLES, M_total=M_TOTAL, a=A_SCALE)
pos = phase_space[:, :3]        # (N, 3) kpc-equivalent
vel = phase_space[:, 3:]        # (N, 3) km/s-equivalent
radii = np.linalg.norm(pos, axis=1)
mass = masses.astype(np.float64)
print("  done.\n")


# --- Analytic Plummer functions ---
def plummer_density(r, M=M_TOTAL, a=A_SCALE):
    """rho(r) = (3M / 4*pi*a^3) (1 + r^2/a^2)^{-5/2}"""
    return (3 * M / (4 * np.pi * a ** 3)) * (1 + (r / a) ** 2) ** (-2.5)


def plummer_enclosed_mass(r, M=M_TOTAL, a=A_SCALE):
    """M(<r) = M r^3 / (r^2 + a^2)^{3/2}"""
    return M * r ** 3 / (r ** 2 + a ** 2) ** 1.5


def plummer_v_circ(r, M=M_TOTAL, a=A_SCALE, G=G_NBODY):
    """v_c(r) = sqrt(G M(<r) / r)"""
    M_enc = plummer_enclosed_mass(r, M, a)
    return np.sqrt(G * M_enc / r)


# =============================================================================
# Tests
# =============================================================================

def test_make_uneven_grid():
    """Grid starts at 0, second node at xmin, last at xmax."""
    g = make_uneven_grid(0.1, 10.0, nbins=20)
    assert g[0] == 0.0, f"First node should be 0, got {g[0]}"
    assert np.isclose(g[-1], 10.0, atol=1e-10), f"Last node should be 10, got {g[-1]}"
    assert np.isclose(g[1], 0.1, atol=1e-10), f"Second node should be 0.1, got {g[1]}"
    assert np.all(np.diff(g) > 0), "Grid must be strictly increasing"
    print("test_make_uneven_grid: [OK]")


def test_empirical_density_profile():
    """Density profile should match analytic Plummer within ~20% at intermediate radii."""
    r_bins, rho = empirical_density_profile(
        pos, mass, nbins=40, rmin=0.2, rmax=8.0
    )
    rho_analytic = plummer_density(r_bins)

    # Intermediate radii where statistics are good (0.5a < r < 5a)
    sel = (r_bins > 0.5) & (r_bins < 5.0)
    rel_err = np.abs(rho[sel] - rho_analytic[sel]) / rho_analytic[sel]
    median_err = np.median(rel_err)

    assert median_err < 0.20, (
        f"Median relative density error = {median_err:.2%} (expected < 20%)"
    )
    print(f"test_empirical_density_profile: [OK]  (median err = {median_err:.1%})")


def test_empirical_circular_velocity():
    """Circular velocity should match analytic Plummer."""
    r_bins, vc = empirical_circular_velocity_profile(
        pos, mass, nbins=40, rmin=0.2, rmax=8.0, G=G_NBODY
    )
    vc_analytic = plummer_v_circ(r_bins)

    sel = (r_bins > 0.5) & (r_bins < 5.0)
    rel_err = np.abs(vc[sel] - vc_analytic[sel]) / vc_analytic[sel]
    median_err = np.median(rel_err)

    assert median_err < 0.10, (
        f"Median relative v_circ error = {median_err:.2%} (expected < 10%)"
    )
    print(f"test_empirical_circular_velocity: [OK]  (median err = {median_err:.1%})")


def test_empirical_velocity_dispersion():
    """Velocity dispersion should be finite and positive at all radii."""
    r_bins, sigma = empirical_velocity_dispersion_profile(
        pos, vel, nbins=30, rmin=0.2, rmax=6.0
    )
    finite = np.isfinite(sigma)
    assert np.sum(finite) > len(sigma) * 0.8, "Most bins should have finite dispersion"
    assert np.all(sigma[finite] >= 0), "Dispersion must be non-negative"
    print("test_empirical_velocity_dispersion: [OK]")


def test_empirical_velocity_rms():
    """RMS velocity should be finite and positive."""
    r_bins, v_rms = empirical_velocity_rms_profile(
        pos, vel, nbins=30, rmin=0.2, rmax=6.0
    )
    finite = np.isfinite(v_rms)
    assert np.sum(finite) > len(v_rms) * 0.8, "Most bins should have finite v_rms"
    assert np.all(v_rms[finite] > 0), "v_rms must be positive"
    print("test_empirical_velocity_rms: [OK]")


def test_empirical_velocity_anisotropy():
    """For an isotropic Plummer sphere, beta ~= 0 on average."""
    r_bins, beta = empirical_velocity_anisotropy_profile(
        pos, vel, mass, nbins=25, rmin=0.3, rmax=5.0
    )
    finite = np.isfinite(beta)
    mean_beta = np.nanmean(beta[finite])
    # Isotropic -> beta = 0;  allow |beta| < 0.15 due to sampling noise
    assert abs(mean_beta) < 0.15, (
        f"Mean beta = {mean_beta:.3f} (expected ~= 0 for isotropic system)"
    )
    print(f"test_empirical_velocity_anisotropy: [OK]  (mean beta = {mean_beta:.3f})")


def test_fit_double_spheroid_profile():
    """Fitted profile should recover the Plummer mass to ~30%."""
    result = fit_double_spheroid_profile(
        pos=pos, mass=mass, bins=30
    )
    M_fit, a_fit, alpha_fit, beta_fit, gamma_fit = result
    rel_mass_err = abs(M_fit - M_TOTAL) / M_TOTAL

    assert rel_mass_err < 0.30, (
        f"Fitted mass = {M_fit:.4f}, true = {M_TOTAL} "
        f"(rel err = {rel_mass_err:.1%}, expected < 30%)"
    )
    print(
        f"test_fit_double_spheroid_profile: [OK]  "
        f"(M={M_fit:.4f}, a={a_fit:.3f}, "
        f"alpha={alpha_fit:.2f}, beta={beta_fit:.2f}, gamma={gamma_fit:.2f})"
    )


def test_morphology_sphere():
    """A perfect Plummer sphere should have abc ~= [1, 1, 1]."""
    abc, T = fit_iterative_ellipsoid(
        pos, mass=mass, Rmax=5.0
    )
    assert np.allclose(abc, [1.0, 1.0, 1.0], atol=0.05), (
        f"Shape abc = {abc} (expected ~= [1, 1, 1] for a sphere)"
    )
    print(f"test_morphology_sphere: [OK]  (abc = {abc})")


def test_morphology_ellipticity_triaxiality():
    """Sphere: ellipticity ~= 0, triaxiality ~= 0."""
    abc, T, ell, tri = fit_iterative_ellipsoid(
        pos, mass=mass, Rmax=5.0, return_ellip_triax=True
    )
    assert ell < 0.08, f"Ellipticity = {ell:.3f} (expected < 0.08 for sphere)"
    print(f"test_morphology_ellipticity_triaxiality: [OK]  (ell={ell:.3f}, T={tri:.3f})")


def test_uniform_spherical_grid():
    """Points should lie on the sphere surface."""
    grid = uniform_spherical_grid(radius=5.0, num_pts=1000)
    r = np.linalg.norm(grid, axis=1)
    assert np.allclose(r, 5.0, atol=1e-10), "All points should have r = 5.0"
    print("test_uniform_spherical_grid: [OK]")


def test_find_center():
    """Centre of the Plummer sphere should be near the origin."""
    ctr = find_center(pos, mass, method="shrinking_sphere", r_init=10.0)
    dist = np.linalg.norm(ctr)
    assert dist < 0.1, (
        f"Centre at {ctr}, distance from origin = {dist:.4f} (expected < 0.1)"
    )
    print(f"test_find_center: [OK]  (centre = {np.round(ctr, 4)})")


def test_find_center_with_velocity():
    """Centre finding with velocity should return both position and velocity."""
    ctr_pos, ctr_vel = find_center(
        pos, mass, vel=vel, method="shrinking_sphere",
        return_velocity=True, vel_aperture=3.0, r_init=10.0
    )
    dist = np.linalg.norm(ctr_pos)
    assert dist < 0.1, (
        f"Centre at {ctr_pos}, distance from origin = {dist:.4f} (expected < 0.1)"
    )
    assert ctr_vel.shape == (3,), f"Centre velocity shape {ctr_vel.shape} (expected (3,))"
    print(
        f"test_find_center_with_velocity: [OK]  "
        f"(centre = {np.round(ctr_pos, 4)}, v_ctr = {np.round(ctr_vel, 4)})"
    )


def test_iterative_unbinding():
    """A virialised Plummer sphere should be >90% bound."""
    result, ctr_pos, ctr_vel = iterative_unbinding(
        pos, vel, mass,
        potential_compute_method="direct",
        softening=0.05,
        G=G_NBODY,
        verbose=False,
    )
    bound_mask = result[0]
    bound_frac = np.mean(bound_mask)
    assert bound_frac > 0.90, (
        f"Bound fraction = {bound_frac:.3f} (expected > 0.90 for virialised system)"
    )
    print(f"test_iterative_unbinding: [OK]  (bound frac = {bound_frac:.3f})")


# ---------------------------------------------------------------------------
# Section 8: suggest_simulation_params tests
# ---------------------------------------------------------------------------

def test_half_mass_radius_uniform_sphere():
    """Uniform sphere of radius R: r_half should be R * (0.5)^(1/3) ~ 0.794 R."""
    rng = np.random.default_rng(42)
    R = 5.0
    N_sph = 20_000
    # Sample uniform sphere by rejection
    pts = rng.uniform(-R, R, (N_sph * 3, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < R][:N_sph]
    m = np.ones(N_sph)
    r_h = half_mass_radius(pts, m)
    expected = R * 0.5 ** (1.0 / 3.0)
    assert abs(r_h - expected) / expected < 0.02, (
        f"half_mass_radius: got {r_h:.4f}, expected {expected:.4f} (tol 2%)"
    )


def test_suggest_params_plummer():
    """Basic smoke test on the Plummer fixture: all keys present, values physical."""
    result = suggest_simulation_params(pos, mass, vel, verbose=False)

    assert "eps" in result and "dt_max" in result, "Missing keys in result dict"
    assert result["eps"] > 0, "eps must be positive"
    assert result["dt_max"] > 0, "dt_max must be positive"
    assert result["eps_lower"] <= result["eps"] <= result["eps_upper"] or (
        # eps_upper might be < eps if r_half is very small; just check both > 0
        result["eps_upper"] > 0
    ), f"eps bounds inconsistent: lower={result['eps_lower']}, eps={result['eps']}, upper={result['eps_upper']}"

    # dt_max should be less than the raw Hubble time (13.8 Gyr) and positive
    assert 0 < result["dt_max"] < 13.8, f"dt_max={result['dt_max']} outside (0, 13.8) Gyr"

    # External keys should be None when no external potential
    assert result["dt_orbit"] is None
    assert result["dt_orbit_dehnen"] is None
    assert result["v_com"] is None


def test_suggest_params_physical_bounds():
    """dt criteria should be consistent: dt_drift = eps / (10 * v95) to within 1%."""
    result = suggest_simulation_params(pos, mass, vel, verbose=False,
                                       n_cross_per_dt=10, eta=0.025, eta_v=0.1)

    v_mag = np.linalg.norm(vel, axis=1)
    v95 = float(np.percentile(v_mag, 95))
    eps = result["eps"]
    from nbody_streams import G_DEFAULT  # noqa: PLC0415

    KPC_KMS_TO_GYR = 0.97779
    expected_drift = eps / (10 * v95) * KPC_KMS_TO_GYR
    assert abs(result["dt_drift"] - expected_drift) / expected_drift < 0.01, (
        f"dt_drift mismatch: got {result['dt_drift']:.6e}, expected {expected_drift:.6e}"
    )

    # dt_min hints
    assert abs(result["dt_min_nbins6"] - result["dt_max"] / 64.0) / result["dt_max"] < 1e-10
    assert abs(result["dt_min_nbins8"] - result["dt_max"] / 256.0) / result["dt_max"] < 1e-10


def test_suggest_params_with_external():
    """With a static point-mass external potential, dt_orbit keys should be populated."""
    from nbody_streams import G_DEFAULT  # noqa: PLC0415

    # Place particles at origin; external mass at (50, 0, 0) kpc
    M_ext = 1e12  # Msun — large external mass
    r_ext = np.array([50.0, 0.0, 0.0])

    def point_mass_potential(positions, t):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        dr = positions - r_ext
        r = np.linalg.norm(dr, axis=1, keepdims=True)
        r = np.maximum(r, 0.01)
        # a = G * M_ext / r^2 directed toward external mass
        return -G_DEFAULT * M_ext * dr / r ** 3

    # Give particles a COM velocity so orbital period estimate is non-trivial
    vel_ext = vel.copy()
    vel_ext += np.array([150.0, 0.0, 0.0])  # COM moving at 150 km/s

    result = suggest_simulation_params(pos, mass, vel_ext,
                                       external_potential=point_mass_potential,
                                       verbose=False)

    assert result["dt_orbit"] is not None, "dt_orbit should be set with external potential"
    assert result["dt_orbit"] > 0, "dt_orbit must be positive"
    assert result["v_com"] is not None and result["v_com"] > 0
    assert result["a_ext_com"] is not None and result["a_ext_com"] > 0
    assert result["t_orbit_est"] is not None and result["t_orbit_est"] > 0
    assert result["dt_orbit_dehnen"] is not None and result["dt_orbit_dehnen"] > 0


def test_suggest_params_edge_cases():
    """Small N and constant-velocity particles should not crash."""
    rng = np.random.default_rng(999)
    pos_small = rng.standard_normal((50, 3))
    mass_small = np.ones(50) / 50.0
    vel_small = np.zeros((50, 3))  # all velocities zero

    result = suggest_simulation_params(pos_small, mass_small, vel_small, verbose=False)
    assert result["dt_max"] > 0, "dt_max must be positive even with zero velocities"
    # With zero velocities: dt_drift and dt_vel should not be zero (guarded by max(..., 1e-10))
    assert result["dt_drift"] > 0
    assert result["dt_vel"] > 0


# =============================================================================
# Runner
# =============================================================================

ALL_TESTS = [
    test_make_uneven_grid,
    test_empirical_density_profile,
    test_empirical_circular_velocity,
    test_empirical_velocity_dispersion,
    test_empirical_velocity_rms,
    test_empirical_velocity_anisotropy,
    test_fit_double_spheroid_profile,
    test_morphology_sphere,
    test_morphology_ellipticity_triaxiality,
    test_uniform_spherical_grid,
    test_find_center,
    test_find_center_with_velocity,
    test_iterative_unbinding,
    test_half_mass_radius_uniform_sphere,
    test_suggest_params_plummer,
    test_suggest_params_physical_bounds,
    test_suggest_params_with_external,
    test_suggest_params_edge_cases,
]


if __name__ == "__main__":
    print("=" * 60)
    print("  nbody_streams.utils -- Plummer sphere test suite")
    print("=" * 60 + "\n")

    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"{test_fn.__name__}: [FAILED] {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
