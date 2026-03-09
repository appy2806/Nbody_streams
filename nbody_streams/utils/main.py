"""
nbody_streams.utils.main
=========================

Utility functions for particle-based analysis of N-body simulations.

Sections
--------
1. Grid / binning helpers
2. Empirical radial profiles (density, circular velocity, dispersions, anisotropy)
3. Density-profile fitting (double-power-law / Zhao 1996)
4. Morphological diagnostics (iterative structure-tensor shape measurement)
5. Spherical grid generators
6. Centre finding (shrinking sphere, density peak / KDE)
7. Iterative boundness (unbinding via energy criterion)

Author: Arpit Arora
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from ._validation import (
    validate_masses,
    validate_nbins,
    validate_positions,
    validate_velocities,
)

# ---------------------------------------------------------------------------
# Optional dependencies (following the pattern in fields.py)
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import curve_fit, minimize, root_scalar
    from scipy.stats import binned_statistic
    from scipy import linalg as sp_linalg

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "SciPy not available. Some utils functions will not work. "
        "Install with: pip install scipy",
        ImportWarning,
    )

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Dummy decorator so decorated functions still run (pure NumPy fallback)
    def jit(nopython=True, cache=True):
        def decorator(func):
            return func
        return decorator

try:
    import agama

    agama.setUnits(mass=1, length=1, velocity=1)
    AGAMA_AVAILABLE = True
except Exception:
    AGAMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "make_uneven_grid",
    # profiles
    "empirical_density_profile",
    "empirical_circular_velocity_profile",
    "empirical_velocity_dispersion_profile",
    "empirical_velocity_rms_profile",
    "empirical_velocity_anisotropy_profile",
    # fitting
    "fit_double_spheroid_profile",
    "fit_dehnen_profile",
    "fit_plummer_profile",
    # morphology
    "fit_iterative_ellipsoid",
    # grids
    "uniform_spherical_grid",
    "spherical_spiral_grid",
    # centre finding
    "find_center",
    "find_center_position",   # deprecated alias
    # boundness
    "compute_iterative_boundness",
    "iterative_unbinding",
]

# Default gravitational constant (kpc, km/s, Msun)
G_DEFAULT = 4.300917270069976e-06


# --- 1. Grid / binning helpers ---

def make_uneven_grid(
    xmin: float,
    xmax: float | None = None,
    nbins: int = 10,
) -> np.ndarray:
    """Create a 1-D grid with unequally spaced nodes.

    The grid starts at 0, the second node sits at *xmin*, and the last
    node sits at *xmax*.  Spacing grows geometrically.

    Parameters
    ----------
    xmin : float
        Location of the first non-zero node (must be > 0).
    xmax : float or None, optional
        Location of the last node (must be > *xmin*).  If *None*, a
        uniform grid with spacing *xmin* is returned.
    nbins : int, optional
        Total number of bins (>= 3).

    Returns
    -------
    x : np.ndarray, shape ``(nbins,)``
    """
    if nbins < 3:
        raise ValueError("nbins must be at least 3.")
    if xmin <= 0:
        raise ValueError("xmin must be positive.")

    if xmax is None:
        return np.linspace(0, xmin * (nbins - 1), nbins)

    if xmax <= xmin:
        raise ValueError("xmax must be greater than xmin.")

    N = nbins - 1  # number of intervals

    # Check feasibility - fall back to uniform if grading impossible
    if xmax <= N * xmin:
        return np.linspace(0, xmax, nbins)

    # Define equation to solve for Z:
    # f(Z) = (exp(Z * 1) - 1) / (exp(Z * N) - 1) - xmin/xmax = 0
    def f(Z):
        ez = np.exp(-Z)
        ezn = np.exp(-Z * N)
        return np.exp(Z * (1 - N)) * (1 - ez) / (1 - ezn) - (xmin / xmax)
    # Solve for Z using a root-finding method
    sol = root_scalar(f, bracket=[1e-8, 100], method="brentq")
    if not sol.converged:
        raise RuntimeError("Failed to find solution for grid spacing parameter Z.")
    Z = sol.root

    # Now compute the grid nodes using the solved Z
    k = np.arange(nbins)
    x = (np.exp(Z * k) - 1) / (np.exp(Z * N) - 1) * xmax
    return x


# --- 2. Empirical radial profiles ---

def empirical_density_profile(
    pos: np.ndarray,
    mass: np.ndarray | float,
    nbins: int = 50,
    rmin: float = 0.1,
    rmax: float = 600,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the mass-density radial profile :math:`\rho(r)`.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)`` or ``(N,)``
        Particle positions (Cartesian) or pre-computed radii.
    mass : scalar or array_like, shape ``(N,)``
        Particle masses.
    nbins : int
        Number of radial bins.
    rmin, rmax : float
        Inner / outer grid nodes passed to :func:`make_uneven_grid`.

    Returns
    -------
    radius : np.ndarray
        Bin centres.
    density : np.ndarray
        Mass density in each shell.
    """
    _, r_p = validate_positions(pos)
    mass_arr = validate_masses(mass, r_p.shape[0])
    validate_nbins(nbins)

    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    V_shells = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    density, _ = np.histogram(r_p, bins=bins, weights=mass_arr)
    density = density / V_shells
    radius = 0.5 * (bins[1:] + bins[:-1])

    return radius, density


def empirical_circular_velocity_profile(
    pos: np.ndarray,
    mass: np.ndarray | float,
    nbins: int = 50,
    rmin: float = 0.1,
    rmax: float = 600,
    G: float = G_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the circular velocity profile :math:`v_{\rm circ}(r) = \sqrt{G\,M(<r)/r}`.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)`` or ``(N,)``
        Particle positions or radii.
    mass : scalar or array_like, shape ``(N,)``
        Particle masses.
    nbins : int
        Number of radial bins.
    rmin, rmax : float
        Inner / outer grid nodes.
    G : float, optional
        Gravitational constant.  Default gives *v_circ* in km/s when
        *pos* is in kpc and *mass* in :math:`M_\odot`.

    Returns
    -------
    radius : np.ndarray
        Bin centres.
    v_circ : np.ndarray
        Circular velocity in each bin.
    """
    _, r_p = validate_positions(pos)
    mass_arr = validate_masses(mass, r_p.shape[0])
    validate_nbins(nbins)

    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    mass_in_bins, _ = np.histogram(r_p, bins=bins, weights=mass_arr)
    M_enclosed = np.cumsum(mass_in_bins)
    radius = 0.5 * (bins[1:] + bins[:-1])

    with np.errstate(divide="ignore", invalid="ignore"):
        v_circ = np.sqrt(G * M_enclosed / radius)
        v_circ = np.where(radius > 0, v_circ, 0.0)

    return radius, v_circ


def empirical_velocity_dispersion_profile(
    pos: np.ndarray,
    vel: np.ndarray,
    nbins: int = 50,
    rmin: float = 0.1,
    rmax: float = 600,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the velocity dispersion :math:`\sigma_v(r)`.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)`` or ``(N,)``
        Particle positions or radii.
    vel : array_like, shape ``(N, 3)`` or ``(N,)``
        Particle velocities or speed magnitudes.
    nbins : int
        Number of radial bins.
    rmin, rmax : float
        Inner / outer grid nodes.

    Returns
    -------
    radius : np.ndarray
        Bin centres.
    vel_disp : np.ndarray
        Velocity dispersion (std) in each radial bin.
    """
    _, r_p = validate_positions(pos)
    vel_arr = validate_velocities(vel, r_p.shape[0])
    validate_nbins(nbins)

    v_magnitude = (
        np.linalg.norm(vel_arr, axis=1) if vel_arr.ndim == 2 else vel_arr
    )

    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    vel_disp, _, _ = binned_statistic(r_p, v_magnitude, statistic="std", bins=bins)
    radius = 0.5 * (bins[1:] + bins[:-1])
    return radius, vel_disp


def empirical_velocity_rms_profile(
    pos: np.ndarray,
    vel: np.ndarray,
    nbins: int = 50,
    rmin: float = 0.1,
    rmax: float = 600,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the root-mean-square velocity profile :math:`v_{\rm rms}(r)`.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)`` or ``(N,)``
        Particle positions or radii.
    vel : array_like, shape ``(N, 3)``
        Particle velocities.
    nbins : int
        Number of radial bins.
    rmin, rmax : float
        Inner / outer grid nodes.

    Returns
    -------
    radius : np.ndarray
        Bin centres.
    vel_rms : np.ndarray
        RMS velocity in each radial bin.
    """
    pos_arr, r_p = validate_positions(pos)
    vel_arr = validate_velocities(vel, r_p.shape[0])
    validate_nbins(nbins)

    v_mag = np.linalg.norm(vel_arr, axis=-1) if vel_arr.ndim >= 2 else vel_arr

    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    binned_sum_v2, _ = np.histogram(r_p, bins=bins, weights=v_mag ** 2)
    binned_counts, _ = np.histogram(r_p, bins=bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        vel_rms = np.sqrt(binned_sum_v2 / binned_counts)

    radius = 0.5 * (bins[1:] + bins[:-1])
    return radius, vel_rms


def empirical_velocity_anisotropy_profile(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray | float | None = None,
    nbins: int = 50,
    rmin: float = 0.1,
    rmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the velocity anisotropy parameter :math:`\beta(r)`.

    .. math::

        \beta(r) = 1 - \frac{\sigma_t^2}{2\,\sigma_r^2}

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
        Particle positions (Cartesian, **not** radii - needed for
        radial/tangential decomposition).
    vel : array_like, shape ``(N, 3)``
        Particle velocities.
    mass : scalar, array_like, or None, optional
        Particle masses.  *None* -> equal mass.
    nbins : int
        Number of radial bins.
    rmin : float
        Minimum radius.
    rmax : float or None
        Maximum radius.  *None* -> 90th percentile of ``|pos|``.

    Returns
    -------
    r_centres : np.ndarray
        Bin centres.
    beta : np.ndarray
        Anisotropy parameter in each bin.
    """
    pos_arr = np.asarray(pos, dtype=float)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3) for anisotropy computation")

    vel_arr = validate_velocities(vel, pos_arr.shape[0])
    if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
        raise ValueError("vel must have shape (N, 3) for anisotropy computation")

    validate_nbins(nbins)

    r = np.linalg.norm(pos_arr, axis=1)
    vr = np.sum(pos_arr * vel_arr, axis=1) / r
    vt2 = np.sum(vel_arr ** 2, axis=1) - vr ** 2

    if mass is None:
        mass_arr = np.ones(len(r))
    else:
        mass_arr = validate_masses(mass, len(r))

    if rmax is None:
        rmax = float(np.percentile(r, 90))

    edges = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    r_centres = 0.5 * (edges[:-1] + edges[1:])

    hist_mass = binned_statistic(r, mass_arr, statistic="sum", bins=edges)[0]

    hist_m_vr = binned_statistic(r, mass_arr * vr, statistic="sum", bins=edges)[0]
    mean_vr = np.divide(
        hist_m_vr, hist_mass, where=hist_mass > 0, out=np.zeros_like(hist_mass)
    )

    hist_m_vr2 = binned_statistic(r, mass_arr * vr ** 2, statistic="sum", bins=edges)[0]
    sigma_r2 = (
        np.divide(
            hist_m_vr2, hist_mass, where=hist_mass > 0, out=np.zeros_like(hist_mass)
        )
        - mean_vr ** 2
    )

    hist_m_vt2 = binned_statistic(r, mass_arr * vt2, statistic="sum", bins=edges)[0]
    sigma_t2 = np.divide(
        hist_m_vt2, hist_mass, where=hist_mass > 0, out=np.zeros_like(hist_mass)
    )

    sigma_r2[sigma_r2 == 0] = np.nan
    beta = 1.0 - sigma_t2 / (2.0 * sigma_r2)

    return r_centres, beta


# --- 3. Density-profile fitting ---

def double_power_law_density(
    mass: float,
    scaleradius: float,
    alpha: float,
    beta: float,
    gamma: float,
    rcut: float | None = None,
    cutoffstrength: float = 2.0,
) -> Callable[[float | np.ndarray], np.ndarray]:
    r"""Construct a Zhao (1996) double-power-law density profile normalised to total mass.

    .. math::

        \rho(r) = \rho_0\, r^{-\gamma}
        \left[1 + \left(\frac{r}{a}\right)^\alpha\right]^{(\gamma-\beta)/\alpha}
        \times f_{\rm cut}(r)

    Parameters
    ----------
    mass : float
        Total mass.
    scaleradius : float
        Scale radius *a*.
    alpha, beta, gamma : float
        Transition steepness, outer slope, inner slope.
    rcut : float or None, optional
        Cutoff radius.
    cutoffstrength : float, optional
        Exponent of the exponential cutoff (default 2).

    Returns
    -------
    rho : callable
        ``rho(r)`` evaluating the density at scalar or array radii.
    """
    from scipy.integrate import quad

    a = float(scaleradius)
    if (beta <= 3.0) and (rcut is None):
        raise ValueError("beta <= 3 requires a finite rcut to normalise total mass.")

    def _base(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                x > 0.0,
                x ** (-gamma) * (1.0 + x ** alpha) ** (-(beta - gamma) / alpha),
                0.0,
            )
        return out

    base_at_a = 2.0 ** (-(beta - gamma) / alpha)

    def _integrand(r):
        x = r / a
        rho_unit = _base(x) / base_at_a
        if rcut is not None and rcut > 0:
            rho_unit *= np.exp(-((r / rcut) ** cutoffstrength))
        return r ** 2 * rho_unit

    upper = float(rcut * 8.0) if (rcut is not None and rcut > 0) else max(a * 1e4, 1e3)

    I, _ = quad(_integrand, 0.0, upper, epsrel=1e-6, limit=200)
    mass_unit = 4.0 * np.pi * I
    if not np.isfinite(mass_unit) or mass_unit <= 0.0:
        raise RuntimeError(
            "Normalisation integral failed; try providing rcut or different slopes."
        )

    rho_a = float(mass) / mass_unit

    def rho(r):
        r_arr = np.asarray(r, dtype=float)
        x = r_arr / a
        rho_vals = rho_a * (_base(x) / base_at_a)
        if rcut is not None and rcut > 0:
            rho_vals *= np.exp(-((r_arr / rcut) ** cutoffstrength))
        return rho_vals

    return rho


def fit_double_spheroid_profile(
    r_centers: np.ndarray = np.array([]),
    rho_vals: np.ndarray = np.array([]),
    pos: np.ndarray = np.array([]),
    mass: np.ndarray | float = np.array([]),
    bins: int = 20,
    axis_y: float = 1.0,
    axis_z: float = 1.0,
    weighting: str | np.ndarray = "uniform",
    plot_results: bool = False,
    return_profiles: bool = False,
    rcut: float | None = None,
    cutoff_strength: float = 2.0,
) -> Union[
    tuple[float, float, float, float, float],
    tuple[
        tuple[float, float, float, float, float],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
]:
    r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.

    If *r_centers* and *rho_vals* are not supplied, the radial density
    profile is estimated from *pos* and *mass* using ellipsoidal
    radii :math:`\tilde r = \sqrt{x^2 + (y/q_y)^2 + (z/q_z)^2}`.

    When ``agama`` is available the fit uses ``agama.Potential(type='Spheroid')``;
    otherwise a pure-Python double-power-law model is used.

    Parameters
    ----------
    r_centers : array_like
        Radii at which densities are known (if empty, derived from *pos*).
    rho_vals : array_like
        Densities at *r_centers*.
    pos : array_like, shape ``(N, 3)``
        Particle positions (used when *r_centers*/*rho_vals* are empty).
    mass : scalar or array_like
        Particle masses.
    bins : int
        Number of radial bins when estimating density from particles.
    axis_y, axis_z : float
        Axis ratios *b/a* and *c/a* for ellipsoidal radius.
    weighting : str or array_like
        Weight scheme: ``'uniform'``, ``'inner'``, ``'outer'``, ``'sqrt'``,
        ``'inverse_sqrt'``, or a custom array of the same length as the profile.
    plot_results : bool
        Show diagnostic plots (requires ``matplotlib``).
    return_profiles : bool
        If *True*, also return raw profile arrays.
    rcut : float or None
        Optional outer truncation radius.
    cutoff_strength : float
        Steepness of the exponential cutoff.

    Returns
    -------
    params : tuple
        ``(M_fit, a_fit, alpha_fit, beta_fit, gamma_fit)``
    profiles : tuple, optional
        ``(r_centers, rho_vals, rho_residuals, r2_rho_vals)`` when
        *return_profiles* is True.
    """
    r_centers = np.asarray(r_centers, dtype=float)
    rho_vals = np.asarray(rho_vals, dtype=float)

    # Derive profile from particles when not supplied
    if len(r_centers) != len(rho_vals) or len(rho_vals) < 2:
        if pos is None or len(np.asarray(pos)) == 0:
            raise ValueError(
                "Either supply r_centers & rho_vals, or pos & mass."
            )
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"pos must be (N, 3), got shape {pos.shape}"
            )

        mass_arr = validate_masses(mass, pos.shape[0])

        x, y, z = pos.T
        r_tilde = np.sqrt(x ** 2 + (y / axis_y) ** 2 + (z / axis_z) ** 2)

        rmin_auto = 0.1
        rmax_auto = float(np.percentile(r_tilde, 90))
        edges = make_uneven_grid(rmin_auto, rmax_auto, nbins=bins + 1)
        r_centers = 0.5 * (edges[:-1] + edges[1:])

        volumes = (
            (4.0 / 3.0) * np.pi * axis_y * axis_z * (edges[1:] ** 3 - edges[:-1] ** 3)
        )
        counts, _ = np.histogram(r_tilde, bins=edges, weights=mass_arr)
        rho_vals = counts / np.maximum(volumes, 1e-18)
        M0 = float(mass_arr.sum())
    else:
        lnr = np.log(r_centers)
        y = rho_vals * r_centers ** 3
        M0 = float(4.0 * np.pi * np.trapezoid(y, x=lnr))

    # Weights
    if isinstance(weighting, str):
        weight_map = {
            "uniform": np.ones_like(r_centers),
            "inner": 1.0 / np.maximum(r_centers ** 2, 1e-18),
            "outer": r_centers ** 2,
            "sqrt": np.sqrt(np.maximum(r_centers, 1e-18)),
            "inverse_sqrt": 1.0 / np.sqrt(np.maximum(r_centers, 1e-18)),
        }
        weights = weight_map.get(weighting, np.ones_like(r_centers))
    else:
        weighting = np.asarray(weighting)
        if len(weighting) != len(r_centers):
            raise ValueError(
                "weighting array length must match the number of profile points."
            )
        weights = weighting

    log_rho_data = np.log10(np.maximum(rho_vals, 1e-12))

    # Objective
    def objective(params):
        logM, loga, alpha, beta, gamma = params
        try:
            if AGAMA_AVAILABLE:
                pot = agama.Potential(
                    type="Spheroid",
                    mass=10 ** logM,
                    scaleRadius=10 ** loga,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    axisRatioY=axis_y,
                    axisRatioZ=axis_z,
                )
                coords = np.column_stack(
                    [r_centers, np.zeros(len(r_centers)), np.zeros(len(r_centers))]
                )
                rho_model = pot.density(coords)
            else:
                rho_fn = double_power_law_density(
                    10 ** logM,
                    10 ** loga,
                    alpha,
                    beta,
                    gamma,
                    rcut=rcut,
                    cutoffstrength=cutoff_strength,
                )
                rho_model = rho_fn(r_centers)

            log_rho_model = np.log10(np.maximum(rho_model, 1e-12))
            return float(np.sum(weights * (log_rho_model - log_rho_data) ** 2))
        except Exception:
            return 1e10

    p0 = [np.log10(M0), np.log10(5.0), 1.0, 3.0, 1.0]
    bounds = [
        (np.log10(M0 * 0.8), np.log10(M0 * 1.2)),
        (np.log10(0.1), np.log10(r_centers[-1])),
        (0.1, np.inf),
        (1.0, np.inf),
        (0.0, np.inf),
    ]

    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
    logM_fit, loga_fit, alpha_fit, beta_fit, gamma_fit = res.x

    M_fit, a_fit = 10 ** logM_fit, 10 ** loga_fit

    # Evaluate best-fit model for residuals
    if AGAMA_AVAILABLE:
        pot_fit = agama.Potential(
            type="Spheroid",
            mass=M_fit,
            scaleRadius=a_fit,
            alpha=alpha_fit,
            beta=beta_fit,
            gamma=gamma_fit,
            axisRatioY=axis_y,
            axisRatioZ=axis_z,
        )
        coords = np.column_stack(
            [r_centers, np.zeros(len(r_centers)), np.zeros(len(r_centers))]
        )
        rho_model = pot_fit.density(coords)
    else:
        rho_fn = double_power_law_density(
            M_fit,
            a_fit,
            alpha_fit,
            beta_fit,
            gamma_fit,
            rcut=rcut,
            cutoffstrength=cutoff_strength,
        )
        rho_model = rho_fn(r_centers)

    rho_residuals = rho_vals - rho_model
    r2_rho_vals = r_centers ** 2 * rho_vals

    # Diagnostic plot (optional)
    if plot_results:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available - skipping plot.", ImportWarning)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=300)

            ax1.loglog(
                r_centers,
                rho_vals,
                c=plt.rcParams["text.color"],
                markersize=6,
                alpha=0.7,
                label="Data",
            )
            ax1.loglog(r_centers, rho_model, "r--", linewidth=2, label="Model")
            ax1.set_xlabel("r")
            ax1.set_ylabel(r"$\rho(r)$")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ss_res = np.sum(rho_residuals ** 2)
            ss_tot = np.sum((rho_vals - np.mean(rho_vals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            ax1.text(
                0.95, 0.95,
                f"$R^2 = {r_squared:.2f}$",
                transform=ax1.transAxes,
                ha="right", va="top", fontweight="bold", fontsize=12,
            )

            r2_rho_model = r_centers ** 2 * rho_model
            ax2.loglog(
                r_centers,
                r2_rho_vals,
                c=plt.rcParams["text.color"],
                markersize=6,
                alpha=0.7,
                label="Data",
            )
            ax2.loglog(r_centers, r2_rho_model, "r--", linewidth=2, label="Model")
            ax2.set_xlabel("r")
            ax2.set_ylabel(r"$r^2 \rho(r)$")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            r2_rho_residuals = r2_rho_vals - r2_rho_model
            ss_res2 = np.sum(r2_rho_residuals ** 2)
            ss_tot2 = np.sum((r2_rho_vals - np.mean(r2_rho_vals)) ** 2)
            r_squared_2 = 1 - (ss_res2 / ss_tot2)
            ax2.text(
                0.95, 0.95,
                f"$R^2 = {r_squared_2:.2f}$",
                transform=ax2.transAxes,
                ha="right", va="top", fontweight="bold", fontsize=12,
            )

            plt.tight_layout()
            plt.show()

    if return_profiles:
        return (
            (M_fit, a_fit, alpha_fit, beta_fit, gamma_fit),
            (r_centers, rho_vals, rho_residuals, r2_rho_vals),
        )
    return (M_fit, a_fit, alpha_fit, beta_fit, gamma_fit)

def fit_dehnen_profile(
    pos: np.ndarray,
    mass: np.ndarray,
    axis_y: float = 1.0,
    axis_z: float = 1.0,
    bins: int = 50,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    r"""Fit a triaxial Dehnen profile by mapping to ellipsoidal radius.

    Computes :math:`\tilde r = \sqrt{x^2 + (y/q_y)^2 + (z/q_z)^2}`,
    bins into log-spaced shells, and fits the Dehnen (1993)
    density model in log-space via ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
        Particle positions.
    mass : array_like, shape ``(N,)``
        Particle masses.
    axis_y, axis_z : float
        Axis ratios *b/a* and *c/a* for the ellipsoidal radius.
    bins : int
        Number of radial bins.

    Returns
    -------
    M_fit : float
        Fitted total mass.
    a_fit : float
        Scale radius.
    gamma_fit : float
        Inner slope :math:`\gamma \in [0, 3)`.
    r_centers : np.ndarray
        Geometric-mean radial bin centres.
    rho_vals : np.ndarray
        Measured density in each bin.
    """
    pos, _ = validate_positions(pos)
    mass = validate_masses(mass, pos.shape[0], non_negative=True)
    validate_nbins(bins)

    x, y, z = pos.T
    r_tilde = np.sqrt(x**2 + (y / axis_y)**2 + (z / axis_z)**2)

    rmin, rmax = np.percentile(r_tilde, [0.1, 99.9])
    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    r_centers = np.sqrt(edges[:-1] * edges[1:])

    counts, _ = np.histogram(r_tilde, edges, weights=mass)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_vals = counts / volumes

    log_rho = np.log10(np.maximum(rho_vals, 1e-12))

    def log_model(r, logM, loga, gamma):
        M = 10**logM
        a = 10**loga
        pref = (3 - gamma) / (4 * np.pi) * M / a**3
        return np.log10(pref * (r / a)**(-gamma) * (1 + r / a)**(gamma - 4))

    p0 = [np.log10(mass.sum()), np.log10(np.median(r_tilde)), 1.0]
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 3])

    popt, _ = curve_fit(log_model, r_centers, log_rho, p0=p0, bounds=bounds)
    M_fit = 10**popt[0]
    a_fit = 10**popt[1]
    gamma_fit = popt[2]

    return M_fit, a_fit, gamma_fit, r_centers, rho_vals


def fit_plummer_profile(
    pos: np.ndarray,
    mass: np.ndarray,
    bins: int = 30,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    r"""Fit a spherical Plummer profile to particle data.

    Bins particles into log-spaced radial shells and fits the Plummer
    density :math:`\rho(r) = \frac{3M}{4\pi b^3}(1 + r^2/b^2)^{-5/2}`
    in log-space via ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
        Particle positions.
    mass : array_like, shape ``(N,)``
        Particle masses.
    bins : int
        Number of radial bins.

    Returns
    -------
    M_fit : float
        Fitted total mass.
    b_fit : float
        Plummer scale radius.
    r_centers : np.ndarray
        Geometric-mean radial bin centres.
    rho_vals : np.ndarray
        Measured density in each bin.
    """
    pos, r = validate_positions(pos)
    mass = validate_masses(mass, pos.shape[0], non_negative=True)
    validate_nbins(bins)

    rmin, rmax = np.percentile(r, [0.1, 99.9])
    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    r_centers = np.sqrt(edges[:-1] * edges[1:])

    counts, _ = np.histogram(r, edges, weights=mass)
    volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_vals = counts / volumes

    log_rho = np.log10(np.maximum(rho_vals, 1e-12))

    def log_model(r, logM, logb):
        M = 10**logM
        b = 10**logb
        return np.log10(3 * M / (4 * np.pi * b**3) * (1 + (r / b)**2)**(-2.5))

    p0 = [np.log10(mass.sum()), np.log10(np.median(r))]
    popt, _ = curve_fit(log_model, r_centers, log_rho, p0=p0)

    M_fit = 10**popt[0]
    b_fit = 10**popt[1]

    return M_fit, b_fit, r_centers, rho_vals


# --- 4. Morphological diagnostics ---

@jit(nopython=True, cache=True)
def _calculate_particle_distances_sq(pos: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances from the origin (numba-accelerated)."""
    return np.sum(pos ** 2, axis=1)


@jit(nopython=True, cache=True)
def _compute_weighted_structure_tensor(
    coords: np.ndarray,
    mass: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted structure (second-moment) tensor S_ij (numba-accelerated).

    S_ij = sum(m_k * w_k * x_i,k * x_j,k) / sum(m_k * w_k)
    """
    if coords.shape[0] == 0:
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    effective_weights = mass * weights
    sum_ew = np.sum(effective_weights)

    if sum_ew == 0:
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    S = np.zeros((3, 3), dtype=coords.dtype)
    for k in range(coords.shape[0]):
        w = effective_weights[k]
        x, y, z = coords[k, 0], coords[k, 1], coords[k, 2]
        S[0, 0] += w * x * x
        S[0, 1] += w * x * y
        S[0, 2] += w * x * z
        S[1, 1] += w * y * y
        S[1, 2] += w * y * z
        S[2, 2] += w * z * z

    S[1, 0] = S[0, 1]
    S[2, 0] = S[0, 2]
    S[2, 1] = S[1, 2]

    return S / sum_ew


@jit(nopython=True, cache=True)
def _calculate_Rsphall_and_extract(
    pos: np.ndarray,
    transform_matrix_cols_bca: np.ndarray,
    q_ratio: float,
    s_ratio: float,
    Rmax_scaled: float,
    Rmin_val: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ellipsoidal radius and particle selection (numba-accelerated).

    Computes the ellipsoidal radius for every particle and returns a
    boolean mask selecting those within ``[Rmin, Rmax]`` (scaled by
    the current axis ratios).
    """
    if pos.shape[0] == 0:
        return (
            np.empty(0, dtype=pos.dtype),
            np.empty(0, dtype=np.bool_),
            np.empty(0, dtype=pos.dtype),
        )

    coords_eigen = np.dot(pos, transform_matrix_cols_bca)

    scales = np.array([q_ratio, s_ratio, 1.0], dtype=pos.dtype)
    epsilon = 1e-9
    if scales[0] < epsilon:
        scales[0] = epsilon
    if scales[1] < epsilon:
        scales[1] = epsilon

    scaled = np.empty_like(coords_eigen)
    for i in range(coords_eigen.shape[0]):
        scaled[i, 0] = coords_eigen[i, 0] / scales[0]
        scaled[i, 1] = coords_eigen[i, 1] / scales[1]
        scaled[i, 2] = coords_eigen[i, 2] / scales[2]

    Rsph_all = np.sqrt(np.sum(scaled ** 2, axis=1))

    mask = Rsph_all < Rmax_scaled
    if Rmin_val > 0:
        prod_qs = q_ratio * s_ratio
        vol_inv = prod_qs ** (1.0 / 3.0) if prod_qs > 1e-9 else 1e-9
        Rmin_scaled = Rmin_val / vol_inv if vol_inv > 1e-9 else Rmin_val * 1e9
        mask &= Rsph_all >= Rmin_scaled

    return Rsph_all, mask, Rsph_all[mask]


def fit_iterative_ellipsoid(
    pos: np.ndarray,
    mass: np.ndarray | None = None,
    vel: np.ndarray | None = None,
    Rmin: float = 0.0,
    Rmax: float = 1.0,
    reduced_structure: bool = True,
    orient_with_momentum: bool = True,
    tol: float = 1e-4,
    max_iter: int = 50,
    verbose: bool = False,
    return_ellip_triax: bool = False,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """Fit an adaptive ellipsoid to a particle distribution and compute shape diagnostics.

    Iteratively selects particles inside an adaptive ellipsoid and diagonalises
    the (optionally reduced/weighted) structure tensor to determine the principal
    axis ratios and principal-axis directions. Iteration continues until the axis
    ratios q = b/a and s = c/a converge within `tol` or until `max_iter` is reached.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
        Particle coordinates.
    mass : array_like, shape ``(N,)``, optional
        Particle masses.  *None* -> unit mass.
    vel : array_like, shape ``(N, 3)``, optional
        Particle velocities.  Required when *orient_with_momentum* is *True*.
    Rmin, Rmax : float
        Inner / outer radii for the initial spherical selection.
    reduced_structure : bool
        Use the iterative reduced inertia tensor
        (weights proportional to 1/R^2_sph).
    orient_with_momentum : bool
        Align the minor axis with the angular-momentum vector each iteration.
    tol : float
        Convergence tolerance on axis-ratio changes.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print iteration diagnostics.
    return_ellip_triax : bool
        If *True*, also return ellipticity and triaxiality.

    Returns
    -------
    abc : np.ndarray, shape ``(3,)``
        Normalised semi-axis lengths ``[1, b/a, c/a]``.
    transform : np.ndarray, shape ``(3, 3)``
        Rows are unit eigenvectors ``[e_a, e_b, e_c]``.
    ellip : float
        ``1 - c/a``.  Only returned when *return_ellip_triax* is *True*.
    triax : float
        ``(a^2-b^2)/(a^2-c^2)``.  Only when *return_ellip_triax* is *True*.
    """
    from scipy import linalg

    pos, _ = validate_positions(pos)  # ensures (N, 3) float array
    n_particles = pos.shape[0]

    mass_arr = validate_masses(mass, n_particles, non_negative=True)
    vel_arr = validate_velocities(vel, n_particles)  # None -> zeros

    use_momentum = False
    if vel is not None and orient_with_momentum:
        use_momentum = True
    elif vel is None and orient_with_momentum and verbose:
        print(
            "Warning: orient_with_momentum=True but vel not provided. "
            "Disabling momentum orientation."
        )

    if not (Rmin >= 0 and Rmax > 0 and Rmax > Rmin):
        raise ValueError("Need Rmin >= 0, Rmax > 0, and Rmax > Rmin.")

    def _nan_result():
        nan_abc = np.full(3, np.nan)
        nan_T = np.full((3, 3), np.nan)
        if return_ellip_triax:
            return nan_abc, nan_T, np.nan, np.nan
        return nan_abc, nan_T

    # Initialise
    q_iter, s_iter = 1.0, 1.0
    transform_bca = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    ).T
    eigval_bca = np.array([1.0, 1.0, 1.0], dtype=float)

    dist_sq = _calculate_particle_distances_sq(pos)
    extract_mask = (dist_sq < Rmax ** 2) & (dist_sq >= Rmin ** 2)

    # Iterate
    for it in range(max_iter):
        if verbose:
            print(f"Iteration {it + 1}/{max_iter}...")

        # 1. Particle selection
        if it > 0 and reduced_structure:
            prod_qs = q_iter * s_iter
            vol_inv = prod_qs ** (1.0 / 3.0) if prod_qs > 1e-9 else 1e-9
            Rmax_sc = Rmax / vol_inv if vol_inv > 1e-9 else Rmax * 1e9

            _, extract_mask, Rsph_ext = _calculate_Rsphall_and_extract(
                pos, transform_bca, q_iter, s_iter, Rmax_sc, Rmin
            )

            if np.sum(extract_mask) < 10:
                if verbose:
                    print("Too few particles in reduced step - stopping.")
                break

            med = np.median(Rsph_ext)
            if med < 1e-9:
                med = 1.0
            factors = Rsph_ext / med
            factors[factors < 1e-6] = 1e-6
            st_weights = 1.0 / (factors ** 2)
        else:
            if it > 0 and not reduced_structure:
                dist_sq = _calculate_particle_distances_sq(pos)
                extract_mask = (dist_sq < Rmax ** 2) & (dist_sq >= Rmin ** 2)

            if np.sum(extract_mask) < 10:
                if verbose:
                    print(
                        f"Too few particles ({np.sum(extract_mask)}) - "
                        f"check Rmin/Rmax."
                    )
                return _nan_result()

            st_weights = np.ones(np.sum(extract_mask), dtype=float)

        cur_pos = pos[extract_mask]
        cur_mass = mass_arr[extract_mask]

        # 2. Structure tensor
        S = _compute_weighted_structure_tensor(cur_pos, cur_mass, st_weights)

        # 3. Diagonalise
        try:
            eigvals, eigvecs = linalg.eigh(S)
        except linalg.LinAlgError:
            if verbose:
                print("eigh failed - stopping.")
            break
        eigvals = np.maximum(eigvals, 1e-12)

        a_sq, b_sq, c_sq = eigvals[2], eigvals[1], eigvals[0]
        e_a = eigvecs[:, 2].copy()
        e_b = eigvecs[:, 1].copy()
        e_c = eigvecs[:, 0].copy()

        # Right-handed system
        if np.linalg.det(np.column_stack((e_a, e_b, e_c))) < 0:
            e_c *= -1.0

        # 4. Optional momentum alignment
        if use_momentum and cur_pos.shape[0] > 0:
            cur_V = vel_arr[extract_mask]
            L = np.sum(
                cur_mass[:, np.newaxis] * np.cross(cur_pos, cur_V), axis=0
            )
            if np.linalg.norm(L) > 1e-9:
                if np.dot(e_c, L) < 0:
                    e_c *= -1.0

                e_a_orig = eigvecs[:, 2].copy()
                if np.abs(np.dot(e_a_orig, e_c)) < 0.9999:
                    e_b = np.cross(e_c, e_a_orig)
                    n_eb = np.linalg.norm(e_b)
                    if n_eb > 1e-9:
                        e_b /= n_eb
                    else:
                        tmp = (
                            np.array([1.0, 0.0, 0.0])
                            if abs(e_c[0]) < 0.9
                            else np.array([0.0, 1.0, 0.0])
                        )
                        e_b = np.cross(e_c, tmp)
                        e_b /= np.linalg.norm(e_b) + 1e-9

                    e_a = np.cross(e_b, e_c)
                    n_ea = np.linalg.norm(e_a)
                    if n_ea > 1e-9:
                        e_a /= n_ea
                    else:
                        tmp = (
                            np.array([1.0, 0.0, 0.0])
                            if abs(e_b[0]) < 0.9
                            else np.array([0.0, 1.0, 0.0])
                        )
                        e_a = np.cross(e_b, tmp)
                        e_a /= np.linalg.norm(e_a) + 1e-9
                else:
                    tmp = (
                        np.array([1.0, 0.0, 0.0])
                        if abs(e_c[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0])
                    )
                    e_b = np.cross(e_c, tmp)
                    e_b /= np.linalg.norm(e_b) + 1e-9
                    e_a = np.cross(e_b, e_c)
                    e_a /= np.linalg.norm(e_a) + 1e-9

        transform_bca = np.column_stack((e_b, e_c, e_a))
        eigval_bca = np.array([b_sq, c_sq, a_sq], dtype=float)

        # 5. Update axis ratios
        if eigval_bca[2] > 1e-12:
            q_new = np.sqrt(eigval_bca[0] / eigval_bca[2])
            s_new = np.sqrt(eigval_bca[1] / eigval_bca[2])
        else:
            q_new, s_new = 1.0, 1.0

        q_new = np.clip(np.nan_to_num(q_new, nan=1.0), 1e-3, 1.0)
        s_new = np.clip(np.nan_to_num(s_new, nan=1.0), 1e-3, q_new)

        # 6. Convergence
        if it > 0:
            q_safe = q_iter if q_iter > 1e-6 else 1.0
            s_safe = s_iter if s_iter > 1e-6 else 1.0
            metric = max((1.0 - q_new / q_safe) ** 2, (1.0 - s_new / s_safe) ** 2)
            if metric < tol ** 2:
                if verbose:
                    print(f"Converged at iteration {it + 1} (metric={metric:.2e}).")
                q_iter, s_iter = q_new, s_new
                break

        q_iter, s_iter = q_new, s_new

        if verbose:
            print(
                f"  sqrt(eigvals): a={np.sqrt(a_sq):.3f}, "
                f"b={np.sqrt(b_sq):.3f}, c={np.sqrt(c_sq):.3f}"
            )
            print(f"  q(b/a)={q_iter:.4f}, s(c/a)={s_iter:.4f}")

        if not reduced_structure and it == 0:
            if verbose:
                print("Single-pass (reduced_structure=False).")
            break
    else:
        if verbose:
            print(f"Reached max_iter ({max_iter}) without convergence.")

    # Final output
    if np.any(np.isnan(eigval_bca)) or np.sum(extract_mask) < 3:
        if verbose:
            print("Warning: invalid eigenvalues or too few particles.")
        return _nan_result()

    a_val = np.sqrt(max(eigval_bca[2], 0.0))
    b_val = np.sqrt(max(eigval_bca[0], 0.0))
    c_val = np.sqrt(max(eigval_bca[1], 0.0))

    e_a_f = transform_bca[:, 2]
    e_b_f = transform_bca[:, 0]
    e_c_f = transform_bca[:, 1]

    pairs = sorted(
        [(a_val, e_a_f), (b_val, e_b_f), (c_val, e_c_f)],
        key=lambda p: p[0],
        reverse=True,
    )
    a_out, e_a_out = pairs[0]
    b_out, e_b_out = pairs[1]
    c_out, e_c_out = pairs[2]

    if a_out > 1e-9:
        abc = np.array([1.0, b_out / a_out, c_out / a_out])
    else:
        abc = np.array([1.0, 1.0, 1.0])
        if verbose:
            print("Warning: major axis near zero - shape ill-defined.")

    transform = np.vstack((e_a_out, e_b_out, e_c_out))

    if not return_ellip_triax:
        return abc, transform

    ellip = (1.0 - c_out / a_out) if a_out > 1e-9 else 0.0
    ellip = float(np.nan_to_num(ellip))

    a2, b2, c2 = a_out ** 2, b_out ** 2, c_out ** 2
    denom = a2 - c2
    if denom > 1e-12:
        triax = (a2 - b2) / denom
    elif abs(a2 - b2) < 1e-12:
        triax = 0.0  # sphere
    else:
        triax = 0.0

    if denom > 1e-12 and abs(b2 - c2) < 1e-12:
        triax = 1.0  # prolate

    triax = float(np.nan_to_num(triax))
    return abc, transform, ellip, triax


# --- 5. Spherical grid generators ---

def uniform_spherical_grid(
    radius: float = 1.0,
    num_pts: int = 500,
) -> np.ndarray:
    """Generate uniformly random points on the surface of a sphere.

    Parameters
    ----------
    radius : float
        Sphere radius.
    num_pts : int
        Number of points.

    Returns
    -------
    xyz : np.ndarray, shape ``(num_pts, 3)``
        Cartesian coordinates.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if not isinstance(num_pts, (int, np.integer)) or num_pts <= 0:
        raise ValueError("num_pts must be a positive integer.")

    phi = np.random.uniform(0, 2 * np.pi, num_pts)
    cos_theta = np.random.uniform(-1, 1, num_pts)
    theta = np.arccos(cos_theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.column_stack((x, y, z))


def spherical_spiral_grid(
    radius: float = 1.0,
    proj: str = "Cart",
) -> np.ndarray:
    """Load a pre-defined spherical spiral grid scaled to a given radius.

    The grid data is read from
    ``nbody_streams/data/spherical_grid_unit.xyz``.

    Parameters
    ----------
    radius : float
        Radius to scale the unit grid to.
    proj : ``'Cart'`` | ``'Sph'`` | ``'Cyl'``
        Coordinate system of the returned array.

        * ``'Cart'`` - Cartesian ``(x, y, z)``.
        * ``'Sph'``  - Spherical ``(r, theta, phi)``.
        * ``'Cyl'``  - Cylindrical ``(R, phi, z)``.

    Returns
    -------
    grid : np.ndarray, shape ``(M, 3)``
    """
    if proj not in ("Cart", "Sph", "Cyl"):
        raise ValueError("proj must be 'Cart', 'Sph', or 'Cyl'.")
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    grid_file = data_dir / "spherical_grid_unit.xyz"
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Spiral grid data not found at {grid_file}. "
            f"Place the file 'spherical_grid_unit.xyz' in {data_dir}."
        )

    XYZ = radius * np.loadtxt(grid_file)

    if proj == "Cart":
        return XYZ

    # Inline coordinate conversions
    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    if proj == "Sph":
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(np.clip(z / np.maximum(r, 1e-30), -1, 1))
        phi = np.arctan2(y, x)
        return np.column_stack((r, theta, phi))

    # proj == "Cyl"
    R = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.column_stack((R, phi, z))


# --- 6. Centre finding ---

def _shrinking_sphere_center(
    pos: np.ndarray,
    mass: np.ndarray,
    r_init: float = 30.0,
    shrink_factor: float = 0.9,
    min_particles: int = 10,
) -> np.ndarray:
    """Shrinking-sphere centre finder.

    Starts from the mass-weighted centroid and iteratively shrinks the
    enclosing sphere, recomputing the centre each step.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
    mass : array_like, shape ``(N,)``
    r_init : float
        Initial sphere radius.
    shrink_factor : float
        Radius reduction per iteration.
    min_particles : int
        Stop when fewer particles remain.

    Returns
    -------
    centre : np.ndarray, shape ``(3,)``
    """
    center = np.average(pos, axis=0, weights=mass)
    radius = r_init
    while True:
        dist2 = np.sum((pos - center) ** 2, axis=1)
        mask = dist2 < radius ** 2
        if np.sum(mask) < min_particles:
            break
        center = np.average(pos[mask], axis=0, weights=mass[mask])
        radius *= shrink_factor
    return center


def _density_peak_center(
    pos: np.ndarray,
    mass: np.ndarray,
    softening: float = 0.1,
    top_fraction: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """Find the density peak by locating the minimum gravitational potential.

    Uses the first available solver in order:
      1. GPU tree code (nbody_streams.tree_gpu) - O(N log N), fastest
      2. pyfalcon CPU tree code               - O(N log N)
      3. agama Multipole BFE                  - approximate
    Raises ImportError if none are available.

    Parameters
    ----------
    pos : np.ndarray, shape ``(N, 3)``
    mass : np.ndarray, shape ``(N,)``
    softening : float
        Gravitational softening length (kpc).
    top_fraction : float
        Fraction of lowest-potential particles used to compute centroid.
    **kwargs
        Extra kwargs forwarded to the tree solver (e.g. ``theta=0.4``).

    Returns
    -------
    centre : np.ndarray, shape ``(3,)``
    """
    theta = kwargs.get("theta", 0.4)
    phi = None

    # --- Try GPU tree ---
    try:
        from ..tree_gpu import tree_gravity_gpu
        import cupy as cp
        pos_cp = cp.asarray(pos, dtype=cp.float32)
        mass_cp = cp.asarray(mass, dtype=cp.float32)
        eps_cp = cp.full(len(pos), float(softening), dtype=cp.float32)
        _, phi_cp = tree_gravity_gpu(
            pos_cp, mass_cp, eps=eps_cp, G=G_DEFAULT, theta=theta,
        )
        phi = cp.asnumpy(phi_cp).astype(float)
    except Exception:
        pass

    # --- Try pyfalcon ---
    if phi is None:
        try:
            import pyfalcon
            _, phi = pyfalcon.gravity(pos, mass * G_DEFAULT, eps=softening, theta=theta)
            phi = np.asarray(phi, dtype=float)
        except Exception:
            pass

    # --- Try agama ---
    if phi is None:
        if not AGAMA_AVAILABLE:
            raise ImportError(
                "density_peak centre finding requires at least one of:\n"
                "  - nbody_streams GPU tree (compile: cd nbody_streams/tree_gpu && make -j)\n"
                "  - pyfalcon  (pip install pyfalcon)\n"
                "  - agama     (pip install agama)\n"
                "Alternatively use method='shrinking_sphere' or method='kde'."
            )
        lmax = kwargs.get("lmax", 8)
        pot = agama.Potential(
            type="Multipole",
            particles=(pos, mass),
            symmetry="n",
            lmax=lmax,
        )
        phi = pot.potential(pos)

    # Centroid of the lowest-potential particles
    n_pick = max(10, int(len(phi) * top_fraction))
    idxs = np.argsort(phi)[:n_pick]
    return np.average(pos[idxs], axis=0, weights=mass[idxs])


def find_center(
    pos: np.ndarray,
    mass: np.ndarray | float | None = None,
    vel: np.ndarray | None = None,
    method: str = "density_peak",
    return_velocity: bool = False,
    vel_aperture: float = 5.0,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Find the centre (and optionally the velocity centre) of a particle distribution.

    Parameters
    ----------
    pos : array_like, shape ``(N, 3)``
        Particle positions.
    mass : scalar, array_like, or None
        Particle masses.
    vel : array_like, shape ``(N, 3)``, optional
        Particle velocities.  Required when *return_velocity* is *True*.
    method : ``'density_peak'`` | ``'shrinking_sphere'`` | ``'kde'``
        Algorithm:

        * ``'density_peak'`` - gravitational potential minimum via tree/BFE.
          Tries GPU tree, then pyfalcon, then agama (in that order). Raises
          ImportError if none are available - does NOT silently fall back.
        * ``'shrinking_sphere'`` - iterative shrinking sphere (fast, robust
          for virialized systems; may be unreliable for tidal debris).
        * ``'kde'`` - Gaussian KDE density peak (slow for large N).

    return_velocity : bool
        If *True*, also return the velocity centre (mass-weighted mean
        velocity of particles within *vel_aperture* kpc of the position centre).
        Requires *vel* to be supplied.
    vel_aperture : float
        Aperture radius (kpc) for velocity centering.
    **kwargs
        Extra parameters forwarded to the chosen method.

        For ``density_peak``:
            ``softening`` (0.1 kpc), ``top_fraction`` (0.01), ``theta`` (0.4),
            ``lmax`` (8, agama fallback only).
        For ``shrinking_sphere``:
            ``r_init`` (30), ``shrink_factor`` (0.9), ``min_particles`` (100).

    Returns
    -------
    centre_pos : np.ndarray, shape ``(3,)``
        Estimated position centre.
    centre_vel : np.ndarray, shape ``(3,)``
        Estimated velocity centre.  Only returned when *return_velocity* is *True*.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 2 and pos.shape[1] == 6:
        # Legacy: accept (N, 6) phase-space; split internally
        if vel is None:
            vel = pos[:, 3:]
        pos = pos[:, :3]

    n = pos.shape[0]
    mass_arr = validate_masses(mass, n)

    if method == "density_peak":
        centre_pos = _density_peak_center(
            pos, mass_arr,
            softening=kwargs.get("softening", 0.1),
            top_fraction=kwargs.get("top_fraction", 0.01),
            **{k: v for k, v in kwargs.items()
               if k in ("theta", "lmax")},
        )

    elif method == "shrinking_sphere":
        centre_pos = _shrinking_sphere_center(
            pos,
            mass_arr,
            r_init=kwargs.get("r_init", 30.0),
            shrink_factor=kwargs.get("shrink_factor", 0.9),
            min_particles=kwargs.get("min_particles", 100),
        )

    elif method == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(pos.T, weights=mass_arr)
        n_sample = min(10_000, n)
        sample = pos[np.random.choice(n, size=n_sample, replace=False)]
        dens = kde(sample.T)
        n_pick = max(10, int(len(dens) * 0.01))
        idxs = np.argsort(dens)[-n_pick:]
        centre_pos = np.average(sample[idxs], axis=0)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose 'density_peak', 'shrinking_sphere', or 'kde'."
        )

    if not return_velocity:
        return centre_pos

    # Velocity centering via aperture selection
    if vel is None:
        raise ValueError("vel must be supplied when return_velocity=True.")
    vel = np.asarray(vel, dtype=float)
    dist2 = np.sum((pos - centre_pos) ** 2, axis=1)
    sel = dist2 < vel_aperture ** 2
    if np.any(sel):
        centre_vel = np.average(vel[sel], axis=0, weights=mass_arr[sel])
    else:
        centre_vel = np.average(vel, axis=0, weights=mass_arr)

    return centre_pos, centre_vel


def find_center_position(
    pos: np.ndarray,
    mass: np.ndarray | float | None = None,
    method: str = "density_peak",
    **kwargs,
) -> np.ndarray:
    """Find the centre of a particle distribution.

    .. deprecated::
        Use :func:`find_center` instead.  This alias will be removed in a
        future release.
    """
    warnings.warn(
        "find_center_position is deprecated; use find_center instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return find_center(pos, mass=mass, method=method, **kwargs)


# --- 7. Iterative boundness ---

def compute_iterative_boundness(*args, **kwargs):
    warnings.warn(
        "compute_iterative_boundness is deprecated; use iterative_unbinding.",
        DeprecationWarning,
        stacklevel=2,
    )
    return iterative_unbinding(*args, **kwargs)

def iterative_unbinding(
    pos_dark: np.ndarray,
    vel_dark: np.ndarray,
    mass_dark: np.ndarray | float,
    pos_star: np.ndarray | None = None,
    vel_star: np.ndarray | None = None,
    mass_star: np.ndarray | float | None = None,
    center_position: np.ndarray | list = [],
    center_velocity: np.ndarray | list = [],
    recursive_iter_converg: int = 50,
    potential_compute_method: str = "tree",
    softening: float = 0.03,
    G: float = G_DEFAULT,
    center_on: str = "dark",
    vel_aperture: float = 5.0,
    tol_frac_change: float = 0.0001,
    verbose: bool = True,
    return_history: bool = False,
    **kwargs,
) -> tuple:
    """Iterative unbinding to determine bound particles.

    Computes the gravitational potential, evaluates total energy per
    particle (E = phi + 1/2 * vel^2), and iteratively removes unbound (E > 0)
    particles, repeating until the bound mass fraction changes by less than
    `tol_frac_change` or `recursive_iter_converg` is reached.

    Supports multi-component systems (e.g. dark matter and stars).

    The centre of the system is determined automatically via a density-peak
    method: the full-system gravitational potential (DM + stars) is computed
    once using the same solver as ``potential_compute_method``, and the
    centroid of the lowest-potential ``center_on`` particles (bottom 1%) is
    taken as the position centre.  The velocity centre is the mass-weighted
    mean velocity of ``center_on`` particles within ``vel_aperture`` of that
    position.  Pass explicit ``center_position`` / ``center_velocity`` to
    bypass automatic centering entirely.

    Parameters
    ----------
    pos_dark : array_like, shape ``(N_d, 3)``
        Dark-matter positions (kpc).
    vel_dark : array_like, shape ``(N_d, 3)``
        Dark-matter velocities (km/s).
    mass_dark : scalar or array_like
        Dark-matter masses (M_sun).
    pos_star, vel_star, mass_star : optional
        Stellar component (same conventions).
    center_position, center_velocity : array_like, shape ``(3,)``
        Pre-computed centre.  If empty, determined automatically.
    recursive_iter_converg : int
        Maximum iterations.
    potential_compute_method : ``'tree'`` | ``'tree_gpu'`` | ``'bfe'`` | ``'direct'`` | ``'direct_gpu'``
        Potential solver used for both centering and unbinding:

        * ``'tree'``       - pyfalcon CPU tree code, O(N log N).
        * ``'tree_gpu'``   - GPU Barnes-Hut tree (nbody_streams.tree_gpu).
        * ``'bfe'``        - Agama multipole expansion.
        * ``'direct'``     - O(N^2) CPU direct summation (practical for N <= ~50 000).
        * ``'direct_gpu'`` - GPU direct summation variant.
    softening : float
        Gravitational softening (kpc).
    G : float
        Gravitational constant.
    center_on : ``'dark'`` | ``'star'`` | ``'both'``
        Which component's particles are used to compute the position centroid
        (lowest-phi subset) and velocity aperture average.  The potential is
        always computed from the full DM + star system regardless.
    vel_aperture : float
        Aperture radius (kpc) for velocity centering.  Only ``center_on``
        particles within this distance of the position centre contribute.
    tol_frac_change : float
        Convergence tolerance on bound-fraction change per iteration.
    verbose : bool
        Print diagnostics.
    return_history : bool
        Return per-iteration bound masks.
    **kwargs
        Extra keyword arguments forwarded to the potential solver,
        e.g. ``theta=0.4`` (tree codes), ``lmax=8`` (bfe).

    Returns
    -------
    results : tuple
        ``(bound_dark,)`` or ``(bound_dark, bound_star)`` - integer
        masks (1 = bound).
    center_position : np.ndarray
    center_velocity : np.ndarray
    """
    # --- Input validation ---
    potential_compute_method = potential_compute_method.lower()

    pos_dark, _ = validate_positions(pos_dark)
    vel_dark = validate_velocities(vel_dark, pos_dark.shape[0])
    mass_dark_arr = validate_masses(mass_dark, pos_dark.shape[0])
    n_dark = pos_dark.shape[0]

    has_stars = pos_star is not None
    if has_stars:
        pos_star, _ = validate_positions(pos_star)
        vel_star = validate_velocities(vel_star, pos_star.shape[0])
        mass_star_arr = validate_masses(mass_star, pos_star.shape[0])
        n_star = pos_star.shape[0]

    if center_on == "star" and not has_stars:
        raise ValueError("center_on='star' requires star data.")

    # --- Resolve potential backend ---
    _use_pyfalcon = False
    _use_tree_gpu = False
    _use_direct = potential_compute_method in ("direct", "direct_gpu")

    if potential_compute_method == "tree":
        try:
            import pyfalcon
            _use_pyfalcon = True
        except ImportError:
            raise ImportError(
                "potential_compute_method='tree' requires pyfalcon. "
                "Install with: pip install pyfalcon\n"
                "Or choose 'tree_gpu', 'bfe', 'direct', or 'direct_gpu'."
            )

    elif potential_compute_method == "tree_gpu":
        try:
            from ..tree_gpu import tree_gravity_gpu as _tree_gravity_gpu
            import cupy as _cp_mod
            _use_tree_gpu = True
            _tree_gpu_fn = _tree_gravity_gpu
            _cp = _cp_mod
        except ImportError as e:
            raise ImportError(
                "potential_compute_method='tree_gpu' requires CuPy and the compiled "
                "GPU tree library. Either:\n"
                "  - CuPy is not installed (pip install cupy-cudaXX)\n"
                "  - libtreeGPU.so has not been built: "
                "cd nbody_streams/tree_gpu && make -j\n"
                f"Original error: {e}"
            ) from e

    elif potential_compute_method == "bfe":
        if not AGAMA_AVAILABLE:
            raise ImportError(
                "potential_compute_method='bfe' requires agama. "
                "Install with: pip install agama\n"
                "Or choose 'tree', 'tree_gpu', 'direct', or 'direct_gpu'."
            )

    if _use_direct:
        from ..species import _emit_performance_warnings
        N_total = n_dark + (n_star if has_stars else 0)
        arch = "gpu" if potential_compute_method == "direct_gpu" else "cpu"
        _emit_performance_warnings(N_total, arch, "direct")
        try:
            if potential_compute_method == "direct_gpu":
                from ..fields import compute_nbody_potential_gpu as _potential_fn
            else:
                from ..fields import compute_nbody_potential_cpu as _potential_fn
        except ImportError:
            raise ImportError(
                f"Cannot import potential function for method "
                f"'{potential_compute_method}'."
            )

    # --- Logger ---
    logger = logging.getLogger("boundness")
    if verbose and not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    theta = kwargs.get("theta", 0.4)
    lmax = kwargs.get("lmax", 8)
    top_fraction = kwargs.get("top_fraction", 0.01)

    # --- Stack full particle arrays ---
    if has_stars:
        pos_all = np.vstack((pos_dark, pos_star))
        vel_all = np.vstack((vel_dark, vel_star))
        mass_all = np.concatenate((mass_dark_arr, mass_star_arr))
    else:
        pos_all = pos_dark.copy()
        vel_all = vel_dark.copy()
        mass_all = mass_dark_arr.copy()

    # Slice selecting which component is used for centroid + velocity averaging.
    # Phi is always computed from the full (DM + star) system so that the
    # gravitational well of the dominant component is always included.
    if center_on == "both" or not has_stars:
        ctr_sl = slice(None)
    elif center_on == "star":
        ctr_sl = slice(n_dark, None)
    else:  # "dark"
        ctr_sl = slice(None, n_dark)

    # --- Centre finding ---
    center_position = np.asarray(center_position, dtype=float)
    center_velocity = np.asarray(center_velocity, dtype=float)

    if center_position.size < 3:
        # Compute full-system phi using the same solver selected for unbinding.
        if _use_pyfalcon:
            _, phi_init = pyfalcon.gravity(
                pos_all, mass_all * G, eps=softening, theta=theta,
            )
            phi_init = np.asarray(phi_init, dtype=float)
        elif _use_tree_gpu:
            _pos_cp = _cp.asarray(pos_all, dtype=_cp.float32)
            _mass_cp = _cp.asarray(mass_all, dtype=_cp.float32)
            _eps_cp = _cp.full(len(pos_all), float(softening), dtype=_cp.float32)
            _, _phi_cp = _tree_gpu_fn(_pos_cp, _mass_cp, eps=_eps_cp, G=G, theta=theta)
            phi_init = _cp.asnumpy(_phi_cp).astype(float)
        elif _use_direct:
            phi_init = _potential_fn(pos_all, mass_all, softening=softening, G=G)
        else:
            # bfe
            _pot_init = agama.Potential(
                type="Multipole",
                particles=(pos_all, mass_all),
                symmetry="n",
                lmax=lmax,
            )
            phi_init = _pot_init.potential(pos_all)

        # Position centroid: mass-weighted average of the lowest-phi
        # center_on particles (bottom top_fraction of the component).
        phi_ctr = phi_init[ctr_sl]
        pos_ctr = pos_all[ctr_sl]
        mass_ctr = mass_all[ctr_sl]
        n_pick = max(10, int(len(phi_ctr) * top_fraction))
        idxs = np.argsort(phi_ctr)[:n_pick]
        center_position = np.average(pos_ctr[idxs], axis=0, weights=mass_ctr[idxs])

    if center_velocity.size < 3:
        # Velocity centre: mass-weighted mean of center_on particles within
        # vel_aperture of the position centre.
        pos_ctr = pos_all[ctr_sl]
        vel_ctr = vel_all[ctr_sl]
        mass_ctr = mass_all[ctr_sl]
        dist2 = np.sum((pos_ctr - center_position) ** 2, axis=1)
        sel = dist2 < vel_aperture ** 2
        if np.any(sel):
            center_velocity = np.average(vel_ctr[sel], axis=0, weights=mass_ctr[sel])
        else:
            center_velocity = np.average(vel_ctr, axis=0, weights=mass_ctr)

    logger.info("Center position (density_peak): %s", np.around(center_position, 2))
    logger.info("Center velocity: %s", np.around(center_velocity, 2))

    pos_rel = pos_all - center_position
    vel_rel = vel_all - center_velocity

    mask_all = np.ones(len(pos_all), dtype=bool)
    bound_history_dm: list[np.ndarray] = []
    bound_history_star: list[np.ndarray] = []
    min_particles = 5

    # --- Iterative unbinding ---
    for i in range(recursive_iter_converg):
        n_bound = int(np.sum(mask_all))
        if n_bound < min_particles:
            logger.info("Stopping: only %d particles remaining.", n_bound)
            break

        # Unbound particles receive a near-zero mass so they still contribute
        # to the potential field at their positions, allowing them to re-bind
        # in early iterations if they are near the energy boundary.
        if _use_pyfalcon:
            mass_src = mass_all.copy()
            mass_src[~mask_all] = 0.01
            _, phi = pyfalcon.gravity(
                pos_rel, mass_src * G, eps=softening, theta=theta
            )

        elif _use_tree_gpu:
            mass_src = mass_all.copy()
            mass_src[~mask_all] = 0.01
            pos_cp = _cp.asarray(pos_rel, dtype=_cp.float32)
            mass_cp = _cp.asarray(mass_src, dtype=_cp.float32)
            eps_cp = _cp.full(len(pos_rel), float(softening), dtype=_cp.float32)
            _, phi_cp = _tree_gpu_fn(
                pos_cp, mass_cp, eps=eps_cp, G=G, theta=theta,
            )
            phi = _cp.asnumpy(phi_cp).astype(float)

        elif _use_direct:
            phi = _potential_fn(
                pos_rel, mass_all * mask_all, softening=softening, G=G,
            )
        else:
            # bfe via agama
            pot = agama.Potential(
                type="Multipole",
                particles=(pos_rel[mask_all], mass_all[mask_all]),
                symmetry="n",
                lmax=lmax,
            )
            phi = pot.potential(pos_rel)

        kin = 0.5 * np.sum(vel_rel ** 2, axis=1)
        bound_mask = (phi + kin) < 0

        bound_history_dm.append(bound_mask[:n_dark].copy())
        if has_stars:
            bound_history_star.append(bound_mask[n_dark:].copy())

        frac_change = np.mean(bound_mask != mask_all)
        logger.info("Iter %d: delta bound mask = %.5f", i, frac_change)
        mask_all = bound_mask
        if frac_change < tol_frac_change:
            logger.info("Converged after %d iterations.", i + 1)
            break

    # --- Assemble output ---
    bound_dark_final = mask_all[:n_dark].astype(int)
    results: list = [bound_dark_final]
    if has_stars:
        results.append(mask_all[n_dark:].astype(int))
    if return_history:
        results.append(bound_history_dm)
        if has_stars:
            results.append(bound_history_star)

    return tuple(results), center_position, center_velocity
