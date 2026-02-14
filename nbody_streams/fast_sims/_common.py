"""Private helpers shared by fast_sims methods.

Functions here are implementation details — not part of the public API.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy import special
from scipy.integrate import solve_ivp

try:
    import agama

    agama.setUnits(mass=1, length=1, velocity=1)
    AGAMA_AVAILABLE = True
except ImportError:
    AGAMA_AVAILABLE = False

__all__: list[str] = []  # nothing public — internal helpers only


# ---------------------------------------------------------------------------
# Velocity dispersion for dynamical friction
# ---------------------------------------------------------------------------

def _compute_vel_disp_from_Potential(
    pot_for_dynFric_sigma,
    grid_r: np.ndarray | None = None,
) -> Callable:
    """Compute a radial velocity-dispersion profile from an Agama potential.

    Parameters
    ----------
    pot_for_dynFric_sigma : agama.Potential
        Potential model used to compute the velocity dispersion profile.
        Ideally an axisymmetric / spherically symmetric model.
    grid_r : np.ndarray, optional
        Radial grid for the dispersion spline.  Defaults to
        ``np.logspace(-1, 2, 16)`` (0.1 – 100 kpc).

    Returns
    -------
    Callable
        ``sigma(r)`` — velocity dispersion at radius *r* (km/s).
    """
    if grid_r is None:
        grid_r = np.logspace(-1, 2, 16)

    try:
        df_host = agama.DistributionFunction(
            type='quasispherical', potential=pot_for_dynFric_sigma,
        )
        grid_sig = agama.GalaxyModel(pot_for_dynFric_sigma, df_host).moments(
            np.column_stack((grid_r, grid_r * 0, grid_r * 0)),
            dens=False, vel=False, vel2=True,
        )[:, 0] ** 0.5
        logspl = agama.Spline(np.log(grid_r), np.log(grid_sig))
        return lambda r: np.exp(logspl(np.log(r)))

    except Exception:
        warnings.warn(
            "Could not compute velocity dispersion from potential; "
            "falling back to precomputed MW profile.",
            RuntimeWarning,
        )
        grid_sig_init = np.array([
            158.34386609, 200.12076947, 208.35638186, 207.53478107,
            197.97276146, 195.18822847, 188.6893688,  183.74527079,
            187.35960162, 193.26190609, 173.27866017, 143.68049751,
            132.84412575, 121.76024275, 106.50314755, 104.28241804,
        ])
        logspl_init = agama.Spline(np.log(grid_r), np.log(grid_sig_init))
        return lambda r: np.exp(logspl_init(np.log(r)))


# ---------------------------------------------------------------------------
# Dynamical friction
# ---------------------------------------------------------------------------

def _dynamical_friction_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    pot_host,
    mass: float,
    sigma_r_func: Callable,
    t: float = 0,
    coulomb_mode: str = 'variable',
    fixed_ln_lambda: float = 3.0,
    core_gamma: float = 0.0,
    r_core: float = 1.0,
) -> np.ndarray:
    """Compute Chandrasekhar dynamical-friction acceleration.

    Parameters
    ----------
    pos, vel : np.ndarray, shape ``(3,)``
        Position and velocity of the satellite.
    pot_host : agama.Potential
        Host potential (for local density).
    mass : float
        Satellite mass (M_sun).
    sigma_r_func : Callable
        ``sigma(r)`` returned by :func:`_compute_vel_disp_from_Potential`.
    t : float
        Evaluation time (Gyr).
    coulomb_mode : ``'fixed'`` or ``'variable'``
        How to compute the Coulomb logarithm.
    fixed_ln_lambda : float
        Used when *coulomb_mode* = ``'fixed'``.
    core_gamma : float
        Power-law index for core-stalling suppression (0 = standard cusp).
    r_core : float
        Core radius (kpc) for the suppression factor.

    Returns
    -------
    np.ndarray, shape ``(3,)``
        Dynamical-friction acceleration vector.
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    if r < 1e-6 or v < 1e-6:
        return np.zeros_like(vel)

    rho = pot_host.density(pos, t=t)
    sig = sigma_r_func(r)
    X = v / (np.sqrt(2) * sig)

    # Coulomb logarithm
    if coulomb_mode == 'fixed':
        ln_lambda = fixed_ln_lambda
    else:
        b_max = r
        b_min = agama.G * mass / (v**2)
        Lambda = b_max / (b_min + 1e-9)
        ln_lambda = np.log(np.maximum(Lambda, 1.1))

    # Chandrasekhar formula
    term_bracket = special.erf(X) - 2.0 / np.sqrt(np.pi) * X * np.exp(-X**2)
    force_magnitude = (
        4 * np.pi * agama.G**2 * mass * rho * ln_lambda * term_bracket
    ) / v**2

    # Core stalling (Read+2006, Petts+2016)
    if core_gamma > 0:
        suppression = np.minimum(1.0, (r / r_core)**core_gamma)
        force_magnitude *= suppression

    return -(vel / v) * force_magnitude


# ---------------------------------------------------------------------------
# Orbit integration with dynamical friction
# ---------------------------------------------------------------------------

def _integrate_orbit_with_dynamical_friction(
    ic: np.ndarray,
    pot_host,
    mass: float,
    time_total: float,
    time_end: float,
    pot_for_dynFric_sigma=None,
    trajsize: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a satellite orbit with optional dynamical friction.

    Dynamical friction is turned off when *mass* = 0.

    Parameters
    ----------
    ic : np.ndarray, shape ``(6,)``
        Initial phase-space position ``[x, y, z, vx, vy, vz]``.
    pot_host : agama.Potential
        Host galaxy potential.
    mass : float
        Fixed satellite mass (M_sun).  Set to 0 to disable friction.
    time_total : float
        Integration duration (Gyr).
    time_end : float
        Present-day epoch (Gyr).
    pot_for_dynFric_sigma : agama.Potential, optional
        Potential for the velocity-dispersion profile.
    trajsize : int
        Number of trajectory points (0 = all adaptive steps).

    Returns
    -------
    times : np.ndarray
        Time array.
    orbit : np.ndarray, shape ``(len(times), 6)``
        Trajectory.
    """
    time_sat, orbit_sat = agama.orbit(
        ic=ic, potential=pot_host,
        time=-time_total, timestart=time_end, trajsize=trajsize,
    )

    if mass == 0:
        return time_sat, orbit_sat

    sigma_r_func = _compute_vel_disp_from_Potential(pot_for_dynFric_sigma)

    def equations_of_motion(t: float, xv: np.ndarray) -> np.ndarray:
        pos, vel = xv[:3], xv[3:6]
        acc = pot_host.force(pos, t=t) + _dynamical_friction_acceleration(
            pos, vel, pot_host, mass, sigma_r_func, t=t,
        )
        return np.hstack((vel, acc))

    sol = solve_ivp(
        equations_of_motion,
        [time_end, time_end - time_total], ic,
        method='DOP853', dense_output=True,
        rtol=1e-8, atol=1e-10,
    )

    return time_sat, sol.sol(time_sat).T


# ---------------------------------------------------------------------------
# Progenitor potential builders
# ---------------------------------------------------------------------------

def _get_prog_GalaxyModel(
    initmass: float,
    scaleradius: float,
    prog_pot_kind: str,
    return_DistribFunc: bool = True,
    **kwargs: Any,
) -> tuple | Any:
    """Create a satellite potential (and optionally its DF).

    Parameters
    ----------
    initmass : float
        Initial satellite mass (M_sun).
    scaleradius : float
        Scale radius (kpc).
    prog_pot_kind : ``'King'``, ``'Plummer'``, or ``'Plummer_withRcut'``
        Profile type.
    return_DistribFunc : bool
        If *True*, return ``(pot, df)``; otherwise just ``pot``.
    **kwargs
        Extra parameters (e.g. ``W0``, ``trunc`` for King).

    Returns
    -------
    pot_sat : agama.Potential
    df_sat : agama.DistributionFunction  *(only when return_DistribFunc=True)*
    """
    kind = prog_pot_kind.lower()

    if kind == 'plummer':
        pot_sat = agama.Potential(
            type='Plummer', mass=initmass, scaleRadius=scaleradius,
        )
    elif kind == 'plummer_withrcut':
        pot_sat = agama.Potential(
            type='Spheroid', mass=initmass, scaleRadius=scaleradius,
            outerCutoffRadius=4 * scaleradius,
            gamma=0, beta=5, alpha=2, cutoffStrength=3,
        )
    elif kind == 'king':
        W0 = kwargs.get('W0', 3)
        trunc = kwargs.get('trunc', 1)
        pot_sat = agama.Potential(
            type='King', mass=initmass,
            scaleRadius=scaleradius, W0=W0, trunc=trunc,
        )
    else:
        raise ValueError(f"Unsupported progenitor potential kind: {prog_pot_kind}")

    if return_DistribFunc:
        return pot_sat, agama.DistributionFunction(
            type='quasispherical', potential=pot_sat,
        )
    return pot_sat


def _find_prog_pot_Nparticles(
    xv: np.ndarray,
    prog: np.ndarray,
    masses: np.ndarray | None = None,
    **potential_kwargs,
) -> tuple[Any, np.ndarray]:
    """Build a progenitor potential from an N-particle snapshot.

    Parameters
    ----------
    xv : np.ndarray, shape ``(N, 6)``
        Phase-space coordinates.
    prog : np.ndarray, shape ``(6,)``
        COM phase-space of the progenitor.
    masses : np.ndarray, shape ``(N,)``, optional
        Particle masses (uniform if *None*).
    **potential_kwargs
        Forwarded to ``agama.Potential``.

    Returns
    -------
    pot_sat : agama.Potential
        Monopole potential built from the particles.
    prog : np.ndarray, shape ``(6,)``
        Progenitor phase-space (unchanged).
    """
    xv = np.asarray(xv)
    if xv.ndim != 2 or xv.shape[1] != 6:
        raise ValueError(f"xv must have shape (N, 6), got {xv.shape}")

    N = len(xv)
    if masses is None:
        masses = np.ones(N) / N
    else:
        masses = np.asarray(masses)
        if len(masses) != N:
            raise ValueError(
                f"masses length ({len(masses)}) != particle count ({N})"
            )

    xv_rel = xv - prog

    pot_params = {
        'type': 'multipole',
        'particles': (xv_rel[:, :3], masses),
        'symmetry': 's',
    }
    pot_params.update(potential_kwargs)

    pot_sat = agama.Potential(**pot_params)
    return pot_sat, prog


# ---------------------------------------------------------------------------
# Perturber potential builder (shared by spray & restricted)
# ---------------------------------------------------------------------------

def _create_perturber_potential(
    add_perturber: dict,
    pot_host,
    time_total: float,
    time_end: float,
    verbose: bool = False,
) -> Any:
    """Build a moving NFW perturber potential on a self-consistent orbit.

    The perturber is rewound from its impact phase-space to the simulation
    start, then integrated forward to present day.

    Parameters
    ----------
    add_perturber : dict
        Must contain ``'mass'``, ``'scaleRadius'``, ``'w_subhalo_impact'``,
        and ``'time_impact'``.
    pot_host : agama.Potential
        Host potential for the perturber orbit.
    time_total, time_end : float
        Simulation time span and end epoch (Gyr).
    verbose : bool
        Print status messages.

    Returns
    -------
    pot_perturber_moving : agama.Potential
        Time-dependent NFW potential centred on the perturber trajectory.
    """
    required_keys = ['w_subhalo_impact', 'time_impact']
    for key in required_keys:
        if key not in add_perturber:
            raise KeyError(f"add_perturber must contain '{key}' key.")

    w_subhalo_impact = np.asarray(add_perturber['w_subhalo_impact'])
    if w_subhalo_impact.ndim != 1 or w_subhalo_impact.shape[0] != 6:
        raise ValueError(
            f"w_subhalo_impact must be shape (6,), got {w_subhalo_impact.shape}"
        )
    if not np.all(np.isfinite(w_subhalo_impact)):
        raise ValueError("w_subhalo_impact must contain only finite values.")

    time_impact = add_perturber['time_impact']
    if not isinstance(time_impact, (int, float)) or not np.isfinite(time_impact):
        raise TypeError("time_impact must be a finite scalar (Gyr).")

    if verbose:
        print(
            f"Adding perturber on self-consistent orbit "
            f"(mass={add_perturber['mass']:.2e} M_sun)."
        )

    # Rewind perturber from impact to simulation start
    w_subhalo_init = agama.orbit(
        potential=pot_host,
        ic=w_subhalo_impact,
        time=time_end - time_total - time_impact,
        timestart=time_end + time_impact,
        trajsize=1,
    )[1][0]

    # Integrate forward over the full simulation window
    traj_perturber = np.column_stack(agama.orbit(
        potential=pot_host,
        ic=w_subhalo_init,
        time=time_total,
        timestart=time_end - time_total,
        trajsize=0,
    ))

    return agama.Potential(
        type='nfw',
        mass=add_perturber['mass'],
        scaleRadius=add_perturber['scaleRadius'],
        center=traj_perturber,
    )
