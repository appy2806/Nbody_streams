"""Particle-spray stream generation.

Implements the Fardal+2015 and Chen+2025 particle-spray techniques for
creating stellar streams from disrupting satellites.  Both methods
rewind the progenitor orbit from its present-day position before
progressively releasing particles at the tidal radius.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d

try:
    import agama

    agama.setUnits(mass=1, length=1, velocity=1)
    AGAMA_AVAILABLE = True
except ImportError:
    AGAMA_AVAILABLE = False

from ._common import (
    _create_perturber_potential,
    _get_prog_GalaxyModel,
)

__all__ = [
    "create_ic_particle_spray_chen2025",
    "create_ic_particle_spray_fardal2015",
    "create_particle_spray_stream",
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Jacobi radius / rotation matrices                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _get_jacobi_rad_vel_mtx(
    pot_host,
    orbit_sat: np.ndarray,
    mass_sat: float,
    G: float | None = None,
    t: float | np.ndarray = 0,
    eigenvalue_method: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobi radius, velocity offset, and rotation matrices.

    Parameters
    ----------
    pot_host : agama.Potential
        Host gravitational potential.
    orbit_sat : array_like, shape ``(N, 6)``
        Satellite orbit ``[x, y, z, vx, vy, vz]`` (kpc, km/s).
    mass_sat : float
        Satellite mass (M_sun).
    G : float, optional
        Gravitational constant.  Defaults to ``agama.G``.
    t : float or array_like, shape ``(N,)``
        Evaluation time(s) for the potential.
    eigenvalue_method : bool
        If *True*, use tidal-tensor eigenvalues; otherwise use the
        radial second derivative (faster, less accurate).

    Returns
    -------
    r_jacobi : np.ndarray, shape ``(N,)``
        Jacobi (tidal) radii (kpc).
    v_jacobi : np.ndarray, shape ``(N,)``
        Characteristic velocity scales (km/s).
    R : np.ndarray, shape ``(N, 3, 3)``
        Rotation matrices (rows: radial, azimuthal, angular-momentum).
    """
    if G is None:
        G = agama.G

    orbit_sat = np.asarray(orbit_sat)
    N = len(orbit_sat)
    pos, vel = orbit_sat[:, :3], orbit_sat[:, 3:6]

    r = np.linalg.norm(pos, axis=1)
    r_sq = r**2 + 1e-50

    L = np.cross(pos, vel)
    L_mag = np.linalg.norm(L, axis=1)
    Omega_sq = (L_mag / r_sq)**2

    der2 = pot_host.eval(pos, der=True, t=t)

    if eigenvalue_method:
        tidal_tensor = np.zeros((N, 3, 3))
        tidal_tensor[:, 0, 0] = der2[:, 0]
        tidal_tensor[:, 1, 1] = der2[:, 1]
        tidal_tensor[:, 2, 2] = der2[:, 2]
        tidal_tensor[:, 0, 1] = tidal_tensor[:, 1, 0] = der2[:, 3]
        tidal_tensor[:, 1, 2] = tidal_tensor[:, 2, 1] = der2[:, 4]
        tidal_tensor[:, 0, 2] = tidal_tensor[:, 2, 0] = der2[:, 5]

        eigenvalues = np.linalg.eigvalsh(tidal_tensor)
        lambda_tidal = eigenvalues[:, -1]
        denominator = lambda_tidal + Omega_sq
    else:
        x, y, z = pos.T
        d2Phi_dr2 = -(
            x**2 * der2[:, 0] + y**2 * der2[:, 1] + z**2 * der2[:, 2]
            + 2 * x * y * der2[:, 3] + 2 * y * z * der2[:, 4]
            + 2 * z * x * der2[:, 5]
        ) / r_sq
        denominator = Omega_sq - d2Phi_dr2

    r_jacobi = (G * mass_sat / abs(denominator))**(1 / 3)
    v_jacobi = np.sqrt(Omega_sq) * r_jacobi

    # Rotation matrices (rows: radial, azimuthal, angular momentum)
    R = np.zeros((N, 3, 3))
    e_r = pos / (r[:, None] + 1e-50)
    e_L = L / (L_mag[:, None] + 1e-50)
    e_phi = np.cross(e_L, e_r, axis=1)
    e_phi_norm = np.linalg.norm(e_phi, axis=1, keepdims=True)
    e_phi = np.divide(e_phi, e_phi_norm, where=e_phi_norm != 0)

    R[:, 0, :] = e_r
    R[:, 1, :] = e_phi
    R[:, 2, :] = e_L

    return r_jacobi, v_jacobi, R


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IC generators                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_ic_particle_spray_chen2025(
    orbit_sat: np.ndarray,
    mass_sat: float,
    rj: np.ndarray,
    R: np.ndarray,
    G: float | None = None,
) -> np.ndarray:
    """Create spray ICs using the Chen+2025 correlated phase-space model.

    Parameters
    ----------
    orbit_sat : np.ndarray, shape ``(N, 6)``
        Satellite orbit (kpc, km/s).
    mass_sat : float
        Satellite mass (M_sun).
    rj : np.ndarray, shape ``(N,)``
        Jacobi radii (kpc).
    R : np.ndarray, shape ``(N, 3, 3)``
        Rotation matrices to the satellite frame.
    G : float, optional
        Gravitational constant (defaults to ``agama.G``).

    Returns
    -------
    ic_stream : np.ndarray, shape ``(2N, 6)``
        Initial conditions for leading/trailing stream particles.

    Notes
    -----
    Covariance matrix and mean offsets are calibrated from N-body
    simulations (Chen et al. 2025).
    """
    if G is None:
        G = agama.G

    N = len(orbit_sat)

    # Expand for leading/trailing arms (2 particles per orbit point)
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points (interleaving positive and negative offsets).
    # R = np.repeat(R, 2, axis=0)  # FIX: Match 2N shape
    r_tidal = np.repeat(rj, 2)

    # Covariance from Chen+2025 calibration
    mean = np.array([1.6, -30, 0, 1, 20, 0]) # [r, phi, theta, vr, alpha, beta]
    cov = np.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ]) # Units: [kpc, deg, deg, km/s, deg, deg]

    # Generate correlated offsets
    rng = np.random.default_rng(0)
    posvel = rng.multivariate_normal(mean, cov, size=2 * N)

    # Convert to physical quantities
    Dr = posvel[:, 0] * r_tidal # Radial offset [kpc]
    phi = np.deg2rad(posvel[:, 1]) # Azimuth [rad]
    theta = np.deg2rad(posvel[:, 2]) # Polar [rad]

    # Escape velocity at each offset radius
    v_esc = np.sqrt(2 * G * mass_sat / np.abs(Dr))
    Dv = posvel[:, 3] * v_esc # Velocity magnitude [kpc/Myr]

    # Velocity angles
    alpha = np.deg2rad(posvel[:, 4]) # Velocity azimuth [rad]
    beta = np.deg2rad(posvel[:, 5]) # Velocity polar [rad]

    # Cartesian offsets in satellite frame
    dx = Dr * np.cos(theta) * np.cos(phi)
    dy = Dr * np.cos(theta) * np.sin(phi)
    dz = Dr * np.sin(theta)
    dvx = Dv * np.cos(beta) * np.cos(alpha)
    dvy = Dv * np.cos(beta) * np.sin(alpha)
    dvz = Dv * np.sin(beta)

    # Construct offset arrays
    offset_pos = np.column_stack([dx, dy, dz])
    offset_vel = np.column_stack([dvx, dvy, dvz])

    # Transform to host frame
    ic_stream = np.tile(orbit_sat, 2).reshape(2 * N, 6)

    # trailing arm
    ic_stream[::2, 0:3] += np.einsum('ni,nij->nj', offset_pos[::2], R)
    ic_stream[::2, 3:6] += np.einsum('ni,nij->nj', offset_vel[::2], R)

    # leading arm
    ic_stream[1::2, 0:3] += np.einsum('ni,nij->nj', -offset_pos[1::2], R)
    ic_stream[1::2, 3:6] += np.einsum('ni,nij->nj', -offset_vel[1::2], R)

    return ic_stream


def create_ic_particle_spray_fardal2015(
    orbit_sat: np.ndarray,
    rj: np.ndarray,
    vj: np.ndarray,
    R: np.ndarray,
    gala_modified: bool = True,
) -> np.ndarray:
    """Create spray ICs using the Fardal+2015 method.

    Parameters
    ----------
    orbit_sat : np.ndarray, shape ``(N, 6)``
        Satellite orbit (kpc, km/s).
    rj : np.ndarray, shape ``(N,)``
        Jacobi radii (kpc).
    vj : np.ndarray, shape ``(N,)``
        Velocity scales (km/s).
    R : np.ndarray, shape ``(N, 3, 3)``
        Rotation matrices to the satellite frame.
    gala_modified : bool
        Use Gala's modified dispersion parameters.

    Returns
    -------
    ic_stream : np.ndarray, shape ``(2N, 6)``
        Initial conditions for leading/trailing stream particles.

    Notes
    -----
    Generates two particles per orbit point (leading + trailing).
    Position/velocity offsets follow asymmetric Gaussian distributions
    (Fardal, M. A., et al. 2015, MNRAS, 452, 301).
    """
    N = len(rj)
    # Expand quantities for leading/trailing arms (2 particles per orbit point).
    rj = np.repeat(rj, 2) * np.tile([1, -1], N) # Alternate signs for arms
    vj = np.repeat(vj, 2) * np.tile([1, -1], N) 
    R = np.repeat(R, 2, axis=0) # Critical: match 2N shape

    # Configure dispersion parameters (Gala-modified or original Fardal values)
    params = {
        'mean_x': 2.0,
        'disp_x': 0.5 if gala_modified else 0.4,
        'disp_z': 0.5,
        'mean_vy': 0.3,
        'disp_vy': 0.5 if gala_modified else 0.4,
        'disp_vz': 0.5,
    }

    # Generate offsets in satellite frame
    rng = np.random.default_rng(0)
    rx = rng.normal(loc=params['mean_x'], scale=params['disp_x'], size=2 * N)
    rz = rng.normal(scale=params['disp_z'], size=2 * N) * rj
    rvy = (
        rng.normal(loc=params['mean_vy'], scale=params['disp_vy'], size=2 * N)
        * vj * (rx if gala_modified else 1)
    )
    rvz = rng.normal(scale=params['disp_vz'], size=2 * N) * vj
    # Scale radial positions
    rx *= rj

    offset_pos = np.column_stack([rx, np.zeros(2 * N), rz])
    offset_vel = np.column_stack([np.zeros(2 * N), rvy, rvz])

    # Transform to host frame
    ic_stream = np.tile(orbit_sat, 2).reshape(2 * N, 6)
    ic_stream[:, :3] += np.einsum('ni,nij->nj', offset_pos, R)
    ic_stream[:, 3:6] += np.einsum('ni,nij->nj', offset_vel, R)

    return ic_stream


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main driver                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_particle_spray_stream(
    pot_host,
    initmass: float,
    sat_cen_present: np.ndarray | tuple,
    scaleradius: float,
    num_particles: int = int(1e4),
    prog_pot_kind: str = 'King',
    time_total: float = 3.0,
    time_end: float = 13.78,
    time_stripping: np.ndarray | None = None,
    save_rate: int = 1,
    dynFric: bool = False,
    pot_for_dynFric_sigma=None,
    gala_modified: bool = True,
    add_perturber: dict[str, Any] | None = None,
    create_ic_method: Callable = create_ic_particle_spray_chen2025,
    verbose: bool = False,
    accuracy_integ: float = 1e-7,
    eigenvalue_method: bool = True,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    r"""Create a stellar stream using the particle-spray method.

    The progenitor orbit is **rewound** from its present-day phase-space
    (``sat_cen_present``) by ``time_total``, then particles are
    progressively released at the tidal radius as the satellite evolves
    forward to ``time_end``.

    Parameters
    ----------
    pot_host : agama.Potential
        Host gravitational potential.
    initmass : float
        Initial satellite mass (M_sun, must be > 0).
    sat_cen_present : array_like, shape ``(6,)``
        Present-day satellite ``[x, y, z, vx, vy, vz]`` (kpc, km/s).
    scaleradius : float
        Satellite scale radius (kpc, must be > 0).
    num_particles : int
        Number of stream particles (must be > 0).
    prog_pot_kind : ``'King'``, ``'Plummer'``, or ``'Plummer_withRcut'``
        Progenitor potential profile.
    time_total : float
        Look-back time to rewind the orbit (Gyr, >= 0).
    time_end : float
        Present-day epoch (Gyr).
    time_stripping : np.ndarray, shape ``(N,)``, optional
        Custom particle-release times, where ``N = num_particles // 2 + 1``.
        Values must lie in ``[time_end - time_total, time_end]``.
        If *None* (default), the progenitor's own time array is used,
        corresponding to **uniform stripping** along the orbit.
    save_rate : int
        Number of output snapshots (1 = final only).
    dynFric : bool
        Enable dynamical friction on the progenitor orbit.
    pot_for_dynFric_sigma : agama.Potential, optional
        Potential for velocity-dispersion computation (DF friction).
    gala_modified : bool
        Use Gala-modified dispersion parameters (Fardal method).
    add_perturber : dict, optional
        Perturber properties.  Must contain ``'mass'`` (M_sun),
        ``'scaleRadius'`` (kpc), ``'w_subhalo_impact'`` (shape ``(6,)``),
        and ``'time_impact'`` (Gyr).  Set to *None* to disable.
    create_ic_method : Callable
        IC generator function (must accept a compatible signature).
    verbose : bool
        Print progress messages.
    accuracy_integ : float
        Orbit integrator accuracy.
    eigenvalue_method : bool
        Tidal-tensor eigenvalue method for Jacobi radius.
    **kwargs
        Extra arguments for the progenitor potential (e.g. ``W0``, ``trunc``).

    Returns
    -------
    dict
        Dictionary containing simulation results with keys:

        - 'times' : np.ndarray of shape (Nsaves,)
            Array of save times.
        - 'prog_xv' : np.ndarray of shape (Nsaves, 6)
            Progenitor positions and velocities at each save time.
            Format: [x, y, z, vx, vy, vz] in [kpc, km/s].
        - 'part_xv' : np.ndarray of shape (Nparticles, Nsaves, 6) or (Nparticles, 6) if save_rate=1
            Stream particle states at each save time.
            Contains NaN values where particles weren't yet released.
            Format: [x, y, z, vx, vy, vz] in [kpc, km/s].

    Notes
    -----
    This function implements the particle-spray method for generating stellar 
    streams from disrupting satellites. The method progressively releases 
    particles from the satellite as it orbits through the host potential.
    
    The particle spray approach models tidal stripping by:
    
    1. Rewinding the satellite orbit by `time_total` 
    2. Progressively releasing particles at tidal radius as satellite evolves
    3. Tracking all particles forward to present day (`time_end`)
    
    The `create_ic_method` parameter allows customization of the initial 
    condition generation, enabling different spray algorithms or modifications 
    to the Chen et al. (2025) approach.
    """
    if not AGAMA_AVAILABLE:
        raise ImportError(
            "Agama is required for particle spray. "
            "Install with: pip install agama"
        )

    # --- Input validation ---
    sat_cen_present = np.asarray(sat_cen_present, dtype=float)
    if sat_cen_present.shape != (6,):
        raise ValueError(
            f"sat_cen_present must have shape (6,), got {sat_cen_present.shape}"
        )
    if initmass <= 0:
        raise ValueError("initmass must be positive.")
    if scaleradius <= 0:
        raise ValueError("scaleradius must be positive.")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if time_total < 0:
        raise ValueError("time_total must be non-negative.")
    if save_rate < 1:
        raise ValueError("save_rate must be >= 1.")
    if accuracy_integ <= 0:
        raise ValueError("accuracy_integ must be positive.")

    if add_perturber is None:
        add_perturber = {'mass': 0, 'scaleRadius': 0.05}

    N = num_particles // 2 + 1

    # --- Rewind progenitor orbit ---
    time_sat, orbit_sat = agama.orbit(
        ic=sat_cen_present, potential=pot_host,
        time=-time_total, timestart=time_end, trajsize=N,
    )

    # Reverse to forward-time order
    time_sat = time_sat[::-1]
    orbit_sat = orbit_sat[::-1]

    # --- Stripping times ---
    if time_stripping is None:
        time_stripping = time_sat
    else:
        time_stripping = np.asarray(time_stripping, dtype=float)
        if time_stripping.shape != (N,):
            raise ValueError(
                f"time_stripping must have length N = num_particles // 2 + 1 "
                f"= {N}, got {time_stripping.shape[0]}"
            )
        t_lo = time_end - time_total
        t_hi = time_end
        if np.any(time_stripping < t_lo) or np.any(time_stripping > t_hi):
            raise ValueError(
                f"time_stripping values must lie in "
                f"[{t_lo:.4f}, {t_hi:.4f}]."
            )

    # --- Progenitor potential ---
    pot_sat = _get_prog_GalaxyModel(
        initmass, scaleradius, prog_pot_kind,
        return_DistribFunc=False, **kwargs,
    )
    pot_sat_moving = agama.Potential(
        potential=pot_sat,
        center=np.column_stack([time_sat, orbit_sat]),
    )

    # --- Perturber (optional) ---
    if add_perturber['mass'] > 0:
        pot_perturber_moving = _create_perturber_potential(
            add_perturber, pot_host, time_total, time_end, verbose=verbose,
        )
        pot_total = agama.Potential(pot_host, pot_sat_moving, pot_perturber_moving)
    else:
        pot_total = agama.Potential(pot_host, pot_sat_moving)

    # --- Generate initial conditions ---
    rj, vj, R = _get_jacobi_rad_vel_mtx(
        pot_host, orbit_sat, initmass,
        t=time_stripping, eigenvalue_method=eigenvalue_method,
    )

    method_args = {
        'orbit_sat': orbit_sat, 'mass_sat': initmass,
        'rj': rj, 'vj': vj, 'R': R, 'gala_modified': gala_modified,
    }

    # Filter to only the parameters the IC method expects
    sig = inspect.signature(create_ic_method)
    filtered_args = {
        k: v for k, v in method_args.items() if k in sig.parameters
    }

    ic_stream = create_ic_method(**filtered_args)
    time_seed = np.repeat(time_stripping, 2)

    # --- Save times ---
    if save_rate > 1:
        save_times = np.linspace(
            time_end - time_total, time_end - 1e-6, save_rate,
        )
    else:
        save_times = time_end - 1e-5 # clip for floating points

    # --- Progenitor interpolation for multi-snapshot ---
    if save_rate > 1:
        if verbose:
            print("Interpolating particle trajectories in time.")
        prog_interp = interp1d(
            time_sat, orbit_sat, axis=0, kind='cubic',
            fill_value='extrapolate',
        )
        prog_xv = prog_interp(save_times)

    # --- Integrate all stream particles ---
    result = agama.orbit(
        potential=pot_total,
        ic=ic_stream[:-2],
        timestart=time_seed[:-2],
        time=time_end - time_seed[:-2],
        dtype=object, # Agama's inbuild trajectory interpolator. 
        accuracy=accuracy_integ,
        verbose=verbose,
    )

    # ======== Particle Trajectory from the interpolator ========
    part_xv = np.stack(
        [orbit(save_times) for orbit in result], axis=0,
    )

    return {
        'times': np.around(save_times, decimals=5) if save_rate > 1 else time_sat,
        'prog_xv': prog_xv if save_rate > 1 else orbit_sat,
        'part_xv': part_xv,
    }
