"""Restricted N-body stream generation.

Evolves test particles in the combined potential of the host galaxy and
an evolving progenitor satellite.  The satellite potential is updated at
regular intervals using a monopole expansion of the remaining bound
particles.

By default the progenitor orbit is **rewound** from its present-day
phase-space before sampling particles; when *xv_init* is provided the
code integrates forward directly (no rewinding).
"""
from __future__ import annotations

import math
import warnings
from typing import Any

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
    _find_prog_pot_Nparticles,
    _get_prog_GalaxyModel,
    _integrate_orbit_with_dynamical_friction,
)

__all__: list[str] = ['run_restricted_nbody']


def run_restricted_nbody(
    pot_host,
    initmass: float,
    sat_cen_present: np.ndarray | tuple,
    scaleradius: float | None = None,
    num_particles: int = int(1e4),
    prog_pot_kind: str = 'King',
    xv_init: np.ndarray | None = None,
    dynFric: bool = False,
    pot_for_dynFric_sigma=None,
    time_total: float = 3.0,
    time_end: float = 0.0,
    step_size: int = 10,
    save_rate: int = 300,
    trajsize_each_step: int = 10,
    add_perturber: dict[str, Any] | None = None,
    verbose: bool = False,
    accuracy_integ: float = 1e-7,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    r"""Run a restricted (collisionless) N-body simulation.

    Test particles representing the satellite's stars move in the
    combined potential of the host galaxy and the (evolving) satellite.
    The satellite potential is rebuilt from the bound particles at every
    ``step_size`` integration steps.

    By default the progenitor orbit is **rewound** from
    ``sat_cen_present`` by ``time_total``; particles are sampled from
    the chosen profile and integrated forward.  When *xv_init* is
    provided the given particles are integrated forward directly
    from ``time_end - time_total`` to ``time_end`` (no rewinding).

    Parameters
    ----------
    pot_host : agama.Potential
        Host gravitational potential.
    initmass : float
        Initial satellite mass (M_sun, must be > 0).
    sat_cen_present : array_like, shape ``(6,)``
        Present-day satellite ``[x, y, z, vx, vy, vz]`` (kpc, km/s).
        When *xv_init* is provided this is taken as the progenitor COM.
    scaleradius : float, optional
        Satellite scale radius (kpc).  Ignored when *xv_init* is given.
    num_particles : int
        Number of test particles.
    prog_pot_kind : ``'King'``, ``'Plummer'``, or ``'Plummer_withRcut'``
        Progenitor potential profile (ignored when *xv_init* is given).
    xv_init : np.ndarray, shape ``(N, 6)``, optional
        Pre-existing particle phase-space.  When provided the code
        integrates forward directly — no rewinding or sampling.
    dynFric : bool
        Enable dynamical friction on the progenitor orbit.
    pot_for_dynFric_sigma : agama.Potential, optional
        Potential for velocity-dispersion computation (DF friction).
    time_total : float
        Look-back time / integration duration (Gyr, >= 0).
    time_end : float
        Present-day epoch (Gyr).
    step_size : int
        Number of ODE steps grouped per potential-update iteration.
    save_rate : int
        Number of interpolated output snapshots (1 = final only).
    trajsize_each_step : int
        Trajectory points saved per integration step.
    add_perturber : dict, optional
        Perturber properties.  Must contain ``'mass'`` (M_sun),
        ``'scaleRadius'`` (kpc), ``'w_subhalo_impact'`` (shape ``(6,)``),
        and ``'time_impact'`` (Gyr).  Set to *None* to disable.
    verbose : bool
        Print progress messages.
    accuracy_integ : float
        Orbit integrator accuracy.
    **kwargs
        Extra arguments for the progenitor potential (e.g. ``W0``, ``trunc``).

    Returns
    -------
    dict
        Dictionary containing simulation results with keys:
        - 'times': array of saved snapshot times (Gyr).
        - 'prog_xv': array of progenitor phase-space at saved times (shape ``(len(times), 6)``).
        - 'part_xv': array of particle phase-space at saved times (shape ``(N, len(times), 6)``).
        - 'bound_mass': array of bound mass at saved times (M_sun).
    """
    if not AGAMA_AVAILABLE:
        raise ImportError(
            "Agama is required for restricted N-body. "
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
    if time_total < 0:
        raise ValueError("time_total must be non-negative.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")
    if save_rate < 1:
        raise ValueError("save_rate must be >= 1.")
    if trajsize_each_step < 1:
        raise ValueError("trajsize_each_step must be >= 1.")
    if accuracy_integ <= 0:
        raise ValueError("accuracy_integ must be positive.")

    if add_perturber is None:
        add_perturber = {'mass': 0, 'scaleRadius': 0.05}

    xv = np.copy(xv_init) if xv_init is not None else None

    if xv is None:
        if scaleradius is None or scaleradius <= 0:
            raise ValueError("scaleradius must be a positive number.")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
    else:
        if xv.ndim != 2 or xv.shape[1] != 6:
            raise ValueError(f"xv_init must have shape (N, 6), got {xv.shape}")
        if scaleradius is not None:
            warnings.warn(
                "scaleradius ignored when xv_init provided", UserWarning,
            )

    # --- Orbit integration / particle sampling ---
    if xv is None:
        # Rewind progenitor with or without dynamical friction
        time_sat, orbit_sat = _integrate_orbit_with_dynamical_friction(
            sat_cen_present, pot_host,
            initmass if dynFric else 0,
            time_total, time_end, pot_for_dynFric_sigma,
        )
        if verbose:
            fric = "" if dynFric else "out"
            print(
                f"Orbit integrated backward for {time_total} Gyr "
                f"with{fric} dynamical friction."
            )

        # Forward-time order
        time_sat = time_sat[::-1]
        orbit_sat = orbit_sat[::-1]

        # Sample particles from the progenitor profile
        pot_sat, df_sat = _get_prog_GalaxyModel(
            initmass, scaleradius, prog_pot_kind, **kwargs,
        )
        xv, masses = agama.GalaxyModel(pot_sat, df_sat).sample(num_particles)
        xv += orbit_sat[0]
    else:
        masses = np.full(len(xv), initmass / len(xv))
        pot_sat, prog_init = _find_prog_pot_Nparticles(
            xv, sat_cen_present, masses=masses,
        )
        time_sat, orbit_sat = agama.orbit(
            ic=prog_init, potential=pot_host,
            time=time_total, timestart=time_end - time_total, trajsize=0,
        )

    # --- Perturber (optional) ---
    pot_perturber_moving = None
    if add_perturber['mass'] > 0:
        pot_perturber_moving = _create_perturber_potential(
            add_perturber, pot_host, time_total, time_end, verbose=verbose,
        )

    # --- Main integration loop ---
    bound_mass = [initmass]
    num_steps = int(np.floor(len(time_sat) / step_size))

    if verbose:
        print(f"No. of timesteps to update progenitor potential: {num_steps}.")

    phase_space_list = []
    time_list = []

    try:
        from tqdm.auto import trange
        loop_iter = trange(num_steps + 1, desc="Simulating Restricted N-body")
    except ImportError:
        loop_iter = range(num_steps + 1)

    for i in loop_iter:
        start_index = i * step_size
        end_index = (i + 1) * step_size

        if end_index >= len(time_sat):
            end_index = -1

        time_begin = time_sat[start_index]
        time_end_step = time_sat[end_index]
        time_step = time_end_step - time_begin

        if math.isclose(time_sat[-1], time_begin, abs_tol=1e-6) or time_sat[-1] < time_begin:
            bound_mass.append(np.sum(masses[bound]))
            continue

        # Build time-dependent total potential
        pot_sat_moving = agama.Potential(
            potential=pot_sat,
            center=np.column_stack([time_sat, orbit_sat]),
        )
        if pot_perturber_moving is not None:
            pot_total = agama.Potential(pot_host, pot_sat_moving, pot_perturber_moving)
        else:
            pot_total = agama.Potential(pot_host, pot_sat_moving)

        # Integrate particles
        traj = agama.orbit(
            ic=xv, potential=pot_total, trajsize=trajsize_each_step,
            verbose=verbose, time=time_step, timestart=time_begin,
            accuracy=accuracy_integ,
        )

        traj_times = traj[0][0]
        traj_phase_space = np.array([t[1] for t in traj])

        phase_space_list.append(traj_phase_space)
        time_list.append(traj_times)

        # Update positions
        xv = traj_phase_space[:, -1]

        # Rebuild satellite potential
        xv_rel = xv - orbit_sat[end_index]
        try:
            pot_sat = agama.Potential(
                type='multipole', particles=(xv_rel[:, 0:3], masses),
                symmetry='s',
            )
        except Exception:
            if verbose:
                print(f"Satellite potential failed at {time_sat[end_index]:.2f} Gyr.")
            return {
                'times': time_sat, 'prog_xv': orbit_sat,
                'part_xv': xv, 'bound_mass': np.hstack(bound_mass),
            }

        # Boundness check
        bound = (
            pot_sat.potential(xv_rel[:, 0:3])
            + 0.5 * np.sum(xv_rel[:, 3:6]**2, axis=1)
        ) < 0
        bound_frac = bound.sum() / xv.shape[0]

        if verbose:
            print(
                f"\rBound Frac: {bound_frac:.4f} at "
                f"T: {time_sat[end_index]:.2f} Gyr.",
                end='\r', flush=True,
            )
        bound_mass.append(np.sum(masses[bound]))

        if end_index == -1:
            break

    if verbose:
        print(
            f"\rBound Frac: {bound_frac:.4f} at "
            f"T: {time_sat[end_index]:.2f} Gyr."
        )

    # --- Post-processing: interpolate to save_times ---
    bound_mass = np.hstack(bound_mass)
    phase_space = np.hstack(phase_space_list)
    time_traj = np.hstack(time_list)

    save_times = (
        np.linspace(time_end - time_total, time_end, save_rate)
        if save_rate > 1 else time_end
    )

    # Bound mass interpolation
    try:
        time_steps_bound = np.hstack([time_sat[::step_size], time_sat[-1]])
        _, unique_inds = np.unique(time_steps_bound, return_index=True)
        bound_mass_interp = interp1d(
            time_steps_bound[unique_inds], bound_mass[unique_inds],
            kind='cubic', fill_value='extrapolate',
        )
        bound_mass_extract = np.minimum(
            bound_mass_interp(save_times),
            bound_mass_interp(save_times)[0],
        )
        bound_mass_out = bound_mass_extract
    except Exception as e:
        if verbose:
            print(f"Bound-mass interpolation failed: {e}. Returning raw.")
        bound_mass_out = (time_sat[::step_size], bound_mass)

    # Phase-space interpolation
    _, unique_inds = np.unique(time_traj, return_index=True)
    phase_space_interp = interp1d(
        time_traj[unique_inds], phase_space[:, unique_inds],
        kind='cubic', axis=1, fill_value='extrapolate',
    )
    orbit_sat_interp = interp1d(
        time_sat, orbit_sat, kind='cubic', axis=0, fill_value='extrapolate',
    )

    return {
        'times': save_times,
        'prog_xv': orbit_sat_interp(save_times),
        'part_xv': phase_space_interp(save_times),
        'bound_mass': bound_mass_out,
    }
