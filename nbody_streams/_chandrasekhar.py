"""
nbody_streams._chandrasekhar
============================
Chandrasekhar dynamical-friction helpers for N-body runs.

Public API
----------
compute_sigma_r        -- velocity-dispersion profile from a host potential
chandrasekhar_friction -- single evaluation of the DF acceleration
make_df_force_extra    -- factory that returns a force_extra closure

Private helpers (not exported)
-------------------------------
_jeans_sigma_r         -- Jeans-equation fallback for sigma(r)
_shrinking_sphere_com  -- iterative shrinking-sphere centre-of-mass
_to_numpy              -- transparent CuPy/NumPy conversion
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import special

try:
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)
    _AGAMA_OK = True
    _G = agama.G  # kpc (km/s)^2 M_sun^-1
except ImportError:
    _AGAMA_OK = False
    _G = 4.300917270069976e-6  # fallback constant

__all__ = [
    "compute_sigma_r",
    "chandrasekhar_friction",
    "make_df_force_extra",
]


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _to_numpy(arr: "np.ndarray | cp.ndarray") -> np.ndarray:
    """Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when
    possible for NumPy; triggers a device-to-host transfer for CuPy)."""
    try:
        return arr.get()
    except AttributeError:
        return np.asarray(arr)


# ---------------------------------------------------------------------------
# Velocity-dispersion profile
# ---------------------------------------------------------------------------

def _jeans_sigma_r(
    pot,
    t_eval: float = 0.0,
    grid_r: np.ndarray | None = None,
) -> Callable:
    """Jeans-equation estimate of the isotropic 1-D velocity dispersion.

    Uses the spherically-symmetric isotropic Jeans equation::

        sigma^2(r) = (1/rho(r)) * integral_r^inf  rho(r') |g_r(r')| dr'

    Works for *any* Agama potential (including time-dependent ones when
    evaluated at *t_eval*).

    Parameters
    ----------
    pot : agama.Potential
        Host potential.
    t_eval : float
        Time at which to evaluate density and force (useful for evolving
        potentials — pick the simulation midpoint).
    grid_r : ndarray, optional
        Log-spaced radial grid for the numerical integration and spline.
        Default: ``np.logspace(-1, 2, 64)`` (0.1 -- 100 kpc).

    Returns
    -------
    Callable
        ``sigma_func(r: float | ndarray) -> ndarray``  [km/s].
    """
    if grid_r is None:
        grid_r = np.logspace(-1, 2, 64)

    xyz = np.column_stack([
        grid_r,
        np.zeros_like(grid_r),
        np.zeros_like(grid_r),
    ])

    rho = np.asarray(pot.density(xyz, t=t_eval))        # (N,) M_sun/kpc^3
    forces = np.asarray(pot.force(xyz, t=t_eval))       # (N, 3) (km/s)^2/kpc
    g_r = np.abs(forces[:, 0])                          # radial magnitude

    integrand = rho * g_r                               # M_sun (km/s)^2 kpc^-4

    # Backward integral: sigma^2(r_i) = (1/rho_i) * trapz(integrand[i:], r[i:])
    sigma2 = np.empty_like(grid_r)
    for i in range(len(grid_r)):
        sigma2[i] = np.trapezoid(integrand[i:], grid_r[i:]) / max(rho[i], 1e-30)

    sigma = np.sqrt(np.maximum(sigma2, 0.0))

    valid = sigma > 0
    log_r = np.log(grid_r[valid])
    log_s = np.log(sigma[valid])

    try:
        logspl = agama.Spline(log_r, log_s)
        r_min, r_max = grid_r[valid].min(), grid_r[valid].max()

        def _sigma(r: "float | np.ndarray") -> np.ndarray:
            r_arr = np.clip(np.asarray(r, dtype=float), r_min, r_max)
            return np.exp(logspl(np.log(r_arr)))

    except Exception:
        from scipy.interpolate import interp1d
        logspl_sp = interp1d(
            log_r, log_s,
            bounds_error=False,
            fill_value=(log_s[0], log_s[-1]),
        )

        def _sigma(r: "float | np.ndarray") -> np.ndarray:
            return np.exp(logspl_sp(np.log(np.asarray(r, dtype=float))))

    return _sigma


def compute_sigma_r(
    pot,
    t_eval: float | None = None,
    grid_r: np.ndarray | None = None,
) -> Callable:
    """Compute a radial velocity-dispersion profile from an Agama potential.

    First attempts Agama's quasispherical distribution-function method
    (fast, self-consistent for spherical potentials).  If that fails
    (non-spherical geometry, time-dependent potential, etc.) falls back to
    the isotropic Jeans equation evaluated at *t_eval*.

    Parameters
    ----------
    pot : agama.Potential
        Host potential.
    t_eval : float, optional
        Time at which to evaluate the potential for the Jeans fallback (and
        for the quasispherical DF grid points).  For a time-evolving
        potential, pass the simulation midpoint ``(t_start + t_end) / 2``.
        Default: ``0.0``.
    grid_r : ndarray, optional
        Radial grid (kpc) for dispersion evaluation.  Default:
        ``np.logspace(-1, 2, 32)``.

    Returns
    -------
    Callable
        ``sigma_func(r: float | ndarray) -> ndarray``  [km/s].
    """
    if not _AGAMA_OK:
        raise ImportError(
            "Agama is required for compute_sigma_r.  "
            "Install it with: pip install agama"
        )

    if t_eval is None:
        t_eval = 0.0

    qs_grid = grid_r if grid_r is not None else np.logspace(-1, 2, 32)

    try:
        df_host = agama.DistributionFunction(
            type="quasispherical", potential=pot,
        )
        xyz = np.column_stack([
            qs_grid,
            np.zeros_like(qs_grid),
            np.zeros_like(qs_grid),
        ])
        grid_sig = agama.GalaxyModel(pot, df_host).moments(
            xyz, dens=False, vel=False, vel2=True,
        )[:, 0] ** 0.5

        if not np.all(np.isfinite(grid_sig)) or np.any(grid_sig <= 0):
            raise ValueError(
                "quasispherical DF returned non-finite or non-positive sigma"
            )

        logspl = agama.Spline(np.log(qs_grid), np.log(grid_sig))
        r_min, r_max = qs_grid.min(), qs_grid.max()

        def _sigma_qs(r: "float | np.ndarray") -> np.ndarray:
            r_arr = np.clip(np.asarray(r, dtype=float), r_min, r_max)
            return np.exp(logspl(np.log(r_arr)))

        return _sigma_qs

    except Exception:
        warnings.warn(
            "quasispherical DF failed; falling back to Jeans equation "
            f"evaluated at t={t_eval:.4g}.",
            RuntimeWarning,
            stacklevel=2,
        )
        jeans_grid = grid_r if grid_r is not None else np.logspace(-1, 2, 64)
        return _jeans_sigma_r(pot, t_eval=t_eval, grid_r=jeans_grid)


# ---------------------------------------------------------------------------
# Shrinking-sphere centre of mass
# ---------------------------------------------------------------------------

def _shrinking_sphere_com(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    n_iter: int = 5,
    frac: float = 0.5,
    min_particles: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative shrinking-sphere centre-of-mass estimator.

    At each iteration the sphere is centred on the current mass-weighted
    centroid and its radius shrunk by *frac*.  This converges quickly to
    the densest region (potential minimum of the satellite).

    Parameters
    ----------
    pos, vel : ndarray, shape ``(N, 3)``
        Particle positions and velocities (NumPy; call ``_to_numpy`` first
        for GPU arrays).
    masses : ndarray, shape ``(N,)``
        Particle masses.
    n_iter : int
        Number of shrinking iterations.
    frac : float
        Radius reduction factor per iteration (0 < frac < 1).
    min_particles : int
        Stop early if fewer than this many particles remain.

    Returns
    -------
    r_com : ndarray, shape ``(3,)``
    v_com : ndarray, shape ``(3,)``
    """
    idx = np.arange(len(pos))

    for _ in range(n_iter):
        p = pos[idx]
        m = masses[idx]
        m_sum = m.sum()

        # Mass-weighted centroid (avoids slow np.average internals)
        r_com = m @ p / m_sum

        r = np.linalg.norm(p - r_com, axis=1)
        keep = r < frac * r.max()

        if keep.sum() < min_particles:
            break
        idx = idx[keep]

    m = masses[idx]
    m_sum = m.sum()
    r_com = m @ pos[idx] / m_sum
    v_com = m @ vel[idx] / m_sum

    return r_com, v_com


# ---------------------------------------------------------------------------
# Core Chandrasekhar formula
# ---------------------------------------------------------------------------

def chandrasekhar_friction(
    r_com: np.ndarray,
    v_com: np.ndarray,
    M_sat: float,
    pot,
    sigma_func: Callable,
    t: float,
    coulomb_mode: str = "variable",
    fixed_ln_lambda: float = 3.0,
    core_gamma: float = 0.0,
    r_core: float = 1.0,
) -> np.ndarray:
    """Chandrasekhar dynamical-friction acceleration at the satellite CoM.

    Computes BT2008 eq. 8.13::

        a_DF = -4*pi*G^2 * M_sat * rho * ln(Lambda) / v^2
               * [erf(X) - (2X/sqrt(pi)) exp(-X^2)]  *  vhat

    where ``X = v / (sqrt(2) * sigma(r))``.

    Parameters
    ----------
    r_com, v_com : ndarray, shape ``(3,)``
        Satellite CoM position [kpc] and velocity [km/s].
    M_sat : float
        Satellite mass [M_sun].
    pot : agama.Potential
        Host potential (for local density ``rho(r, t)``).
    sigma_func : Callable
        ``sigma(r) -> float`` [km/s].
    t : float
        Current simulation time.
    coulomb_mode : ``'variable'`` or ``'fixed'``
        How the Coulomb logarithm is computed.

        * ``'variable'`` -- ``ln(Lambda) = ln(r * v^2 / (G * M_sat))``
          clipped to ≥ ln(1.1).
        * ``'fixed'`` -- constant ``fixed_ln_lambda``.

    fixed_ln_lambda : float
        Used when *coulomb_mode* = ``'fixed'``.
    core_gamma : float
        Power-law index for Read+2006 core-stalling suppression
        (0 = standard cusp; ~1--2 for constant-density cores).
    r_core : float
        Core radius [kpc] for the suppression factor.

    Returns
    -------
    ndarray, shape ``(3,)``
        DF acceleration [km/s per kpc/(km/s)] == [(km/s)^2/kpc].
    """
    r = float(np.linalg.norm(r_com))
    v = float(np.linalg.norm(v_com))

    if r < 1e-6 or v < 1e-6:
        return np.zeros(3)

    rho = float(pot.density(r_com, t=t))
    sigma = float(sigma_func(r))
    X = v / (np.sqrt(2.0) * sigma)

    # Coulomb logarithm
    if coulomb_mode == "fixed":
        ln_lambda = fixed_ln_lambda
    else:
        b_min = _G * M_sat / (v ** 2 + 1e-30)
        Lambda = r / (b_min + 1e-9)
        ln_lambda = float(np.log(max(Lambda, 1.1)))

    # Chandrasekhar bracket factor
    bracket = special.erf(X) - (2.0 / np.sqrt(np.pi)) * X * np.exp(-(X ** 2))

    a_mag = (
        4.0 * np.pi * (_G ** 2) * M_sat * rho * ln_lambda * bracket
    ) / (v ** 2)

    # Core-stalling suppression (Read+2006, Petts+2016)
    if core_gamma > 0.0:
        a_mag *= min(1.0, (r / r_core) ** core_gamma)

    return -(v_com / v) * a_mag


# ---------------------------------------------------------------------------
# force_extra factory
# ---------------------------------------------------------------------------

def make_df_force_extra(
    pot,
    M_sat: float,
    t_start: float,
    t_end: float,
    *,
    coulomb_mode: str = "variable",
    fixed_ln_lambda: float = 3.0,
    core_gamma: float = 0.0,
    r_core: float = 1.0,
    update_interval: int = 10,
    shrink_n_iter: int = 5,
    shrink_frac: float = 0.5,
    sigma_grid_r: np.ndarray | None = None,
) -> Callable:
    """Build a ``force_extra`` closure that applies Chandrasekhar dynamical
    friction to the satellite particles.

    The closure is compatible with the ``force_extra`` parameter of
    :func:`~nbody_streams.run_nbody_gpu`, :func:`~nbody_streams.run_nbody_cpu`,
    and :func:`~nbody_streams.tree_gpu.run_gpu_tree.run_nbody_gpu_tree`.

    **How it works**

    1. On the first call the satellite CoM is found with the shrinking-sphere
       estimator (see :func:`_shrinking_sphere_com`).
    2. Between correction steps a kinematic *predictor* advances the cached
       CoM:  ``r_com += v_com*dt + 0.5*a_df*dt^2``  and
       ``v_com += a_df*dt``.
    3. Every *update_interval* steps the shrinking-sphere corrector reruns
       and the cached state is updated.
    4. The DF acceleration is evaluated at the corrected/predicted CoM and
       applied uniformly to **all** particles (the satellite is treated as a
       single rigid body for the purposes of friction).

    **Performance**

    The shrinking-sphere step costs ~500 µs for N=10 000 particles
    (5 iterations, pure NumPy).  Amortized over *update_interval* = 10
    steps this is ~50 µs per step — roughly 1 % overhead vs a GPU step.

    **GPU paths**

    On GPU paths the integrators pass CuPy arrays for ``pos`` and ``vel``.
    The closure calls :func:`_to_numpy` internally, so no user action is
    required.

    Parameters
    ----------
    pot : agama.Potential
        Host potential used for density, force, and sigma(r).
    M_sat : float
        Total satellite mass [M_sun].
    t_start, t_end : float
        Integration interval.  Used to evaluate sigma(r) at the midpoint.
    coulomb_mode : ``'variable'`` or ``'fixed'``
        Coulomb logarithm mode (see :func:`chandrasekhar_friction`).
    fixed_ln_lambda : float
        Only used when *coulomb_mode* = ``'fixed'``.
    core_gamma : float
        Core-stalling suppression index (0 = off).
    r_core : float
        Core radius [kpc] for core-stalling suppression.
    update_interval : int
        Correct the CoM with shrinking sphere every this many steps.
        Default 10.
    shrink_n_iter : int
        Shrinking-sphere iteration count.  Default 5.
    shrink_frac : float
        Radius reduction per shrinking-sphere iteration.  Default 0.5.
    sigma_grid_r : ndarray, optional
        Custom radial grid for sigma(r) computation.

    Returns
    -------
    Callable
        Closure with signature
        ``force_extra(pos, vel, masses, t) -> ndarray (N, 3)``.
        Pass it directly as ``force_extra=`` to any integrator.

    Raises
    ------
    ImportError
        If Agama is not available.
    ValueError
        If *M_sat* <= 0 or *update_interval* < 1.

    Examples
    --------
    >>> from nbody_streams._chandrasekhar import make_df_force_extra
    >>> import agama
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0)
    >>> df_force = make_df_force_extra(pot, M_sat=1e9, t_start=0.0, t_end=5.0)
    >>> result = run_nbody_gpu(xv, masses, ..., force_extra=df_force)
    """
    if not _AGAMA_OK:
        raise ImportError(
            "Agama is required for make_df_force_extra.  "
            "Install it with: pip install agama"
        )
    if M_sat <= 0:
        raise ValueError(f"M_sat must be positive, got {M_sat}")
    if update_interval < 1:
        raise ValueError(f"update_interval must be >= 1, got {update_interval}")

    # Build sigma(r) once at the midpoint time
    t_mid = 0.5 * (t_start + t_end)
    sigma_func = compute_sigma_r(pot, t_eval=t_mid, grid_r=sigma_grid_r)

    # Mutable closure state (dict avoids nonlocal for Python 3.10 compat)
    _state: dict = {
        "step": 0,
        "initialized": False,
        "t_prev": t_start,
        "r_com": np.zeros(3),
        "v_com": np.zeros(3),
        "a_df": np.zeros(3),
    }

    def _force_extra(
        pos: "np.ndarray | cp.ndarray",
        vel: "np.ndarray | cp.ndarray",
        masses: "np.ndarray | cp.ndarray",
        t: float,
    ) -> np.ndarray:
        pos_np = _to_numpy(pos)     # (N, 3)
        vel_np = _to_numpy(vel)     # (N, 3)
        m_np = _to_numpy(masses)    # (N,)

        step = _state["step"]
        dt = t - _state["t_prev"] if step > 0 else 0.0

        # ---- CoM update ------------------------------------------------
        if not _state["initialized"] or (step % update_interval == 0):
            # Corrector: shrinking sphere
            r_com, v_com = _shrinking_sphere_com(
                pos_np, vel_np, m_np,
                n_iter=shrink_n_iter,
                frac=shrink_frac,
            )
            _state["r_com"] = r_com
            _state["v_com"] = v_com
            _state["initialized"] = True
        else:
            # Predictor: kinematic extrapolation using cached DF accel
            a = _state["a_df"]
            r_com = _state["r_com"] + _state["v_com"] * dt + 0.5 * a * dt ** 2
            v_com = _state["v_com"] + a * dt
            _state["r_com"] = r_com
            _state["v_com"] = v_com

        # ---- Chandrasekhar acceleration --------------------------------
        a_df = chandrasekhar_friction(
            r_com=r_com,
            v_com=v_com,
            M_sat=M_sat,
            pot=pot,
            sigma_func=sigma_func,
            t=t,
            coulomb_mode=coulomb_mode,
            fixed_ln_lambda=fixed_ln_lambda,
            core_gamma=core_gamma,
            r_core=r_core,
        )
        _state["a_df"] = a_df
        _state["t_prev"] = t
        _state["step"] = step + 1

        # Apply uniform DF acceleration to all particles
        return np.broadcast_to(a_df, pos_np.shape).copy()

    return _force_extra
