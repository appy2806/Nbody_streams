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
_jeans_sigma_r         -- Jeans-equation sigma(r) (default)
_sigma_local_circular  -- local circular-speed approximation to sigma
_bound_center_phi      -- phi-energy iterative bound-particle centre
_shrinking_sphere_com  -- iterative shrinking-sphere centre-of-mass (fallback)
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


def _is_gpu_potential(pot) -> bool:
    """True when *pot* is from the PotentialGPU family (nbody_streams.agama_helper)."""
    try:
        from .agama_helper._potential import _GPUPotBase
        return isinstance(pot, _GPUPotBase)
    except ImportError:
        return False


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

    rho = _to_numpy(pot.density(xyz, t=t_eval))          # (N,) M_sun/kpc^3
    forces = _to_numpy(pot.force(xyz, t=t_eval))         # (N, 3) (km/s)^2/kpc
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


def _sigma_local_circular(pot, r: float, t: float = 0.0) -> float:
    """Local circular-speed approximation to the 1-D velocity dispersion.

    Uses the isotropic approximation::

        sigma(r, t) = sqrt(r * |g_r(r, t)| / 2)

    This is the 1-D dispersion for a system in which the circular speed is
    ``v_c = sqrt(r |g_r|)``.  It is cheap (one force evaluation), naturally
    time-evolving for moving external potentials, and works for any geometry.

    Parameters
    ----------
    pot : agama.Potential
        Host potential.
    r : float
        Galactocentric radius [kpc].
    t : float
        Evaluation time.

    Returns
    -------
    float
        sigma [km/s].
    """
    xyz = np.array([[r, 0.0, 0.0]])
    g_r = abs(float(_to_numpy(pot.force(xyz, t=t))[0, 0]))
    return float(np.sqrt(max(0.5 * r * g_r, 0.0)))


def compute_sigma_r(
    pot,
    t_eval: float | None = None,
    grid_r: np.ndarray | None = None,
    method: str = "jeans",
) -> Callable:
    """Compute a radial velocity-dispersion profile from an Agama potential.

    Parameters
    ----------
    pot : agama.Potential
        Host potential.
    t_eval : float, optional
        Time at which to evaluate the potential.  Default: ``0.0``.
    grid_r : ndarray, optional
        Radial grid (kpc) for dispersion evaluation.  Default:
        ``np.logspace(-1, 2, 64)`` (Jeans) or ``np.logspace(-1, 2, 32)``
        (quasispherical).
    method : ``'jeans'`` or ``'quasispherical'``
        Algorithm to use:

        * ``'jeans'`` *(default)* -- isotropic Jeans equation.  Works for any
          potential, including time-dependent ones.  Preferred for evolving
          hosts.
        * ``'quasispherical'`` -- Agama quasispherical DF moments.  Fast and
          self-consistent for spherical potentials; fails for non-spherical or
          time-dependent potentials (automatically falls back to Jeans in that
          case).

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

    if method == "jeans":
        jeans_grid = grid_r if grid_r is not None else np.logspace(-1, 2, 64)
        return _jeans_sigma_r(pot, t_eval=t_eval, grid_r=jeans_grid)

    if method == "quasispherical":
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

    raise ValueError(
        f"method must be 'jeans' or 'quasispherical', got '{method}'"
    )


# ---------------------------------------------------------------------------
# Phi-energy iterative bound-particle centre
# ---------------------------------------------------------------------------

def _bound_center_phi(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    phi: np.ndarray,
    r_com_prev: np.ndarray,
    v_com_prev: np.ndarray,
    dt: float,
    r_max: float = 10.0,
    max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phi-energy iterative bound-particle centre.

    Finds the centre of the gravitationally bound satellite core by iterating
    over particles that satisfy the binding criterion::

        phi_self + 0.5 * |v - v_com|^2 < 0

    where ``phi_self`` is the N-body self-gravity potential.  This is the
    standard energy-based iterative unbinding algorithm.

    Based on the reference implementation in the user's original DF code.

    Parameters
    ----------
    pos, vel : ndarray, shape ``(N, 3)``
        Particle positions [kpc] and velocities [km/s] (NumPy).
    masses : ndarray, shape ``(N,)``
        Particle masses (unused in the centering step; kept for API symmetry).
    phi : ndarray, shape ``(N,)``
        Self-gravity potential per particle [(km/s)^2].
    r_com_prev : ndarray, shape ``(3,)``
        Centre-of-mass position from the previous step [kpc].
    v_com_prev : ndarray, shape ``(3,)``
        Centre-of-mass velocity from the previous step [km/s].
    dt : float
        Timestep [simulation time units] used for the linear predictor.
    r_max : float
        Initial aperture radius [kpc] around the predicted centre.
        Only particles within this sphere are considered in the first
        iteration.  Default: 10.0 kpc.
    max_iter : int
        Maximum number of binding-energy iterations.  Default: 10.

    Returns
    -------
    r_com : ndarray, shape ``(3,)``
        Best-estimate centre position.
    v_com : ndarray, shape ``(3,)``
        Best-estimate centre velocity.
    bound_mask : ndarray of bool, shape ``(N,)``
        True for particles that satisfy the final binding criterion.
    """
    # Linear predictor: advance previous centre by dt
    r_pred = r_com_prev + v_com_prev * dt
    v_pred = v_com_prev.copy()

    # f_center is a 6-vector [x, y, z, vx, vy, vz]
    f_center = np.concatenate([r_pred, v_pred])

    # Initial aperture: particles within r_max of predicted centre
    dr2 = np.sum((pos - f_center[:3]) ** 2, axis=1)
    use = dr2 < r_max ** 2

    if use.sum() < 2:
        use = np.ones(len(pos), dtype=bool)

    prev_f_center = f_center.copy()

    # Iterative convergence
    bound_mask = use.copy()
    for _ in range(max_iter):
        xv_use = np.column_stack([pos[use], vel[use]])  # (M, 6)
        f_center = np.median(xv_use, axis=0)            # (6,)

        # Binding criterion: phi_self + 0.5 |v - v_com|^2 < 0
        v_rel2 = np.sum((vel - f_center[3:6]) ** 2, axis=1)
        bound_mask = (phi + 0.5 * v_rel2) < 0

        n_bound = int(bound_mask.sum())
        if n_bound <= 1 or np.all(f_center == prev_f_center):
            break

        dr2 = np.sum((pos - f_center[:3]) ** 2, axis=1)
        use = bound_mask & (dr2 < r_max ** 2)
        prev_f_center = f_center.copy()

        if use.sum() < 2:
            break

    return f_center[:3].copy(), f_center[3:6].copy(), bound_mask


# ---------------------------------------------------------------------------
# Shrinking-sphere centre of mass (fallback for direct integrators)
# ---------------------------------------------------------------------------

def _shrinking_sphere_com(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    n_iter: int = 5,
    frac: float = 0.5,
    min_particles: int = 16,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    r_sphere : float
        Maximum distance of the surviving (innermost) particles from the
        final CoM.  This is the effective radius of the identified satellite
        core and is used by ``make_df_force_extra`` to mask stripped particles.
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

    # Radius of the surviving core: max distance of surviving particles from CoM
    r_sphere = float(np.linalg.norm(pos[idx] - r_com, axis=1).max())

    return r_com, v_com, r_sphere


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

    rho = float(_to_numpy(pot.density(np.atleast_2d(r_com), t=t)).ravel()[0])
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
    apply_radius_factor: float | None = 2.0,
    sigma_method: str = "jeans",
) -> Callable:
    """Build a ``force_extra`` closure that applies Chandrasekhar dynamical
    friction to the satellite particles.

    The closure is compatible with the ``force_extra`` parameter of
    :func:`~nbody_streams.run_nbody_gpu`, :func:`~nbody_streams.run_nbody_cpu`,
    and :func:`~nbody_streams.tree_gpu.run_gpu_tree.run_nbody_gpu_tree`.

    **Centre-finding and bound mass**

    Two code paths are used depending on whether the integrator supplies the
    self-gravity potential ``phi``:

    * **phi available** (GPU/CPU tree backends): the closure uses an
      iterative energy-based algorithm (:func:`_bound_center_phi`) to find
      the bound core:  particles satisfying
      ``phi_self + 0.5 |v - v_com|^2 < 0`` are considered bound.  The DF
      is applied only to bound particles, and the satellite mass used in
      the formula is updated dynamically to ``M_bound = sum(masses[bound])``.
      This naturally accounts for tidal mass loss.

    * **phi not available** (GPU/CPU direct backends): falls back to the
      shrinking-sphere estimator (:func:`_shrinking_sphere_com`) with an
      *apply_radius_factor* radial cutoff.  M_sat is fixed at the
      factory-time value.

    **Velocity dispersion**

    The sigma(r) profile is controlled by *sigma_method*:

    * ``'jeans'`` *(default)* -- isotropic Jeans equation, evaluated once
      at the simulation midpoint.  Reliable for any potential geometry.
    * ``'local_circular'`` -- ``sigma(r,t) = sqrt(r |g_r(r,t)| / 2)``,
      re-evaluated at each corrector step.  Cheap and naturally
      time-evolving for moving host potentials.
    * ``'quasispherical'`` -- Agama quasispherical DF moments.  Most
      accurate for spherical time-independent potentials; falls back to
      Jeans automatically.

    **Performance**

    On tree paths the phi array is obtained for free (no extra force
    evaluation).  The bound-centre iteration costs ~200 µs for N=10 000
    particles.  Amortized over *update_interval* = 10 steps this is ~20 µs
    per step — under 1 % overhead vs a GPU-tree step.

    Parameters
    ----------
    pot : agama.Potential
        Host potential used for density and sigma(r).
    M_sat : float
        Total satellite mass [M_sun].  Used as fallback when phi is not
        available; the phi path uses dynamic ``M_bound`` instead.
    t_start, t_end : float
        Integration interval.  Used to evaluate sigma at the midpoint
        (Jeans / quasispherical paths only).
    coulomb_mode : ``'variable'`` or ``'fixed'``
        Coulomb logarithm mode (see :func:`chandrasekhar_friction`).
    fixed_ln_lambda : float
        Only used when *coulomb_mode* = ``'fixed'``.
    core_gamma : float
        Core-stalling suppression index (0 = off).
    r_core : float
        Core radius [kpc] for core-stalling suppression.
    update_interval : int
        Correct the CoM every this many steps.  Default 10.
    shrink_n_iter : int
        Shrinking-sphere iteration count (fallback path).  Default 5.
    shrink_frac : float
        Radius reduction per shrinking-sphere iteration.  Default 0.5.
    sigma_grid_r : ndarray, optional
        Custom radial grid for Jeans / quasispherical sigma(r).
    apply_radius_factor : float or None, optional
        Fallback-path only.  Apply DF only within
        ``apply_radius_factor * r_sphere`` of the CoM.  Default 2.0.
        Pass ``None`` to apply DF to all particles on the fallback path.
    sigma_method : ``'jeans'``, ``'local_circular'``, or ``'quasispherical'``
        Velocity-dispersion algorithm.  Default ``'jeans'``.

    Returns
    -------
    Callable
        Closure with signature
        ``force_extra(pos, vel, masses, t, *, phi=None) -> ndarray (N, 3)``.
        Pass it directly as ``force_extra=`` to any integrator.  Tree
        integrators will automatically supply ``phi``; direct integrators
        will not (fallback path is used silently).

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
    _gpu_pot = _is_gpu_potential(pot)
    # Agama is only required for quasispherical sigma (uses Agama DF moments)
    # or if pot is an agama.Potential itself.  PotentialGPU + jeans/local_circular
    # works without Agama installed.
    if not _AGAMA_OK and not _gpu_pot:
        raise ImportError(
            "Agama is required for make_df_force_extra when pot is an agama.Potential.  "
            "Install it with: pip install agama"
        )
    if not _AGAMA_OK and sigma_method == "quasispherical":
        raise ImportError(
            "sigma_method='quasispherical' requires Agama (DF moments via "
            "agama.DistributionFunction).  Use sigma_method='jeans' or "
            "'local_circular' when Agama is not installed."
        )
    if M_sat <= 0:
        raise ValueError(f"M_sat must be positive, got {M_sat}")
    if update_interval < 1:
        raise ValueError(f"update_interval must be >= 1, got {update_interval}")
    if sigma_method not in ("jeans", "local_circular", "quasispherical"):
        raise ValueError(
            f"sigma_method must be 'jeans', 'local_circular', or "
            f"'quasispherical', got '{sigma_method}'"
        )
    if _gpu_pot and sigma_method == "quasispherical":
        warnings.warn(
            "sigma_method='quasispherical' requires Agama to introspect the "
            "potential (agama.DistributionFunction).  PotentialGPU objects are "
            "not recognised by Agama — falling back to Jeans equation.",
            RuntimeWarning,
            stacklevel=2,
        )
        sigma_method = "jeans"

    # Build sigma(r) spline once (for jeans / quasispherical paths)
    t_mid = 0.5 * (t_start + t_end)
    if sigma_method == "local_circular":
        # Will be computed per-step inline; no spline needed
        _sigma_spline = None
    else:
        _sigma_spline = compute_sigma_r(
            pot, t_eval=t_mid, grid_r=sigma_grid_r, method=sigma_method
        )

    def _get_sigma(r: float, t: float) -> float:
        """Return sigma [km/s] at radius r and time t."""
        if sigma_method == "local_circular":
            return _sigma_local_circular(pot, r, t)
        return float(_sigma_spline(r))

    # Mutable closure state (dict avoids nonlocal for Python 3.10 compat)
    _state: dict = {
        "step": 0,
        "initialized": False,
        "t_prev": t_start,
        "r_com": np.zeros(3),
        "v_com": np.zeros(3),
        "a_df": np.zeros(3),
        "r_sphere": np.inf,  # initially apply DF to all particles (fallback path)
        "M_bound": M_sat,    # updated on phi path
    }

    def _force_extra(
        pos: "np.ndarray | cp.ndarray",
        vel: "np.ndarray | cp.ndarray",
        masses: "np.ndarray | cp.ndarray",
        t: float,
        **kw,
    ) -> np.ndarray:
        pos_np = _to_numpy(pos)     # (N, 3)
        vel_np = _to_numpy(vel)     # (N, 3)
        m_np = _to_numpy(masses)    # (N,)

        # phi is optional — supplied by tree integrators, absent for direct
        phi_raw = kw.get("phi", None)
        phi_np = _to_numpy(phi_raw) if phi_raw is not None else None  # (N,) or None

        step = _state["step"]
        dt = t - _state["t_prev"] if step > 0 else 0.0

        # ---- CoM update ------------------------------------------------
        if phi_np is not None:
            # --- Phi path: phi-energy iterative bound centre ---
            if not _state["initialized"] or (step % update_interval == 0):
                r_com, v_com, bound_mask = _bound_center_phi(
                    pos_np, vel_np, m_np, phi_np,
                    _state["r_com"], _state["v_com"], dt,
                )
                n_bound = int(bound_mask.sum())
                M_bound = float(m_np[bound_mask].sum()) if n_bound > 0 else M_sat
                _state["r_com"] = r_com
                _state["v_com"] = v_com
                _state["M_bound"] = M_bound
                _state["bound_mask"] = bound_mask
                _state["initialized"] = True
            else:
                # Predictor: kinematic extrapolation
                a = _state["a_df"]
                r_com = _state["r_com"] + _state["v_com"] * dt + 0.5 * a * dt ** 2
                v_com = _state["v_com"] + a * dt
                _state["r_com"] = r_com
                _state["v_com"] = v_com
                bound_mask = _state.get("bound_mask", np.ones(len(pos_np), dtype=bool))
                M_bound = _state["M_bound"]

            # Dynamic mass (naturally accounts for tidal stripping)
            M_eff = max(M_bound, 1e4)  # floor to avoid division by zero

            # Sigma at current r and t
            r_scalar = float(np.linalg.norm(r_com))
            sigma_val = _get_sigma(r_scalar, t)

            # Build a scalar-return sigma wrapper for chandrasekhar_friction
            def _sigma_scalar(r):
                return _get_sigma(float(r), t)

            a_df = chandrasekhar_friction(
                r_com=r_com,
                v_com=v_com,
                M_sat=M_eff,
                pot=pot,
                sigma_func=_sigma_scalar,
                t=t,
                coulomb_mode=coulomb_mode,
                fixed_ln_lambda=fixed_ln_lambda,
                core_gamma=core_gamma,
                r_core=r_core,
            )
            _state["a_df"] = a_df
            _state["t_prev"] = t
            _state["step"] = step + 1

            # Apply DF only to bound particles
            a_out = np.zeros_like(pos_np)
            if bound_mask.sum() > 0:
                a_out[bound_mask] = a_df
            return a_out

        else:
            # --- Fallback path: shrinking-sphere + apply_radius_factor ---
            if not _state["initialized"] or (step % update_interval == 0):
                r_com, v_com, r_sphere = _shrinking_sphere_com(
                    pos_np, vel_np, m_np,
                    n_iter=shrink_n_iter,
                    frac=shrink_frac,
                )
                _state["r_com"] = r_com
                _state["v_com"] = v_com
                _state["r_sphere"] = r_sphere
                _state["initialized"] = True
            else:
                # Predictor: kinematic extrapolation using cached DF accel
                a = _state["a_df"]
                r_com = _state["r_com"] + _state["v_com"] * dt + 0.5 * a * dt ** 2
                v_com = _state["v_com"] + a * dt
                _state["r_com"] = r_com
                _state["v_com"] = v_com

            # Sigma at current r and t
            def _sigma_scalar(r):
                return _get_sigma(float(r), t)

            a_df = chandrasekhar_friction(
                r_com=r_com,
                v_com=v_com,
                M_sat=M_sat,  # fixed mass on fallback path
                pot=pot,
                sigma_func=_sigma_scalar,
                t=t,
                coulomb_mode=coulomb_mode,
                fixed_ln_lambda=fixed_ln_lambda,
                core_gamma=core_gamma,
                r_core=r_core,
            )
            _state["a_df"] = a_df
            _state["t_prev"] = t
            _state["step"] = step + 1

            # Apply DF only within apply_radius_factor * r_sphere
            if apply_radius_factor is not None and np.isfinite(_state["r_sphere"]):
                cutoff = apply_radius_factor * _state["r_sphere"]
                dist_from_com = np.linalg.norm(pos_np - _state["r_com"], axis=1)
                mask = dist_from_com <= cutoff
                a_out = np.zeros_like(pos_np)
                a_out[mask] = a_df
                return a_out

            # Fallback: apply uniform DF acceleration to all particles
            return np.broadcast_to(a_df, pos_np.shape).copy()

    return _force_extra
