"""
nbody_streams.sim
Unified high-level simulation entry point for multi-species N-body runs.

Public API
----------
run_simulation : single function with ``architecture`` and ``method`` flags
                 that dispatches to the appropriate backend.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from .species import (
    Species,
    PerformanceWarning,
    _build_particle_arrays,
    _validate_species,
    _split_by_species,
    _emit_performance_warnings,
)
from .run import run_nbody_gpu, run_nbody_cpu, G_DEFAULT
from ._chandrasekhar import make_df_force_extra

try:
    from .tree_gpu.run_gpu_tree import run_nbody_gpu_tree
    _TREE_GPU_OK = True
except ImportError:
    _TREE_GPU_OK = False


def run_simulation(
    phase_space: np.ndarray,
    species: list[Species],
    time_start: float,
    time_end: float,
    dt: float,
    G: float = G_DEFAULT,
    architecture: Literal["cpu", "gpu"] = "gpu",
    method: Literal["direct", "tree"] = "direct",
    external_potential=None,
    dynamical_friction: bool = False,
    output_dir: str = "./output",
    save_snapshots: bool = True,
    snapshots: int = 100,
    num_files_to_write: int = 1,
    restart_interval: int = 1000,
    continue_run: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    debug_energy: bool = False,
    **kwargs,
) -> dict[str, NDArray]:
    """
    Run a direct N-body simulation with one or more particle species.

    This is the primary high-level entry point.  It validates the species
    list, builds combined per-particle mass / softening arrays, emits
    performance warnings when particle counts exceed recommended thresholds,
    dispatches to the appropriate backend, and returns the final phase-space
    coordinates split back into a per-species dictionary.

    Parameters
    ----------
    phase_space : ndarray, shape (N_total, 6)
        Initial conditions ``[x, y, z, vx, vy, vz]`` for **all** particles
        concatenated in the same order as *species*.
    species : list[Species]
        Ordered list of species definitions.  The total particle count must
        equal ``phase_space.shape[0]``.
    time_start, time_end : float
        Integration time interval.
    dt : float
        Fixed timestep.
    G : float, optional
        Gravitational constant.  Default: kpc / (km/s) / Msun units.
    architecture : {'cpu', 'gpu'}, optional
        Compute backend.  Default: ``'gpu'``.
    method : {'direct', 'tree'}, optional
        Gravity solver algorithm.

        * ``'direct'`` - O(N^2) pairwise summation (GPU or CPU).
        * ``'tree'`` - hierarchical tree algorithm.  CPU: pyfalcon falcON O(N);
          GPU: Barnes-Hut tree code (requires ``libtreeGPU.so`` built from
          ``nbody_streams/tree_gpu/``).

        Default: ``'direct'``.
    external_potential : agama.Potential or PotentialGPU or None, optional
        Time-varying external potential.  Two backends are supported:

        * **agama.Potential** — standard Agama C++ potential; evaluated on
          CPU with host↔device position transfers on the GPU paths.
        * **PotentialGPU** (``nbody_streams.agama_helper.PotentialGPU``) —
          GPU-native potential; force is evaluated directly on the GPU array
          with no host transfer.  Requires CuPy + nvcc.

        Default: ``None``.
    dynamical_friction : bool, optional
        Apply Chandrasekhar dynamical friction to the satellite CoM motion.
        Requires ``external_potential`` to be set.  When *external_potential*
        is a ``PotentialGPU``, the DF computation remains CPU-side (sigma(r),
        bound-centre iteration, Chandrasekhar formula); only the density/force
        lookups delegate to the GPU.  ``sigma_method='quasispherical'`` is
        not available for ``PotentialGPU`` and silently falls back to Jeans.
        Default: ``False``.

        .. note:: Not yet implemented.  Will raise ``NotImplementedError``
            if set to ``True``.  Tracked for a future release.

    output_dir : str, optional
        Directory for snapshot and restart files.  Default: ``'./output'``.
    save_snapshots : bool, optional
        Write HDF5 snapshots to disk.  Default: ``True``.
    snapshots : int, optional
        Number of evenly-spaced snapshots.  Default: 100.
    num_files_to_write : int, optional
        Split output across this many HDF5 files.  Default: 1.
    restart_interval : int, optional
        Save a restart checkpoint every this many steps.  Default: 1000.
    continue_run : bool, optional
        Resume from an existing restart file.  Default: ``False``.
    verbose : bool, optional
        Print progress information.  Default: ``True``.
    debug_energy : bool, optional
        Print virial ratio Q=KE/|PE| and relative energy drift dE/E alongside
        each progress line.  Available for all backends:

        * GPU tree / CPU tree -- potential returned for free; zero extra cost.
        * GPU direct / CPU direct -- one extra O(N^2) potential pass per
          output interval; meaningful overhead for large N.

        Default: ``False``.
    **kwargs
        Backend-specific advanced options.  Commonly used:

        * ``theta`` (float, default 0.6) -- Tree opening angle.  Only used
          for ``method='tree'``.
        * ``nthreads`` (int or None, default None) -- CPU thread count for
          direct summation.  ``None`` = auto.
        * ``external_update_interval`` (int, default 1) -- Recompute external
          forces every this many steps.  GPU only.
        * ``precision`` (str, default ``'float32_kahan'``) -- Floating-point
          precision for GPU direct force computation.  One of
          ``'float32'``, ``'float64'``, ``'float32_kahan'``.  Ignored for
          tree backends (which are always float32 internally).

        Dynamical-friction options (only used when ``dynamical_friction=True``):

        * ``df_M_sat`` (float) -- Satellite mass [M_sun].  Default: total
          mass of all species.  On tree paths, used as the initial value; the
          effective mass is updated dynamically to the bound-particle mass.
        * ``df_coulomb_mode`` (str, default ``'variable'``) -- Coulomb
          logarithm mode: ``'variable'`` (``ln(r v^2 / G M_sat)``) or
          ``'fixed'``.
        * ``df_fixed_ln_lambda`` (float, default 3.0) -- Fixed ln(Λ) when
          ``df_coulomb_mode='fixed'``.
        * ``df_core_gamma`` (float, default 0.0) -- Core-stalling suppression
          index (0 = off; ~1–2 for constant-density cores).
        * ``df_r_core`` (float, default 1.0) -- Core radius [kpc].
        * ``df_update_interval`` (int, default 10) -- Correct CoM every N
          steps.
        * ``df_sigma_method`` (str, default ``'jeans'``) -- Velocity
          dispersion algorithm: ``'jeans'`` (Jeans equation, recommended for
          time-evolving potentials), ``'local_circular'`` (``sqrt(r|g_r|/2)``,
          cheap and time-evolving), or ``'quasispherical'`` (Agama DF moments,
          best for spherical static potentials).
        * ``df_apply_radius_factor`` (float or None, default 2.0) -- Fallback-
          path (direct integrators) only: apply DF within this many core
          radii.  On tree paths, the phi-energy criterion is used instead.
        * ``df_shrink_n_iter`` (int, default 5) -- Shrinking-sphere iterations
          (fallback path).
        * ``df_shrink_frac`` (float, default 0.5) -- Shrinking-sphere radius
          reduction (fallback path).
        * ``df_sigma_grid_r`` (ndarray or None) -- Custom radial grid for
          Jeans/quasispherical sigma(r).

        GPU tree only (``architecture='gpu', method='tree'``):

        * ``nleaf`` (int, default 64) -- Minimum leaf node size.
        * ``ncrit`` (int, default 64) -- Group criticality threshold.
        * ``level_split`` (int, default 5) -- Level at which tree splits groups.
        * ``step_timeout_s`` (float, default 60.0) -- Per-step watchdog timeout
          in seconds; raises ``RuntimeError`` if a step exceeds this.

    Returns
    -------
    dict[str, ndarray]
        Final phase-space coordinates, keyed by species name.
        Each value has shape ``(N_k, 6)``.

    Raises
    ------
    ValueError
        If *species* list is inconsistent with *phase_space* shape.
    FileExistsError
        If snapshot files already exist in *output_dir* and *overwrite* is
        ``False`` and *continue_run* is ``False``.
    ImportError
        If ``architecture='gpu'``, ``method='tree'``, and ``libtreeGPU.so``
        has not been built.  If CuPy is unavailable and
        ``architecture='gpu'``.

    Examples
    --------
    Single-species dark-matter simulation:

    >>> from nbody_streams import run_simulation, Species, make_plummer_sphere
    >>> xv, masses = make_plummer_sphere(1000, M_total=1e9, a=1.0)
    >>> dm = Species.dark(N=1000, mass=1e6, softening=0.1)
    >>> result = run_simulation(xv, [dm], 0.0, 1.0, 1e-3,
    ...                         architecture='cpu', method='direct',
    ...                         save_snapshots=False, verbose=False)
    >>> result['dark'].shape
    (1000, 6)

    Two-species dark-matter + stars:

    >>> xv_dm, _ = make_plummer_sphere(800, M_total=1e9, a=2.0)
    >>> xv_st, _ = make_plummer_sphere(200, M_total=1e7, a=0.5)
    >>> xv_all = np.vstack([xv_dm, xv_st])
    >>> dm   = Species.dark(N=800, mass=1.25e6, softening=0.2)
    >>> star = Species.star(N=200, mass=5e4,   softening=0.05)
    >>> result = run_simulation(xv_all, [dm, star], 0.0, 0.5, 1e-3,
    ...                         architecture='cpu', method='direct',
    ...                         save_snapshots=False, verbose=False)
    >>> result.keys()
    dict_keys(['dark', 'star'])

    Using the GPU tree backend with a non-default opening angle:

    >>> result = run_simulation(xv, [dm], 0.0, 1.0, 1e-3,
    ...                         architecture='gpu', method='tree',
    ...                         save_snapshots=False, verbose=False,
    ...                         theta=0.5)

    Notes
    -----
    **Softening kernels by backend** (hardcoded; not user-configurable via
    ``run_simulation``):

    * GPU direct  -- ``'spline'`` kernel
    * CPU direct  -- ``'spline'`` kernel
    * CPU tree (pyfalcon/falcON) -- ``dehnen_k1`` (integer 1)
    * GPU tree (Barnes-Hut) -- Plummer softening hardcoded in C++

    **Low-level API for full control**

    ``run_simulation`` is the recommended entry point for most use cases.
    Experienced users who need finer control -- custom softening kernels,
    alternative floating-point precision, per-particle softening, or
    ``external_update_interval`` on the CPU -- can call the backend
    functions directly:

    * :func:`run_nbody_gpu` -- GPU direct-sum integrator
    * :func:`run_nbody_cpu` -- CPU direct / falcON tree integrator
    * :func:`~nbody_streams.tree_gpu.run_gpu_tree.run_nbody_gpu_tree` -- GPU Barnes-Hut tree

    **Performance guidance** (warnings are also emitted automatically):

    * CPU direct  > 20 000 particles -> very slow (O(N^2)).
    * GPU direct  > 500 000 particles -> slow at this scale.
    * Any method  > 2 000 000 particles -> use GPU+Tree (``architecture='gpu', method='tree'``).
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if architecture not in ("cpu", "gpu"):
        raise ValueError(
            f"architecture must be 'cpu' or 'gpu', got '{architecture}'"
        )
    if method not in ("direct", "tree"):
        raise ValueError(f"method must be 'direct' or 'tree', got '{method}'")
    if architecture == "gpu" and method == "tree" and not _TREE_GPU_OK:
        raise ImportError(
            "GPU tree-code requires libtreeGPU.so to be built first:\n\n"
            "    cd nbody_streams/tree_gpu && make -j$(nproc)\n\n"
            "Then re-import nbody_streams."
        )

    phase_space = np.asarray(phase_space, dtype=np.float64)
    if phase_space.ndim != 2 or phase_space.shape[1] != 6:
        raise ValueError(
            f"phase_space must be shape (N, 6), got {phase_space.shape}"
        )

    _validate_species(phase_space, species)

    if dynamical_friction and external_potential is None:
        raise ValueError(
            "dynamical_friction=True requires external_potential to be set. "
            "The Chandrasekhar DF formula needs host density and sigma(r) from "
            "the external potential."
        )

    N_total = phase_space.shape[0]

    # ------------------------------------------------------------------
    # Build combined arrays
    # ------------------------------------------------------------------
    mass_arr, softening_arr = _build_particle_arrays(species)

    # ------------------------------------------------------------------
    # Performance warnings
    # ------------------------------------------------------------------
    _emit_performance_warnings(N_total, architecture, method)

    # Warn if satellite is massive enough that DF should be enabled
    if external_potential is not None and not dynamical_friction:
        M_total_sat = float(mass_arr.sum())
        if M_total_sat > 1e10:
            import warnings as _warnings
            _warnings.warn(
                f"Total satellite mass is {M_total_sat:.2e} M_sun and an external "
                f"potential is set, but dynamical_friction=False.  At this mass "
                f"scale the host-satellite dynamical friction timescale is short "
                f"(<~1 Gyr).  Consider setting dynamical_friction=True — note this "
                f"will add a small CPU-side overhead (~1%%) per step.",
                PerformanceWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Pull advanced options from kwargs with backend-appropriate defaults.
    # These span multiple backends so we handle them explicitly.
    # ------------------------------------------------------------------
    theta = kwargs.pop("theta", 0.6)
    nthreads = kwargs.pop("nthreads", None)
    external_update_interval = kwargs.pop("external_update_interval", 1)
    precision = kwargs.pop("precision", "float32_kahan")

    # ------------------------------------------------------------------
    # Dynamical friction: build force_extra closure if requested.
    # ALL df_* kwargs are consumed unconditionally so they never leak to
    # backends regardless of whether dynamical_friction is True or False.
    # ------------------------------------------------------------------
    _force_extra = kwargs.pop("force_extra", None)
    df_M_sat = kwargs.pop("df_M_sat", float(mass_arr.sum()))
    df_coulomb_mode = kwargs.pop("df_coulomb_mode", "variable")
    df_fixed_ln_lambda = kwargs.pop("df_fixed_ln_lambda", 3.0)
    df_core_gamma = kwargs.pop("df_core_gamma", 0.0)
    df_r_core = kwargs.pop("df_r_core", 1.0)
    df_update_interval = kwargs.pop("df_update_interval", 10)
    df_shrink_n_iter = kwargs.pop("df_shrink_n_iter", 5)
    df_shrink_frac = kwargs.pop("df_shrink_frac", 0.5)
    df_sigma_grid_r = kwargs.pop("df_sigma_grid_r", None)
    df_apply_radius_factor = kwargs.pop("df_apply_radius_factor", 2.0)
    df_sigma_method = kwargs.pop("df_sigma_method", "jeans")
    if dynamical_friction:
        _df_closure = make_df_force_extra(
            pot=external_potential,
            M_sat=df_M_sat,
            t_start=time_start,
            t_end=time_end,
            coulomb_mode=df_coulomb_mode,
            fixed_ln_lambda=df_fixed_ln_lambda,
            core_gamma=df_core_gamma,
            r_core=df_r_core,
            update_interval=df_update_interval,
            shrink_n_iter=df_shrink_n_iter,
            shrink_frac=df_shrink_frac,
            sigma_grid_r=df_sigma_grid_r,
            apply_radius_factor=df_apply_radius_factor,
            sigma_method=df_sigma_method,
        )
        if _force_extra is not None:
            # Compose: existing force_extra + DF.
            # Forward **kw (including phi= from tree integrators) to both closures.
            _existing = _force_extra

            def _force_extra(pos, vel, masses, t, **kw):
                return _existing(pos, vel, masses, t, **kw) + _df_closure(pos, vel, masses, t, **kw)
        else:
            _force_extra = _df_closure

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if architecture == "gpu" and method == "tree":
        # Remaining kwargs are forwarded to run_nbody_gpu_tree, which
        # accepts extra tree-tuning params: nleaf, ncrit, level_split,
        # step_timeout_s.  An unrecognised key will raise TypeError there.
        final_xv = run_nbody_gpu_tree(
            phase_space=phase_space,
            masses=mass_arr,
            time_start=time_start,
            time_end=time_end,
            dt=dt,
            softening=softening_arr,
            G=G,
            theta=theta,
            external_potential=external_potential,
            external_update_interval=external_update_interval,
            force_extra=_force_extra,
            output_dir=output_dir,
            save_snapshots=save_snapshots,
            snapshots=snapshots,
            num_files_to_write=num_files_to_write,
            restart_interval=restart_interval,
            continue_run=continue_run,
            overwrite=overwrite,
            verbose=verbose,
            debug_energy=debug_energy,
            species=species,
            **kwargs,
        )
    elif architecture == "gpu":  # direct
        if kwargs:
            raise TypeError(
                f"run_simulation() got unexpected keyword argument(s) for "
                f"architecture='gpu', method='direct': {sorted(kwargs)}"
            )
        final_xv = run_nbody_gpu(
            phase_space=phase_space,
            masses=mass_arr,
            time_start=time_start,
            time_end=time_end,
            dt=dt,
            softening=softening_arr,
            G=G,
            precision=precision,
            kernel="spline",
            external_potential=external_potential,
            external_update_interval=external_update_interval,
            force_extra=_force_extra,
            output_dir=output_dir,
            save_snapshots=save_snapshots,
            snapshots=snapshots,
            num_files_to_write=num_files_to_write,
            restart_interval=restart_interval,
            continue_run=continue_run,
            overwrite=overwrite,
            verbose=verbose,
            debug_energy=debug_energy,
            species=species,
        )
    else:  # architecture == "cpu"
        if kwargs:
            raise TypeError(
                f"run_simulation() got unexpected keyword argument(s) for "
                f"architecture='cpu': {sorted(kwargs)}"
            )
        # CPU tree uses integer kernel 1 (dehnen_k1); direct uses 'spline'
        cpu_kernel = 1 if method == "tree" else "spline"
        final_xv = run_nbody_cpu(
            phase_space=phase_space,
            masses=mass_arr,
            time_start=time_start,
            time_end=time_end,
            dt=dt,
            softening=softening_arr,
            G=G,
            method=method,
            theta=theta,
            kernel=cpu_kernel,
            nthreads=nthreads,
            external_potential=external_potential,
            force_extra=_force_extra,
            output_dir=output_dir,
            save_snapshots=save_snapshots,
            snapshots=snapshots,
            num_files_to_write=num_files_to_write,
            restart_interval=restart_interval,
            continue_run=continue_run,
            debug_energy=debug_energy,
            overwrite=overwrite,
            verbose=verbose,
            species=species,
        )

    # ------------------------------------------------------------------
    # Split output by species
    # ------------------------------------------------------------------
    return _split_by_species(final_xv, species)
