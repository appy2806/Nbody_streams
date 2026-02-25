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


def run_simulation(
    phase_space: np.ndarray,
    species: list[Species],
    time_start: float,
    time_end: float,
    dt: float,
    G: float = G_DEFAULT,
    architecture: Literal["cpu", "gpu"] = "gpu",
    method: Literal["direct", "tree"] = "direct",
    precision: Literal["float32", "float64", "float32_kahan"] = "float32_kahan",
    kernel: str = "dehnen_k1",
    theta: float = 0.6,
    nthreads: int | None = None,
    external_potential=None,
    external_update_interval: int = 1,
    output_dir: str = "./output",
    save_snapshots: bool = True,
    snapshots: int = 100,
    num_files_to_write: int = 1,
    restart_interval: int = 1000,
    continue_run: bool = False,
    verbose: bool = True,
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

        * ``'direct'`` — O(N²) pairwise summation (GPU or CPU).
        * ``'tree'`` — hierarchical tree algorithm (CPU only via pyfalcon;
          GPU tree is *not yet implemented*).

        Default: ``'direct'``.
    precision : {'float32', 'float64', 'float32_kahan'}, optional
        Floating-point precision for GPU force computation.  Ignored for CPU.
        Default: ``'float32_kahan'``.
    kernel : str, optional
        Softening kernel.  For CPU direct: one of ``'newtonian'``,
        ``'plummer'``, ``'dehnen_k1'``, ``'dehnen_k2'``, ``'spline'``.
        For CPU tree: integer kernel index (passed through to pyfalcon).
        For GPU: same string options as CPU direct.
        Default: ``'dehnen_k1'``.
    theta : float, optional
        Tree opening angle.  Only used for ``method='tree'``.  Default: 0.6.
    nthreads : int or None, optional
        CPU thread count for direct summation.  ``None`` = auto.
    external_potential : agama.Potential or None, optional
        Time-varying external potential (requires Agama).  Default: ``None``.
    external_update_interval : int, optional
        Recompute external forces every this many steps.  GPU only.
        Default: 1.
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

    Returns
    -------
    dict[str, ndarray]
        Final phase-space coordinates, keyed by species name.
        Each value has shape ``(N_k, 6)``.

    Raises
    ------
    ValueError
        If *species* list is inconsistent with *phase_space* shape.
    NotImplementedError
        If ``architecture='gpu'`` and ``method='tree'``.
    ImportError
        If CuPy is unavailable and ``architecture='gpu'``.

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

    Notes
    -----
    **Performance guidance** (warnings are also emitted automatically):

    * CPU direct  > 20 000 particles → very slow (O N²).
    * GPU direct  > 500 000 particles → slow at this scale.
    * Any method  > 2 000 000 particles → requires GPU+Tree (coming soon).
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
    if architecture == "gpu" and method == "tree":
        raise NotImplementedError(
            "GPU tree-code (architecture='gpu', method='tree') is not yet "
            "implemented.  Use architecture='cpu', method='tree' or "
            "architecture='gpu', method='direct' instead."
        )

    phase_space = np.asarray(phase_space, dtype=np.float64)
    if phase_space.ndim != 2 or phase_space.shape[1] != 6:
        raise ValueError(
            f"phase_space must be shape (N, 6), got {phase_space.shape}"
        )

    _validate_species(phase_space, species)

    N_total = phase_space.shape[0]

    # ------------------------------------------------------------------
    # Build combined arrays
    # ------------------------------------------------------------------
    mass_arr, softening_arr = _build_particle_arrays(species)

    # ------------------------------------------------------------------
    # Performance warnings
    # ------------------------------------------------------------------
    _emit_performance_warnings(N_total, architecture, method)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if architecture == "gpu":
        final_xv = run_nbody_gpu(
            phase_space=phase_space,
            masses=mass_arr,
            time_start=time_start,
            time_end=time_end,
            dt=dt,
            softening=softening_arr,
            G=G,
            precision=precision,
            kernel=kernel,
            external_potential=external_potential,
            external_update_interval=external_update_interval,
            output_dir=output_dir,
            save_snapshots=save_snapshots,
            snapshots=snapshots,
            num_files_to_write=num_files_to_write,
            restart_interval=restart_interval,
            continue_run=continue_run,
            verbose=verbose,
            species=species,
        )
    else:  # architecture == "cpu"
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
            kernel=kernel,
            nthreads=nthreads,
            external_potential=external_potential,
            output_dir=output_dir,
            save_snapshots=save_snapshots,
            snapshots=snapshots,
            num_files_to_write=num_files_to_write,
            restart_interval=restart_interval,
            continue_run=continue_run,
            verbose=verbose,
            species=species,
        )

    # ------------------------------------------------------------------
    # Split output by species
    # ------------------------------------------------------------------
    return _split_by_species(final_xv, species)
