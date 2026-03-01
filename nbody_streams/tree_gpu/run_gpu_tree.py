"""
nbody_streams.tree_gpu.run_gpu_tree
====================================
KDK leapfrog integrator using the GPU Barnes-Hut tree code.

Public API
----------
run_nbody_gpu_tree
    Matches the call signature of ``run_nbody_gpu`` (direct-sum) but routes
    all force evaluations through ``tree_gravity_gpu``.  Positions and
    velocities are kept in float32 on the GPU (consistent with the float32
    tree-code internals).

_StepWatchdog
    Background-thread watchdog that fires a KeyboardInterrupt in the main
    thread if a single integration step exceeds a timeout.  Used internally
    by ``run_nbody_gpu_tree``; exposed for advanced users who run the tree
    loop manually.
"""

from __future__ import annotations

import ctypes as _ct
import datetime
import threading
import time as pytime
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False

try:
    import agama as _agama
    _AGAMA_OK = True
except ImportError:
    _AGAMA_OK = False

try:
    from ..nbody_io import _save_snapshot, _save_restart, _load_restart
except ImportError:
    try:
        from nbody_streams.nbody_io import _save_snapshot, _save_restart, _load_restart
    except ImportError:
        _save_snapshot = _save_restart = _load_restart = None  # type: ignore[assignment]

try:
    from ..species import Species, _build_particle_arrays, _split_by_species
except ImportError:
    from nbody_streams.species import Species, _build_particle_arrays, _split_by_species  # type: ignore[no-redef]

from ._force import tree_gravity_gpu, TreeGPU, G_DEFAULT

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class _StepWatchdog:
    """
    Background-thread watchdog for GPU leapfrog loops.

    Per-step overhead is ~1 μs (two lock acquires + monotonic timestamp).
    If the deadline is exceeded the watchdog fires ``KeyboardInterrupt`` in
    the main thread — this works even when the main thread is blocked inside
    a C extension (``cudaDeviceSynchronize``).

    Use as a context manager around each integration step::

        watchdog = _StepWatchdog(timeout_s=60.0)
        for step in range(n_steps):
            with watchdog:
                acc, phi = tree_gravity_gpu(...)
                vel += 0.5 * dt * acc
                pos += dt * vel
        watchdog.close()
    """

    def __init__(self, timeout_s: float = 60.0):
        self._timeout  = timeout_s
        self._deadline = None
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._main_tid = threading.main_thread().ident
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="step-watchdog")
        self._thread.start()

    def _run(self):
        while not self._stop.wait(timeout=0.25):
            with self._lock:
                dl = self._deadline
            if dl is not None and pytime.monotonic() > dl:
                _ct.pythonapi.PyThreadState_SetAsyncExc(
                    _ct.c_ulong(self._main_tid),
                    _ct.py_object(KeyboardInterrupt),
                )
                with self._lock:
                    self._deadline = None   # fire only once

    def __enter__(self):
        with self._lock:
            self._deadline = pytime.monotonic() + self._timeout
        return self

    def __exit__(self, *_):
        with self._lock:
            self._deadline = None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Main integrator
# ---------------------------------------------------------------------------

def run_nbody_gpu_tree(
    phase_space: np.ndarray,
    masses: np.ndarray,
    time_start: float,
    time_end: float,
    dt: float,
    softening: float | np.ndarray,
    G: float = G_DEFAULT,
    # Tree parameters
    theta: float = 0.6,
    nleaf: int = 64,
    ncrit: int = 64,
    level_split: int = 5,
    step_timeout_s: float = 60.0,
    # External potential
    external_potential=None,
    external_update_interval: int = 1,
    # I/O
    output_dir: str = "./output",
    save_snapshots: bool = True,
    snapshots: int = 100,
    num_files_to_write: int = 1,
    restart_interval: int = 1000,
    continue_run: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    # Multi-species (preferred path when using run_simulation)
    species: list[Species] | None = None,
) -> np.ndarray:
    """
    Run a GPU-tree N-body simulation with KDK leapfrog integration.

    Force evaluations use the GPU Barnes-Hut tree code (O(N log N)).
    Positions and velocities are kept in float32 on the GPU.  Energy
    diagnostics are computed in float64.

    Parameters
    ----------
    phase_space : ndarray, shape (N, 6)
        Initial conditions ``[x, y, z, vx, vy, vz]``.
    masses : ndarray, shape (N,)
        Per-particle masses.
    time_start, time_end : float
        Integration interval.
    dt : float
        Fixed timestep.
    softening : float or ndarray, shape (N,)
        Plummer softening length(s).  Scalar → uniform; array → per-particle.
    G : float, optional
        Gravitational constant.  Default: kpc / (km/s)² / Msun.
    theta : float, optional
        Barnes-Hut opening angle.  0.75 (fast) – 0.5 (accurate).  Default 0.6.
    nleaf : int, optional
        Max particles per leaf cell.  Must be in {16, 24, 32, 48, 64}.
    ncrit : int, optional
        Walk group size.  Default 64.
    level_split : int, optional
        Tree level for spatial grouping.  Default 5.
    step_timeout_s : float, optional
        Seconds before the watchdog fires a KeyboardInterrupt for a hanging
        CUDA kernel.  Default 60.
    external_potential : agama.Potential, optional
        External time-varying potential (Agama).  Evaluated once per timestep
        (or every ``external_update_interval`` steps) and added to tree forces.
    external_update_interval : int, optional
        Evaluate external potential every N steps.  Default 1.
    output_dir : str, optional
        Snapshot output directory.  Default ``'./output'``.
    save_snapshots : bool, optional
        Whether to write HDF5 snapshots.  Default True.
    snapshots : int, optional
        Number of snapshots to write.  Default 100.
    num_files_to_write : int, optional
        Distribute snapshots across this many HDF5 files.  Default 1.
    restart_interval : int, optional
        Save a restart checkpoint every N steps.  Default 1000.
    continue_run : bool, optional
        Resume from an existing restart file.  Default False.
    overwrite : bool, optional
        Delete existing snapshot files before starting.  Default False.
    verbose : bool, optional
        Print energy / ETA diagnostics.  Default True.
    species : list[Species], optional
        Multi-species descriptor list.  When provided, ``masses`` and
        ``softening`` are built from it internally (they are ignored).

    Returns
    -------
    ndarray, shape (N, 6)
        Final phase-space coordinates in float64.

    Raises
    ------
    ImportError
        If CuPy is not installed.
    FileExistsError
        If snapshot files already exist and ``overwrite=False``.
    NotImplementedError
        If an external potential is requested but Agama is not installed.
    """
    if not _CUPY_OK:
        raise ImportError("CuPy is required for run_nbody_gpu_tree.")

    # ── Build mass / softening from species if provided ───────────────────────
    if species is not None:
        masses, softening = _build_particle_arrays(species)

    masses    = np.asarray(masses,    dtype=np.float64)
    softening = np.asarray(softening, dtype=np.float64) if hasattr(softening, '__len__') else float(softening)

    N         = phase_space.shape[0]
    n_steps   = max(1, round((time_end - time_start) / dt))
    snap_every = max(1, n_steps // snapshots) if save_snapshots and snapshots > 0 else 0
    e_every   = max(1, n_steps // 50)   # energy output ~50 times during run

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Overwrite / continue guard ────────────────────────────────────────────
    if save_snapshots and not continue_run:
        existing = sorted(output_path.glob("snapshot*.h5"))
        if existing:
            if overwrite:
                for f in existing:
                    f.unlink()
                if verbose:
                    print(f"Removed {len(existing)} existing snapshot file(s) in '{output_dir}'.")
            else:
                raise FileExistsError(
                    f"Output directory '{output_dir}' already contains snapshot files: "
                    f"{[f.name for f in existing]}. "
                    "Pass overwrite=True to delete them, or continue_run=True to resume."
                )

    # ── External potential ────────────────────────────────────────────────────
    if external_potential is not None and not _AGAMA_OK:
        raise NotImplementedError(
            "external_potential requires Agama. Install it with: "
            "pip install agama @ git+https://github.com/GalacticDynamics-Oxford/Agama.git"
        )

    # ── GPU arrays — float32 (tree code requires float32) ─────────────────────
    mass_f32 = cp.asarray(masses,    dtype=cp.float32)
    mass_f64 = cp.asarray(masses,    dtype=cp.float64)

    if np.isscalar(softening):
        eps_gpu  = cp.full(N, float(softening), dtype=cp.float32)
    else:
        eps_gpu  = cp.asarray(softening, dtype=cp.float32)

    pos = cp.asarray(phase_space[:, :3], dtype=cp.float32)
    vel = cp.asarray(phase_space[:, 3:], dtype=cp.float32)

    # ── Resume from restart ───────────────────────────────────────────────────
    start_step = 0
    snap_idx   = 0
    if continue_run and _load_restart is not None:
        restart_data = _load_restart(output_dir)
        if restart_data is not None:
            (pv_np, t_rest, step_rest, snapctr_rest,
             _m_r, _s_r, _snames_r, _sN_r) = restart_data
            pos        = cp.asarray(pv_np[:, :3], dtype=cp.float32)
            vel        = cp.asarray(pv_np[:, 3:], dtype=cp.float32)
            start_step = int(step_rest) + 1
            snap_idx   = int(snapctr_rest)
            if verbose:
                print(f"[RESTART] Resumed from step {step_rest}  snap_idx={snap_idx}")

    # ── Pre-allocate tree handle ───────────────────────────────────────────────
    tree     = TreeGPU(N, eps=0.0, theta=theta, verbose=False)
    watchdog = _StepWatchdog(timeout_s=step_timeout_s)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _energy(vel_: cp.ndarray, phi_: cp.ndarray) -> tuple[float, float]:
        v2 = cp.sum(vel_.astype(cp.float64) ** 2, axis=1)
        KE = 0.5 * float(cp.sum(mass_f64 * v2))
        PE = 0.5 * float(cp.sum(mass_f64 * phi_.astype(cp.float64)))
        return KE, PE

    def _ext_acc(pos_: cp.ndarray, t_: float) -> cp.ndarray:
        """Evaluate external potential and return acceleration (float32, GPU)."""
        if external_potential is None:
            return None
        xyz_cpu = cp.asnumpy(pos_.astype(cp.float64))
        acc_ext_cpu = external_potential.force(xyz_cpu).astype(np.float32)
        return cp.asarray(acc_ext_cpu, dtype=cp.float32)

    def _save_snap_now(step: int, t_: float):
        if not save_snapshots or _save_snapshot is None:
            return
        pv_np = cp.asnumpy(cp.concatenate([pos, vel], axis=1)).astype(np.float64)
        _save_snapshot(
            pv_np, snap_idx, t_, output_dir,
            species=species if species is not None else None,
            time_step=dt,
            num_files_to_write=num_files_to_write if num_files_to_write > 1 else None,
            total_expected_snapshots=snapshots,
        )

    def _save_restart_now(step: int, t_: float):
        if _save_restart is None:
            return
        pv_np  = cp.asnumpy(cp.concatenate([pos, vel], axis=1)).astype(np.float64)
        m_np   = cp.asnumpy(mass_f32).astype(np.float64)
        s_np   = cp.asnumpy(eps_gpu).astype(np.float64)
        snames = [s.name for s in species] if species else None
        sN     = [s.N    for s in species] if species else None
        _save_restart(pv_np, float(t_), step, output_dir, snap_idx,
                      mass_arr=m_np, softening_arr=s_np,
                      species_names=snames, species_N=sN)

    def _nan_gate(step: int, label: str) -> bool:
        """Return True (abort) if NaN detected in current acc."""
        if not cp.any(cp.isnan(acc[:, 0])):
            return False
        print(f"\n[ERROR] NaN in forces at step {step} ({label}). Aborting.")
        return True

    # ── Initial forces ─────────────────────────────────────────────────────────
    if verbose:
        print(f"run_nbody_gpu_tree: N={N:,}  steps={n_steps}  dt={dt}  theta={theta}")
        print(f"  Output: {output_dir}")

    t_now = time_start + start_step * dt

    with watchdog:
        acc, phi = tree_gravity_gpu(
            pos, mass_f32, eps_gpu, G=G,
            theta=theta, nleaf=nleaf, ncrit=ncrit, level_split=level_split,
            tree=tree,
        )

    # Add external forces at t=0
    _acc_ext = _ext_acc(pos, t_now)
    if _acc_ext is not None:
        acc = acc + _acc_ext

    # Initial snapshot
    if start_step == 0 and save_snapshots:
        _save_snap_now(0, t_now)
        snap_idx += 1

    # ── Energy storage ─────────────────────────────────────────────────────────
    n_e     = (n_steps - start_step) // e_every + 2
    t_arr   = np.zeros(n_e)
    KE_arr  = np.zeros(n_e)
    PE_arr  = np.zeros(n_e)
    E_arr   = np.zeros(n_e)
    KE0, PE0 = _energy(vel, phi)
    t_arr[0] = t_now
    KE_arr[0] = KE0;  PE_arr[0] = PE0;  E_arr[0] = KE0 + PE0
    E_ref    = E_arr[0]
    e_idx    = 1

    # ── KDK loop ───────────────────────────────────────────────────────────────
    t_wall0 = pytime.perf_counter()
    aborted = False

    try:
        for step in range(start_step, n_steps):
            t_now = time_start + (step + 1) * dt

            with watchdog:
                vel  = vel  + (0.5 * dt) * acc
                pos  = pos  + dt         * vel
                acc, phi = tree_gravity_gpu(
                    pos, mass_f32, eps_gpu, G=G,
                    theta=theta, nleaf=nleaf, ncrit=ncrit, level_split=level_split,
                    tree=tree,
                )

            # External potential (evaluated every external_update_interval steps)
            if external_potential is not None and (step + 1) % external_update_interval == 0:
                _acc_ext = _ext_acc(pos, t_now)
            if _acc_ext is not None:
                acc = acc + _acc_ext

            with watchdog:
                vel  = vel  + (0.5 * dt) * acc

            # ── Energy diagnostics ────────────────────────────────────────────
            if (step + 1) % e_every == 0 and e_idx < n_e:
                KE, PE = _energy(vel, phi)
                dE = (KE + PE - E_ref) / abs(E_ref) if E_ref != 0 else 0.0
                t_arr[e_idx]  = t_now
                KE_arr[e_idx] = KE;  PE_arr[e_idx] = PE;  E_arr[e_idx] = KE + PE

                if verbose:
                    elapsed    = pytime.perf_counter() - t_wall0
                    done_steps = step + 1 - start_step
                    left_steps = n_steps - step - 1
                    eta_s      = elapsed / max(done_steps, 1) * left_steps
                    eta_str    = str(datetime.timedelta(seconds=int(eta_s)))
                    eta_wall   = datetime.datetime.now() + datetime.timedelta(seconds=eta_s)
                    print(f"  step {step+1:6d}/{n_steps}  t={t_now:.4f}  "
                          f"Q={KE/abs(PE):.3f}  dE/E={dE:+.2e}  ETA {eta_str} ({eta_wall:%H:%M})",
                          flush=True)
                e_idx += 1

            # ── Snapshot ──────────────────────────────────────────────────────
            if snap_every > 0 and (step + 1) % snap_every == 0:
                if _nan_gate(step, "snap"):
                    aborted = True;  break
                _save_snap_now(step, t_now)
                snap_idx += 1

            # ── Restart checkpoint ────────────────────────────────────────────
            if (step + 1) % restart_interval == 0:
                if _nan_gate(step, "restart"):
                    aborted = True;  break
                _save_restart_now(step, t_now)

    except KeyboardInterrupt:
        print(f"\n[WATCHDOG] Step exceeded {step_timeout_s:.0f}s at step {step}. "
              "GPU kernel likely deadlocked. Saving restart and aborting.")
        _save_restart_now(step, time_start + step * dt)
        aborted = True

    # ── Final restart + summary ────────────────────────────────────────────────
    if not aborted:
        _save_restart_now(n_steps - 1, time_end)

    watchdog.close()
    tree.close()

    elapsed_total = pytime.perf_counter() - t_wall0
    status = "ABORTED" if aborted else "Done"
    if verbose:
        steps_done = (step + 1 - start_step) if not aborted else max(step - start_step, 1)
        ms_per_step = elapsed_total / max(steps_done, 1) * 1e3
        print(f"\n{status}. Wall time: {elapsed_total:.1f} s  ({ms_per_step:.1f} ms/step)")

    # Return float64 phase-space (consistent with run_nbody_gpu)
    pv = cp.concatenate([pos, vel], axis=1)
    return cp.asnumpy(pv).astype(np.float64)
