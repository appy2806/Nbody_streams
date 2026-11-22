"""
nbody_streams.tree_gpu.run_gpu_tree
====================================
KDK leapfrog integrator using the GPU Barnes-Hut tree code.

Public API
----------
run_nbody_gpu_tree
    Matches the call signature of ``run_nbody_gpu`` (direct-sum) but routes
    all force evaluations through ``tree_gravity_gpu``.  Positions and
    velocities are integrated in float64 on the GPU; the float32 tree sees
    positions relative to a moving centre (see ``run_nbody_gpu_tree``).

    Optional ``debug_energy=True`` activates energy / virial diagnostics
    printed alongside the standard progress line.  Energy is essentially
    free for the tree code (potential comes back from every force call), so
    the overhead is only the float64 reduction — ~0.5 ms for N=1M.

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
    # Plain tqdm (not tqdm.auto): the auto backend picks a notebook widget that
    # often renders nothing on JupyterHub-over-browser sessions. The terminal
    # bar works reliably everywhere.
    from tqdm import trange as _trange, tqdm as _tqdm_cls
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False

try:
    from ..nbody_io import _save_snapshot, _save_restart, _load_restart, _IOClock
except ImportError:
    try:
        from nbody_streams.nbody_io import _save_snapshot, _save_restart, _load_restart, _IOClock
    except ImportError:
        _save_snapshot = _save_restart = _load_restart = _IOClock = None  # type: ignore[assignment]

try:
    from ..species import Species, _build_particle_arrays
except ImportError:
    from nbody_streams.species import Species, _build_particle_arrays  # type: ignore[no-redef]

from ._force import tree_gravity_gpu, TreeGPU, G_DEFAULT, _max_n, _IDX_CAP, get_memory_fraction

# GPU memory the integrator allocates per particle beyond what the TreeGPU
# guard counts (the float64 pos/vel, float32/float64 masses and float32
# softening, 64 B, already exist when it checks and count as used): the
# float64 total acceleration (24), the previous step's float32 acc and phi,
# alive during the next tree call (16), the (N, 6) float64 pos/vel
# concatenation for snapshots and restarts (48), and CuPy pool fragmentation
# between them (~40: measured, the pool peaks at 232 B/particle with
# snapshots, 196 without).  Plus the float64 external acceleration (24, only
# with external_potential or force_extra; the potential's own scratch is not
# counted).
_RUN_BYTES_PER_PARTICLE     = 128
_RUN_EXT_BYTES_PER_PARTICLE = 24

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Fused leapfrog updates
# ---------------------------------------------------------------------------
# Each replaces two or three CuPy passes and their temporaries; positions,
# velocities and the total acceleration stay float64 and are updated in place.
if _CUPY_OK:
    # first half kick + drift
    _kick_drift = cp.ElementwiseKernel(
        "float64 a, float64 h, float64 dt", "float64 v, float64 x",
        "v += h * a; x += dt * v;", "tree_kick_drift")
    # total acceleration (float32 self-gravity + float64 external) + second half kick
    _kick_ext = cp.ElementwiseKernel(
        "float32 a_self, float64 a_ext, float64 h", "float64 a, float64 v",
        "a = (double)a_self + a_ext; v += h * a;", "tree_kick_ext")
    _kick_self = cp.ElementwiseKernel(
        "float32 a_self, float64 h", "float64 a, float64 v",
        "a = (double)a_self; v += h * a;", "tree_kick_self")


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
    force_extra=None,
    # I/O
    output_dir: str = "./output",
    save_snapshots: bool = True,
    snapshots: int = 100,
    num_files_to_write: int = 1,
    restart_interval: int | None = 1000,
    continue_run: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    debug_energy: bool = False,
    # Multi-species (preferred path when using run_simulation)
    species: list[Species] | None = None,
) -> np.ndarray:
    """
    Run a GPU-tree N-body simulation with KDK leapfrog integration.

    Force evaluations use the GPU Barnes-Hut tree code (O(N log N)).

    Precision: positions and velocities are integrated in float64.  The
    float32 tree receives ``pos - c``, where ``c`` is the position of the
    particle with the lowest self-gravity potential at the previous force
    evaluation (the density peak of the bound progenitor).  Self-gravity is
    translation invariant, so ``c`` does not change the physics; it keeps
    float32 separations precise for sub-pc structure far from the origin.
    The external potential is evaluated at the float64 positions.

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
        Plummer softening length(s).  Scalar -> uniform; array -> per-particle.
    G : float, optional
        Gravitational constant.  Default: kpc / (km/s)^2 / Msun.
    theta : float, optional
        Barnes-Hut opening angle.  0.75 (fast) - 0.5 (accurate).  Default 0.6.
    nleaf : int, optional
        Max particles per leaf cell.  Must be in {16, 24, 32, 48, 64}.
    ncrit : int, optional
        Walk group size.  Default 64.
    level_split : int, optional
        Unused; accepted for backward compatibility.  Default 5.
    step_timeout_s : float, optional
        Seconds before the watchdog fires a KeyboardInterrupt for a hanging
        CUDA kernel.  Default 60.
    external_potential : agama.Potential, optional
        External time-varying potential (Agama).  Evaluated once per timestep
        (or every ``external_update_interval`` steps) and added to tree forces.

        .. note:: **Dynamical friction is not included.**
            The host is treated as a smooth field with no back-reaction.
            Safe for M_sat < ~1e9 Msun at r > 10 kpc (t_df >> t_Hubble).
            For LMC-class objects (M_sat > 1e10 Msun), friction is important
            and should be added explicitly.

    external_update_interval : int, optional
        Evaluate external potential every N steps.  Default 1.
    output_dir : str, optional
        Snapshot output directory.  Default ``'./output'``.
    save_snapshots : bool, optional
        Whether to write HDF5 snapshots.  Default True.
    snapshots : int, optional
        Number of snapshots to write.  Default 100.  Output steps are
        ``np.round(np.linspace(0, n_steps, snapshots))`` — the same schedule
        used by ``run_nbody_gpu`` / ``run_nbody_cpu`` — so exactly
        ``snapshots`` datasets are written with 0-based ids
        ``000..snapshots-1``, the first at ``time_start`` and the last at
        ``time_end``, even when ``snapshots`` does not divide the step count.
    num_files_to_write : int, optional
        Distribute snapshots across this many HDF5 files.  Default 1.
    restart_interval : int or None, optional
        Save a restart checkpoint every N steps and at the end.  Default 1000.
        ``None`` or 0: no restart files at all (at N ~ 1e8 each is several
        GB).
    continue_run : bool, optional
        Resume from an existing restart file.  Default False.
    overwrite : bool, optional
        Delete existing snapshot files before starting.  Default False.
    verbose : bool, optional
        Print header, progress (rate/ETA), and snapshot messages.  Default True.
    debug_energy : bool, optional
        Print virial ratio Q=K/|W| and relative energy drift dE/E alongside
        the progress line.  Requires ~0.5 ms extra per progress print (float64
        reduction over all particles).  Default False.
        Note: unlike direct-sum integrators, potential phi is returned for free
        by the tree code, so this flag carries negligible overhead.
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
    ValueError
        If N particles do not fit in 80% of the free GPU memory (tree plus
        integrator arrays; see docs/tree_gpu.md, "GPU memory layout") or
        exceed the tree's 2^29-particle index limit.
    """
    if not _CUPY_OK:
        raise ImportError("CuPy is required for run_nbody_gpu_tree.")

    # ── Build mass / softening from species if provided ───────────────────────
    if species is not None:
        masses, softening = _build_particle_arrays(species)

    masses    = np.asarray(masses,    dtype=np.float64)
    softening = (np.asarray(softening, dtype=np.float64)
                 if hasattr(softening, '__len__') else float(softening))

    N          = phase_space.shape[0]
    n_steps    = max(1, round((time_end - time_start) / dt))

    # Snapshot schedule — must stay identical to the direct-sum backend
    # (``run_nbody_gpu`` in nbody_streams/run.py).  Evenly spaced step indices
    # from 0 to n_steps inclusive, so exactly ``snapshots`` datasets are
    # written with ids 000..snapshots-1 and the last one lands on n_steps
    # (i.e. time_end) even when snapshots does not divide n_steps.
    if snapshots > 1:
        snapshot_steps = np.round(np.linspace(0, n_steps, snapshots)).astype(int)
    else:
        snapshot_steps = np.array([n_steps], dtype=int)

    # Progress printed every ~5% of steps (matches run_nbody_gpu cadence)
    progress_every = max(1, n_steps // 20)

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
    def _is_gpu_potential(pot) -> bool:
        """True when pot is a PotentialGPU family object (no CPU round-trip needed)."""
        try:
            from ..agama_helper._potential import _GPUPotBase
            return isinstance(pot, _GPUPotBase)
        except ImportError:
            return False

    if external_potential is not None and not _AGAMA_OK and not _is_gpu_potential(external_potential):
        raise NotImplementedError(
            "external_potential requires Agama (or a PotentialGPU from nbody_streams.agama_helper). "
            "Install Agama with: "
            "pip install agama @ git+https://github.com/GalacticDynamics-Oxford/Agama.git"
        )

    # ── GPU arrays: float64 integrator state, float32 tree inputs ─────────────
    mass_f32 = cp.asarray(masses, dtype=cp.float32)
    mass_f64 = cp.asarray(masses, dtype=cp.float64)

    if np.isscalar(softening):
        eps_gpu = cp.full(N, float(softening), dtype=cp.float32)
    else:
        eps_gpu = cp.asarray(softening, dtype=cp.float32)

    # Own C-contiguous copies: the loop updates them in place, and must not
    # write into a CuPy phase_space passed by the caller.
    pos = cp.array(phase_space[:, :3], dtype=cp.float64, order="C")
    vel = cp.array(phase_space[:, 3:], dtype=cp.float64, order="C")

    # ── Resume from restart ───────────────────────────────────────────────────
    start_step = 0
    snap_idx   = None
    if continue_run and _load_restart is not None:
        restart_data = _load_restart(output_dir)
        if restart_data is not None:
            (pv_np, t_rest, step_rest, snapctr_rest,
             _m_r, _s_r, _snames_r, _sN_r) = restart_data
            pos        = cp.array(pv_np[:, :3], dtype=cp.float64, order="C")
            vel        = cp.array(pv_np[:, 3:], dtype=cp.float64, order="C")
            start_step = int(step_rest) + 1
            snap_idx   = int(snapctr_rest)
            if verbose:
                print(f"Resuming from step {start_step}, snap_idx={snap_idx}")

    # Not resuming (or no restart file found): derive the counter from start_step
    if snap_idx is None:
        snap_idx = int(np.searchsorted(snapshot_steps, start_step, side="left"))

    remaining_steps = n_steps - start_step

    # ── Header ────────────────────────────────────────────────────────────────
    if verbose:
        print("=" * 80)
        print("GPU Tree N-body Integration (Barnes-Hut)")
        print("=" * 80)
        print(f"Particles: {N:,}")
        if species is not None:
            for s in species:
                print(f"  [{s.name}] N={s.N:,}  eps={s.softening if np.isscalar(s.softening) else 'variable'}")
        print(f"Time: {time_start:.3e} -> {time_end:.3e}  (dt={dt:.3e})")
        print(f"Steps: {n_steps:,}  ({remaining_steps:,} remaining)")
        print(f"Tree: theta={theta}  nleaf={nleaf}  ncrit={ncrit}")
        print(f"Softening: {softening if np.isscalar(softening) else 'per-particle'}")
        print(f"External potential: {'Yes' if external_potential is not None else 'No'}")
        if external_potential is not None:
            print(f"  Update interval: every {external_update_interval} steps")
        print(f"Snapshots: {len(snapshot_steps)}  "
              f"(steps {snapshot_steps[0]:,}..{snapshot_steps[-1]:,})")
        print(f"Restart files: every {restart_interval} steps" if restart_interval else "Restart files: off")
        print(f"Watchdog timeout: {step_timeout_s:.0f} s/step")
        print(f"debug_energy: {debug_energy}")
        print("=" * 80)

    # ── Pre-allocate tree handle ───────────────────────────────────────────────
    extra = _RUN_BYTES_PER_PARTICLE + (
        _RUN_EXT_BYTES_PER_PARTICLE if (external_potential is not None or force_extra is not None) else 0)
    cap = _max_n(extra)
    if N > cap:
        raise ValueError(
            f"N={N:,} exceeds the GPU tree run limit ({cap:,}), set by "
            + (f"{get_memory_fraction():.0%} of free GPU memory (tree plus integrator arrays; "
               "raise it with nbody_streams.tree_gpu.set_memory_fraction)" if cap < _IDX_CAP
               else "the 29-bit particle index (2^29 particles)") + ".")
    tree     = TreeGPU(N, eps=0.0, theta=theta, verbose=False)
    watchdog = _StepWatchdog(timeout_s=step_timeout_s)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _energy(vel_: cp.ndarray, phi_: cp.ndarray) -> tuple[float, float]:
        """KE + PE in float64. ~0.5 ms for N=1M (two reductions)."""
        v2 = cp.sum(vel_.astype(cp.float64) ** 2, axis=1)
        KE = 0.5 * float(cp.sum(mass_f64 * v2))
        PE = 0.5 * float(cp.sum(mass_f64 * phi_.astype(cp.float64)))
        return KE, PE

    def _ext_acc(pos_: cp.ndarray, t_: float) -> cp.ndarray | None:
        """Evaluate external acceleration for either agama.Potential or PotentialGPU.

        PotentialGPU: force() accepts a CuPy array directly — no host transfer.
        agama.Potential: positions moved to CPU, result brought back to GPU.
        Both paths take and return float64 CuPy arrays (absolute positions).
        """
        if external_potential is None:
            return None
        if _is_gpu_potential(external_potential):
            # PotentialGPU: stays on GPU entirely
            return cp.asarray(external_potential.force(pos_, t=t_), dtype=cp.float64)
        # agama.Potential: CPU round-trip
        return cp.asarray(external_potential.force(cp.asnumpy(pos_), t=t_),
                          dtype=cp.float64)

    def _self_gravity(pos_: cp.ndarray, centre_: cp.ndarray):
        """Tree self-gravity on float32 positions relative to ``centre_``.

        Returns float32 (acc, phi) and the next centre: the float64 position
        of the most bound particle (min phi).  Everything stays on the device.
        """
        acc_, phi_ = tree_gravity_gpu(
            pos_, mass_f32, eps_gpu, G=G,
            theta=theta, nleaf=nleaf, ncrit=ncrit, level_split=level_split,
            tree=tree, center=centre_,
        )
        return acc_, phi_, pos_[cp.argmin(phi_)]

    io_clock = _IOClock(sync=cp.cuda.Device().synchronize)   # output wall time

    def _save_snap_now(t_: float):
        if not save_snapshots or _save_snapshot is None:
            return
        with io_clock:
            pv_np = cp.asnumpy(cp.concatenate([pos, vel], axis=1))
            _save_snapshot(
                pv_np, snap_idx, t_, output_dir,
                species=species,
                time_step=dt,
                num_files_to_write=num_files_to_write if num_files_to_write > 1 else None,
                total_expected_snapshots=snapshots,
            )

    def _save_restart_now(step: int, t_: float):
        if _save_restart is None or not restart_interval:
            return
        with io_clock:
            pv_np  = cp.asnumpy(cp.concatenate([pos, vel], axis=1))
            m_np   = cp.asnumpy(mass_f32).astype(np.float64)
            s_np   = cp.asnumpy(eps_gpu).astype(np.float64)
            snames = [s.name for s in species] if species else None
            sN     = [s.N    for s in species] if species else None
            _save_restart(pv_np, float(t_), step, output_dir, snap_idx,
                          mass_arr=m_np, softening_arr=s_np,
                          species_names=snames, species_N=sN)

    def _nan_gate(step: int, label: str) -> bool:
        if not cp.any(cp.isnan(acc[:, 0])):
            return False
        print(f"\n[ERROR] NaN in forces at step {step} ({label}). Aborting.")
        return True

    # ── Initial forces ─────────────────────────────────────────────────────────
    if verbose:
        print("\nComputing initial forces...")

    t_now = time_start + start_step * dt

    # Initial tree centre: mass-weighted centre of the system.  From the first
    # force evaluation on, the centre follows the minimum of phi.
    centre = (mass_f64[:, None] * pos).sum(axis=0) / mass_f64.sum()

    with watchdog:
        acc_self, phi, centre = _self_gravity(pos, centre)

    def _ext_total(acc_ext_, t_):
        """External + extra (non-conservative) acceleration, float64, or None.

        pos and vel are CuPy float64 arrays on the tree-GPU path; phi is passed
        for free from the tree force evaluation.
        """
        if force_extra is None:
            return acc_ext_
        extra = cp.asarray(force_extra(pos, vel, masses, t_, phi=phi), dtype=cp.float64)
        return extra if acc_ext_ is None else acc_ext_ + extra

    # Total acceleration (float64), updated in place every step.
    _acc_ext = _ext_acc(pos, t_now)
    acc = acc_self.astype(cp.float64)
    a_ext = _ext_total(_acc_ext, t_now)
    if a_ext is not None:
        acc += a_ext

    # Energy reference (only computed if debug_energy, but always store E_ref)
    E_ref = 0.0
    if debug_energy:
        KE0, PE0 = _energy(vel, phi)
        E_ref = KE0 + PE0

    # Initial snapshot (only when the schedule actually starts at start_step)
    if snap_idx < len(snapshot_steps) and snapshot_steps[snap_idx] == start_step:
        if save_snapshots:
            _save_snap_now(t_now)
            if verbose:
                print(f"Saved snapshot id={snap_idx:03d} at step "
                      f"{start_step}, time {t_now:.4e}")
        snap_idx += 1

    if verbose:
        print("\nStarting integration...")

    # ── KDK loop ───────────────────────────────────────────────────────────────
    t_wall0  = pytime.perf_counter()
    io_clock.total = 0.0      # count output from here on (the initial snapshot is setup)
    aborted  = False
    step     = start_step   # ensure defined for KeyboardInterrupt handler

    # Progress printing helper — routes through tqdm.write when tqdm is active
    # so the bar line is not clobbered.
    if _TQDM_OK and verbose:
        _loop      = _trange(start_step, n_steps, desc="N-body simulation", unit="step")
        _vprint    = _tqdm_cls.write
    else:
        _loop      = range(start_step, n_steps)
        def _vprint(msg: str) -> None:  # type: ignore[misc]
            print(msg, flush=True)

    try:
        for step in _loop:
            current_step = step + 1
            t_now = time_start + current_step * dt

            with watchdog:
                _kick_drift(acc, 0.5 * dt, dt, vel, pos)
                acc_self, phi, centre = _self_gravity(pos, centre)

            # External potential
            if external_potential is not None and current_step % external_update_interval == 0:
                _acc_ext = _ext_acc(pos, t_now)

            # Extra non-conservative forces (e.g. dynamical friction).
            a_ext = _ext_total(_acc_ext, t_now)

            with watchdog:
                if a_ext is None:
                    _kick_self(acc_self, 0.5 * dt, acc, vel)
                else:
                    _kick_ext(acc_self, a_ext, 0.5 * dt, acc, vel)

            # ── Progress update (every ~5% of steps) ──────────────────────────
            # When tqdm is active it already shows step count / rate / ETA,
            # so only emit the manual line when debug_energy adds extra info
            # or when running without tqdm.
            if verbose and current_step % progress_every == 0:
                _emit_manual = (not _TQDM_OK) or debug_energy
                if _emit_manual:
                    elapsed    = pytime.perf_counter() - t_wall0 - io_clock.total
                    done_steps = current_step - start_step
                    left_steps = n_steps - current_step
                    rate       = done_steps / elapsed if elapsed > 0 else 0.0
                    avg_ms     = elapsed / max(done_steps, 1) * 1e3
                    eta_s      = left_steps / rate if rate > 0 else 0.0

                    line = (f"  Step {current_step:>6}/{n_steps} | "
                            f"t={t_now:.4e} | "
                            f"Snapshots: {snap_idx}/{snapshots} | "
                            f"{rate:.1f} steps/s | "
                            f"avg {avg_ms:.1f}ms/step | "
                            f"ETA {eta_s:.0f}s")

                    if debug_energy and E_ref != 0.0:
                        KE, PE = _energy(vel, phi)
                        dE = (KE + PE - E_ref) / abs(E_ref)
                        Q  = KE / abs(PE) if PE != 0.0 else float("nan")
                        line += f" | Q={Q:.3f}  dE/E={dE:+.2e}"

                    _vprint(line)

            # ── Snapshot ──────────────────────────────────────────────────────
            while (snap_idx < len(snapshot_steps)
                   and current_step >= snapshot_steps[snap_idx]):
                if save_snapshots:
                    if _nan_gate(step, "snap"):
                        aborted = True;  break
                    _save_snap_now(t_now)
                    if verbose:
                        _vprint(f"Saved snapshot id={snap_idx:03d} at step "
                                f"{current_step}, time {t_now:.4e}")
                snap_idx += 1
            if aborted:
                break

            # ── Restart checkpoint ────────────────────────────────────────────
            if restart_interval and current_step % restart_interval == 0:
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

    cp.cuda.Device().synchronize()
    elapsed_total = pytime.perf_counter() - t_wall0
    t_io          = io_clock.total
    t_steps       = max(elapsed_total - t_io, 1e-12)   # integration only
    steps_done    = (step + 1 - start_step) if not aborted else max(step - start_step, 1)
    avg_ms        = t_steps / max(steps_done, 1) * 1e3
    status        = "ABORTED" if aborted else "Done"

    if verbose:
        print(f"\n{status}. Wall time: {elapsed_total:.1f} s  "
              f"(integration {t_steps:.1f} s: {avg_ms:.1f} ms/step, "
              f"{steps_done / t_steps:.1f} steps/s  |  output {t_io:.1f} s)")

    pv = cp.concatenate([pos, vel], axis=1)
    return cp.asnumpy(pv)
