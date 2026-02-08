#!/usr/bin/env python3
"""
nbody_streams.run

GPU-accelerated N-body simulation with leapfrog (KDK) integration.

Provides high-performance N-body integration using CUDA-accelerated direct summation
for self-gravity, with optional external time-evolving potentials via Agama.

Requirements
------------
- nbody_forces_gpu module (for GPU self-gravity)
- CuPy (for GPU integration)
- Agama (optional, for external potentials)
- NumPy

Author: Arpit Arora
Date: Sept 2025
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Literal
import warnings
import time as pytime
import h5py # For HDF5 snapshot output (if needed)

from .fields import compute_nbody_forces_gpu, compute_nbody_forces_cpu 
from .io import _save_snapshot, _save_restart, _load_restart, _update_snapshot_times

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration disabled.", ImportWarning)

try:
    import agama
    AGAMA_AVAILABLE = True
except ImportError:
    AGAMA_AVAILABLE = False

try:
    import pyfalcon
    HAS_FALCON = True
except ImportError:
    HAS_FALCON = False
    warnings.warn("pyfalcon not available. Tree code is disabled. Please use a limited number of particles.", ImportWarning)

# ============================================================================
# CONSTANTS AND TYPE DEFINITIONS
# ============================================================================

# Default gravitational constant (kpc, km/s, Msun units)
G_DEFAULT = 4.300917270069976e-06 # double precision value for accuracy, but we will use float32 in GPU kernel for performance
# Kernel type mapping
KERNEL_TYPES = Literal['newtonian', 'plummer', 'dehnen_k1', 'dehnen_k2', 'spline']
 
# Define this once at the top of your function or module
_PRECISION_MAP = {
    'float64': (cp.float64, np.float64),
    'float32': (cp.float32, np.float32),
    'float32_kahan': (cp.float32, np.float32)  # Same storage as float32!
}
NBODY_UNITS = {
    'kpc': 1.0, # length unit
    'Msun': 1.0, # mass unit
    'kpc / (km/s)': 1.0, # time unit (derived from length and velocity) 
    'km/s': 1, # velocity unit
    'G': G_DEFAULT, # gravitational constant in these units
}

# ============================================================================
# INTERNAL I/O FUNCTIONS
# ============================================================================

def _save_snapshot_simplistic(
    phase_space: np.ndarray,
    snap_index: int,
    time: float,
    output_dir: Path,
    **kwargs,
) -> None:
    """
    Fallback saver: Stores data in a simple .npy format.
    
    Parameters
    ----------
    phase_space : np.ndarray, shape (N, 6)
        Phase space [x, y, z, vx, vy, vz].
    snapshot_num : int
        Snapshot index.
    time : float
        Current time.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"snap_{snapshot_num:04d}.nbody"
    header = f"t={time:.12e}\nx y z vx vy vz"
    np.savetxt(filename, phase_space, header=header, fmt='%.12e')

# ==========================================================================
# ACCELERATION COMPUTATION FUNCTIONS
# ==========================================================================

def _compute_accelerations_gpu(
    pos_gpu: cp.ndarray,
    mass: np.ndarray,
    softening: float | np.ndarray,
    G: float,
    precision: Literal['float32', 'float64', 'float32_kahan'],
    kernel_type: str,
    external_potential: 'agama.Potential | None',
    time: float,
    external_update_interval: int,
    step: int,
    cached_external_acc: cp.ndarray | None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    """
    Compute total accelerations on GPU.
    
    Combines self-gravity (GPU) with optional external potential (CPU).
    External forces can be cached and updated less frequently.
    
    Parameters
    ----------
    pos_gpu : cp.ndarray, shape (N, 3)
        Positions on GPU (float64).
    mass : np.ndarray, shape (N,)
        Particle masses (float32).
    softening : float | np.ndarray
        Softening length(s).
    G : float
        Gravitational constant.
    kernel_type : str
        Force kernel type.
    external_potential : agama.Potential | None
        External potential.
    time : float
        Current time.
    external_update_interval : int
        Update external forces every N steps.
    step : int
        Current step number.
    cached_external_acc : cp.ndarray | None
        Cached external accelerations.
    
    Returns
    -------
    acc_total : cp.ndarray, shape (N, 3), float64
        Total accelerations on GPU.
    new_cached_external : cp.ndarray | None
        Updated cached external accelerations (or None if not updated).
    """
    
    # precision mapping - default to float32 for GPU performance, but allow float64 for accuracy if needed
    prec_key = precision.lower()
    if prec_key not in _PRECISION_MAP:
        raise ValueError(f"Precision must be one of {list(_PRECISION_MAP.keys())}")     
    
    dtype, dtype_np = _PRECISION_MAP[prec_key]


    # Convert mass to GPU if needed
    mass_gpu = cp.asarray(mass, dtype=dtype)
    
    # Self-gravity on GPU (keep everything on GPU!)
    acc_self_gpu = compute_nbody_forces_gpu(
        pos_gpu.astype(dtype),  # Convert on GPU
        mass_gpu,
        softening,
        G,
        precision,
        kernel_type,
        return_cupy=True  # Return CuPy array directly
    ).astype(cp.float64, copy=False) # Convert to float64
    
    # External potential (if provided)
    if external_potential is not None:
        # Update external forces at specified interval
        if step % external_update_interval == 0 or cached_external_acc is None:
            pos_cpu_f64 = pos_gpu.get()  # Only transfer when needed
            acc_ext_cpu = external_potential.force(pos_cpu_f64, t=time)
            new_cached_external = cp.asarray(acc_ext_cpu, dtype=cp.float64)
        else:
            new_cached_external = cached_external_acc
        
        acc_total = acc_self_gpu + new_cached_external
        return acc_total, new_cached_external
    else:
        return acc_self_gpu, None

def _compute_accelerations_tree(
    pos_cpu: np.ndarray,
    mass: np.ndarray,
    softening: float | np.ndarray,
    theta: float,
    kernel: int,
    G: float,
    external_potential: agama.Potential | None,
    time: float,
) -> np.ndarray:
    """
    Compute accelerations using tree algorithm (pyfalcon).
    
    Parameters
    ----------
    pos_cpu : np.ndarray
        Particle positions, shape (N, 3).
    mass : np.ndarray
        Particle masses, shape (N,).
    softening : float | np.ndarray
        Softening length(s).
    theta : float
        Opening angle for tree algorithm.
    kernel : int
        Kernel type for pyfalcon.
    G : float
        Gravitational constant.
    external_potential : agama.Potential | None
        External time-dependent potential.
    time : float
        Current simulation time.
    
    Returns
    -------
    np.ndarray
        Total accelerations, shape (N, 3).
    """
    # Self-gravity from pyfalcon tree
    acc, _ = pyfalcon.gravity(
        pos_cpu, 
        G * mass, 
        eps=softening, 
        theta=theta,
        kernel=kernel
    )
    
    # Add external potential if provided
    if external_potential is not None:
        acc += external_potential.force(pos_cpu, t=time)
    
    return acc

def _compute_accelerations_cpu(
    pos_cpu: np.ndarray,
    mass: np.ndarray,
    softening: float | np.ndarray,
    G: float,
    kernel_type: str,
    nthreads: int | None,
    external_potential: agama.Potential | None,
    time: float,
) -> np.ndarray:
    """
    Compute accelerations using direct O(N^2) pairwise summation.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3).
    masses : np.ndarray
        Particle masses, shape (N,).
    softening : float | np.ndarray
        Softening length(s).
    G : float
        Gravitational constant.
    kernel_type : str
        Kernel type ('newtonian', 'plummer', 'spline', 'dehnen_k1', 'dehnen_k2').
    nthreads : int | None
        Number of threads for parallelization.
    external_potential : agama.Potential | None
        External time-dependent potential.
    time : float
        Current simulation time.
    
    Returns
    -------
    np.ndarray
        Total accelerations, shape (N, 3).
    """
    # Self-gravity from direct summation
    acc = compute_nbody_forces_cpu(
        pos=pos_cpu,
        mass=mass,
        softening=softening,
        G=G,
        kernel=kernel_type,
        nthreads=nthreads,
    )
    
    # Add external potential if provided
    if external_potential is not None:
        acc += external_potential.force(pos_cpu, t=time)
    return acc

# ===========================================================================
# PUBLIC API FUNCTIONS
# ===========================================================================

def run_nbody_gpu(
    phase_space: np.ndarray,
    masses: np.ndarray,
    time_start: float,
    time_end: float,
    dt: float,
    softening: float | np.ndarray,
    G: float = G_DEFAULT,
    precision: Literal['float32', 'float64', 'float32_kahan'] = 'float32_kahan',
    kernel: Literal['newtonian', 'plummer', 'dehnen_k1', 'dehnen_k2', 'spline'] = 'spline',
    external_potential: 'agama.Potential | None' = None,
    external_update_interval: int = 1,
    output_dir: str = "./output",
    snapshots: int = 10,
    num_files_to_write: int = 1,
    restart_interval: int = 1000,
    continue_run: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run GPU-accelerated N-body simulation with leapfrog (KDK) integration.
    
    Uses CUDA for self-gravity computation and CuPy for integration on GPU.
    Positions and velocities maintained in double precision (float64), while
    force calculations use single precision (float32) for speed.
    
    Parameters
    ----------
    phase_space : np.ndarray, shape (N, 6)
        Initial conditions [x, y, z, vx, vy, vz] in double precision.
    masses : np.ndarray, shape (N,)
        Particle masses (will be converted to float32 for force calculation).
    time_start : float
        Start time.
    time_end : float
        End time.
    dt : float
        Timestep.
    softening : float | np.ndarray
        Gravitational softening length. Scalar or per-particle array.
    G : float, optional
        Gravitational constant. Default: 4.30092e-6 (kpc, km/s, Msun units).
    precision : {'float32', 'float64', 'float32_kahan'}, optional
        Floating point precision for acceleration computation. Default is 'float32_kahan'.
        'float32_kahan' uses Kahan summation for improved accuracy in float32 mode.
    kernel : str, optional
        Force softening kernel. Options: 'newtonian', 'plummer', 'dehnen_k1',
        'dehnen_k2', 'spline'. Default: 'spline'.
    external_potential : agama.Potential | None, optional
        External time-dependent potential. Default: None.
    external_update_interval : int, optional
        Update external forces every N steps (reduces Agama overhead).
        Default: 1 (update every step). Try 5-10 for slowly-varying potentials.
    output_dir : str, optional
        Output directory. Default: './output'.
    snapshots : int, optional
        Number of snapshots to save (evenly spaced). Default: 10.
    num_files_to_write : int, optional
        Number of snapshot files to write (for load balancing). Default: 1.
    restart_interval : int, optional
        Save restart file every N steps. Default: 1000.
    continue_run : bool, optional
        Resume from restart file if exists. Default: False.
    verbose : bool, optional
        Print progress information. Default: True.
    
    Returns
    -------
    np.ndarray, shape (N, 6)
        Final phase space coordinates.
    
    Raises
    ------
    ImportError
        If required packages (CuPy, nbody_forces_gpu) not available.
    ValueError
        If input arrays have invalid shapes.
    
    Notes
    -----
    **Integration Scheme:**
    Uses symplectic kick-drift-kick (KDK) leapfrog integrator, which is
    second-order accurate and conserves energy well for Hamiltonian systems.
    
    **GPU Memory Management:**
    - Positions/velocities kept on GPU throughout integration
    - Only transferred to CPU for snapshots and restart files
    - External forces computed on CPU (Agama) and transferred to GPU
    
    **Performance:**
    - Self-gravity: ~3-50ms depending on N (10K-80K particles)
    - External forces: ~10-20ms on 16 CPU cores (if Agama used)
    - KDK operations: ~0.01ms (negligible)
    
    Use `external_update_interval > 1` to reduce overhead if external
    potential changes slowly compared to self-gravity.
    
    **Unit Consistency:**
    Ensure G matches your unit system. Common values:
    - kpc, km/s, Msun: G = 4.30092e-6
    - kpc, Msun, Gyr: G = 4.302e-6
    - N-body units: G = 1.0
    
    Examples
    --------
    Simple self-gravity simulation:
    
    >>> import numpy as np
    >>> N = 10000
    >>> phase_space = np.random.randn(N, 6)
    >>> masses = np.ones(N)
    >>> final = run_nbody_gpu(phase_space, masses, 0.0, 0.1, 1e-4, 
    ...                       softening=0.01, snapshots=10)
    
    With external potential (Agama):
    
    >>> import agama
    >>> agama.setUnits(mass=1, length=1, velocity=1)
    >>> pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0)
    >>> final = run_nbody_gpu(phase_space, masses, 0.0, 0.1, 1e-4,
    ...                       softening=0.01, external_potential=pot,
    ...                       external_update_interval=5)
    
    Resume from crash:
    
    >>> final = run_nbody_gpu(phase_space, masses, 0.0, 0.1, 1e-4,
    ...                       softening=0.01, continue_run=True)
    
    See Also
    --------
    nbody_forces_gpu.compute_nbody_forces_gpu : GPU force computation
    """
    # Validate dependencies
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy required for GPU integration. Install: pip install cupy-cuda13x")
        
    # Validate inputs
    phase_space = np.asarray(phase_space, dtype=np.float64)
    if phase_space.ndim != 2 or phase_space.shape[1] != 6:
        raise ValueError(f"phase_space must be (N, 6), got {phase_space.shape}")
    
    masses = np.asarray(masses, dtype=np.float32)
    N = phase_space.shape[0]
    
    if masses.shape[0] != N:
        raise ValueError(f"masses must have length N={N}, got {masses.shape[0]}")
    
    if external_potential is not None and not AGAMA_AVAILABLE:
        raise ImportError("Agama required for external_potential. Install Agama.")
        
    output_path = Path(output_dir)

    start_step = 0
    time = time_start
    snapshot_counter = None

    # Load restart if requested
    if continue_run:
        restart_data = _load_restart(output_path)
        if restart_data is not None:
            xv, time, start_step, saved_snap_counter = restart_data
            snapshot_counter = int(saved_snap_counter) # Update snapshot counter from restart file
            if verbose:
                print(f"✓ Resuming from step {start_step}, time {time:.6e}")
    else:
        xv = phase_space.copy()  # Ensure we have a local copy to modify
    
    
    # Compute steps reliably (use round to avoid off-by-one)
    total_steps = int(round((time_end - time_start) / dt))
    remaining_steps = total_steps - start_step
        
    # Compute snapshot steps (evenly spaced from 0 to total_steps)
    if snapshots > 1:
        snapshot_steps = np.round(np.linspace(0, total_steps, snapshots)).astype(int)
    else:
        snapshot_steps = np.array([total_steps], dtype=int)

     # If not resuming from restart, initialize snapshot_counter from start_step
    if snapshot_counter is None:
        # For a fresh run, start at 0
        # For a resumed run, count how many snapshots should have been written already
        snapshot_counter = int(np.searchsorted(snapshot_steps, start_step, side="left"))  

    if verbose:
        print("="*80)
        print("GPU N-body Integration")
        print("="*80)
        print(f"Particles: {N:,}")
        print(f"Time: {time_start:.3e} → {time_end:.3e} (dt={dt:.3e})")
        print(f"Steps: {total_steps:,} ({remaining_steps:,} remaining)")
        print(f"Kernel: {kernel}, Softening: {softening if np.isscalar(softening) else 'variable'}")
        print(f"External potential: {'Yes' if external_potential is not None else 'No'}")
        if external_potential is not None:
            print(f"  Update interval: every {external_update_interval} steps")
        print(f"Snapshots: {snapshots} (every ~{total_steps//snapshots:,} steps)")
        print(f"Restart files: every {restart_interval} steps")
        print("="*80)

    # Transfer to GPU
    if verbose:
        print("\nTransferring data to GPU...")
        
    pos_gpu = cp.asarray(xv[:, :3], dtype=cp.float64)
    vel_gpu = cp.asarray(xv[:, 3:6], dtype=cp.float64)
    mass_gpu = cp.asarray(masses, dtype=cp.float64) # float32 is enough. 
    dt_gpu = cp.float64(dt)
    
    # Initial acceleration (with warmup)
    if verbose:
        print("Computing initial forces (compiling CUDA kernel)...")
    
    acc_gpu, cached_external_acc = _compute_accelerations_gpu(  
        pos_gpu, mass_gpu, softening, G, precision, kernel,
        external_potential, time, external_update_interval, start_step, None     
    )
    
    # Warmup complete, start timing
    if verbose:
        print("\nStarting integration...")
    
    # Save initial snapshot if requested (global index at snapshot_steps[snapshot_counter] == start_step)
    if snapshot_counter < len(snapshot_steps) and snapshot_steps[snapshot_counter] == start_step:
        xv_cpu = np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)])
        if verbose:
            print(f"writing snapshot snap: {snapshot_counter} at step {start_step}, time {time:.6e}...")
        _save_snapshot(xv_cpu, snapshot_counter, time, output_path,
                       mass_dark=masses[0],  # dark mass in this code path
                       num_files_to_write=num_files_to_write,
                       total_expected_snapshots=snapshots)
        _update_snapshot_times(output_path, snapshot_counter, time)
        snapshot_counter += 1
    
    # ============================================================================
    # Main integration loop: iterate over the remaining steps (1..remaining_steps)
    # ============================================================================
    t_start = pytime.perf_counter()
    for step_i in range(1, remaining_steps + 1):
        current_step = start_step + step_i
        
        # === KDK Leapfrog (all on GPU) ===
        
        # Kick (half-step)
        vel_gpu += acc_gpu * (dt_gpu / 2)
        
        # Drift (full-step)
        pos_gpu += vel_gpu * dt_gpu
        
        # Update time
        time += dt
        
        # Compute new accelerations
        acc_gpu, cached_external_acc = _compute_accelerations_gpu(
            pos_gpu, mass_gpu, softening, G, precision, kernel,
            external_potential, time, external_update_interval,
            current_step, cached_external_acc
        )
        
        # Kick (half-step)
        vel_gpu += acc_gpu * (dt_gpu / 2)
        
        # === I/O Operations ===
        # === snapshots (handle possibly multiple snapshot indices that fall here) ===
        while snapshot_counter < len(snapshot_steps) and current_step >= snapshot_steps[snapshot_counter]:
            xv_cpu = np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)])
            _save_snapshot(xv_cpu, snapshot_counter, time, output_path,
                           num_files_to_write=num_files_to_write,
                           mass_dark=masses[0],  # dark mass in this code path
                           total_expected_snapshots=snapshots)
            _update_snapshot_times(output_path, snapshot_counter, time)
            if verbose:
                print(f"Saved snapshot id={snapshot_counter:03d} at step {current_step}, time {time:.6e}...")
            
            snapshot_counter += 1

        # Progress update
        if verbose and step_i % max(1, remaining_steps // 20) == 0:
            elapsed = pytime.perf_counter() - t_start
            rate = step_i / elapsed if elapsed > 0 else 0
            eta = (remaining_steps - step_i) / rate if rate > 0 else 0
            avg_step_time = elapsed / step_i if step_i > 0 else 0
            print(f"  Step {current_step:>6}/{total_steps} | "
                f"t={time:.4e} | "
                f"Snapshots: {snapshot_counter}/{len(snapshot_steps)} | "
                f"{rate:.1f} steps/s | "
                f"avg {avg_step_time*1000:.1f}ms/step | "
                f"ETA {eta:.0f}s")
                
        # Save restart file
        if current_step > 0 and (current_step % restart_interval) == 0:
            xv_cpu = np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)])
            _save_restart(xv_cpu, time, current_step, output_path, snapshot_counter)
    # =============================================================================
    # End of integration loop
    # =============================================================================

    # ensure final snapshot saved (if last snapshot maps to total_steps and wasn't saved)
    if snapshot_counter < len(snapshot_steps) and snapshot_steps[-1] == total_steps:
        xv_cpu = np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)])
        if verbose:
            print(f"<=Saving final snapshot snap.{snapshot_counter:03d} at step {total_steps}, time {time:.6e}=>")
        _save_snapshot(xv_cpu, snapshot_counter, time, output_path,
                       mass_dark=masses[0],  # dark mass in this code path
                       num_files_to_write=num_files_to_write,
                       total_expected_snapshots=snapshots)
        _update_snapshot_times(output_path, snapshot_counter, time)
        snapshot_counter += 1
    
    # when saving final restart (optional), include snapshot_counter
    _save_restart(np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)]), time, total_steps, output_path, snapshot_counter)

    if verbose:
        t_end = pytime.perf_counter()
        total_time = t_end - t_start
        avg_step_time = total_time / remaining_steps if remaining_steps > 0 else 0
        
        print("\n" + "="*80)
        print("Integration Complete")
        print("="*80)
        print(f"Final time: {time:.6e}")
        print(f"Total wall time: {total_time:.2f} s")
        print(f"Steps per second: {remaining_steps / total_time:.1f}")
        print(f"Average step time: {avg_step_time*1000:.1f} ms")
        print(f"Snapshots saved: {snapshot_counter}")
        print("="*80)
    
    # Return final state
    xv_final = np.hstack([cp.asnumpy(pos_gpu), cp.asnumpy(vel_gpu)])
    return xv_final

def run_nbody_cpu(
    phase_space: np.ndarray,
    masses: np.ndarray,
    time_start: float,
    time_end: float,
    dt: float,
    softening: float | np.ndarray,
    G: float = 4.302e-6,
    method: str = "direct",
    theta: float = 0.6,
    kernel: int | str = 1,
    nthreads: int | None = None,
    external_potential: agama.Potential | None = None,
    output_dir: str = "./",
    snapshots: int = 1,
    num_files_to_write: int = 1,
    restart_interval: int = 1000,
    continue_run: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run CPU-NUMBA accelerated N-body simulation with leapfrog (KDK) integration.
    
    Parameters
    ----------
    phase_space : np.ndarray
        Initial phase space coordinates, shape (N, 6) with columns [x, y, z, vx, vy, vz].
    masses : np.ndarray
        Particle masses, shape (N,).
    time_start : float
        Starting time in Gyr | kpc/(km/s).
    time_end : float
        End time in Gyr | kpc/(km/s).
    dt : float
        Time step in Gyr | kpc/(km/s).
    softening : float | np.ndarray
        Gravitational softening length. Can be a single value or array of shape (N,).
    G : float, optional
        Gravitational constant (default: 4.302e-6 kpc^3 / (Msun Gyr^2)).
    method : {'tree', 'direct'}, optional
        Gravity solver method (default: 'tree'):
        - 'tree': Fast tree algorithm via pyfalcon (good for >20K particles)
        - 'direct': Direct O(N^2) pairwise summation (faster for <20K particles)
    theta : float, optional
        Opening angle for tree algorithm (only for method='tree', default: 0.6).
    kernel : int | str, optional
        Softening kernel:
        - For method='tree': integer kernel type for pyfalcon (default: 1)
        - For method='direct': string kernel type (default: 'spline')
          Options: 'newtonian', 'plummer', 'spline', 'dehnen_k1', 'dehnen_k2'
    nthreads : int | None, optional
        Number of threads for parallelization (only for method='direct', default: None = auto).
    external_potential : agama.Potential | None, optional
        External time-dependent potential (default: None).
    output_dir : str, optional
        Directory for output files (default: "./").
    snapshots : int, optional
        Number of snapshots to save. If 1, only saves final state (default: 1).
    num_files_to_write : int, optional
        Number of snapshot files to write (for load balancing). Default: 1.
    restart_interval : int, optional
        Save restart file every N steps (default: 1000).
    continue_run : bool, optional
        Continue from restart file if it exists (default: False).
    verbose : bool, optional
        low-key outputs for sanity (default: True).
        
    Returns
    -------
    np.ndarray
        Final phase space coordinates, shape (N, 6).
    
    Notes
    -----
    Uses kick-drift-kick (KDK) leapfrog integration scheme, which is symplectic
    and second-order accurate. Snapshots are saved as 'snap_XXX.nbody' files.
    Restart files are saved as 'restart.npz' for crash recovery.
    
    **Method Selection:**
    - Use 'tree' for large particle counts (>20K). The FMM scales as O(N).
    - Use 'direct' for small particle counts (<20K). Scales as O(N^2) but with 
      better accuracy and faster wall time for small N when parallelized.
    - Direct method requires the 'compute_characteristics_ut' package.
    
    **IMPORTANT - Unit Consistency:**
    The gravitational constant G must be consistent with your unit system. The default
    value assumes:
    - Positions in kpc
    - Velocities in kpc/Gyr
    - Masses in Msun
    - Time in Gyr
    
    Common unit systems and corresponding G values:
    - **kpc, Msun, Gyr:** G = 4.302e-6 kpc^3 / (Msun Gyr^2)  [DEFAULT]
    - **kpc, Msun, Myr:** G = 4.302 kpc^3 / (Msun Myr^2)
    - **kpc, Msun, kpc/(km/s):** G = 4.30092e-6 kpc^3 / (Msun * (kpc/(km/s))^2)
    - **pc, Msun, Myr:** G = 4.302e-3 pc^3 / (Msun Myr^2)
    
    If using an external_potential, ensure it uses the same unit system and G value.
    You can access agama's G value via `agama.G` if needed.
    
    Examples
    --------
    >>> # Simple self-gravity simulation with tree method (default)
    >>> phase_space = np.random.randn(1000, 6)
    >>> masses = np.ones(1000) * 1e5
    >>> final_state = run_nbody(phase_space, masses, 0.0, 0.1, 1e-4, softening=0.01)
    
    >>> # Small system with direct method for better accuracy
    >>> final_state = run_nbody(phase_space, masses, 0.0, 0.1, 1e-4, 
    ...                          softening=0.01, method='direct', 
    ...                          kernel='spline', nthreads=48)
    
    >>> # With external time-evolving potential (matching agama's G)
    >>> pot = agama.Potential(type='Dehnen', mass=1e12, scaleRadius=10.0)
    >>> final_state = run_nbody(phase_space, masses, 0.0, 0.1, 1e-4, 
    ...                          softening=0.01, G=agama.G, 
    ...                          external_potential=pot)
    
    >>> # Different unit system (kpc, Msun, Myr)
    >>> final_state = run_nbody(phase_space, masses, 0.0, 100.0, 0.1, 
    ...                          softening=0.01, G=4.302)
    """
    # Validate method
    if method not in ['tree', 'direct']:
        raise ValueError(f"method must be 'tree' or 'direct', got '{method}'")
    
    if method == 'tree' and not HAS_FALCON:
        raise ImportError(
            "method='tree' requires 'git@github.com:GalacticDynamics-Oxford/pyfalcon.git' package. "
            "Install it or use method='direct' instead."
        )
    
    # Set default kernel for direct method if not specified
    if method == 'direct' and isinstance(kernel, int):
        kernel = 'spline'

    N = phase_space.shape[0]
    output_path = Path(output_dir)

    start_step = 0
    time = time_start
    snapshot_counter = None

    # Load restart if requested
    if continue_run:
        restart_data = _load_restart(output_path)
        if restart_data is not None:
            xv, time, start_step, saved_snap_counter = restart_data
            snapshot_counter = int(saved_snap_counter) # Update snapshot counter from restart file
            if verbose:
                print(f"✓ Resuming from step {start_step}, time {time:.6e}")
    else:
        xv = phase_space.copy()  # Ensure we have a local copy to modify
    
    # Compute steps reliably (use round to avoid off-by-one)
    total_steps = int(round((time_end - time_start) / dt))
    remaining_steps = total_steps - start_step
        
    # Compute snapshot steps (evenly spaced from 0 to total_steps)
    if snapshots > 1:
        snapshot_steps = np.round(np.linspace(0, total_steps, snapshots)).astype(int)
    else:
        snapshot_steps = np.array([total_steps], dtype=int)

     # If not resuming from restart, initialize snapshot_counter from start_step
    if snapshot_counter is None:
        # For a fresh run, start at 0
        # For a resumed run, count how many snapshots should have been written already
        snapshot_counter = int(np.searchsorted(snapshot_steps, start_step, side="left"))  

    if verbose:
        print("="*80)
        print("CPU N-body Integration")
        print("="*80)
        print(f"Particles: {N:,}")
        print(f"Time: {time_start:.3e} → {time_end:.3e} (dt={dt:.3e})")
        print(f"Steps: {total_steps:,} ({remaining_steps:,} remaining)")
        print(f"Kernel: {kernel}, Softening: {softening if np.isscalar(softening) else 'variable'}")
        print(f"External potential: {'Yes' if external_potential is not None else 'No'}")
        if external_potential is not None:
            print(f"  Update interval: every {external_update_interval} steps")
        print(f"Snapshots: {snapshots} (every ~{total_steps//snapshots:,} steps)")
        print(f"Restart files: every {restart_interval} steps")
        print("="*80)
    
    # Initial acceleration (with warmup)
    if verbose:
        print("Computing initial forces (compiling NUMBA kernel)...")
    
    # Select acceleration computation method
    if method == 'tree':
        if verbose: print(f"Using tree method (theta={theta}, kernel={kernel})")
        compute_acc = lambda pos, t: _compute_accelerations_tree(
            pos, masses, softening, theta, kernel, G, external_potential, t
        )
    else:  # method == 'direct'
        if verbose: print(f"Using direct method (kernel={kernel}, nthreads={nthreads})")
        compute_acc = lambda pos, t: _compute_accelerations_cpu(
            pos, masses, softening, G, kernel, nthreads, external_potential, t
        )
    
    # Initial acceleration
    acc_cpu = compute_acc(xv[:, :3], time)
    
    # Warmup complete, start timing
    if verbose:
        print("\nStarting integration...")
    
    # Save initial snapshot if requested (global index at snapshot_steps[snapshot_counter] == start_step)
    if snapshot_counter < len(snapshot_steps) and snapshot_steps[snapshot_counter] == start_step:
        if verbose:
            print(f"writing snapshot snap: {snapshot_counter} at step {start_step}, time {time:.6e}...")
        _save_snapshot(xv, snapshot_counter, time, output_path,
                       mass_dark=masses[0],  # no dark mass in this code path
                       num_files_to_write=num_files_to_write,
                       total_expected_snapshots=snapshots)
        _update_snapshot_times(output_path, snapshot_counter, time)
        snapshot_counter += 1
    
    # ============================================================================
    # Main integration loop: iterate over the remaining steps (1..remaining_steps)
    # ============================================================================
    t_start = pytime.perf_counter()
    for step_i in range(1, remaining_steps + 1):
        current_step = start_step + step_i

        # Kick-Drift-Kick leapfrog
        
        # Kick (half-step)
        xv[:, 3:6] += acc_cpu * (dt / 2)  
        
        # Drift (full-step)
        xv[:, 0:3] += xv[:, 3:6] * dt  
        
        # Update time before force computation
        time += dt  

        # Recompute accelerations at new positions and time
        acc_cpu = compute_acc(xv[:, :3], time)
        
        # Kick (half-step)
        xv[:, 3:6] += acc_cpu * (dt / 2)  
        
        # === I/O Operations ===
        # === snapshots (handle possibly multiple snapshot indices that fall here) ===
        while snapshot_counter < len(snapshot_steps) and current_step >= snapshot_steps[snapshot_counter]:
            _save_snapshot(xv, snapshot_counter, time, output_path,
                           mass_dark=masses[0],  # no dark mass in this code path
                           num_files_to_write=num_files_to_write,
                           total_expected_snapshots=snapshots)
            _update_snapshot_times(output_path, snapshot_counter, time)
            if verbose:
                print(f"Saved snapshot id={snapshot_counter:03d} at step {current_step}, time {time:.6e}...")
            
            snapshot_counter += 1

        # Progress update
        if verbose and step_i % max(1, remaining_steps // 50) == 0:
            elapsed = pytime.perf_counter() - t_start
            rate = step_i / elapsed if elapsed > 0 else 0
            eta = (remaining_steps - step_i) / rate if rate > 0 else 0
            avg_step_time = elapsed / step_i if step_i > 0 else 0
            print(f"  Step {current_step:>6}/{total_steps} | "
                f"t={time:.4e} | "
                f"Snapshots: {snapshot_counter}/{len(snapshot_steps)} | "
                f"{rate:.1f} steps/s | "
                f"avg {avg_step_time*1000:.1f}ms/step | "
                f"ETA {eta:.0f}s")
                
        # Save restart file
        if current_step > 0 and (current_step % restart_interval) == 0:
            _save_restart(xv, time, current_step, output_path, snapshot_counter)

    # =============================================================================
    # End of integration loop
    # =============================================================================
    
     # ensure final snapshot saved (if last snapshot maps to total_steps and wasn't saved)
    if snapshot_counter < len(snapshot_steps) and snapshot_steps[-1] == total_steps:
        if verbose:
            print(f"<=Saving final snapshot snap.{snapshot_counter:03d} at step {total_steps}, time {time:.6e}=>")
        _save_snapshot(xv, snapshot_counter, time, output_path,
                       mass_dark=masses[0],  # no dark mass in this code path
                       num_files_to_write=num_files_to_write,
                       total_expected_snapshots=snapshots)
        _update_snapshot_times(output_path, snapshot_counter, time)
        snapshot_counter += 1
    
    # when saving final restart (optional), include snapshot_counter
    _save_restart(xv, time, total_steps, output_path, snapshot_counter)

    if verbose:
        t_end = pytime.perf_counter()
        total_time = t_end - t_start
        avg_step_time = total_time / remaining_steps if remaining_steps > 0 else 0
        
        print("\n" + "="*80)
        print("Integration Complete")
        print("="*80)
        print(f"Final time: {time:.6e}")
        print(f"Total wall time: {total_time:.2f} s")
        print(f"Steps per second: {remaining_steps / total_time:.1f}")
        print(f"Average step time: {avg_step_time*1000:.1f} ms")
        print(f"Snapshots saved: {snapshot_counter}")
        print("="*80)
    
    # Return final state
    return xv

# ============================================================================
# INITIAL CONDITIONS GENERATORS
# ============================================================================

def make_plummer_sphere(
    N: int,
    M_total: float = 1.0,
    a: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Plummer sphere in virial equilibrium.
    
    Parameters
    ----------
    N : int
        Number of particles.
    M_total : float
        Total mass.
    a : float
        Plummer scale radius.
    seed : int
        Random seed.
    
    Returns
    -------
    phase_space : np.ndarray, shape (N, 6)
        Positions and velocities.
    masses : np.ndarray, shape (N,)
        Particle masses (equal mass).
    """
    rng = np.random.default_rng(seed)
    
    # Sample radii from Plummer profile
    u = rng.random(N)
    r = a / np.sqrt(u**(-2/3) - 1)
    
    # Isotropic angles
    theta = np.arccos(2 * rng.random(N) - 1)
    phi = 2 * np.pi * rng.random(N)
    
    # Positions
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Velocities from distribution function (simplified)
    G = 1.0  # N-body units
    v_esc = np.sqrt(2 * G * M_total / np.sqrt(r**2 + a**2))
    
    v_mag = np.zeros(N)
    for i in range(N):
        q = 0.0
        while True:
            q = rng.random()
            g = rng.random()
            if g < q**2 * (1 - q**2)**3.5:
                break
        v_mag[i] = q * v_esc[i]
    
    # Isotropic velocity directions
    theta_v = np.arccos(2 * rng.random(N) - 1)
    phi_v = 2 * np.pi * rng.random(N)
    
    vx = v_mag * np.sin(theta_v) * np.cos(phi_v)
    vy = v_mag * np.sin(theta_v) * np.sin(phi_v)
    vz = v_mag * np.cos(theta_v)
    
    phase_space = np.column_stack([x, y, z, vx, vy, vz])
    masses = np.full(N, M_total / N, dtype=np.float32)
    
    return phase_space, masses


def place_on_orbit(
    phase_space: np.ndarray,
    r_peri: float,
    r_apo: float,
    potential: 'agama.Potential',
) -> np.ndarray:
    """
    Place system on orbit in external potential.
    
    Parameters
    ----------
    phase_space : np.ndarray, shape (N, 6)
        System in its rest frame.
    r_peri : float
        Pericenter radius.
    r_apo : float
        Apocenter radius.
    potential : agama.Potential
        External potential.
    
    Returns
    -------
    np.ndarray, shape (N, 6)
        System shifted to orbit.
    """
    # Start at apocenter (x-axis)
    r_center = r_apo
    
    # Compute circular velocity at geometric mean radius
    r_circ = np.sqrt(r_peri * r_apo)
    v_circ = (-r_circ * potential.force(np.array([[r_circ, 0, 0]]))[0, 0])**0.5
    
    # Tangential velocity at apocenter (energy/angular momentum match)
    v_tang = v_circ * np.sqrt(2 * r_circ / r_apo - 1)
    
    # Shift system
    xv_orbit = phase_space.copy()
    xv_orbit[:, 0] += r_center  # Shift x
    xv_orbit[:, 4] += v_tang    # Add tangential velocity (y-direction)
    
    return xv_orbit

__all__ = [
    'run_nbody_gpu',
    'run_nbody_cpu',
    'G_DEFAULT',
    'NBODY_UNITS',
]

if __name__ == "__main__":
    print("="*80)
    print("GPU N-body Integration - Test Run")
    print("="*80)
    
    # Test 1: Plummer sphere in isolation
    print("\n### Test 1: Plummer sphere (self-gravity only) ###\n")
    
    N = 10_000
    xv, masses = make_plummer_sphere(N, M_total=1e6, a=1.0)
    
    final = run_nbody_gpu(
        phase_space=xv,
        masses=masses,
        time_start=0.0,
        time_end=0.01,
        dt=1e-4,
        softening=0.01,
        G=1.0,
        kernel='spline',
        snapshots=5,
        restart_interval=50,
        output_dir="./test_plummer",
        verbose=True,
    )
    
    print(f"\nFinal state check:")
    print(f"  COM position: {final[:, :3].mean(axis=0)}")
    print(f"  COM velocity: {final[:, 3:6].mean(axis=0)}")
    print(f"  RMS position: {np.std(final[:, :3]):.3f}")
    print(f"  RMS velocity: {np.std(final[:, 3:6]):.3f}")
    
    # Test 2: With external potential (if Agama available)
    if AGAMA_AVAILABLE:
        print("\n### Test 2: Plummer with external NFW (on orbit) ###\n")
        
        # External NFW potential
        agama.setUnits(mass=1, length=1, velocity=1)
        pot_nfw = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0) 
        
        # Simple orbit: place cluster at (30, 0, 0) with tangential velocity
        xv_orbit = xv.copy()
        xv_orbit[:, 0] += 30.0  # Shift 30 kpc in x
        xv_orbit[:, 4] += 150.0  # Add 150 km/s in y (tangential)
        
        print(f"Initial orbit state:")
        print(f"  COM position: {xv_orbit[:, :3].mean(axis=0)}")
        print(f"  COM velocity: {xv_orbit[:, 3:6].mean(axis=0)}")
        
        final_orbit = run_nbody_gpu(    
            phase_space=xv_orbit,
            masses=masses,
            time_start=0.0,
            time_end=0.01,
            dt=1e-4,
            softening=0.01,
            G=1.0,
            kernel='spline',
            external_potential=pot_nfw,
            external_update_interval=1,  # Update external forces every 5 steps
            snapshots=5,
            restart_interval=50,
            output_dir="./test_orbit",
            verbose=True,
        )
        
        print(f"\nFinal orbit state:")
        print(f"  COM position: {final_orbit[:, :3].mean(axis=0)}")
        print(f"  COM velocity: {final_orbit[:, 3:6].mean(axis=0)}")
        print(f"  COM displacement: {np.linalg.norm(final_orbit[:, :3].mean(axis=0) - xv_orbit[:, :3].mean(axis=0)):.3f} kpc")

    print("="*80)
    print("CPU N-body Integration - Direct Run")
    print("="*80)
    
    N = 10_000
    xv, masses = make_plummer_sphere(N, M_total=1e6, a=1.0)
    
    final = run_nbody_cpu(
        phase_space=xv,
        masses=masses,
        time_start=0.0,
        time_end=0.01,
        dt=1e-4,
        softening=0.01,
        G=1.0,
        kernel='spline',
        method='direct',
        snapshots=5,
        restart_interval=50,
        output_dir="./tests_nbody/test_plummer_cpu",
        verbose=True,
    )
    
    print(f"\nFinal state check:")
    print(f"  COM position: {final[:, :3].mean(axis=0)}")
    print(f"  COM velocity: {final[:, 3:6].mean(axis=0)}")
    print(f"  RMS position: {np.std(final[:, :3]):.3f}")
    print(f"  RMS velocity: {np.std(final[:, 3:6]):.3f}")
    
    print("\n" + "="*80)
    print("Tests complete! ✓")
    print("="*80)
