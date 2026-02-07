#!/usr/bin/env python3
"""
GPU-accelerated N-body gravitational fields computation using CuPy.

This module provides high-performance direct N-body calculation on NVIDIA GPUs
using raw CUDA kernels compiled with nvcc. Achieves 50-100x speedup over Numba parallelized
CPU implementations for typical particle counts (10K-100K).

Requirements
------------
- NVIDIA GPU with CUDA support (compute capability >= 6.0)
- CuPy: https://cupy.dev/
- NumPy

Examples
--------
Basic usage with default parameters:

>>> import numpy as np
>>> from nbody_forces_gpu import compute_nbody_forces_gpu
>>> 
>>> # Generate random particle distribution
>>> N = 20000
>>> pos = np.random.randn(N, 3).astype(np.float32)
>>> mass = np.ones(N, dtype=np.float32)
>>> 
>>> # Compute gravitational accelerations on GPU
>>> acc = compute_nbody_forces_gpu(pos, mass, softening=0.01)
>>> print(f"Acceleration shape: {acc.shape}")  # (20000, 3)

With different softening kernels:

>>> # Plummer softening
>>> acc = compute_nbody_forces_gpu(pos, mass, softening=0.01, kernel='plummer')
>>> 
>>> # Dehnen kernel (falcON default)
>>> acc = compute_nbody_forces_gpu(pos, mass, softening=0.01, kernel='dehnen_k1')

Variable softening per particle:

>>> softening_array = np.linspace(0.005, 0.02, N)
>>> acc = compute_nbody_forces_gpu(pos, mass, softening=softening_array)

Notes
-----
- All computations use single precision (float32) for optimal GPU performance
- Input arrays are automatically converted to float32 if needed
- The function handles memory transfers to/from GPU internally
- For repeated calls, consider keeping data on GPU (see advanced usage)

Author: Arpit Arora
Date: Sept 2025
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Literal, Union
import warnings
from nbody_cuda_kernels import *  # Import kernel templates

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn(
        "CuPy not available. GPU acceleration disabled. "
        "Install with: pip install cupy-cudaxxx",
        ImportWarning
    )

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. CPU fallback disabled.", ImportWarning)

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

# ============================================================================
# NUMBA CPU VERSION (Parallelized fallback)
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True, inline='always')
    def _get_force_kernel(r2: float, h: float, kernel_id: int) -> float:
        """
        Selectable force kernel function.
        Returns the 1/r^3 equivalent factor for the force calculation.

        kernel_id:
        0 = Newtonian (regularized)
        1 = Plummer
        2 = Spline (Monaghan 1992)
        3 = Dehnen k=1 (falcON default)
        4 = Dehnen k=2
        """
        # --- 0: Newtonian (with tiny regularization for safety) ---
        if kernel_id == 0:
            return 1.0 / (r2* np.sqrt(r2))

        # --- 1: Plummer Kernel ---
        if kernel_id == 1:
            denom = r2 + h * h
            return 1.0 / (denom * np.sqrt(denom))
    
        # --- 2: Dehnen k=1 (C2 correction, falcON default) ---
        # Pot1 = [(x^2 + h^2)^(-1/2) + 0.5*h^2*(x^2 + h^2)^(-3/2)] 
        # kernel_force = -dPot/dx = x * [(x^2 + h^2)^(-3/2) + (3/2)*h^2*(x^2 + h^2)^(-5/2)].
        # Remember the kernel is 1/x of that for 1/L^3 units.
        if kernel_id == 2:  # P1
            denom = r2 + h * h
            sqrt_denom = np.sqrt(denom)
            term1 = 1.0 / (denom * sqrt_denom)           # 1/(r²+h²)^(3/2) - same as Plummer
            term2 = 1.5 * h * h / (denom * denom * sqrt_denom)  # 1.5*h²/(r²+h²)^(5/2)
            return term1 + term2
    
        # --- 3: Dehnen k=2 (C4 correction) ---
        # Pot1 = [(x^2 + h^2)^(-1/2) + (1/2)*h^2*(x^2 + h^2)^(-3/2) + (3/4)*h^4*(x^2 + h^2)^(-5/2)] 
        # kernel_force = -dPot/dx = x * [(x^2 + h^2)^(-3/2) + (3/2)*h^2*(x^2 + h^2)^(-5/2) + (15/4)*h^4*(x^2 + h^2)^(-7/2)].
        if kernel_id == 3:  # P2
            denom = r2 + h * h
            sqrt_denom = np.sqrt(denom)
            hh = h * h
            term1 = 1.0 / (denom * sqrt_denom)                    # base term
            term2 = 1.5 * hh / (denom * denom * sqrt_denom)       # h² correction
            term3 = 3.75 * hh * hh / (denom * denom * denom * sqrt_denom)  # h⁴ correction
            return term1 + term2 + term3
        
        r = np.sqrt(r2)    

        # --- Below are all compact kernels. ---
        # For all compact-support kernels, the force is Newtonian for r >= h
        if r > h:
            return 1.0 / (r * r * r)

        hinv = 1.0 / h
        h3inv = hinv * hinv * hinv
        q = r * hinv
        q2 = q * q
        
        # --- 4: Spline Kernel --- (Monaghan 1992)
        if kernel_id == 4:
            if q <= 0.5:
                return h3inv * (10.666666666666666 + q2 * (-38.4 + 32.0 * q))
            else:
                return h3inv * (21.333333333333333 - 48.0 * q + 38.4 * q2
                            - 10.666666666666667 * q2 * q
                            - 0.06666666666666667 / (q2 * q))
                
        # Fallback for safety, though the wrapper should prevent this
        return 0.0

    @njit(parallel=True, fastmath=True, cache=True)
    def _compute_forces_cpu(pos: np.ndarray, mass: np.ndarray, h: np.ndarray, 
                            kernel_id: int, r_eps: float) -> np.ndarray:
        """
        CPU parallelized force computation with Numba.
        
        Parameters
        ----------
        pos : np.ndarray, shape (N, 3)
            Particle positions.
        mass : np.ndarray, shape (N,)
            Particle masses.
        h : np.ndarray, shape (N,)
            Softening lengths for each particle.
        kernel_id : int
            Identifier for the gravitational softening kernel to use.
            Maps to a specific kernel function (e.g., Plummer, spline, etc.).
        r_eps : float
            Small value added to distances to avoid divide-by-zero errors.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Computed accelerations for each particle in Cartesian coordinates.
        """
        
        N = pos.shape[0]
        forces = np.zeros((N, 3), dtype=pos.dtype)

        for i in prange(N):
            fx, fy, fz = 0.0, 0.0, 0.0
            xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
            hi = h[i]

            for j in range(N):
                if i == j:
                    continue
                
                # cache mass and h for j to reduce indexing.
                mj = mass[j]
                hj = h[j]
                
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                dz = pos[j, 2] - zi

                r2 = dx*dx + dy*dy + dz*dz + r_eps*r_eps
                            
                # The softening length h_ij is the max of the two particles
                # combine softening lengths
                h_ij = hi if hi >= hj else hj

                # Get the force factor from our unified kernel function
                kernel_val = _get_force_kernel(r2, h_ij, kernel_id)

                factor = mj * kernel_val

                fx += factor * dx
                fy += factor * dy
                fz += factor * dz

            forces[i, 0] = fx
            forces[i, 1] = fy
            forces[i, 2] = fz

        return forces
    
    @njit(fastmath=True, cache=True, inline='always')
    def _get_potential_kernel(r2: float, h: float, kernel_id: int) -> float:
        """
        CPU potential kernel matching GPU version.
        Returns -1/r equivalent for potential Φ(r).
        """
        r = np.sqrt(r2)
        
        if kernel_id == 0:  # Newtonian
            return -1.0 / r if r > 0 else 0.0
        
        if kernel_id == 1:  # Plummer
            return -1.0 / np.sqrt(r2 + h * h)
        
        if kernel_id == 2:  # Dehnen k=1
            h2 = h * h
            denom = r2 + h2
            sqrt_denom = np.sqrt(denom)
            inv_sqrt = 1.0 / sqrt_denom
            inv_d32 = inv_sqrt / denom
            return -inv_sqrt - 0.5 * h2 * inv_d32
        
        if kernel_id == 3:  # Dehnen k=2
            h2 = h * h
            h4 = h2 * h2
            denom = r2 + h2
            sqrt_denom = np.sqrt(denom)
            inv_sqrt = 1.0 / sqrt_denom
            inv_d32 = inv_sqrt / denom
            inv_d52 = inv_d32 / sqrt_denom
            return -inv_sqrt - 0.5 * h2 * inv_d32 - 0.375 * h4 * inv_d52
        
        if kernel_id == 4:  # Spline
            if h == 0.0 or r >= h:
                return -1.0 / r
            
            hinv = 1.0 / h
            q = r * hinv
            
            if q < 1e-8:
                return -2.8 * hinv
            
            q2 = q * q
            
            if q <= 0.5:
                return (-2.8 + q2 * (5.33333333333333333 + q2 * q2 * (6.4 * q - 9.6))) * hinv
            
            if q <= 1.0:
                return (
                    -3.2
                    + 0.066666666666666666666 / q
                    + q2 * (10.666666666666666666666 + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q)))
                ) * hinv
            
            return -1.0 / r
        
        return 0.0
    
    
    @njit(parallel=True, fastmath=True, cache=True)
    def _compute_potential_cpu(pos: np.ndarray, mass: np.ndarray, h: np.ndarray,
                               kernel_id: int, r_eps: float) -> np.ndarray:
        """
        CPU parallelized potential computation with Numba.
        
        Parameters
        ----------
        pos : np.ndarray, shape (N, 3)
            Particle positions.
        mass : np.ndarray, shape (N,)
            Particle masses.
        h : np.ndarray, shape (N,)
            Softening lengths.
        kernel_id : int
            Kernel type identifier.
        r_eps : float
            Regularization parameter.
        
        Returns
        -------
        np.ndarray, shape (N,)
            Potential at each particle.
        """
        N = pos.shape[0]
        potential = np.zeros(N, dtype=pos.dtype)
        eps2 = r_eps * r_eps
        
        for i in prange(N):
            xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
            hi = h[i]
            pot_i = 0.0
            
            for j in range(N):
                if i == j:
                    continue
                
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                dz = pos[j, 2] - zi
                
                r2 = dx * dx + dy * dy + dz * dz + eps2
                
                # Effective softening: max of pair
                hj = h[j]
                h_eff = hi if hi >= hj else hj
                
                kernel_val = _get_potential_kernel(r2, h_eff, kernel_id)
                pot_i += mass[j] * kernel_val
            
            potential[i] = pot_i
        
        return potential

# ============================================================================
# CUDA KERNEL MANAGEMENT - UNIFIED
# ============================================================================

# Type specifications for float vs double
_TYPE_SPECS = {
    'float32': {
        'T': 'float', 
        'RSQRT': 'rsqrtf', 
        'SQRT': 'sqrtf', 
        'FMA': 'fmaf', 
        'FMAX': 'fmaxf'
    },
    'float64': {
        'T': 'double', 
        'RSQRT': 'rsqrt', 
        'SQRT': 'sqrt', 
        'FMA': 'fma', 
        'FMAX': 'fmax'
    },
}

# Kernel cache - stores compiled kernels
_NBODY_KERNEL_CACHE = {}

# Template and kernel name mapping
_NBODY_KERNEL_CONFIG = {
    # Forces kernels
    ('force', 'float32', False): (_NBODY_KERNEL_TEMPLATE, 'nbody_forces_kernel'),
    ('force', 'float64', False): (_NBODY_KERNEL_TEMPLATE, 'nbody_forces_kernel'),
    ('force', 'float32', True):  (_KAHAN_KERNEL_TEMPLATE, 'nbody_forces_kahan_kernel'),
    ('force', 'float64', True):  (_KAHAN_KERNEL_TEMPLATE, 'nbody_forces_kahan_kernel'),
    
    # Potential kernels
    ('potential', 'float32', False): (_POTENTIAL_KERNEL_TEMPLATE, 'nbody_potential_kernel'),
    ('potential', 'float64', False): (_POTENTIAL_KERNEL_TEMPLATE, 'nbody_potential_kernel'),
    ('potential', 'float32', True):  (_POTENTIAL_KAHAN_KERNEL_TEMPLATE, 'nbody_potential_kahan_kernel'),
    ('potential', 'float64', True):  (_POTENTIAL_KAHAN_KERNEL_TEMPLATE, 'nbody_potential_kahan_kernel'),
}


def _get_kernel(kernel_type='force', precision='float64', use_kahan=False):
    """
    Get compiled CUDA kernel for forces or potential.
    
    Parameters
    ----------
    kernel_type : str
        Either 'force' or 'potential'
    precision : str
        Either 'float32' or 'float64' (or 'float32_kahan' for backwards compat)
    use_kahan : bool
        Whether to use Kahan summation. Ignored if precision='float32_kahan'
    
    Returns
    -------
    kernel : cp.RawKernel
        Compiled CUDA kernel
    """
    # Handle legacy 'float32_kahan' precision string
    if precision == 'float32_kahan':
        precision = 'float32'
        use_kahan = True
    
    # Validate inputs
    if kernel_type not in ['force', 'potential']:
        raise ValueError(f"kernel_type must be 'force' or 'potential', got {kernel_type}")
    if precision not in ['float32', 'float64']:
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision}")
    
    # Create cache key
    cache_key = f"{kernel_type}_{precision}{'_kahan' if use_kahan else ''}"
    
    # Check cache
    if cache_key in _NBODY_KERNEL_CACHE:
        return _NBODY_KERNEL_CACHE[cache_key]
    
    # Get template and kernel name
    config_key = (kernel_type, precision, use_kahan)
    if config_key not in _NBODY_KERNEL_CONFIG:

        raise ValueError(f"No kernel configuration for {config_key}")
    
    template, kernel_name = _NBODY_KERNEL_CONFIG[config_key]
    
    # Get type specs
    type_specs = _TYPE_SPECS[precision]
    
    # Format template with type specs
    source = template.format(**type_specs)
    
    # Compile kernel
    kernel = cp.RawKernel(source, kernel_name, backend='nvcc')
    
    # Cache and return
    _NBODY_KERNEL_CACHE[cache_key] = kernel
    return kernel

def _input_validation_allocation_gpu(pos, mass, softening, precision, kernel):
    """Helper function to validate inputs and allocate GPU arrays for computations."""

    if not CUPY_AVAILABLE:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install cupy-cudaxxx"
        )
    
    # precision mapping - default to float32 for GPU performance, but allow float64 for accuracy if needed
    prec_key = precision.lower()
    if prec_key not in _PRECISION_MAP:
        raise ValueError(f"Precision must be one of {list(_PRECISION_MAP.keys())}")     
    
    dtype, dtype_np = _PRECISION_MAP[prec_key]

    # Validate and convert inputs - handle both NumPy and CuPy
    if isinstance(pos, cp.ndarray):
        pos = pos.astype(dtype, copy=False)  # Keep on GPU
    else:
        pos = np.asarray(pos, dtype=dtype_np)
        pos = cp.asarray(pos)  # Transfer to GPU
    
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
    
    N = pos.shape[0]
    
    # Handle mass - keep on GPU
    if np.isscalar(mass):
        mass_gpu = cp.full(N, mass, dtype=dtype)
    else:
        if isinstance(mass, cp.ndarray):
            mass_gpu = mass.astype(dtype, copy=False)
        else:
            mass_arr = np.asarray(mass, dtype=dtype_np)
            mass_gpu = cp.asarray(mass_arr)
        
        if mass_gpu.shape[0] != N:
            raise ValueError(f"mass must have length N={N}, got {mass_gpu.shape[0]}")
    
    # Handle softening - keep on GPU
    if np.isscalar(softening):
        h_gpu = cp.full(N, softening, dtype=dtype)
    else:
        if isinstance(softening, cp.ndarray):
            h_gpu = softening.astype(dtype, copy=False)
        else:
            h_arr = np.asarray(softening, dtype=dtype_np)
            h_gpu = cp.asarray(h_arr)
        
        if h_gpu.shape[0] != N:
            raise ValueError(f"softening must have length N={N}, got {h_gpu.shape[0]}")
    
    # Map kernel name to ID
    kernel_map = {
        'newtonian': 0,
        'plummer': 1,
        'dehnen_k1': 2,
        'dehnen_k2': 3,
        'spline': 4,
    }
    
    if kernel.lower() not in kernel_map:
        raise ValueError(
            f"Invalid kernel '{kernel}'. Must be one of: {list(kernel_map.keys())}"
        )
    
    kernel_id = kernel_map[kernel.lower()]
    
    # Transfer to GPU (Structure of Arrays layout for coalesced memory access)
    x_gpu = cp.ascontiguousarray(pos[:, 0])
    y_gpu = cp.ascontiguousarray(pos[:, 1])
    z_gpu = cp.ascontiguousarray(pos[:, 2])

    return x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, dtype, dtype_np, kernel_id, N

# ============================================================================
# PUBLIC API
# ============================================================================

def compute_nbody_forces_gpu(
    pos: ArrayLike,
    mass: ArrayLike | float,
    softening: float | ArrayLike = 0.0,
    G: float = G_DEFAULT,
    precision: Literal['float32', 'float64', 'float32_kahan'] = 'float32_kahan',
    kernel: KERNEL_TYPES = 'spline',
    return_cupy: bool = False
) -> NDArray:
    """
    Compute direct N-body gravitational forces using GPU acceleration.
    
    Calculates pairwise gravitational interactions between all particles using
    CUDA-accelerated computation. Supports multiple softening kernels and achieves
    50-100x speedup over Numba parallelized CPU implementations.
    
    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions in Cartesian coordinates. Will be converted to float32.
    mass : array_like, shape (N,) or scalar
        Particle masses. If scalar, all particles have equal mass.
        Will be converted to float32.
    softening : float or array_like, shape (N,), optional
        Gravitational softening length(s). If scalar, uniform softening is applied.
        If array, per-particle softening is used (effective softening is max of pair).
        Default is 0.0 (pure Newtonian with small regularization).
    G : float, optional
        Gravitational constant. Default units of
        [kpc, km/s, Msun]. Use 1.0 for N-body units.
    precision : {'float32', 'float64', 'float32_kahan'}, optional
        Floating point precision for computation. Default is 'float64' for maximum accuracy.
        'float32_kahan' uses Kahan summation for improved accuracy in float32 mode.
    kernel : {'newtonian', 'plummer', 'dehnen_k1', 'dehnen_k2', 'spline'}, optional
        Softening kernel type:
        
        - 'newtonian': Pure 1/r^2 gravity with regularization (default for softening=0)
        - 'plummer': Plummer softening, 1/(r^2 + h^2)^(3/2)
        - 'dehnen_k1': Dehnen P1 kernel with C2 continuity (falcON default)
        - 'dehnen_k2': Dehnen P2 kernel with C4 continuity
        - 'spline': Cubic spline kernel (Monaghan 1992), compact support at h
        
        Default is 'spline'.
    return_cupy : bool, optional
        If True, return CuPy array (data stays on GPU). If False (default),
        return NumPy array (data copied to CPU). Use True for chaining GPU operations.
    
    Returns
    -------
    accelerations : ndarray, shape (N, 3)
        Gravitational accelerations for each particle in Cartesian coordinates.
        Returned as NumPy array (CPU) by default, or CuPy array (GPU) if return_cupy=True.
        dtype is picked by precision argument (float32 or float64).
    
    Raises
    ------
    ImportError
        If CuPy is not installed.
    ValueError
        If input arrays have incorrect shapes or types.
    RuntimeError
        If CUDA error occurs during computation.
    
    Notes
    -----
    - All computations use single precision (float32) for optimal GPU performance
    - Complexity is O(N^2), suitable for N ~ 10K-100K particles
    - For N > 100K, consider tree codes (Barnes-Hut) or FMM methods
    - 1M particles typically takes several seconds on RTX3080 GPU
    - Memory requirement: ~40 bytes per particle on GPU
    - Self-interactions are automatically excluded (particle i does not feel force from itself)
    
    Performance
    -----------
    Typical performance on NVIDIA RTX 3080 Laptop GPU:
    
    - N=10K:  ~8ms  (125 Gint/s, ~125 steps/sec)
    - N=20K:  ~13ms (124 Gint/s, ~77 steps/sec)
    - N=40K:  ~26ms (123 Gint/s, ~38 steps/sec)
    - N=80K:  ~51ms (124 Gint/s, ~20 steps/sec)
    
    Scales approximately as O(N^2) for force computation.
    
    Examples
    --------
    Basic usage with uniform mass and softening:
    
    >>> import numpy as np
    >>> pos = np.random.randn(10000, 3).astype(np.float32)
    >>> mass = 1.0  # Solar masses
    >>> acc = compute_nbody_forces_gpu(pos, mass, softening=0.01)
    
    Variable mass distribution:
    
    >>> mass = np.random.lognormal(0, 1, size=10000).astype(np.float32)
    >>> acc = compute_nbody_forces_gpu(pos, mass, softening=0.01, kernel='plummer')
    
    Per-particle softening (adaptive softening):
    
    >>> softening = np.linspace(0.005, 0.02, 10000).astype(np.float32)
    >>> acc = compute_nbody_forces_gpu(pos, mass, softening=softening)
    
    Keep data on GPU for integration loop:
    
    >>> acc_gpu = compute_nbody_forces_gpu(pos, mass, softening=0.01, return_cupy=True)
    >>> # acc_gpu is CuPy array, stays on GPU for further operations
    
    See Also
    --------
    numpy.linalg.norm : Compute magnitudes of acceleration vectors
    
    References
    ----------
    .. [1] Dehnen, W. (2001). "Towards optimal softening in three-dimensional
           N-body codes - I. Minimizing the force error." MNRAS, 324, 273-291.
    .. [2] Monaghan, J. J. (1992). "Smoothed particle hydrodynamics."
           ARA&A, 30, 543-574.
    """

    # 1. Validate inputs and allocate GPU arrays
    x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, dtype, dtype_np, kernel_id, N = (
        _input_validation_allocation_gpu(
            pos,
            mass,
            softening,
            precision,
            kernel,
        )
    )
       
    # Allocate output arrays on GPU
    ax_gpu = cp.empty(N, dtype=dtype)
    ay_gpu = cp.empty(N, dtype=dtype)
    az_gpu = cp.empty(N, dtype=dtype)

    # 2. Get the right kernel (it compiles only once per precision type)
    _nbody_force_kernel = _get_kernel('force', precision, use_kahan=(precision=='float32_kahan'))

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    eps2 = 1e-16  # Regularization parameter
    
    _nbody_force_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, ax_gpu, ay_gpu, az_gpu,
         dtype(G), dtype(eps2), cp.int32(kernel_id), cp.int32(N))
    )
    
    acc_gpu = cp.empty((N, 3), dtype=dtype)
    acc_gpu[:, 0] = ax_gpu
    acc_gpu[:, 1] = ay_gpu
    acc_gpu[:, 2] = az_gpu
    # Convert back to Array of Structures layout
    if return_cupy:    
        return acc_gpu
    else:
        return cp.asnumpy(acc_gpu)

def compute_nbody_potential_gpu(
    pos: ArrayLike,
    mass: ArrayLike | float,
    softening: float | ArrayLike = 0.0,
    G: float = G_DEFAULT,
    precision: Literal['float32', 'float64', 'float32_kahan'] = 'float32_kahan',
    kernel: KERNEL_TYPES = 'spline',
    return_cupy: bool = False
) -> NDArray[np.float32]:
    """
    Compute gravitational potential at each particle location on GPU.
    
    Calculates Φ(r_i) = G * Σ_j m_j * kernel(|r_i - r_j|) for all particles.
    Useful for energy diagnostics and post-processing. Self-potential is
    automatically excluded.
    
    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    mass : array_like, shape (N,) or scalar
        Particle masses.
    softening : float or array_like, shape (N,), optional
        Softening length(s). Default: 0.0.
    G : float, optional
        Gravitational constant. Default: 4.30092e-6 (kpc, km/s, Msun).
    precision : {'float32', 'float64', 'float32_kahan'}, optional
        Floating point precision for computation. Default is 'float64' for maximum accuracy.
        'float32_kahan' uses Kahan summation for improved accuracy in float32 mode.
    kernel : str, optional
        Potential kernel: 'newtonian', 'plummer', 'dehnen_k1', 'dehnen_k2', 'spline'.
        Default: 'spline'.
    return_cupy : bool, optional
        Return CuPy array (GPU) if True, else NumPy array (CPU). Default: False.
    
    Returns
    -------
    potentials : ndarray, shape (N,)
        Gravitational potential at each particle. Dtype float32.
    
    Raises
    ------
    ImportError
        If CuPy not available.
    
    Notes
    -----
    - Self-potential excluded: Φ_i does not include contribution from particle i itself
    - Total potential energy: PE = 0.5 * sum(m_i * Φ_i) [factor 0.5 avoids double counting]
    - Faster than force calculation (~20-30% faster for same N)
    - Same O(N²) complexity
    
    Performance
    -----------
    Slightly faster than force computation (~20-30% speedup):
    
    - N=10K:  ~6ms  (vs ~8ms for forces)
    - N=20K:  ~10ms (vs ~13ms for forces)
    - N=80K:  ~40ms (vs ~51ms for forces)
    
    Examples
    --------
    Compute potential for energy diagnostics:
    
    >>> import numpy as np
    >>> pos = np.random.randn(10000, 3).astype(np.float32)
    >>> mass = np.ones(10000, dtype=np.float32)
    >>> pot = compute_nbody_potential_gpu(pos, mass, softening=0.01)
    >>> 
    >>> # Total potential energy
    >>> PE = 0.5 * np.sum(mass * pot)
    
    With Dehnen kernel (matching falcON):
    
    >>> pot = compute_nbody_potential_gpu(pos, mass, softening=0.01, kernel='dehnen_k1')
    
    See Also
    --------
    compute_nbody_potential_cpu : CPU fallback version
    nbody_forces_gpu.compute_nbody_forces_gpu : Compute forces (accelerations)
    """
    
    # 1. Validate inputs and allocate GPU arrays
    x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, dtype, dtype_np, kernel_id, N = (
        _input_validation_allocation_gpu(
            pos,
            mass,
            softening,
            precision,
            kernel,
        )
    )
   
    # Allocate output arrays on GPU
    pot_gpu = cp.empty(N, dtype=dtype)
    
    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    eps2 = 1e-16  # No regularization for potential (i.e., no softening)

     # 2. Get the right kernel (it compiles only once per precision type)
    _nbody_potential_kernel = _get_kernel('potential', precision, use_kahan=(precision=='float32_kahan'))
   
    _nbody_potential_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, pot_gpu,
         dtype(G), dtype(eps2), cp.int32(kernel_id), cp.int32(N))
    )
    
    if return_cupy:
        return pot_gpu.astype(dtype)  # Ensure correct dtype on GPU
    else:
        return cp.asnumpy(pot_gpu.astype(dtype))  # Transfer to CPU with correct dtype


def compute_nbody_forces_cpu(
    pos: ArrayLike,
    mass: ArrayLike | float,
    softening: float | ArrayLike = 0.0,
    G: float = G_DEFAULT,
    kernel: KERNEL_TYPES = 'spline',
    nthreads: int | None = None,
    dtype: str = 'float64'
) -> NDArray: 
    """
    Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.
    Comparable speeds to fastest FMM/Tree-PM codes for <100_000 bodies parallelized with 48 cores.
    CAUTION: Even with optimizations and kernels, the direct-Nbody is infact O(N^2). Be aware!!
    
    User is responsible for correct pos, mass unit conversions...
    
    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    mass : array_like, shape (N,) or float
        Particle masses. If float, all particles have the same mass.
    softening : float or array_like, shape (N,), optional
        Softening length(s). If scalar, a same-valued array of length N is used.
        Defaults to 0.0, which implies pure Newtonian gravity.
    G : float, optional
        Gravitational constant. Defaults to 4.30092e-6. [Length: kpc, vel: km/s, Mass: Msun]
    kernel : {'newtonian', 'plummer', 'spline', 'dehnen_k1', 'dehnen_k2'}, optional
        Softening kernel to use:
        - 'newtonian': Pure 1/r^2 gravity with small epsilon regularization
        - 'plummer': Plummer softening (always softened)
        - 'dehnen_k1': Dehnen P1 (C2 correction) kernel (falcON default)
        - 'dehnen_k2': Dehnen P2 (C4 correction) kernel 
        COMPACT SUPPORT KERNELS:
        - 'spline': Cubic spline kernel (Monaghan 1992), compact support.
    nthreads : int or None, optional
        Number of threads for numba parallel loops.
    dtype : {'float64', 'float32'} or numpy dtype, optional
        Floating type to use for computation. Default 'float64'.

    Returns
    -------
    forces : ndarray, shape (N, 3)
        Gravitational accelerations on each particle.

    Notes
    -----
    - Spline kernels have compact support at radius `softening` and become
      exactly Newtonian for r >= softening.
    - The Plummer kernel is always softened and never becomes purely Newtonian.
    - For momentum conservation, softening lengths are combined using max().
    - Takes about 200secs for 1M bodies on 48 cores. 
    """
    # Validate dtype
    if isinstance(dtype, str):
        if dtype not in ('float64', 'float32'):
            raise ValueError("dtype must be 'float64' or 'float32'")
        dtype_np = np.float64 if dtype == 'float64' else np.float32
    else:
        dtype_np = np.dtype(dtype).type
    
    # Convert inputs and ensure contiguous arrays of chosen dtype
    pos = np.ascontiguousarray(np.asarray(pos, dtype=dtype_np))
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (N, 3)")
        
    N = pos.shape[0]

    # Convert mass
    if np.isscalar(mass): mass = np.full(N, mass, dtype=dtype_np)

    else:
        mass = np.ascontiguousarray(np.asarray(mass, dtype=dtype_np))    
        if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
            raise ValueError("mass must be shape (N,)")
    
    # Set numba threads if requested
    if nthreads is not None:
        set_num_threads(int(nthreads))
    
    # Prepare softening/smoothing parameter: ALWAYS an array of length N
    if np.isscalar(softening):
        soft_arr = np.full(N, float(softening), dtype=dtype_np)
    else:
        soft_arr = np.ascontiguousarray(np.asarray(softening, dtype=dtype_np))
        if soft_arr.shape[0] != N:
            raise ValueError("softening array must have length N")

    if np.any(soft_arr == 0):
        print("Warning: softening=0 detected, Using kernel='newtonian'.")
        kernel = 'newtonian'
    
    # Map the kernel string to a unique integer ID
    kernel_map = {
        'newtonian': 0,
        'plummer': 1,
        'dehnen_k1': 2,
        'dehnen_k2': 3,
        'spline': 4,
    }
    
    if kernel.lower() not in kernel_map:
        print(f'{kernel} not found in available. using spline kernel.')
        kernel = 'spline'
    
    kernel_id = kernel_map[kernel.lower()]    
    
    # Automatically select r_eps appropriate for dtype
    if dtype_np is np.float32:
        r_eps = dtype_np(1e-4)
    else:
        r_eps = dtype_np(1e-6)
    
    # func call to compute forces
    return G * _compute_forces_cpu(pos, mass, soft_arr, kernel_id, r_eps)

def compute_nbody_potential_cpu(
    pos: ArrayLike,
    mass: ArrayLike | float,
    softening: float | ArrayLike = 0.0,
    G: float = G_DEFAULT,
    kernel: KERNEL_TYPES = 'spline',
    nthreads: int | None = None,
    dtype: str = 'float64'
) -> NDArray:
    """
    Compute gravitational potential using CPU parallelization (Numba).
    
    Fallback version for systems without GPU or for validation.
    Same API as GPU version but uses CPU parallelization.
    
    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    mass : array_like, shape (N,) or scalar
        Particle masses.
    softening : float or array_like, shape (N,), optional
        Softening length(s). Default: 0.0.
    G : float, optional
        Gravitational constant. Default: 4.30092e-6.
    kernel : str, optional
        Potential kernel type. Default: 'spline'.
    nthreads : int, optional
        Number of CPU threads. Default: None (use all).
    dtype : str, optional
        'float32' or 'float64'. Default: 'float64'.
    
    Returns
    -------
    potentials : ndarray, shape (N,)
        Gravitational potential at each particle.
    
    Examples
    --------
    >>> import numpy as np
    >>> pos = np.random.randn(1000, 3)
    >>> mass = np.ones(1000)
    >>> pot = compute_nbody_potential_cpu(pos, mass, softening=0.01, nthreads=16)
    """
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba required for CPU version. Install: pip install numba")
    
    # Validate dtype
    dtype_np = np.float64 if dtype == 'float64' else np.float32
    
    # Convert inputs
    pos = np.ascontiguousarray(pos, dtype=dtype_np)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N, 3), got {pos.shape}")
    
    N = pos.shape[0]
    
    # Handle mass
    if np.isscalar(mass):
        mass_arr = np.full(N, mass, dtype=dtype_np)
    else:
        mass_arr = np.ascontiguousarray(mass, dtype=dtype_np)
        if mass_arr.shape[0] != N:
            raise ValueError(f"mass must be length N={N}")
    
    # Handle softening
    if np.isscalar(softening):
        h_arr = np.full(N, softening, dtype=dtype_np)
    else:
        h_arr = np.ascontiguousarray(softening, dtype=dtype_np)
        if h_arr.shape[0] != N:
            raise ValueError(f"softening must be length N={N}")
    
    # Map kernel
    kernel_map = {
        'newtonian': 0,
        'plummer': 1,
        'dehnen_k1': 2,
        'dehnen_k2': 3,
        'spline': 4,
    }
    
    if kernel.lower() not in kernel_map:
        raise ValueError(f"Invalid kernel '{kernel}'")
    
    kernel_id = kernel_map[kernel.lower()]
    
    # Set threads
    if nthreads is not None:
        set_num_threads(nthreads)
    
    # Regularization
    r_eps = 1e-6 if dtype_np == np.float64 else 1e-4
    
    # Compute
    potential = _compute_potential_cpu(pos, mass_arr, h_arr, kernel_id, r_eps)
    
    return G * potential


def get_gpu_info() -> dict:
    """
    Get information about available GPU(s).
    
    Returns
    -------
    info : dict
        Dictionary containing:
        - 'available': bool, whether GPU is available
        - 'device_name': str, GPU model name
        - 'compute_capability': tuple of int (major, minor), CUDA compute capability as (major, minor)
        - 'memory_total': int, total GPU memory in bytes
        - 'memory_free': int, free GPU memory in bytes
    
    Examples
    --------
    >>> info = get_gpu_info()
    >>> if info['available']:
    ...     print(f"GPU: {info['device_name']}")
    ...     print(f"Memory: {info['memory_free'] / 1e9:.1f} GB free")
    """
    if not CUPY_AVAILABLE:
        return {'available': False}
    
    try:
        device = cp.cuda.Device()
        mem_info = cp.cuda.runtime.memGetInfo()
        return {
            'available': True,
            'device_name': cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8'),
            'compute_capability': device.compute_capability,
            'memory_total': mem_info[1],
            'memory_free': mem_info[0],
        }
    except Exception as e:
        print(e)
        return {'available': False, 'error': str(e)}


# ============================================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================================

__all__ = [
    'compute_nbody_forces_gpu',
    'compute_nbody_forces_cpu',
    'compute_nbody_potential_gpu',
    'compute_nbody_potential_cpu',
    'get_gpu_info',
    'G_DEFAULT',
]

if __name__ == "__main__":
    import time
    print("\n" + "="*80)
    print("N-body Force Benchmark - GPU vs CPU")
    print("="*80)
    
    info = get_gpu_info()
    if info['available']:
        print(f"\n✓ GPU Available: {info['device_name']}")
        print(f"  Memory: {info['memory_free']/1e9:.1f} / {info['memory_total']/1e9:.1f} GB free")
        print(f"  Compute: {info['compute_capability']}")
    else:
        print("\n✗ No GPU available")
        # exit(1)
    
    # Test parameters
    N = 10_000
    n_warmup = 1
    n_bench = 5
    
    print(f"\nTesting with N={N:,} particles")
    print(f"Warmup: {n_warmup} iterations")
    print(f"Benchmark: {n_bench} iterations")
    
    np.random.seed(42)
    pos_cpu = np.random.randn(N, 3)
    mass_cpu = np.ones(N)
    
    # =========================================================================
    # GPU Test - Data Already on GPU (realistic scenario)
    # =========================================================================
    if CUPY_AVAILABLE:
        print("\n" + "="*80)
        print("GPU Test (data on GPU)")
        print("="*80)
        
        # Transfer once to GPU
        pos_gpu = cp.asarray(pos_cpu)
        mass_gpu = cp.asarray(mass_cpu)
        
        # Warmup (compile kernel)
        print("Warming up...")
        for _ in range(n_warmup):
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, softening=0.01, kernel='spline', return_cupy=True
            )
            cp.cuda.Stream.null.synchronize()  # Wait for GPU
        
        # Benchmark
        print("Benchmarking...")
        times = []
        for _ in range(n_bench):
            cp.cuda.Stream.null.synchronize()  # Ensure previous work done
            t0 = time.perf_counter()
            
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, softening=0.01, kernel='spline', return_cupy=True
            )
            
            cp.cuda.Stream.null.synchronize()  # Wait for completion
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        tgpu = np.median(times)
        tgpu_std = np.std(times)
        
        print(f"✓ GPU computation successful")
        print(f"  Median time: {tgpu*1000:.2f} ± {tgpu_std*1000:.2f} ms")
        print(f"  Throughput: {N*N/tgpu/1e9:.1f} Ginteractions/s")
        print(f"  All times: {[f'{t*1000:.1f}' for t in times]} ms")
        
        # Save for comparison
        acc_gpu_for_compare = acc_gpu.get()
    
    # =========================================================================
    # GPU Test - Including Transfers (worst case scenario)
    # =========================================================================
    if CUPY_AVAILABLE:
        print("\n" + "="*80)
        print("GPU Test (with CPU↔GPU transfers)")
        print("="*80)
        
        # Warmup
        for _ in range(n_warmup):
            _ = compute_nbody_forces_gpu(
                pos_cpu, mass_cpu, softening=0.01, kernel='spline', return_cupy=False
            )
        
        # Benchmark
        times_with_transfer = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            acc_cpu_result = compute_nbody_forces_gpu(
                pos_cpu, mass_cpu, softening=0.01, kernel='spline', return_cupy=False
            )
            
            t1 = time.perf_counter()
            times_with_transfer.append(t1 - t0)
        
        tgpu_transfer = np.median(times_with_transfer)
        tgpu_transfer_std = np.std(times_with_transfer)
        
        print(f"✓ GPU computation successful")
        print(f"  Median time: {tgpu_transfer*1000:.2f} ± {tgpu_transfer_std*1000:.2f} ms")
        print(f"  Throughput: {N*N/tgpu_transfer/1e9:.1f} Ginteractions/s")
        print(f"  Transfer overhead: {(tgpu_transfer - tgpu)*1000:.2f} ms ({(tgpu_transfer/tgpu - 1)*100:.1f}%)")
    
    # =========================================================================
    # CPU Test
    # =========================================================================
    if NUMBA_AVAILABLE:
        print("\n" + "="*80)
        print("CPU Test (16 threads, Numba)")
        print("="*80)
        
        # Warmup
        print("Warming up...")
        for _ in range(n_warmup):
            _ = compute_nbody_forces_cpu(
                pos_cpu, mass_cpu, softening=0.01, nthreads=16, kernel='spline', dtype ='float64',
            )
        
        # Benchmark
        print("Benchmarking...")
        times_cpu = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            acc_cpu = compute_nbody_forces_cpu(
                pos_cpu, mass_cpu, softening=0.01, nthreads=16, kernel='spline',    dtype ='float64',
            )
            
            t1 = time.perf_counter()
            times_cpu.append(t1 - t0)
        
        tcpu = np.median(times_cpu)
        tcpu_std = np.std(times_cpu)
        
        print(f"✓ CPU computation successful")
        print(f"  Median time: {tcpu*1000:.1f} ± {tcpu_std*1000:.1f} ms")
        print(f"  Throughput: {N*N/tcpu/1e9:.1f} Ginteractions/s")
    
    # =========================================================================
    # Comparison
    # =========================================================================
    if CUPY_AVAILABLE and NUMBA_AVAILABLE:
        print("\n" + "="*80)
        print("Comparison")
        print("="*80)
        
        # Numerical accuracy
        diff = np.abs(acc_cpu - acc_gpu_for_compare)
        rel_err = diff / np.maximum(np.abs(acc_cpu), 1e-20)
        print(f"\nNumerical accuracy:")
        print(f"  Max abs error: {diff.max()}")
        print(f"  Mean abs error: {diff.mean()}")
        print(f"  Max rel error: {rel_err.max()}")
        print(f"  Mean rel error: {rel_err.mean()}")
        
        # Performance
        print(f"\nPerformance:")
        print(f"  GPU (data on GPU):       {tgpu*1000:.2f} ms")
        print(f"  GPU (with transfers):    {tgpu_transfer*1000:.2f} ms")
        print(f"  CPU (16 threads):        {tcpu*1000:.1f} ms")
        print(f"\nSpeedup:")
        print(f"  GPU/CPU (no transfer):   {tcpu/tgpu:.1f}x")
        print(f"  GPU/CPU (with transfer): {tcpu/tgpu_transfer:.1f}x")
    
    print("\n" + "="*80)
    print("Benchmark complete! ✓")
    print("="*80)

    print("="*80)
    print("N-body Potential - Self Test (GPU + CPU)")
    print("="*80)
    
    print(f"\nTesting with N={N:,} particles...")
    
    np.random.seed(42)
    pos = np.random.randn(N, 3)
    mass = np.ones(N)
    
    # Test GPU
    if CUPY_AVAILABLE:
        print("\n--- GPU Test ---")
        pos_f32 = pos.astype(np.float32)
        mass_f32 = mass.astype(np.float32)
        
        # Warmup
        _ = compute_nbody_potential_gpu(pos_f32, mass_f32, softening=0.01)
        
        # Benchmark
        t0 = time.perf_counter()
        pot_gpu = compute_nbody_potential_gpu(pos_f32, mass_f32, softening=0.01, kernel='spline')
        t1 = time.perf_counter()
        tgpu = t1 - t0
        print(f"✓ GPU computation successful")
        print(f"  Time: {tgpu*1000:.1f} ms")
        print(f"  Throughput: {N*N/tgpu/1e9:.1f} Ginteractions/s")
        print(f"  Mean potential: {pot_gpu.mean():.3e}")
        
        PE_gpu = 0.5 * np.sum(mass_f32 * pot_gpu)
        print(f"  Total PE: {PE_gpu:.6e}")
    
    # Test CPU
    if NUMBA_AVAILABLE:
        print("\n--- CPU Test (16 threads) ---")
        
        # Warmup
        _ = compute_nbody_potential_cpu(pos, mass, softening=0.01, nthreads=16)
        
        # Benchmark
        t0 = time.perf_counter()
        pot_cpu = compute_nbody_potential_cpu(pos, mass, softening=0.01, nthreads=16, kernel='spline')
        t1 = time.perf_counter()
        tcpu = t1 - t0
        print(f"✓ CPU computation successful")
        print(f"  Time: {tcpu*1000:.1f} ms")
        print(f"  Throughput: {N*N/(tcpu)/1e9:.1f} Ginteractions/s")
        print(f"  Mean potential: {pot_cpu.mean():.3e}")
        
        PE_cpu = 0.5 * np.sum(mass * pot_cpu)
        print(f"  Total PE: {PE_cpu:.6e}")
    
    # Compare
    if CUPY_AVAILABLE and NUMBA_AVAILABLE:
        print("\n--- Comparison ---")
        diff = np.abs(pot_cpu - pot_gpu.astype(np.float64))
        rel_err = diff / np.maximum(np.abs(pot_cpu), 1e-20)
        print(f"  Max abs error: {diff.max():.3e}")
        print(f"  Max rel error: {rel_err.max():.3e}")
        print(f"  Mean rel error: {rel_err.mean():.3e}")
        print(f" SPEEDUP: GPU / CPU = {tcpu / tgpu:.2f}x")
    
    print("\n" + "="*80)
    print("Self-test passed! ✓")
    print("="*80)
