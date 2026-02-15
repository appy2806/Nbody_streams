#!/usr/bin/env python3
"""
nbody_streams.fields
GPU-accelerated N-body gravitational fields computation using CuPy.

This module provides high-performance direct N-body calculation on NVIDIA GPUs
using raw CUDA kernels compiled with nvcc. Achieves 50-100x speedup over Numba parallelized
CPU implementations for typical particle counts (10K-100K).

Requirements
------------
- NVIDIA GPU with CUDA support (compute capability >= 6.0)
- CuPy: https://cupy.dev/
- NumPy

Author: Arpit Arora
Date: Sept 2025

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
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Literal, Union
import warnings

try :
    from .cuda_kernels import *  # Import kernel templates
except ImportError:
    from cuda_kernels import *  # Fallback for direct script execution

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
    warnings.warn("Numba not available. CPU code disabled.", ImportWarning)

# ============================================================================
# CONSTANTS AND TYPE DEFINITIONS
# ============================================================================

# Default gravitational constant (kpc, km/s, Msun units)
G_DEFAULT = 4.300917270069976e-06 # double precision value for accuracy, but we will use float32 in GPU kernel for performance
# Kernel type mapping
KERNEL_TYPES = Literal['newtonian', 'plummer', 'dehnen_k1', 'dehnen_k2', 'spline']

KERNEL_MAP = {
    'newtonian': 0,
    'plummer': 1,
    'dehnen_k1': 2,
    'dehnen_k2': 3,
    'spline': 4,
}

 
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
        'FMAX': 'fmaxf',
        'USE_FLOAT4': True,  # >>>>>>> NEW: Flag for future float4 optimization
    },
    'float64': {
        'T': 'double', 
        'RSQRT': 'rsqrt', 
        'SQRT': 'sqrt', 
        'FMA': 'fma', 
        'FMAX': 'fmax',
        'USE_FLOAT4': False,  # >>>>>>> NEW: No float4 optimization for double precision
    },
}

# Kernel cache - stores compiled kernels
_NBODY_KERNEL_CACHE = {}

# Template and kernel name mapping
_NBODY_KERNEL_CONFIG = {
    # Forces kernels
    ('force', 'float32', False): (_NBODY_KERNEL_TEMPLATE_FLOAT4, 'nbody_forces_kernel_float4'),
    ('force', 'float64', False): (_NBODY_KERNEL_TEMPLATE, 'nbody_forces_kernel'),
    ('force', 'float32', True):  (_KAHAN_KERNEL_TEMPLATE_FLOAT4, 'nbody_forces_kahan_kernel_float4'),
    ('force', 'float64', True):  (_KAHAN_KERNEL_TEMPLATE, 'nbody_forces_kahan_kernel'),
    
    # Potential kernels
    ('potential', 'float32', False): (_POTENTIAL_KERNEL_TEMPLATE_FLOAT4, 'nbody_potential_kernel_float4'),
    ('potential', 'float64', False): (_POTENTIAL_KERNEL_TEMPLATE, 'nbody_potential_kernel'),
    ('potential', 'float32', True):  (_POTENTIAL_KAHAN_KERNEL_TEMPLATE_FLOAT4, 'nbody_potential_kahan_kernel_float4'),
    ('potential', 'float64', True):  (_POTENTIAL_KAHAN_KERNEL_TEMPLATE, 'nbody_potential_kahan_kernel'),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

    #  Find architecture available across NVIDIA GPUs automatically
    cc = cp.cuda.Device().compute_capability
    arch_flag = f'-arch=sm_{cc}'

    options = (
        '-O3', # Maximum optimization
        '--use_fast_math', # secret sauce for performance boost, but can reduce accuracy in some cases.
        arch_flag,  # Target GPU determined automatically!
        '--ptxas-options=-v' # Optional: Shows register usage in console
    )
    
    # Compile kernel
    kernel = cp.RawKernel(source, kernel_name, options=options, backend='nvcc')
    
    # Cache and return
    _NBODY_KERNEL_CACHE[cache_key] = kernel
    return kernel

def _validate_inputs(pos, mass, softening, precision, kernel):
    """
    Validate inputs and prepare CPU arrays.
    Works regardless of whether CuPy is available.
    Returns CPU numpy arrays ready for use or GPU transfer.
    """
    # Precision mapping
    prec_key = precision.lower()
    if prec_key not in _PRECISION_MAP:
        raise ValueError(f"Precision must be one of {list(_PRECISION_MAP.keys())}")
    
    dtype_np = _PRECISION_MAP[prec_key][1]  # Only use numpy dtype
    dtype = _PRECISION_MAP[prec_key][0]  if CUPY_AVAILABLE else None  # CuPy dtype if available, else None
    
    # Validate pos (accept both numpy and cupy, but validate on CPU metadata)
    if CUPY_AVAILABLE and isinstance(pos, cp.ndarray):
        # Just validate shape without transferring data
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        N = pos.shape[0]
        pos_cpu = None  # Signal that data is already on GPU
        
    else:
        pos_cpu = np.asarray(pos, dtype=dtype_np)
        if pos_cpu.ndim != 2 or pos_cpu.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos_cpu.shape}")
        N = pos_cpu.shape[0]
    
    # Handle mass
    if np.isscalar(mass):
        mass_cpu = None  # Signal to create on target device
        mass_scalar = float(mass)
    else:
        if CUPY_AVAILABLE and isinstance(mass, cp.ndarray):
            if mass.shape[0] != N:
                raise ValueError(f"mass must have length N={N}, got {mass.shape[0]}")
            mass_cpu = None
            mass_scalar = None
        else:
            mass_cpu = np.asarray(mass, dtype=dtype_np)
            if mass_cpu.shape[0] != N:
                raise ValueError(f"mass must have length N={N}, got {mass_cpu.shape[0]}")
            mass_scalar = None
    
    # Handle softening
    if np.isscalar(softening):
        h_cpu = None
        h_scalar = float(softening)
    else:
        if CUPY_AVAILABLE and isinstance(softening, cp.ndarray):
            if softening.shape[0] != N:
                raise ValueError(f"softening must have length N={N}, got {softening.shape[0]}")
            h_cpu = None
            h_scalar = None
        else:
            h_cpu = np.asarray(softening, dtype=dtype_np)
            if h_cpu.shape[0] != N:
                raise ValueError(f"softening must have length N={N}, got {h_cpu.shape[0]}")
            h_scalar = None
        
    if kernel.lower() not in KERNEL_MAP:
        raise ValueError(
            f"Invalid kernel '{kernel}'. Must be one of: {list(KERNEL_MAP.keys())}"
        )
    
    kernel_id = KERNEL_MAP[kernel.lower()]
    
    return {
        'N': N,
        'pos_cpu': pos_cpu,
        'mass_cpu': mass_cpu,
        'mass_scalar': mass_scalar,
        'h_cpu': h_cpu,
        'h_scalar': h_scalar,
        'kernel_id': kernel_id,
        'dtype_np': dtype_np,
        'dtype': dtype,
        'original_pos': pos,
        'original_mass': mass,
        'original_h': softening,
    }

def _prepare_gpu_arrays(validated, precision, skip_validation=False):
    """
    Transfer validated CPU data to GPU or use existing GPU arrays.
    Only called by GPU functions.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy required")
    
    dtype, dtype_np = _PRECISION_MAP[precision.lower()]
    
    if skip_validation:
        # Fast path - assume everything is already GPU arrays of correct type
        pos = validated  # Just the pos array in this case
        N = pos.shape[0]
        # FIX: Make contiguous arrays instead of strided views
        x_gpu = cp.ascontiguousarray(pos[:, 0])
        y_gpu = cp.ascontiguousarray(pos[:, 1])
        z_gpu = cp.ascontiguousarray(pos[:, 2])
        return x_gpu, y_gpu, z_gpu, None, None, N
    
    N = validated['N']
    
    # Handle pos
    if validated['pos_cpu'] is None:
        # Already on GPU
        pos_gpu = validated['original_pos'].astype(dtype, copy=False)
    else:
        pos_gpu = cp.asarray(validated['pos_cpu'])
    
    x_gpu = cp.ascontiguousarray(pos_gpu[:, 0])
    y_gpu = cp.ascontiguousarray(pos_gpu[:, 1])
    z_gpu = cp.ascontiguousarray(pos_gpu[:, 2])
    
    # Handle mass
    if validated['mass_scalar'] is not None:
        mass_gpu = cp.full(N, validated['mass_scalar'], dtype=dtype)
    elif validated['mass_cpu'] is None:
        mass_gpu = validated['original_mass'].astype(dtype, copy=False)
    else:
        mass_gpu = cp.asarray(validated['mass_cpu'])
    
    # Handle softening
    if validated['h_scalar'] is not None:
        h_gpu = cp.full(N, validated['h_scalar'], dtype=dtype)
    elif validated['h_cpu'] is None:
        h_gpu = validated['original_h'].astype(dtype, copy=False)
    else:
        h_gpu = cp.asarray(validated['h_cpu'])
    
    return x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, N

def _prepare_cpu_arrays(validated):
    """
    Prepare CPU arrays from validated data.
    Only called by CPU functions.
    """
    N = validated['N']
    dtype_np = validated['dtype_np']
    
    # Handle pos
    if validated['pos_cpu'] is None:
        # Was on GPU, need to transfer
        if CUPY_AVAILABLE and isinstance(validated['original_pos'], cp.ndarray):
            pos = cp.asnumpy(validated['original_pos']).astype(dtype_np)
        else:
            raise ValueError("pos is on GPU but CuPy not available")
    else:
        pos = np.ascontiguousarray(validated['pos_cpu'])
    
    # Handle mass
    if validated['mass_scalar'] is not None:
        mass = np.full(N, validated['mass_scalar'], dtype=dtype_np)
    elif validated['mass_cpu'] is None:
        if CUPY_AVAILABLE and isinstance(validated['original_mass'], cp.ndarray):
            mass = cp.asnumpy(validated['original_mass']).astype(dtype_np)
        else:
            raise ValueError("mass is on GPU but CuPy not available")
    else:
        mass = np.ascontiguousarray(validated['mass_cpu'])
    
    # Handle softening
    if validated['h_scalar'] is not None:
        h = np.full(N, validated['h_scalar'], dtype=dtype_np)
    elif validated['h_cpu'] is None:
        if CUPY_AVAILABLE and isinstance(validated['original_h'], cp.ndarray):
            h = cp.asnumpy(validated['original_h']).astype(dtype_np)
        else:
            raise ValueError("h is on GPU but CuPy not available")
    else:
        h = np.ascontiguousarray(validated['h_cpu'])
    
    return pos, mass, h, N

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
    return_cupy: bool = False,
    skip_validation: bool = False,
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
    skip_validation : bool, optional
        If True, skip input validation (faster but less safe). Default is False.
    
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

    if not CUPY_AVAILABLE:
        raise ImportError("CuPy required")   

    # 1. Validate inputs and allocate GPU arrays
    if skip_validation:
        # Ultra-fast path for hot loops
        # Assume: pos, mass, softening are already GPU arrays
        x_gpu, y_gpu, z_gpu, _, _, N = _prepare_gpu_arrays(
            pos, precision, skip_validation=True
        )
        mass_gpu = mass
        h_gpu = softening
        dtype, dtype_np = _PRECISION_MAP[precision.lower()]
        kernel_id = KERNEL_MAP[kernel.lower()]
    else:
        validated = _validate_inputs(pos, mass, softening, precision, kernel)
        kernel_id = validated['kernel_id']
        dtype = validated['dtype']
        dtype_np = validated['dtype_np']   
        x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, N = _prepare_gpu_arrays(validated, precision)

    # >>>>>>> NEW: Detect if we should use float4
    # Update the use_float4 check:
    use_float4 = (precision in ['float32', 'float32_kahan'])  # ← Include Kahan now!

    # use_float4 = (precision in ['float32', 'float32_kahan'] and 
    #               precision != 'float32_kahan')  # Keep Kahan separate for now

    # Allocate output arrays on GPU
    ax_gpu = cp.empty(N, dtype=dtype)
    ay_gpu = cp.empty(N, dtype=dtype)
    az_gpu = cp.empty(N, dtype=dtype)

    # 2. Get the right kernel (it compiles only once per precision type)
    _nbody_force_kernel = _get_kernel('force', precision, use_kahan=(precision=='float32_kahan'))

    # Launch kernel
    threads_per_block = 128  # Use the TILE_SIZE defined in cuda_kernels.py
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    eps2 = dtype(1e-15)  # Regularization parameter
    
    # >>>>>>> NEW: Different launch for float4 vs regular
    if use_float4:
        # For float4 optimization, we would need to prepare data differently and call a different kernel.
        # Interleave x,y,z,mass into float4 structure
        # Create float4 array - CuPy needs view casting, not structured dtype
        pos_mass_gpu = cp.empty((N, 4), dtype=cp.float32)
        pos_mass_gpu[:, 0] = x_gpu
        pos_mass_gpu[:, 1] = y_gpu
        pos_mass_gpu[:, 2] = z_gpu
        pos_mass_gpu[:, 3] = mass_gpu
        
        # Reinterpret as float4* for kernel (view as contiguous memory)
        pos_mass_gpu = pos_mass_gpu.ravel()
        
        _nbody_force_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (pos_mass_gpu, h_gpu, ax_gpu, ay_gpu, az_gpu,
            dtype(G), eps2, cp.int32(kernel_id), cp.int32(N))
        )

    else:
        _nbody_force_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, ax_gpu, ay_gpu, az_gpu,
            dtype(G), eps2, cp.int32(kernel_id), cp.int32(N))
        )
    
    # Assemble result into (N, 3) array
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
    return_cupy: bool = False,
    skip_validation: bool = False,
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
    skip_validation : bool, optional
        If True, skip input validation (faster but less safe). Default is False.
    
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
    
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy required")   

    # 1. Validate inputs and allocate GPU arrays
    if skip_validation:
        # Ultra-fast path for hot loops
        # Assume: pos, mass, softening are already GPU arrays
        x_gpu, y_gpu, z_gpu, _, _, N = _prepare_gpu_arrays(
            pos, precision, skip_validation=True
        )
        mass_gpu = mass
        h_gpu = softening
        dtype, dtype_np = _PRECISION_MAP[precision.lower()]
        kernel_id = KERNEL_MAP[kernel.lower()]
    else:
        validated = _validate_inputs(pos, mass, softening, precision, kernel)
        kernel_id = validated['kernel_id']
        dtype = validated['dtype']
        dtype_np = validated['dtype_np']   
        x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, N = _prepare_gpu_arrays(validated, precision)

    # >>>>>>> NEW: Detect if we should use float4
    # Update the use_float4 check:
    use_float4 = (precision in ['float32', 'float32_kahan'])  # ← Include Kahan now!

    # Allocate output arrays on GPU
    pot_gpu = cp.empty(N, dtype=dtype)
    
    # Launch kernel
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    eps2 = dtype(1e-15)  # No regularization for potential (i.e., no softening)

     # 2. Get the right kernel (it compiles only once per precision type)
    _nbody_potential_kernel = _get_kernel('potential', precision, use_kahan=(precision=='float32_kahan'))

    # >>>>>>> NEW: Different launch for float4 vs regular
    if use_float4:
        # For float4 optimization, we would need to prepare data differently and call a different kernel.
        # Interleave x,y,z,mass into float4 structure
        # Create float4 array - CuPy needs view casting, not structured dtype
        pos_mass_gpu = cp.empty((N, 4), dtype=cp.float32)
        pos_mass_gpu[:, 0] = x_gpu
        pos_mass_gpu[:, 1] = y_gpu
        pos_mass_gpu[:, 2] = z_gpu
        pos_mass_gpu[:, 3] = mass_gpu
        
        # Reinterpret as float4* for kernel (view as contiguous memory)
        pos_mass_gpu = pos_mass_gpu.ravel()
        
        _nbody_potential_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (pos_mass_gpu, h_gpu, pot_gpu,
            dtype(G), eps2, cp.int32(kernel_id), cp.int32(N))
        )

    else:
        _nbody_potential_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (x_gpu, y_gpu, z_gpu, mass_gpu, h_gpu, pot_gpu,
            dtype(G), eps2, cp.int32(kernel_id), cp.int32(N))
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
    precision: str = 'float64'
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
    precision : str, optional
        Floating point precision to use for computation. Default 'float64'.

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
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba required for CPU version. Install: pip install numba")

    # Validate dtype
    validated = _validate_inputs(pos, mass, softening, precision, kernel)
    pos_cpu, mass_cpu, h_cpu, N = _prepare_cpu_arrays(validated)
    kernel_id = validated['kernel_id']    
    dtype_np = validated['dtype_np']

    eps2 = 1e-15  # Regularization parameter for Newtonian kernel (if used)
        
    # func call to compute forces
    return G * _compute_forces_cpu(pos_cpu, mass_cpu, h_cpu, kernel_id, dtype_np(eps2))

def compute_nbody_potential_cpu(
    pos: ArrayLike,
    mass: ArrayLike | float,
    softening: float | ArrayLike = 0.0,
    G: float = G_DEFAULT,
    kernel: KERNEL_TYPES = 'spline',
    nthreads: int | None = None,
    precision: str = 'float64'
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
    precision : str, optional
        Floating point precision to use for computation. Default: 'float64'.
    
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
    validated = _validate_inputs(pos, mass, softening, precision, kernel)
    pos_cpu, mass_cpu, h_cpu, N = _prepare_cpu_arrays(validated)
    kernel_id = validated['kernel_id']    
    dtype_np = validated['dtype_np']

    eps2 = 1e-15  # Regularization parameter for Newtonian kernel (if used)
    
    # Compute
    potential = _compute_potential_cpu(pos_cpu, mass_cpu, h_cpu, kernel_id, dtype_np(eps2))
    
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
    import argparse, time
    import sys, os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='N-body GPU/CPU Benchmark Suite')
    parser.add_argument('-N', '--num-particles', type=int, default=25_600,
                       help='Number of particles (default: 25,600)')
    parser.add_argument('--n-warmup', type=int, default=2,
                       help='Number of warmup iterations (default: 1)')
    parser.add_argument('--n-bench', type=int, default=5,
                       help='Number of benchmark iterations (default: 5)')
    parser.add_argument('--cpu-threads', type=int, default=0,
                       help='Number of CPU threads (default: 0 = all available)')
    args = parser.parse_args()
    
    N = args.num_particles
    n_warmup = args.n_warmup
    n_bench = args.n_bench
    cpu_threads = (
        os.cpu_count() if args.cpu_threads == 0 else args.cpu_threads
        )
    
    print("\n" + "="*80)
    print("N-BODY GPU/CPU COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    
    # GPU Info
    info = get_gpu_info()
    if info['available']:
        print(f"\n✓ GPU Available: {info['device_name']}")
        print(f"  Memory: {info['memory_free']/1e9:.1f} / {info['memory_total']/1e9:.1f} GB free")
        print(f"  Compute Capability: {info['compute_capability']}")
    else:
        print("\n✗ No GPU available")
    
    print(f"\nTest Configuration:")
    print(f"  Particles: {N:,}")
    print(f"  Warmup iterations: {n_warmup}")
    print(f"  Benchmark iterations: {n_bench}")
    print(f"  CPU threads: {cpu_threads}")
    print(f"  Kernel: spline")
    print(f"  Softening: 0.01")
    
    # Generate test data (deterministic)
    np.random.seed(42)
    pos_f64 = np.random.randn(N, 3)
    mass_f64 = np.ones(N)
    pos_f32 = pos_f64.astype(np.float32)
    mass_f32 = mass_f64.astype(np.float32)
    
    # Storage for results
    results = {}
    
    # =========================================================================
    # SECTION 1: GPU FORCE BENCHMARKS (ALL PRECISIONS)
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 1: GPU FORCE COMPUTATION - PRECISION COMPARISON")
    print("="*80)

    precisions_to_test = ['float32', 'float32_kahan', 'float64']

    for precision in precisions_to_test:
        if not CUPY_AVAILABLE:
            print(f"\n✗ Skipping {precision} (CuPy not available)")
            continue
            
        print(f"\n{'-'*80}")
        print(f"Testing: {precision.upper()}")
        print(f"{'-'*80}")
        
        # Get dtype for this precision
        dtype, dtype_np = _PRECISION_MAP[precision.lower()]
        
        # Prepare GPU data with matching precision
        if precision == 'float64':
            pos_gpu = cp.asarray(pos_f64)
            mass_gpu = cp.asarray(mass_f64)
        else:
            pos_gpu = cp.asarray(pos_f32)
            mass_gpu = cp.asarray(mass_f32)

        # Pre-create softening array for skip_validation path
        h_gpu = cp.full(N, 0.01, dtype=dtype)
        
        # Test 1: Data already on GPU (no validation) - FAST PATH
        print(f"\n[{precision}] GPU - Data on GPU, no validation (FAST PATH):")
        
        # Warmup
        for _ in range(n_warmup):
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, h_gpu,  # ← Pass h_gpu, not scalar
                kernel='spline', precision=precision,
                return_cupy=True, skip_validation=True
            )
            cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_bench):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, h_gpu,  # ← Pass h_gpu, not scalar
                kernel='spline', precision=precision,
                return_cupy=True, skip_validation=True
            )
            
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        t_gpu_novalidation = np.median(times)
        results[f'{precision}_gpu_novalidation'] = {
            'time': t_gpu_novalidation,
            'throughput': N*N/t_gpu_novalidation/1e9,
            'times': times,
            'acc': acc_gpu.get()
        }
        
        print(f"  ✓ Median time: {t_gpu_novalidation*1000:.2f} ms")
        print(f"  ✓ Throughput: {N*N/t_gpu_novalidation/1e9:.1f} Ginteractions/s")
        print(f"  ✓ All times: {[f'{t*1000:.1f}' for t in times]} ms")
        
        # Test 2: Data on GPU with validation - NORMAL PATH
        print(f"\n[{precision}] GPU - Data on GPU, with validation (NORMAL PATH):")
        
        # Warmup
        for _ in range(n_warmup):
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, softening=0.01,  # ← Can use scalar here
                kernel='spline', precision=precision,
                return_cupy=True, skip_validation=False
            )
            cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_bench):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            
            acc_gpu = compute_nbody_forces_gpu(
                pos_gpu, mass_gpu, softening=0.01,  # ← Can use scalar here
                kernel='spline', precision=precision,
                return_cupy=True, skip_validation=False
            )
            
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        t_gpu_validation = np.median(times)
        results[f'{precision}_gpu_validation'] = {
            'time': t_gpu_validation,
            'throughput': N*N/t_gpu_validation/1e9,
            'times': times,
        }
        
        validation_overhead = (t_gpu_validation - t_gpu_novalidation) * 1000
        validation_percent = (t_gpu_validation/t_gpu_novalidation - 1) * 100
        
        print(f"  ✓ Median time: {t_gpu_validation*1000:.2f} ms")
        print(f"  ✓ Throughput: {N*N/t_gpu_validation/1e9:.1f} Ginteractions/s")
        print(f"  ✓ Validation overhead: {validation_overhead:.2f} ms ({validation_percent:.1f}%)")
        
        # Test 3: Data on CPU (includes transfer) - WORST CASE
        print(f"\n[{precision}] GPU - Data on CPU (with transfer - WORST CASE):")
        
        # Use correct CPU precision
        pos_cpu_test = pos_f64 if precision == 'float64' else pos_f32
        mass_cpu_test = mass_f64 if precision == 'float64' else mass_f32
        
        # Warmup
        for _ in range(n_warmup):
            _ = compute_nbody_forces_gpu(
                pos_cpu_test, mass_cpu_test, softening=0.01,
                kernel='spline', precision=precision,
                return_cupy=False
            )
        
        # Benchmark
        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            acc_cpu_result = compute_nbody_forces_gpu(
                pos_cpu_test, mass_cpu_test, softening=0.01,
                kernel='spline', precision=precision,
                return_cupy=False
            )
            
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        t_gpu_transfer = np.median(times)
        results[f'{precision}_gpu_transfer'] = {
            'time': t_gpu_transfer,
            'throughput': N*N/t_gpu_transfer/1e9,
            'times': times,
        }
        
        transfer_overhead = (t_gpu_transfer - t_gpu_novalidation) * 1000
        transfer_percent = (t_gpu_transfer/t_gpu_novalidation - 1) * 100
        
        print(f"  ✓ Median time: {t_gpu_transfer*1000:.2f} ms")
        print(f"  ✓ Throughput: {N*N/t_gpu_transfer/1e9:.1f} Ginteractions/s")
        print(f"  ✓ Transfer overhead: {transfer_overhead:.2f} ms ({transfer_percent:.1f}%)")
        
    # =========================================================================
    # SECTION 2: GPU POTENTIAL BENCHMARKS (ALL PRECISIONS)
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 2: GPU POTENTIAL COMPUTATION - PRECISION COMPARISON")
    print("="*80)

    for precision in precisions_to_test:
        if not CUPY_AVAILABLE:
            continue
            
        print(f"\n{'-'*80}")
        print(f"Testing: {precision.upper()}")
        print(f"{'-'*80}")
        
        # Prepare data (CPU arrays for potential - it handles GPU transfer internally)
        if precision == 'float64':
            pos_test = pos_f64
            mass_test = mass_f64
        else:
            pos_test = pos_f32
            mass_test = mass_f32
        
        # Warmup
        for _ in range(n_warmup):
            _ = compute_nbody_potential_gpu(
                pos_test, mass_test, softening=0.01,
                kernel='spline', precision=precision
            )
        
        # Benchmark
        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            pot_gpu = compute_nbody_potential_gpu(
                pos_test, mass_test, softening=0.01,
                kernel='spline', precision=precision
            )
            
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        t_pot_gpu = np.median(times)
        results[f'{precision}_potential_gpu'] = {
            'time': t_pot_gpu,
            'throughput': N*N/t_pot_gpu/1e9,
            'times': times,
            'potential': pot_gpu
        }
        
        print(f"  ✓ Median time: {t_pot_gpu*1000:.2f} ms")
        print(f"  ✓ Throughput: {N*N/t_pot_gpu/1e9:.1f} Ginteractions/s")
        print(f"  ✓ Mean potential: {pot_gpu.mean():.6e}")
    
    # =========================================================================
    # SECTION 3: CPU BENCHMARKS (FLOAT64 ONLY)
    # =========================================================================
    if NUMBA_AVAILABLE:
        print("\n" + "="*80)
        print(f"SECTION 3: CPU COMPUTATION ({cpu_threads} threads, float64)")
        print("="*80)
        
        # Force computation
        print(f"\n{'-'*80}")
        print("CPU Force Computation")
        print(f"{'-'*80}")

        # Warmup
        for _ in range(n_warmup):
            _ = compute_nbody_forces_cpu(
                pos_f64, mass_f64, softening=0.01,
                nthreads=cpu_threads, kernel='spline', precision='float64',
            )
        
        # Benchmark
        times_cpu = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            acc_cpu = compute_nbody_forces_cpu(
                pos_f64, mass_f64, softening=0.01,
                nthreads=cpu_threads, kernel='spline', precision='float64',
            )
            
            t1 = time.perf_counter()
            times_cpu.append(t1 - t0)
        
        t_cpu_force = np.median(times_cpu)
        results['cpu_force'] = {
            'time': t_cpu_force,
            'throughput': N*N/t_cpu_force/1e9,
            'times': times_cpu,
            'acc': acc_cpu
        }
        
        print(f"  ✓ Median time: {t_cpu_force*1000:.1f} ms")
        print(f"  ✓ Throughput: {N*N/t_cpu_force/1e9:.1f} Ginteractions/s")
        
        # Potential computation
        print(f"\n{'-'*80}")
        print("CPU Potential Computation")
        print(f"{'-'*80}")
        
        # Warmup
        for _ in range(n_warmup):
            _ = compute_nbody_potential_cpu(
                pos_f64, mass_f64, softening=0.01,
                nthreads=cpu_threads, kernel='spline'
            )
        
        # Benchmark
        times_cpu = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            
            pot_cpu = compute_nbody_potential_cpu(
                pos_f64, mass_f64, softening=0.01,
                nthreads=cpu_threads, kernel='spline'
            )
            
            t1 = time.perf_counter()
            times_cpu.append(t1 - t0)
        
        t_cpu_potential = np.median(times_cpu)
        results['cpu_potential'] = {
            'time': t_cpu_potential,
            'throughput': N*N/t_cpu_potential/1e9,
            'times': times_cpu,
            'potential': pot_cpu
        }
        
        print(f"  ✓ Median time: {t_cpu_potential*1000:.1f} ms")
        print(f"  ✓ Throughput: {N*N/t_cpu_potential/1e9:.1f} Ginteractions/s")
        print(f"  ✓ Mean potential: {pot_cpu.mean():.6e}")
    
    # =========================================================================
    # SECTION 4: PHYSICS VALIDATION (CONSERVATION LAWS & ACCURACY)
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 4: PHYSICS VALIDATION & NUMERICAL ACCURACY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("Net Force Conservation (isolated system)")
    print("-"*80)
    print("For an isolated system, net force should be ≈ 0 (momentum conservation)")
    print()
    
    for precision in precisions_to_test:
        if f'{precision}_gpu_novalidation' not in results:
            continue
        
        acc = results[f'{precision}_gpu_novalidation']['acc']
        
        # Net force (sum of all accelerations × mass)
        if precision == 'float64':
            net_force = np.sum(acc * mass_f64[:, np.newaxis], axis=0)
        else:
            net_force = np.sum(acc * mass_f32[:, np.newaxis], axis=0)
        
        net_force_mag = np.linalg.norm(net_force)
        
        # Typical force magnitude
        force_magnitudes = np.linalg.norm(acc, axis=1)
        if precision == 'float64':
            force_magnitudes *= mass_f64
        else:
            force_magnitudes *= mass_f32
        typical_force = np.median(force_magnitudes)
        
        relative_error = net_force_mag / (typical_force * N)
        
        # Get machine epsilon
        if precision == 'float64':
            eps = np.finfo(np.float64).eps
        else:
            eps = np.finfo(np.float32).eps
        
        print(f"[{precision.upper()}]")
        print(f"  Net force magnitude: {net_force_mag:.6e}")
        print(f"  Net force components: [{net_force[0]:.6e}, {net_force[1]:.6e}, {net_force[2]:.6e}]")
        print(f"  Relative to typical force: {relative_error:.6e}")
        print(f"  Machine epsilon (ε): {eps:.6e}")
        print(f"  Error / (N × ε): {relative_error / eps:.1f}")
        
        # Check if within acceptable bounds
        if relative_error < 1000 * eps:
            print(f"  ✓ PASS - Net force consistent with floating point precision")
        else:
            print(f"  ⚠ WARNING - Net force larger than expected")
        print()
    
    # CPU validation
    if NUMBA_AVAILABLE:
        acc_cpu = results['cpu_force']['acc']
        net_force_cpu = np.sum(acc_cpu * mass_f64[:, np.newaxis], axis=0)
        net_force_mag_cpu = np.linalg.norm(net_force_cpu)
        
        force_magnitudes_cpu = np.linalg.norm(acc_cpu, axis=1) * mass_f64
        typical_force_cpu = np.median(force_magnitudes_cpu)
        relative_error_cpu = net_force_mag_cpu / (typical_force_cpu * N)
        
        print(f"[CPU FLOAT64]")
        print(f"  Net force magnitude: {net_force_mag_cpu:.6e}")
        print(f"  Relative to typical force: {relative_error_cpu:.6e}")
        print(f"  Error / (N × ε): {relative_error_cpu / np.finfo(np.float64).eps:.1f}")
        
        if relative_error_cpu < 1000 * np.finfo(np.float64).eps:
            print(f"  ✓ PASS - Net force consistent with floating point precision")
        else:
            print(f"  ⚠ WARNING - Net force larger than expected")
        print()
    
    # =========================================================================
    # SECTION 5: GPU vs CPU ACCURACY COMPARISON
    # =========================================================================
    if CUPY_AVAILABLE and NUMBA_AVAILABLE:
        print("\n" + "-"*80)
        print("GPU vs CPU Numerical Accuracy (Forces)")
        print("-"*80)
        
        for precision in precisions_to_test:
            if f'{precision}_gpu_novalidation' not in results:
                continue
            
            acc_gpu = results[f'{precision}_gpu_novalidation']['acc']
            
            # Convert CPU to same precision for comparison
            if precision == 'float64':
                acc_cpu_compare = acc_cpu
            else:
                acc_cpu_compare = acc_cpu.astype(np.float32)
                        
            diff = np.abs(acc_cpu_compare - acc_gpu)
            rel_err = diff / np.maximum(np.abs(acc_cpu_compare), 1e-30)
            
            print(f"\n[{precision.upper()} vs CPU float64]")
            print(f"  Max absolute error: {diff.max():.6e}")
            print(f"  Mean absolute error: {diff.mean():.6e}")
            print(f"  Max relative error: {rel_err.max():.6e}")
            print(f"  Mean relative error: {rel_err.mean():.6e}")
            print(f"  RMS error: {np.sqrt(np.mean(diff**2)):.6e}")
        
        print("\n" + "-"*80)
        print("GPU vs CPU Numerical Accuracy (Potential)")
        print("-"*80)
        
        for precision in precisions_to_test:
            if f'{precision}_potential_gpu' not in results:
                continue
            
            pot_gpu = results[f'{precision}_potential_gpu']['potential']
            
            # Convert CPU to same precision for comparison
            if precision == 'float64':
                pot_cpu_compare = pot_cpu
            else:
                pot_cpu_compare = pot_cpu.astype(np.float32)
            
            diff = np.abs(pot_cpu_compare - pot_gpu)
            rel_err = diff / np.maximum(np.abs(pot_cpu_compare), 1e-30)
            
            print(f"\n[{precision.upper()} vs CPU float64]")
            print(f"  Max absolute error: {diff.max():.6e}")
            print(f"  Mean absolute error: {diff.mean():.6e}")
            print(f"  Max relative error: {rel_err.max():.6e}")
            print(f"  Mean relative error: {rel_err.mean():.6e}")
            
            # Total energy comparison
            if precision == 'float64':
                PE_gpu = 0.5 * np.sum(mass_f64 * pot_gpu)
                PE_cpu = 0.5 * np.sum(mass_f64 * pot_cpu)
            else:
                PE_gpu = 0.5 * np.sum(mass_f32 * pot_gpu)
                PE_cpu = 0.5 * np.sum(mass_f32 * pot_cpu.astype(np.float32))
            
            PE_diff = abs(PE_gpu - PE_cpu)
            PE_rel = PE_diff / abs(PE_cpu)
            
            print(f"  Total PE (GPU): {PE_gpu:.10e}")
            print(f"  Total PE (CPU): {PE_cpu:.10e}")
            print(f"  Relative PE error: {PE_rel:.6e}")
    
    # =========================================================================
    # SECTION 6: PERFORMANCE SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 6: PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("Force Computation Performance")
    print("-"*80)
    print(f"\n{'Method':<30} {'Time (ms)':<12} {'Throughput (Gint/s)':<20} {'Speedup':<10}")
    print("-"*80)
    
    # CPU baseline
    if NUMBA_AVAILABLE:
        t_baseline = results['cpu_force']['time']
        print(f"{'CPU (float64, 16 threads)':<30} {t_baseline*1000:>10.1f}   {results['cpu_force']['throughput']:>18.1f}   {'1.0x':>8}")
    
    # GPU results
    for precision in precisions_to_test:
        if f'{precision}_gpu_novalidation' not in results:
            continue
        
        # No validation
        t = results[f'{precision}_gpu_novalidation']['time']
        tp = results[f'{precision}_gpu_novalidation']['throughput']
        speedup = t_baseline / t if NUMBA_AVAILABLE else 0
        print(f"{'GPU ' + precision + ' (no valid)':<30} {t*1000:>10.2f}   {tp:>18.1f}   {speedup:>8.1f}x")
        
        # With validation
        t = results[f'{precision}_gpu_validation']['time']
        tp = results[f'{precision}_gpu_validation']['throughput']
        speedup = t_baseline / t if NUMBA_AVAILABLE else 0
        print(f"{'GPU ' + precision + ' (validated)':<30} {t*1000:>10.2f}   {tp:>18.1f}   {speedup:>8.1f}x")
        
        # With transfer
        t = results[f'{precision}_gpu_transfer']['time']
        tp = results[f'{precision}_gpu_transfer']['throughput']
        speedup = t_baseline / t if NUMBA_AVAILABLE else 0
        print(f"{'GPU ' + precision + ' (w/ transfer)':<30} {t*1000:>10.2f}   {tp:>18.1f}   {speedup:>8.1f}x")
        print()
    
    print("\n" + "-"*80)
    print("Potential Computation Performance")
    print("-"*80)
    print(f"\n{'Method':<30} {'Time (ms)':<12} {'Throughput (Gint/s)':<20} {'Speedup':<10}")
    print("-"*80)
    
    # CPU baseline
    if NUMBA_AVAILABLE:
        t_baseline = results['cpu_potential']['time']
        print(f"{'CPU (float64, 16 threads)':<30} {t_baseline*1000:>10.1f}   {results['cpu_potential']['throughput']:>18.1f}   {'1.0x':>8}")
    
    # GPU results
    for precision in precisions_to_test:
        if f'{precision}_potential_gpu' not in results:
            continue
        
        t = results[f'{precision}_potential_gpu']['time']
        tp = results[f'{precision}_potential_gpu']['throughput']
        speedup = t_baseline / t if NUMBA_AVAILABLE else 0
        print(f"{'GPU ' + precision:<30} {t*1000:>10.2f}   {tp:>18.1f}   {speedup:>8.1f}x")
    
    # =========================================================================
    # SECTION 7: PRECISION COMPARISON
    # =========================================================================
    if CUPY_AVAILABLE and 'float32_gpu_novalidation' in results and 'float64_gpu_novalidation' in results:
        print("\n" + "="*80)
        print("SECTION 7: PRECISION TRADE-OFFS")
        print("="*80)
        
        t32 = results['float32_gpu_novalidation']['time']
        t32k = results.get('float32_kahan_gpu_novalidation', {}).get('time', 0)
        t64 = results['float64_gpu_novalidation']['time']
        
        acc32 = results['float32_gpu_novalidation']['acc']
        acc32k = results.get('float32_kahan_gpu_novalidation', {}).get('acc', None)
        acc64 = results['float64_gpu_novalidation']['acc']
        
        print("\nForce Computation:")
        print(f"  float32:        {t32*1000:.2f} ms  (1.00x)")
        if t32k > 0:
            print(f"  float32_kahan:  {t32k*1000:.2f} ms  ({t32k/t32:.2f}x slower)")
        print(f"  float64:        {t64*1000:.2f} ms  ({t64/t32:.2f}x slower)")
        
        print("\nAccuracy vs float64:")
        diff32 = np.abs(acc64.astype(np.float32) - acc32)
        print(f"  float32 max error:        {diff32.max():.6e}")
        print(f"  float32 mean rel error:   {(diff32 / np.maximum(np.abs(acc64.astype(np.float32)), 1e-30)).mean():.6e}")
        
        if acc32k is not None:
            diff32k = np.abs(acc64.astype(np.float32) - acc32k)
            print(f"  float32_kahan max error:  {diff32k.max():.6e}")
            print(f"  float32_kahan mean rel:   {(diff32k / np.maximum(np.abs(acc64.astype(np.float32)), 1e-30)).mean():.6e}")
            print(f"\nKahan improvement factor: {diff32.mean() / diff32k.mean():.2f}x better accuracy")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE ✓")
    print("="*80)
    print(f"\nTest completed successfully for N={N:,} particles")
    print(f"All results saved in 'results' dictionary")
