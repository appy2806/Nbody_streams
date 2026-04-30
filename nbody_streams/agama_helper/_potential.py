"""
_potential.py
~~~~~~~~~~~~~~~~
GPU-accelerated evaluation of Agama potentials using CuPy.

Kernels live in ``_*_kernel.cu`` compiled once via ``cp.RawModule``
with nvcc and ``--use_fast_math``.

API mirrors agama.Potential:
    .potential(xyz, t=0.)   -> (N,)      [km/s]^2
    .force(xyz, t=0.)       -> (N,3)     [km/s]^2/kpc    (= -gradPhi)
    .density(xyz, t=0.)    -> (N,)      [Msol/kpc^3]
    .forceDeriv(xyz, t=0.)  -> (force(N,3), deriv(N,6))
                               deriv[i] = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
                               (matches agama.Potential.forceDeriv exactly)
    .evalDeriv(xyz, t=0.)   -> (phi(N,), force(N,3), deriv(N,6))

All methods accept:
    - CuPy or NumPy arrays
    - Shape (3,) for a single point  -> output shape is squeezed accordingly
    - Shape (N,3) for N points

Units (Agama convention): mass=Msol, length=kpc, velocity=km/s
    PHI in (km/s)^2,  force in (km/s)^2/kpc,  rho in Msol/kpc^3

Spline accuracy: Agama-compatible quintic splines with log-scaling of the
radial coefficients (replicates MultipoleInterp1d from potential_multipole.cpp).
Requires dPhi/dr data : use _DR coefficient files.

Requirements:
    CuPy >= 10.0,  nvcc accessible,  CUDA GPU
    scipy (for quintic spline construction and Lambert W : falls back to
    invPhi0=0 if unavailable, still quintic but without inner asymptote scaling)
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Tuple, Union
from scipy.interpolate import CubicSpline, CubicHermiteSpline

import numpy as np

# ---------------------------------------------------------------------------
# CuPy guard
# ---------------------------------------------------------------------------

try:
    import cupy as cp
except ImportError as _err:
    raise ImportError(
        "CuPy is required for GPU potential evaluation.\n"
        "Install with:  pip install cupy-cuda12x  (adjust CUDA version)\n"
        f"Original error: {_err}"
    ) from _err

# ---------------------------------------------------------------------------
# Compile the CUDA module once (lazy, cached on first use)
# ---------------------------------------------------------------------------

_KERNEL_FILE = Path(__file__).parent / "_multipole_potential_kernel.cu"
_MODULE: cp.RawModule | None = None

_CYLSPL_KERNEL_FILE = Path(__file__).parent / "_cylspl_potential_kernel.cu"
_CYLSPL_MODULE: cp.RawModule | None = None

# Radius-sort threshold.
# On modern GPUs (L40/A100 with 96 MB L2 cache), the poly array for any
# realistic lmax fits entirely in L2 without sorting.  argsort+scatter adds
# ~0.15 ms at N=100k and ~0.76 ms at N=1M : pure overhead with no cache benefit.
# Sorting would only help if poly_size > GPU L2, which requires lmax > ~50 or
# a grid with >100k nodes.  Set to a very large value to disable effectively.
_SORT_THRESHOLD = 999_999_999


# ---------------------------------------------------------------------------
# Mixin : shared by all GPU potential classes
# ---------------------------------------------------------------------------

class _GPUPotBase:
    """
    Mixin giving every GPU potential class ``+`` composition and ``sum()`` support.

    ``pot_a + pot_b``  ->  ``CompositePotentialGPU([pot_a, pot_b])``
    ``pot_a + composite``  ->  flattened composite (avoids nesting)
    ``sum([pot_a, pot_b, pot_c])``  ->  works via ``__radd__(0)``
    """
    def __add__(self, other):
        from_self  = self._components  if isinstance(self,  CompositePotentialGPU) else [self]
        from_other = other._components if isinstance(other, CompositePotentialGPU) else [other]
        return CompositePotentialGPU(from_self + from_other)

    def __radd__(self, other):
        if other == 0:          # support sum([pot1, pot2, ...])
            return self
        return NotImplemented


def _get_module() -> cp.RawModule:
    global _MODULE
    if _MODULE is None:
        if not _KERNEL_FILE.exists():
            raise FileNotFoundError(
                f"CUDA kernel file not found: {_KERNEL_FILE}\n"
                "Expected _multipole_potential_kernel.cu in the same directory as _potential.py."
            )
        _MODULE = cp.RawModule(
            code=_KERNEL_FILE.read_text(),
            backend="nvcc",
            options=("--use_fast_math", "-std=c++14"),
        )
    return _MODULE


def _get_kernel(name: str) -> cp.RawKernel:
    return _get_module().get_function(name)


def _get_cylspl_module() -> cp.RawModule:
    global _CYLSPL_MODULE
    if _CYLSPL_MODULE is None:
        if not _CYLSPL_KERNEL_FILE.exists():
            raise FileNotFoundError(
                f"CylSpline CUDA kernel file not found: {_CYLSPL_KERNEL_FILE}\n"
                "Expected _cylspl_potential_kernel.cu in the same directory as _potential.py."
            )
        _CYLSPL_MODULE = cp.RawModule(
            code=_CYLSPL_KERNEL_FILE.read_text(),
            backend="nvcc",
            # -Xptxas -O0: disable PTX-level (ptxas) optimizations.
            # Required on sm_90 (Hopper) with CUDA 12/13: the ptxas optimizer
            # miscompiles the bicubic Hermite evaluation loop in this kernel,
            # producing catastrophically wrong potential values for interior grid
            # cells (while grid-node evaluations remain exact).  -G (full debug)
            # and -Xptxas -O0 both suppress the bug; -Xptxas -O0 is cheaper
            # because it only disables PTX assembly-time optimizations, not
            # nvcc-level transformations.  Performance impact is small because
            # the kernel is memory-bandwidth bound.
            options=("--use_fast_math", "-std=c++14", "-Xptxas", "-O0"),
        )
    return _CYLSPL_MODULE


def _get_cylspl_kernel(name: str) -> cp.RawKernel:
    return _get_cylspl_module().get_function(name)


# ---------------------------------------------------------------------------
# Physical constants (Agama units: Msol, kpc, km/s)
# ---------------------------------------------------------------------------

_G_AGAMA  = 4.30091727067736e-06                  # kpc (km/s)^2 Msol^-1
_INV_4PIG = 1.0 / (4.0 * math.pi * _G_AGAMA)      # Msol kpc^-1 (km/s)^-2

_THREADS_PER_BLOCK = 256


# ---------------------------------------------------------------------------
# Helpers: normalise xyz input / squeeze output
# ---------------------------------------------------------------------------

def _prep_xyz(xyz) -> Tuple[cp.ndarray, bool]:
    """Accept (3,) or (N,3) numpy/cupy array. Returns (cp float64 (N,3), was_single)."""
    xyz = cp.asarray(xyz, dtype=cp.float64)
    if xyz.ndim == 1:
        if xyz.shape[0] != 3:
            raise ValueError(f"Single-point xyz must have shape (3,), got {xyz.shape}")
        return xyz.reshape(1, 3), True
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {xyz.shape}")
    return cp.ascontiguousarray(xyz), False


# ---------------------------------------------------------------------------
# CPU preprocessing helpers (quintic spline + log-scaling)
# ---------------------------------------------------------------------------

def _compute_invPhi0(phi_l0: np.ndarray,
                     dphi_dr_l0: np.ndarray,
                     r_grid: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Agama's ``invPhi0`` parameter for the l=0 log-scaling,
    plus inner power-law extrapolation coefficients (s, U, W).

    Replicates ``computeExtrapolationCoefs(v=0, inner)`` from
    ``potential_multipole.cpp``, lines 509-558.

    Returns
    -------
    invPhi0 : float
        1/W  if the inner asymptote warrants it, otherwise 0.
    s, U, W : float
        Inner extrapolation: Phi_0(r) = U*(r/r0)^s + W  for r < r0.
    """
    try:
        from scipy.special import lambertw as _lambertw
    except ImportError:
        return 0.0, 0.0, 0.0, phi_l0[0]

    Phi1, Phi2   = phi_l0[0],    phi_l0[1]
    dPhi1, dPhi2 = dphi_dr_l0[0], dphi_dr_l0[1]   # dPhi/dr (not dPhi/dlogr)
    r1, r2       = r_grid[0],    r_grid[1]

    lnr  = np.log(r2 / r1)
    num1 = r1 * dPhi1    # = dPhi/d(logr) at r1

    SAFETY = 100.0 * np.finfo(np.float64).eps
    if (abs(num1) == 0.0 or
            abs(Phi1 - Phi2) < SAFETY * max(abs(Phi1), abs(Phi2))):
        return 0.0, 0.0, 0.0, Phi1

    A = lnr * num1 / (Phi1 - Phi2)

    if not np.isfinite(A) or A >= 0.0:
        return 0.0, 0.0, 0.0, Phi1

    sqrt_eps = np.sqrt(np.finfo(np.float64).eps)
    if abs(A + 1.0) < sqrt_eps:
        s = 0.0
    else:
        lw_arg = A * np.exp(A)
        branch = 0 if A > -1.0 else -1
        lw_val = float(np.real(_lambertw(lw_arg, k=branch)))
        s = (A - lw_val) / lnr   # v=0

    if not np.isfinite(s):
        s = 2.0  # constant-density-core fallback

    # Compute U, W : matches C++ lines 543-548 (no early return for s<=0)
    v = 0  # l=0 inner extrapolation
    if s != v:
        U = (r1 * dPhi1 - v * Phi1) / (s - v)
        W = (r1 * dPhi1 - s * Phi1) / (v - s)
    else:
        U = r1 * dPhi1 - v * Phi1
        W = Phi1

    # v=0 special: compare power-law vs cubic-polynomial prediction for dPhi2
    # (C++ lines 550-561 : always runs, may override s to 2)
    dPhi2a = U * s * np.exp(s * lnr) / r2 if s != 0 else r1 * dPhi1 / r2
    dPhi2b = (r2 / r1 * (6.0*r1*(Phi2-Phi1)/(r2-r1) - dPhi1*(2*r1+r2))) / (2*r2+r1)
    if abs(dPhi2 - dPhi2b) < abs(dPhi2 - dPhi2a):
        # Adopt constant-density-core (s=2) extrapolation
        s = 2.0
        U = 0.5 * r1 * dPhi1
        W = Phi1 - U

    # Now check s>0 (matches C++ caller line 1653: invPhi0 = s>0 ? 1./W : 0)
    if s <= 0.0:
        return 0.0, float(s), float(U), float(W)

    invPhi0 = 1.0 / W

    # Safety: reset if any grid point violates the log-scaling requirement
    if np.any(phi_l0 * invPhi0 >= 1.0):
        invPhi0 = 0.0

    return float(invPhi0), float(s), float(U), float(W)


def _compute_outer_extrap(phi_l0: np.ndarray,
                          dphi_dr_l0: np.ndarray,
                          r_grid: np.ndarray) -> Tuple[float, float, float]:
    """
    Outer power-law extrapolation for l=0: Phi(r) = W*(r/r_N)^(-1) + U*(r/r_N)^s
    for r > r_N (last grid point).

    Replicates ``computeExtrapolationCoefs(v=-1, outer)`` from
    ``potential_multipole.cpp``.  The W term is the Keplerian -GM/r component
    (zero Laplacian); U*(r/r_N)^s is the residual whose density contribution
    is U*s*(s+1)*(r/r_N)^s / r^2 / (4piG).

    Returns
    -------
    outer_s, outer_U, outer_W : float
    """
    try:
        from scipy.special import lambertw as _lambertw
    except ImportError:
        return 0.0, 0.0, float(phi_l0[-1])

    v    = -1          # outer l=0: Keplerian falloff
    Phi1  = float(phi_l0[-1])
    Phi2  = float(phi_l0[-2])
    dPhi1 = float(dphi_dr_l0[-1])   # dPhi/dr (not dPhi/dlogr) at last point
    r1    = float(r_grid[-1])
    r2    = float(r_grid[-2])

    lnr  = np.log(r2 / r1)          # < 0 since r2 < r1
    num1 = r1 * dPhi1                # dPhi/d(logr) at r1
    num2 = v * Phi1                  # = -Phi1
    den1 = Phi1
    den2 = Phi2 * np.exp(-v * lnr)  # Phi2 * (r2/r1)

    SAFETY = 100.0 * np.finfo(float).eps
    roundoff = (abs(num1 - num2) < max(abs(num1), abs(num2)) * SAFETY or
                abs(den1 - den2) < max(abs(den1), abs(den2)) * SAFETY)

    if not np.isfinite(Phi1) or roundoff or den1 == den2:
        # Near-Keplerian: only 1/r term, density = 0
        return 0.0, 0.0, Phi1

    A = lnr * (num1 - num2) / (den1 - den2)

    if not np.isfinite(A) or A >= 0:
        return 0.0, 0.0, Phi1

    # x = (s-v)*lnr satisfies x = A*(1 - exp(x)); solve via Lambert W
    s = (float(v) if abs(A + 1) < np.sqrt(np.finfo(float).eps) else
         v + (A - float(np.real(_lambertw(A * np.exp(A), k=0 if A > -1 else -1)))) / lnr)

    # Safeguard: outer density must decay (s < 0); fallback gives rho ~ r^-4
    if not np.isfinite(s) or s >= 0:
        s = -2.0

    if s != v:
        U = (r1 * dPhi1 - v * Phi1) / (s - v)
        W = (r1 * dPhi1 - s * Phi1) / (v - s)
    else:
        U = r1 * dPhi1 - v * Phi1
        W = Phi1

    return float(s), float(U), float(W)


def _solve_quintic_d2(logr: np.ndarray,
                      phi_sc: np.ndarray,
                      dphi_dlogr_sc: np.ndarray) -> np.ndarray:
    """
    Compute second derivatives for Agama-compatible quintic C2 splines.

    Replicates ``constructQuinticSpline(xval, fval, fder)`` from
    ``math_spline.cpp``, lines 473-505.  The tridiagonal system imposes
    4th-derivative = 0 at both endpoints (the "standard" quintic spline).

    Parameters
    ----------
    logr          : (nR,)       x-grid (log r values)
    phi_sc        : (nR, n_lm)  function values (possibly log-scaled)
    dphi_dlogr_sc : (nR, n_lm)  first derivatives w.r.t. logr (log-scaled)

    Returns
    -------
    d2 : (nR, n_lm)   second derivatives w.r.t. logr
    """
    from scipy.linalg import solve_banded

    n, n_lm = phi_sc.shape

    diag = np.zeros(n)
    sup  = np.zeros(n)   # sup[i]  = A[i, i+1]
    sub  = np.zeros(n)   # sub[i]  = A[i, i-1]
    rhs  = np.zeros((n, n_lm))

    # Interior rows
    for i in range(n):
        if i > 0:
            hi = 1.0 / (logr[i] - logr[i-1])
            diag[i] += hi * 3.0
            sup[i-1] = -hi                          # A[i-1, i]
            rhs[i] -= (20.0*(phi_sc[i]-phi_sc[i-1])*hi
                       - 12.0*dphi_dlogr_sc[i]
                       - 8.0 *dphi_dlogr_sc[i-1]) * hi * hi
        if i < n - 1:
            hi = 1.0 / (logr[i+1] - logr[i])
            diag[i] += hi * 3.0
            sub[i+1] = -hi                          # A[i+1, i]
            rhs[i] += (20.0*(phi_sc[i+1]-phi_sc[i])*hi
                       - 12.0*dphi_dlogr_sc[i]
                       - 8.0 *dphi_dlogr_sc[i+1]) * hi * hi

    # Boundary conditions: 4th derivative = 0 at endpoints
    hi0 = 1.0 / (logr[1] - logr[0])
    sup[0]   = -2.0 * hi0
    rhs[0]   = (30.0*(phi_sc[1]-phi_sc[0])*hi0
                - 14.0*dphi_dlogr_sc[1]
                - 16.0*dphi_dlogr_sc[0]) * hi0 * hi0

    hin = 1.0 / (logr[n-1] - logr[n-2])
    sub[n-1] = -2.0 * hin
    rhs[n-1] = (-30.0*(phi_sc[n-1]-phi_sc[n-2])*hin
                + 14.0*dphi_dlogr_sc[n-2]
                + 16.0*dphi_dlogr_sc[n-1]) * hin * hin

    # Pack banded storage (scipy convention):
    #   ab[0, j] = A[j-1, j]  (superdiag)
    #   ab[1, j] = A[j, j]    (diagonal)
    #   ab[2, j] = A[j+1, j]  (subdiag)
    # sup[i] = A[i, i+1] -> ab[0, i+1] = sup[i]
    # sub[i] = A[i, i-1] -> ab[2, i-1] = sub[i]
    ab = np.zeros((3, n))
    ab[0, 1:]   = sup[:n-1]
    ab[1, :]    = diag
    ab[2, :n-1] = sub[1:]

    return solve_banded((1, 1), ab, rhs)    # (nR, n_lm)


# ---------------------------------------------------------------------------
# CPU preprocessing: MultipoleCoefs -> GPU-ready arrays
# ---------------------------------------------------------------------------

def _build_multipole_data(coefs, prune_threshold: float = 1e-16) -> dict:
    """
    Convert a MultipoleCoefs dataclass into quintic-spline coefficient arrays
    for the GPU, replicating Agama's ``MultipoleInterp1d`` constructor exactly.

    Spline representation (Agama-compatible)
    ----------------------------------------
    Agama applies a log-scaling before fitting the quintic:
      - l=0 term: Phi_scaled  = log(invPhi0 - 1/Phi_0)
      - l>0 terms: Phi_scaled = Phi_c / Phi_0

    The quintic C2 spline is fit on these scaled values; second derivatives
    are determined by the tridiagonal system (constructQuinticSpline).

    Polynomial layout per interval:
        C(s) = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))   s in [0,1]
    with boundary conditions C(0)=f0, C'(0)=m0, C''(0)=q0,
                              C(1)=f1, C'(1)=m1, C''(1)=q1

    Returns
    -------
    dict with keys:
        poly        : (n_lm, n_intervals, 6) float64
        logr_min    : float
        dlogr       : float
        inv_dlogr   : float
        n_intervals : int
        n_lm        : int
        lmax        : int
        lm_l        : (n_lm,) int32
        lm_m        : (n_lm,) int32
        r_grid      : (nR,) float64
        log_scaling : int  (0 or 1)
        invPhi0     : float
    """
    if coefs.dphi_dr is None:
        raise ValueError(
            "MultipoleCoefs.dphi_dr is None; dPhi/dr data is required.\n"
            "Use a .coef_mul_DR file (not plain .coef_mul)."
        )

    r_grid  = np.asarray(coefs.R_grid,  dtype=np.float64)
    phi     = np.asarray(coefs.phi,     dtype=np.float64)   # (nR, n_lm)
    dphi_dr = np.asarray(coefs.dphi_dr, dtype=np.float64)   # (nR, n_lm)
    nR      = len(r_grid)
    n_lm    = phi.shape[1]

    if nR < 2:
        raise ValueError(f"Need at least 2 radial grid points, got {nR}.")

    logr    = np.log(r_grid)
    dlogr_v = np.diff(logr)
    dlogr   = float(dlogr_v.mean())
    max_err = float(np.max(np.abs(dlogr_v - dlogr)) / dlogr)

    # dPhi/dr -> dPhi/d(logr) = r * dPhi/dr
    dphi_dlogr = dphi_dr * r_grid[:, np.newaxis]    # (nR, n_lm)
    # Keep a reference to dphi/dr at l=0 for _compute_invPhi0 (updated during resample)
    dphi_dr_l0 = dphi_dr[:, 0]

    if max_err > 1e-3:
        # Non-log-uniform grid (e.g. King tidal cutoff causes clustered spacing).
        # GPU O(1) interval lookup and poly building both require log-uniform grids.
        # Use at least 1000 points so the clustered boundary region (e.g. tidal cutoff)
        # has enough intervals to represent the sharp density drop accurately.
        nR_resamp      = max(nR, 1000)
        logr_new       = np.linspace(logr[0], logr[-1], nR_resamp)
        r_new          = np.exp(logr_new)

        spl = CubicHermiteSpline(logr, phi, dphi_dlogr)
        phi_new        = spl(logr_new)
        dphi_dlogr_new = spl(logr_new, 1)

        dphi_dr_l0  = dphi_dlogr_new[:, 0] / r_new   # updated for _compute_invPhi0
        phi         = phi_new
        dphi_dlogr  = dphi_dlogr_new
        r_grid      = r_new
        logr        = logr_new
        nR          = nR_resamp
        dlogr_v     = np.diff(logr)
        dlogr       = float(dlogr_v.mean())
        max_err     = float(np.max(np.abs(dlogr_v - dlogr)) / dlogr)
    elif max_err > 1e-6:
        warnings.warn(
            f"Radial grid is not perfectly log-uniform "
            f"(max relative spacing error = {max_err:.2e}). "
            "O(1) grid lookup may be slightly inaccurate.",
            stacklevel=3,
        )

    logr_min    = float(logr[0])
    inv_dlogr   = 1.0 / dlogr
    n_intervals = nR - 1

    # Outer power-law extrapolation from the last two unscaled grid points.
    # Must be computed before log-scaling modifies phi_l0.
    outer_s, outer_U, outer_W = _compute_outer_extrap(phi[:, 0], dphi_dr_l0, r_grid)

    lm_l = np.array([l for l, m in coefs.lm_labels], dtype=np.int32)
    lm_m = np.array([m for l, m in coefs.lm_labels], dtype=np.int32)
    lmax = int(lm_l.max())

    if lmax > 32:
        raise ValueError(f"lmax={lmax} exceeds the kernel's compile-time limit of 32.")

    # Sort lm pairs by (|m|, l, cos-before-sin) so the kernel can run the
    # Legendre recurrence sequentially (on-the-fly) instead of pre-filling a
    # 289-element Plm_arr.  (l=0,m=0) maps to sort key (0,0,0) -> stays first.
    _srt = np.lexsort((lm_m < 0, lm_l, np.abs(lm_m)))
    lm_l       = lm_l[_srt]
    lm_m       = lm_m[_srt]
    phi        = phi[:, _srt]
    dphi_dlogr = dphi_dlogr[:, _srt]

    # ------------------------------------------------------------------
    # Zero-coefficient pruning: drop (l,m) columns where all phi values
    # are below threshold.  Column 0 (monopole) is always kept : Agama's
    # log-scaling and inner-boundary code depend on it.
    # For axisymmetric potentials (type='Disk', lmax=32): 1089 -> 17 terms.
    # ------------------------------------------------------------------
    keep_mask = np.ones(n_lm, dtype=bool)
    if prune_threshold > 0:
        keep_mask[1:] = np.any(np.abs(phi[:, 1:]) >= prune_threshold, axis=0)
    if not np.all(keep_mask):
        n_pruned = int(n_lm - keep_mask.sum())
        phi          = phi[:, keep_mask]
        dphi_dlogr   = dphi_dlogr[:, keep_mask]
        lm_l         = lm_l[keep_mask]
        lm_m         = lm_m[keep_mask]
        n_lm         = int(keep_mask.sum())
        lmax         = int(lm_l.max())

    # ------------------------------------------------------------------
    # Log-scaling (Agama: only when all Phi[l=0] < 0)
    # ------------------------------------------------------------------
    phi_l0      = phi[:, 0]
    log_scaling = bool(np.all(phi_l0 < 0.0))
    invPhi0     = 0.0

    inner_s = 0.0
    inner_U = 0.0
    inner_W = float(phi_l0[0])  # default: constant at boundary value

    if log_scaling:
        invPhi0, inner_s, inner_U, inner_W = _compute_invPhi0(
            phi_l0, dphi_dr_l0, r_grid)

    phi_sc         = phi.copy()
    dphi_dlogr_sc  = dphi_dlogr.copy()

    if log_scaling:
        # c=0 (l=0, m=0): Phi_scaled = log(invPhi0 - 1/Phi_0)
        denom0                = phi_l0 * (invPhi0 * phi_l0 - 1.0)
        phi_sc[:, 0]          = np.log(invPhi0 - 1.0 / phi_l0)
        dphi_dlogr_sc[:, 0]   = dphi_dlogr[:, 0] / denom0

        # c>0: Phi_scaled = Phi_c / Phi_0,  dPhi_scaled = (dPhi_c - ratio*dPhi_0)/Phi_0
        phi0               = phi_l0[:, np.newaxis]           # (nR, 1)
        dphi0_dlr          = dphi_dlogr[:, 0:1]              # (nR, 1) : actual dPhi_0/d(logr)
        ratio              = phi[:, 1:] / phi0               # (nR, n_lm-1)
        phi_sc[:, 1:]       = ratio
        dphi_dlogr_sc[:, 1:] = (dphi_dlogr[:, 1:] - ratio * dphi0_dlr) / phi0

    # ------------------------------------------------------------------
    # Quintic spline: solve for second derivatives on the (scaled) grid
    # ------------------------------------------------------------------
    d2phi_sc = _solve_quintic_d2(logr, phi_sc, dphi_dlogr_sc)   # (nR, n_lm)

    # ------------------------------------------------------------------
    # Build 6-coefficient Horner polynomials per (lm, interval)
    #   C(s) = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))
    # Boundary conditions:
    #   C(0)=f0, C'(0)=m0 (=dC/ds|0), C''(0)=q0 (=d2C/ds^2|0)
    #   C(1)=f1, C'(1)=m1,           C''(1)=q1
    # where  m = dphi_dlogr_scaled * h,  q = d2phi_sc * h^2
    # ------------------------------------------------------------------
    h  = dlogr
    ni = n_intervals

    f0 = phi_sc[:ni, :]                       # (ni, n_lm)
    f1 = phi_sc[1:, :]
    m0 = dphi_dlogr_sc[:ni, :] * h           # dC/ds at s=0
    m1 = dphi_dlogr_sc[1:,  :] * h
    q0 = d2phi_sc[:ni, :] * (h * h)          # d2C/ds^2 at s=0
    q1 = d2phi_sc[1:,  :] * (h * h)
    df = f1 - f0

    a0 = f0
    a1 = m0
    a2 = 0.5 * q0
    a3 = 10.0*df - 6.0*m0 - 4.0*m1 - 1.5*q0 + 0.5*q1
    a4 = -15.0*df + 8.0*m0 + 7.0*m1 + 1.5*q0 - q1
    a5 = 6.0*df - 3.0*m0 - 3.0*m1 + 0.5*(q1 - q0)

    # (ni, n_lm, 6) -> transpose -> (n_lm, ni, 6)
    poly = np.ascontiguousarray(
        np.stack([a0, a1, a2, a3, a4, a5], axis=-1).transpose(1, 0, 2)
    ).astype(np.float64)

    return dict(
        poly        = poly,
        logr_min    = logr_min,
        dlogr       = dlogr,
        inv_dlogr   = inv_dlogr,
        n_intervals = n_intervals,
        n_lm        = n_lm,
        lmax        = lmax,
        lm_l        = lm_l,
        lm_m        = lm_m,
        r_grid      = r_grid,
        log_scaling = int(log_scaling),
        invPhi0     = float(invPhi0),
        inner_s     = float(inner_s),
        inner_U     = float(inner_U),
        inner_W     = float(inner_W),
        outer_s     = float(outer_s),
        outer_U     = float(outer_U),
        outer_W     = float(outer_W),
    )


# ---------------------------------------------------------------------------
# MultipolePotentialGPU
# ---------------------------------------------------------------------------

class MultipolePotentialGPU(_GPUPotBase):
    """
    GPU evaluator for an Agama Multipole BFE potential.

    Replicates Agama's ``MultipoleInterp1d`` with quintic C2 splines and
    log-scaling of radial coefficients (see potential_multipole.cpp).

    API matches ``agama.Potential``:
        .potential(xyz, t=0.)   -> Phi
        .force(xyz, t=0.)       -> -gradPhi
        .density(xyz, t=0.)     -> rho = div(grad(Phi)) / (4piG)
        .forceDeriv(xyz, t=0.)  -> (force, deriv)
              deriv layout: [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
        .evalDeriv(xyz, t=0.)   -> (phi, force, deriv)

    Parameters
    ----------
    coefs : MultipoleCoefs
        Must include ``dphi_dr`` (use ``_DR`` coefficient files).
    """

    def __init__(self, coefs, prune_threshold: float = 1e-16) -> None:
        data = _build_multipole_data(coefs, prune_threshold=prune_threshold)

        self._d_poly = cp.asarray(data["poly"].ravel(), dtype=cp.float64)
        self._d_lm_l = cp.asarray(data["lm_l"],        dtype=cp.int32)
        self._d_lm_m = cp.asarray(data["lm_m"],        dtype=cp.int32)

        self._logr_min    = float(data["logr_min"])
        self._dlogr       = float(data["dlogr"])
        self._inv_dlogr   = float(data["inv_dlogr"])
        self._n_intervals = int(data["n_intervals"])
        self._n_lm        = int(data["n_lm"])
        self._lmax        = int(data["lmax"])
        self._log_scaling = int(data["log_scaling"])
        self._invPhi0     = float(data["invPhi0"])
        self._inner_s     = float(data["inner_s"])
        self._inner_U     = float(data["inner_U"])
        self._inner_W     = float(data["inner_W"])
        self._outer_s     = float(data["outer_s"])
        self._outer_U     = float(data["outer_U"])
        self._outer_W     = float(data["outer_W"])
        self._inv_4piG    = _INV_4PIG

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path], **kw) -> "MultipolePotentialGPU":
        """Load from an Agama .coef_mul_DR file or HDF5 archive."""
        try:
            from nbody_streams.agama_helper import read_coefs
        except ImportError:
            from _coefs import read_coefs
        return cls(read_coefs(str(path)), **kw)

    @classmethod
    def from_agama(cls, pot, **kw) -> "MultipolePotentialGPU":
        """
        Build from an ``agama.Potential`` by exporting its coefficients.

        Works for potentials that Agama stores internally as Multipole
        expansions - e.g. ``Multipole``, ``King``, ``Spheroid``.

        Analytic types (NFW, Plummer, Dehnen, ...) do NOT export their
        parameters (Agama writes only ``type=NFW  # other params not stored``).
        For those, construct the GPU class directly::

            gpu = NFWPotentialGPU(mass=1e12, scaleRadius=20)

        or accept the potential as-is using the analytic GPU classes in
        ``_analytic_potentials.py``.
        """
        shm = "/dev/shm"
        tmp_dir  = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) \
                       else tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"agama_gpu_{uuid.uuid4().hex}.coef")
        try:
            pot.export(tmp_path)
            # Check if this is an actual expansion file (has "Coefficients" block)
            with open(tmp_path) as _f:
                _head = _f.read(512)
            if "Coefficients" not in _head:
                raise ValueError(
                    "from_agama(): the potential does not export expansion coefficients.\n"
                    "Agama analytic types (NFW, Plummer, Dehnen, ...) do not store their\n"
                    "parameters on export.  Construct the GPU class directly instead:\n"
                    "  NFWPotentialGPU(mass=..., scaleRadius=...)\n"
                    "For King/Spheroid, Agama materialises them as Multipole : those work."
                )
            return cls.from_file(tmp_path, **kw)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    # ---- kernel launch helpers ----------------------------------------------

    def _common_args(self):
        """Scalar args common to all kernels.

        All int/double scalars are explicitly typed (np.int32 / np.float64).
        Without this, CuPy wraps Python int as np.intp (int64) and Python float
        as np.float64.  np.float64 is fine; but np.intp (8 bytes) for a kernel
        `int` (4 bytes) parameter causes the CUDA driver to read the wrong bytes
        on sm_90 / Hopper, corrupting every subsequent kernel argument.
        """
        return (self._d_poly,
                np.float64(self._logr_min),
                np.float64(self._dlogr),
                np.float64(self._inv_dlogr),
                np.int32(self._n_intervals),
                np.int32(self._n_lm),
                np.int32(self._lmax),
                self._d_lm_l, self._d_lm_m,
                np.int32(self._log_scaling),
                np.float64(self._invPhi0),
                np.float64(self._inner_s),
                np.float64(self._inner_U),
                np.float64(self._inner_W),
                np.float64(self._outer_s),
                np.float64(self._outer_U),
                np.float64(self._outer_W))

    def _launch_eval(self, d_x, d_y, d_z, N: int, do_grad: bool):
        """Run potential-only or potential+gradient kernel."""
        phi_out  = cp.empty(N, dtype=cp.float64)
        grad_out = cp.empty(3 * N, dtype=cp.float64) if do_grad \
                   else cp.empty(0, dtype=cp.float64)

        kname  = "multipole_force_kernel" if do_grad else "multipole_potential_kernel"
        blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _get_kernel(kname)(
            (blocks,), (_THREADS_PER_BLOCK,),
            (d_x, d_y, d_z) + self._common_args() + (phi_out, grad_out, np.int32(N)),
        )
        return phi_out, (grad_out if do_grad else None)

    def _launch_hess(self, d_x, d_y, d_z, N: int):
        """Run potential+gradient+Hessian kernel."""
        phi_out  = cp.empty(N,     dtype=cp.float64)
        grad_out = cp.empty(3 * N, dtype=cp.float64)
        hess_out = cp.empty(6 * N, dtype=cp.float64)

        blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _get_kernel("multipole_hess_kernel")(
            (blocks,), (_THREADS_PER_BLOCK,),
            (d_x, d_y, d_z) + self._common_args() + (phi_out, grad_out, hess_out, np.int32(N)),
        )
        return phi_out, grad_out, hess_out

    def _launch_density(self, d_x, d_y, d_z, N: int):
        """Run density kernel."""
        rho_out = cp.empty(N, dtype=cp.float64)

        blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _get_kernel("multipole_density_kernel")(
            (blocks,), (_THREADS_PER_BLOCK,),
            (d_x, d_y, d_z) + self._common_args() + (np.float64(self._inv_4piG), rho_out, np.int32(N)),
        )
        return rho_out

    def _unpack_xyz(self, xyz):
        """Return (d_x, d_y, d_z, N, was_single, sort_idx_or_None).

        When N >= _SORT_THRESHOLD, particles are sorted by radius so that threads
        in the same warp land in the same radial interval -> L1/L2 cache reuse on
        the poly coefficient array.  The caller must call _desort() on outputs.
        """
        arr, was_single = _prep_xyz(xyz)
        N   = arr.shape[0]

        if N >= _SORT_THRESHOLD:
            r2   = arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2
            sidx = cp.argsort(r2)        # sort ascending by r^2
            arr  = arr[sidx]             # reordered (N,3)
        else:
            sidx = None

        d_x = cp.ascontiguousarray(arr[:, 0])
        d_y = cp.ascontiguousarray(arr[:, 1])
        d_z = cp.ascontiguousarray(arr[:, 2])
        return d_x, d_y, d_z, N, was_single, sidx

    @staticmethod
    def _desort(arr: cp.ndarray, sidx: cp.ndarray) -> cp.ndarray:
        """Scatter sorted output back to the original particle order."""
        out = cp.empty_like(arr)
        out[sidx] = arr
        return out

    # ---- public Agama-compatible API ----------------------------------------

    def potential(self, xyz, t: float = 0.0) -> cp.ndarray:
        """Evaluate Phi.  Returns (N,) or scalar CuPy float64 in (km/s)^2."""
        d_x, d_y, d_z, N, single, sidx = self._unpack_xyz(xyz)
        phi, _ = self._launch_eval(d_x, d_y, d_z, N, do_grad=False)
        if sidx is not None:
            phi = self._desort(phi, sidx)
        return phi[0] if single else phi

    def force(self, xyz, t: float = 0.0) -> cp.ndarray:
        """Evaluate F = -gradPhi.  Returns (3,) or (N,3) CuPy float64 in (km/s)^2/kpc."""
        d_x, d_y, d_z, N, single, sidx = self._unpack_xyz(xyz)
        _, grad = self._launch_eval(d_x, d_y, d_z, N, do_grad=True)
        if sidx is not None:
            grad = self._desort(grad.reshape(N, 3), sidx).ravel()
        force = -grad.reshape(N, 3)
        return force[0] if single else force

    def density(self, xyz, t: float = 0.0) -> cp.ndarray:
        """Evaluate rho = div(grad(Phi))/(4piG).  Returns (N,) or scalar CuPy float64 in Msol/kpc^3."""
        d_x, d_y, d_z, N, single, sidx = self._unpack_xyz(xyz)
        rho = self._launch_density(d_x, d_y, d_z, N)
        if sidx is not None:
            rho = self._desort(rho, sidx)
        return rho[0] if single else rho

    def forceDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Force and force derivatives, matching ``agama.Potential.forceDeriv``.

        Returns
        -------
        force : (3,) or (N,3) CuPy float64
            F = -gradPhi in (km/s)^2/kpc
        deriv : (6,) or (N,6) CuPy float64
            [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
            = -[Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
        """
        d_x, d_y, d_z, N, single, sidx = self._unpack_xyz(xyz)
        phi_raw, grad_raw, hess_raw = self._launch_hess(d_x, d_y, d_z, N)

        if sidx is not None:
            phi_raw  = self._desort(phi_raw, sidx)
            grad_raw = self._desort(grad_raw.reshape(N, 3), sidx).ravel()
            hess_raw = self._desort(hess_raw.reshape(N, 6), sidx).ravel()

        force = -grad_raw.reshape(N, 3)
        deriv = -hess_raw.reshape(N, 6)

        if single:
            return force[0], deriv[0]
        return force, deriv

    def evalDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Potential, force, and force derivatives in one kernel call.

        Returns
        -------
        phi   : (N,) or scalar : Phi in (km/s)^2
        force : (N,3) or (3,) : F = -gradPhi
        deriv : (N,6) or (6,) : force derivatives (see forceDeriv)
        """
        d_x, d_y, d_z, N, single, sidx = self._unpack_xyz(xyz)
        phi_raw, grad_raw, hess_raw = self._launch_hess(d_x, d_y, d_z, N)

        if sidx is not None:
            phi_raw  = self._desort(phi_raw, sidx)
            grad_raw = self._desort(grad_raw.reshape(N, 3), sidx).ravel()
            hess_raw = self._desort(hess_raw.reshape(N, 6), sidx).ravel()

        force = -grad_raw.reshape(N, 3)
        deriv = -hess_raw.reshape(N, 6)

        if single:
            return phi_raw[0], force[0], deriv[0]
        return phi_raw, force, deriv

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        """
        Agama-compatible eval : returns any combination of potential, acceleration,
        and its derivatives.

        Matches ``agama.Potential.eval(xyz, pot=False, acc=False, der=False, t=0)``.

        Returns a single array when only one quantity is requested, otherwise a
        tuple in the order (phi, force, deriv) for whichever subset is requested.

        Parameters
        ----------
        pot : bool   : include potential Phi
        acc : bool   : include acceleration (= force = -gradPhi)
        der : bool   : include force derivatives (-d2Phi/dx_i dx_j, shape (N,6))
        """
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        if der:
            phi_r, force_r, deriv_r = self.evalDeriv(xyz, t)
            results = []
            if pot: results.append(phi_r)
            if acc: results.append(force_r)
            if der: results.append(deriv_r)
        elif acc:
            force_r = self.force(xyz, t)
            results = []
            if pot: results.append(self.potential(xyz, t))
            results.append(force_r)
        else:
            results = [self.potential(xyz, t)]
        return results[0] if len(results) == 1 else tuple(results)


# ---------------------------------------------------------------------------
# CylSpline preprocessing helpers
# ---------------------------------------------------------------------------

# Agama normalization constants : match _cylspl_potential_kernel.cu exactly
_CS_PREFACT = np.array([
    0.2820947917738782,   0.3454941494713355,   0.1287580673410632,
    0.02781492157551894,  0.004214597070904597, 0.0004911451888263050,
    4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
    1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11,
    1.790656309174350e-12, 7.299068453727266e-14, 2.751209457796109e-15,
    9.643748535232993e-17, 3.159120301003413e-18,
])
_CS_COEF = np.array([
     0.2820947917738782, -0.3454941494713355,  0.3862742020231896,
    -0.4172238236327841,  0.4425326924449826, -0.4641322034408582,
     0.4830841135800662, -0.5000395635705506,  0.5154289843972843,
    -0.5295529414924496,  0.5426302919442215, -0.5548257538066191,
     0.5662666637421912, -0.5770536647012670,  0.5872677968601020,
    -0.5969753602424046,  0.6062313441538353,
])
_CS_MUL0 = 3.5449077018110318   # 2*sqrt(pi)
_CS_MUL1 = 5.0132565706694072   # 2*sqrt(2*pi)


def _sph_harm_agama(lmax: int, m: int, ct: float, st: float) -> np.ndarray:
    """
    Compute Agama-normalized P_l^m(ct, st) for l = m .. lmax.

    Matches the corrected sphHarm_device in _cylspl_potential_kernel.cu exactly.
    ct = cos(theta) = z/r,  st = sin(theta) = R/r.

    Returns array of shape (lmax - m + 1,).
    """
    n = lmax - m + 1
    result = np.zeros(n)

    # P_m^m = COEF[m] * st^m
    val = _CS_COEF[m] * float(st) ** m
    result[0] = val
    if lmax == m:
        return result

    prefact = _CS_PREFACT[m]
    Plm1 = result[0] / prefact        # un-normalized P_m^m
    Plm  = ct * (2*m + 1) * Plm1     # un-normalized P_{m+1}^m
    Plm2 = 0.0

    # l = m+1: N_{m+1,m} = N_{m,m} * sqrt((2m+3)/(2m+1) / (2m+1))
    prefact *= math.sqrt((2*m + 3) / (2*m + 1) / (2.0*m + 1.0))
    result[1] = Plm * prefact

    for l in range(m + 2, lmax + 1):
        Plm_new = (ct * (2*l - 1) * Plm - (l + m - 1) * Plm2) / (l - m)
        prefact *= math.sqrt((2*l + 1) / (2*l - 1) * (l - m) / (l + m))
        result[l - m] = Plm_new * prefact
        Plm2 = Plm1
        Plm1 = Plm
        Plm  = Plm_new

    return result


def _determine_asympt_cylspline(
    R_grid: np.ndarray,
    z_grid: np.ndarray,
    phi_dict: dict,
    mmax: int,
    lmax_fit: int = 8,
) -> Tuple[np.ndarray, float]:
    """
    Fit outer PowerLaw multipole coefficients matching Agama's determineAsympt().

    Parameters
    ----------
    R_grid   : (nR,) radial grid (original, before asinh scaling) [kpc]
    z_grid   : (nz,) z grid (full range or half-space) [kpc]
    phi_dict : {m: (nR, nz) array} : Fourier amplitude tables (original, not log-scaled)
    mmax     : max azimuthal order present in the data
    lmax_fit : number of spherical harmonics to fit (Agama: LMAX_EXTRAPOLATION=8)

    Returns
    -------
    W_outer  : (lmax_fit+1)^2 array indexed as W[l*(l+1)+m]
    r0_outer : reference radius = min(R_grid[-1], max(|z_grid|))
    """
    from scipy.linalg import lstsq as _lstsq

    nR   = len(R_grid)
    nz   = len(z_grid)
    zsym = (z_grid[0] == 0)
    mmax_fit = min(lmax_fit, mmax)

    # --- boundary points (replicates Agama's determineAsympt loop) ----------
    Rp, zp, iRp, izp = [], [], [], []

    for iR in range(nR - 1):
        # top z edge
        Rp.append(R_grid[iR]); zp.append(z_grid[nz-1]); iRp.append(iR); izp.append(nz-1)
        if zsym:
            Rp.append(R_grid[iR]); zp.append(-z_grid[nz-1]); iRp.append(iR); izp.append(nz-1)
        else:
            # bottom z edge
            Rp.append(R_grid[iR]); zp.append(z_grid[0]); iRp.append(iR); izp.append(0)

    for iz in range(nz):
        # max R edge
        Rp.append(R_grid[nR-1]); zp.append(z_grid[iz]); iRp.append(nR-1); izp.append(iz)
        if zsym and iz > 0:
            Rp.append(R_grid[nR-1]); zp.append(-z_grid[iz]); iRp.append(nR-1); izp.append(iz)

    Rp   = np.array(Rp,  dtype=np.float64)
    zp   = np.array(zp,  dtype=np.float64)
    rp   = np.hypot(Rp, zp)
    ct_p = zp / rp     # cos(theta) = z/r
    st_p = Rp / rp     # sin(theta) = R/r

    r0 = min(float(R_grid[-1]), float(np.max(np.abs(z_grid))))
    ncoefs  = (lmax_fit + 1) ** 2
    W = np.zeros(ncoefs)
    npoints = len(Rp)

    for m in range(-mmax_fit, mmax_fit + 1):
        if m not in phi_dict:
            continue
        phi_m = phi_dict[m]    # (nR, nz)
        absm  = abs(m)
        ncols = lmax_fit - absm + 1
        MUL   = _CS_MUL0 if m == 0 else _CS_MUL1

        M   = np.zeros((npoints, ncols))
        rhs = np.zeros(npoints)

        for p in range(npoints):
            rhs[p] = float(phi_m[iRp[p], izp[p]])
            Plm = _sph_harm_agama(lmax_fit, absm, float(ct_p[p]), float(st_p[p]))
            for l_idx in range(ncols):
                l = absm + l_idx
                M[p, l_idx] = Plm[l_idx] * (rp[p] / r0) ** (-(l + 1)) * MUL

        sol, _, _, _ = _lstsq(M, rhs)
        for l_idx in range(ncols):
            l = absm + l_idx
            W[l * (l + 1) + m] = float(sol[l_idx])

    # Safeguard: if W[0] is not finite, fall back to average monopole amplitude
    if not np.isfinite(W[0]):
        phi0 = phi_dict.get(0)
        if phi0 is not None:
            avg = float(np.mean(
                [phi0[iRp[p], izp[p]] * rp[p] / r0 for p in range(npoints)]
            ))
        else:
            avg = 0.0
        W[:] = 0.0
        W[0] = avg

    return W, float(r0)


def _setup_cubic2d_nodes(lR: np.ndarray, lz: np.ndarray, fval: np.ndarray) -> np.ndarray:
    """
    Build CubicSpline2d node data (fval, fx, fy, fxy) for one harmonic.

    Replicates Agama's CubicSpline2d constructor with:
      deriv_xmin=0 (clamped at R=0), deriv_xmax=NAN (natural),
      deriv_ymin=NAN, deriv_ymax=NAN (natural in z).

    Parameters
    ----------
    lR   : (nR,) asinh-scaled R grid
    lz   : (nz,) asinh-scaled z grid
    fval : (nR, nz) values on the grid (possibly log-scaled)

    Returns
    -------
    nodes : (nR, nz, 4) float64 : [fval, fx, fy, fxy] per node
    """
    from scipy.interpolate import CubicSpline as _CS

    nR, nz = fval.shape
    fx  = np.zeros((nR, nz))
    fy  = np.zeros((nR, nz))
    fxy = np.zeros((nR, nz))

    # Step 1: natural cubic spline in lz for each R-row -> fy = df/dlz
    for i in range(nR):
        spl = _CS(lz, fval[i, :], bc_type='natural')
        fy[i, :] = spl(lz, 1)

    # Step 2: clamped-left cubic spline in lR for each z-column -> fx = df/dlR
    # bc_type=((1, 0.0), 'natural'): f'=0 at lR[0], f''=0 at lR[-1]
    for j in range(nz):
        spl = _CS(lR, fval[:, j], bc_type=((1, 0.0), 'natural'))
        fx[:, j] = spl(lR, 1)

    # Step 3: clamped-left cubic spline in lR for each z-column -> fxy = d(fy)/dlR
    # Same BCs as step 2 (matches Agama's: isFinite(deriv_xmin)?0:NAN, NAN)
    for j in range(nz):
        spl = _CS(lR, fy[:, j], bc_type=((1, 0.0), 'natural'))
        fxy[:, j] = spl(lR, 1)

    return np.stack([fval, fx, fy, fxy], axis=-1).astype(np.float64)   # (nR, nz, 4)


def _build_cylspline_data(coefs) -> dict:
    """
    Preprocess a CylSplineCoefs object into GPU-ready arrays.

    Replicates the CylSpline C++ constructor from potential_cylspline.cpp:
      - Rscale from -Mtot/Phi0
      - asinh coordinate transform
      - optional log-scaling of m=0 term and normalization of m not 0
      - CubicSpline2d node arrays (fval, fx, fy, fxy) per harmonic
      - outer PowerLaw multipole fit via determineAsympt

    Returns
    -------
    dict with keys:
      Rscale, lR_grid, lz_grid, node_arr, m_arr, n_harm, mmax,
      log_scaling, W_outer, r0_outer, lmax_outer, mmax_outer
    """
    R_grid   = np.asarray(coefs.R_grid, dtype=np.float64)
    z_grid_0 = np.asarray(coefs.z_grid, dtype=np.float64)
    phi_orig = {m: np.asarray(v, dtype=np.float64) for m, v in coefs.phi.items()}

    nR       = len(R_grid)
    mmax     = max(abs(m) for m in phi_orig)

    # ---------- Handle z-grid symmetry (matches Agama CylSpline constructor) --
    zsym = (z_grid_0[0] == 0)   # half-space: z >= 0 only

    if zsym:
        # Mirror grid: [-z_N, ..., -z_1, 0, z_1, ..., z_N]
        z_grid = np.concatenate([-z_grid_0[:0:-1], z_grid_0])
        nz_0   = len(z_grid_0)
        phi_dict = {}
        for m, tab in phi_orig.items():
            # tab is (nR, nz_0) covering z >= 0
            # Mirror: Phi(R, -z) = Phi(R, z) for all m terms
            full = np.empty((nR, 2 * nz_0 - 1))
            full[:, nz_0 - 1:] = tab           # z >= 0 half
            full[:, :nz_0]     = tab[:, ::-1]  # z <= 0 half (mirror)
            phi_dict[m] = full
        # Phi0 at R=0, z=0 -> first column of mirrored grid at iz = nz_0-1
        Phi0 = phi_dict[0][0, nz_0 - 1]
    else:
        z_grid   = z_grid_0
        phi_dict = phi_orig
        nz       = len(z_grid)
        # Phi0 at R=0, iz = (nz+1)//2  (matches C++: (sizez+1)/2)
        iz_mid   = (nz + 1) // 2
        Phi0     = phi_dict[0][0, iz_mid]

    nz = len(z_grid)

    # ---------- Outer asymptote fit (on original unscaled grids) ---------------
    W_outer, r0_outer = _determine_asympt_cylspline(R_grid, z_grid, phi_dict, mmax)
    lmax_outer  = 8
    mmax_outer  = min(lmax_outer, mmax)

    # Mtot ~ -(asymptOuter at R_max, z=0) * R_max
    # PowerLaw monopole at r = R_max: Phi_mono = W[0] * (R_max/r0)^{-1} * MUL0
    # Mtot = -Phi_mono * R_max = -W_outer[0] * r0 * MUL0
    Mtot = -W_outer[0] * r0_outer * _CS_MUL0

    if Phi0 < 0.0 and Mtot > 0.0:
        Rscale = -Mtot / Phi0
    else:
        Rscale = float(R_grid[nR // 2])

    # ---------- Asinh-scaled coordinate grids ----------------------------------
    lR_grid = np.arcsinh(R_grid / Rscale)
    lz_grid = np.arcsinh(z_grid / Rscale)

    # ---------- Log-scaling flag (all Phi[m=0] < 0?) --------------------------
    log_scaling = bool(np.all(phi_dict[0] < 0.0))

    # ---------- Build spline nodes for every harmonic --------------------------
    m_values = sorted(phi_dict.keys())
    n_harm   = len(m_values)
    m_arr    = np.array(m_values, dtype=np.int32)

    phi0_tab = phi_dict[0]   # (nR, nz) m=0 amplitudes : needed for normalisation

    all_nodes = []
    for m in m_values:
        fval = phi_dict[m]   # (nR, nz)

        if log_scaling:
            if m == 0:
                fval_sc = np.log(-fval)                # log(-Phi_0)
            else:
                fval_sc = fval / phi0_tab              # Phi_m / Phi_0
        else:
            fval_sc = fval

        nodes = _setup_cubic2d_nodes(lR_grid, lz_grid, fval_sc)   # (nR, nz, 4)
        all_nodes.append(nodes)

    node_arr = np.stack(all_nodes, axis=0)   # (n_harm, nR, nz, 4)

    return dict(
        Rscale      = float(Rscale),
        lR_grid     = lR_grid.astype(np.float64),
        lz_grid     = lz_grid.astype(np.float64),
        node_arr    = node_arr.astype(np.float64),
        m_arr       = m_arr,
        n_harm      = n_harm,
        mmax        = int(mmax),
        log_scaling = int(log_scaling),
        W_outer     = W_outer.astype(np.float64),
        r0_outer    = float(r0_outer),
        lmax_outer  = int(lmax_outer),
        mmax_outer  = int(mmax_outer),
    )


# ---------------------------------------------------------------------------
# CylSplinePotentialGPU
# ---------------------------------------------------------------------------

class CylSplinePotentialGPU(_GPUPotBase):
    """
    GPU evaluator for an Agama CylSpline potential.

    Replicates Agama's CylSpline::evalCyl with 2D bicubic Hermite splines in
    asinh-scaled coordinates (lR, lz), log-scaling of the m=0 term, and a
    PowerLaw outer asymptotic (lmax=8 fit to grid boundary).

    API matches ``agama.Potential``:
        .potential(xyz, t=0.)   -> Phi
        .force(xyz, t=0.)       -> -gradPhi
        .density(xyz, t=0.)     -> rho  (computed via Laplacian of Hessian)
        .forceDeriv(xyz, t=0.)  -> (force, deriv)
        .evalDeriv(xyz, t=0.)   -> (phi, force, deriv)

    Parameters
    ----------
    coefs : CylSplineCoefs
        Parsed coefficient object (from ``read_cylspl_coefs``).
    """

    def __init__(self, coefs) -> None:
        data = _build_cylspline_data(coefs)

        nR = len(data["lR_grid"])
        nz = len(data["lz_grid"])

        self._d_lR_grid  = cp.asarray(data["lR_grid"])
        self._d_lz_grid  = cp.asarray(data["lz_grid"])
        self._d_node_arr = cp.asarray(data["node_arr"].ravel())
        self._d_m_arr    = cp.asarray(data["m_arr"], dtype=cp.int32)
        self._d_W_outer  = cp.asarray(data["W_outer"])

        self._Rscale      = float(data["Rscale"])
        self._nR          = nR
        self._nz          = nz
        self._n_harm      = int(data["n_harm"])
        self._mmax        = int(data["mmax"])
        self._log_scaling = int(data["log_scaling"])
        self._r0_outer    = float(data["r0_outer"])
        self._lmax_outer  = int(data["lmax_outer"])
        self._mmax_outer  = int(data["mmax_outer"])
        self._inv_4piG    = _INV_4PIG

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_file(cls, path, **kw) -> "CylSplinePotentialGPU":
        """Load from an Agama .coef_cylsp file."""
        try:
            from nbody_streams.agama_helper import read_coefs as _rc
        except ImportError:
            from _coefs import read_coefs as _rc
        return cls(_rc(str(path)), **kw)

    # ---- kernel launch helpers ----------------------------------------------

    def _common_args(self):
        # Scalars must be explicitly typed: CuPy wraps Python int as np.intp
        # (int64) but the kernel declares them as int (int32).  On sm_90 /
        # Hopper the CUDA driver reads the wrong bytes, corrupting base-index
        # calculations (base = mm * nR * nz) and producing garbage / inf output.
        # Explicit np.int32 / np.float64 casts guarantee the correct 4/8-byte
        # values regardless of CuPy or driver version.
        return (
            np.float64(self._Rscale),
            self._d_lR_grid, np.int32(self._nR),
            self._d_lz_grid, np.int32(self._nz),
            self._d_node_arr,
            np.int32(self._n_harm),
            self._d_m_arr,
            np.int32(self._mmax),
            np.int32(self._log_scaling),
            self._d_W_outer, np.float64(self._r0_outer),
            np.int32(self._lmax_outer), np.int32(self._mmax_outer),
        )

    def _unpack_xyz(self, xyz):
        arr, was_single = _prep_xyz(xyz)
        N   = arr.shape[0]
        d_x = cp.ascontiguousarray(arr[:, 0])
        d_y = cp.ascontiguousarray(arr[:, 1])
        d_z = cp.ascontiguousarray(arr[:, 2])
        return d_x, d_y, d_z, N, was_single

    def _launch_eval(self, d_x, d_y, d_z, N: int, do_grad: bool):
        phi_out  = cp.empty(N,     dtype=cp.float64)
        grad_out = cp.empty(3 * N, dtype=cp.float64) if do_grad \
                   else cp.empty(0, dtype=cp.float64)

        kname  = "cylspl_force_kernel" if do_grad else "cylspl_potential_kernel"
        blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        if do_grad:
            _get_cylspl_kernel(kname)(
                (blocks,), (_THREADS_PER_BLOCK,),
                (d_x, d_y, d_z) + self._common_args() + (phi_out, grad_out, np.int32(N)),
            )
        else:
            _get_cylspl_kernel(kname)(
                (blocks,), (_THREADS_PER_BLOCK,),
                (d_x, d_y, d_z) + self._common_args() + (phi_out, np.int32(N)),
            )
        return phi_out, (grad_out if do_grad else None)

    def _launch_hess(self, d_x, d_y, d_z, N: int):
        phi_out  = cp.empty(N,     dtype=cp.float64)
        grad_out = cp.empty(3 * N, dtype=cp.float64)
        hess_out = cp.empty(6 * N, dtype=cp.float64)

        blocks = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _get_cylspl_kernel("cylspl_hess_kernel")(
            (blocks,), (_THREADS_PER_BLOCK,),
            (d_x, d_y, d_z) + self._common_args() + (phi_out, grad_out, hess_out, np.int32(N)),
        )
        return phi_out, grad_out, hess_out

    # ---- public Agama-compatible API ----------------------------------------

    def potential(self, xyz, t: float = 0.0) -> cp.ndarray:
        d_x, d_y, d_z, N, single = self._unpack_xyz(xyz)
        phi, _ = self._launch_eval(d_x, d_y, d_z, N, do_grad=False)
        return phi[0] if single else phi

    def force(self, xyz, t: float = 0.0) -> cp.ndarray:
        d_x, d_y, d_z, N, single = self._unpack_xyz(xyz)
        _, grad = self._launch_eval(d_x, d_y, d_z, N, do_grad=True)
        force = -grad.reshape(N, 3)
        return force[0] if single else force

    def density(self, xyz, t: float = 0.0) -> cp.ndarray:
        """Evaluate rho via Laplacian: rho = -(Hxx+Hyy+Hzz)/(4piG)."""
        d_x, d_y, d_z, N, single = self._unpack_xyz(xyz)
        _, _, hess = self._launch_hess(d_x, d_y, d_z, N)
        H = hess.reshape(N, 6)
        rho = -(H[:, 0] + H[:, 1] + H[:, 2]) * self._inv_4piG
        return rho[0] if single else rho

    def forceDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray]:
        d_x, d_y, d_z, N, single = self._unpack_xyz(xyz)
        _, grad_raw, hess_raw = self._launch_hess(d_x, d_y, d_z, N)
        force = -grad_raw.reshape(N, 3)
        deriv = -hess_raw.reshape(N, 6)
        if single:
            return force[0], deriv[0]
        return force, deriv

    def evalDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        d_x, d_y, d_z, N, single = self._unpack_xyz(xyz)
        phi_raw, grad_raw, hess_raw = self._launch_hess(d_x, d_y, d_z, N)
        force = -grad_raw.reshape(N, 3)
        deriv = -hess_raw.reshape(N, 6)
        if single:
            return phi_raw[0], force[0], deriv[0]
        return phi_raw, force, deriv

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        if der:
            phi_r, force_r, deriv_r = self.evalDeriv(xyz, t)
            results = []
            if pot: results.append(phi_r)
            if acc: results.append(force_r)
            if der: results.append(deriv_r)
        elif acc:
            force_r = self.force(xyz, t)
            results = []
            if pot: results.append(self.potential(xyz, t))
            results.append(force_r)
        else:
            results = [self.potential(xyz, t)]
        return results[0] if len(results) == 1 else tuple(results)


# ---------------------------------------------------------------------------
# CompositePotentialGPU
# ---------------------------------------------------------------------------

class CompositePotentialGPU(_GPUPotBase):
    """
    Sum of GPU potential components : mirrors agama.Potential composite.

    Each component may be any object exposing the standard GPU potential API.
    """

    def __init__(self, components: list) -> None:
        if not components:
            raise ValueError("CompositePotentialGPU requires at least one component.")
        self._components = list(components)

    def __repr__(self):
        return f"CompositePotentialGPU({len(self._components)} components)"

    @classmethod
    def from_agama(cls, pot) -> "CompositePotentialGPU":
        """
        Build from an agama.Potential by materialising it as a single Multipole
        expansion via ``pot.export()``.

        This works for any Agama potential that *Agama itself* can write as a
        Multipole coefficient file : in practice: ``Multipole``, ``King``,
        ``Spheroid``, or any composite that Agama auto-expands on export.

        Analytic potentials (NFW, Plummer, ...) export only ``type=NFW`` with no
        parameters, so ``from_agama`` raises a clear error for those.  Build
        the analytic GPU class directly instead.

        Note on multi-section exports (composite agama.Potential with multiple
        components): Agama writes one ``[Potential]`` block per component.  The
        nbody_streams ``read_coefs`` parser handles only single-component files;
        multi-section files must be passed to Agama directly.  This method will
        raise a ``ValueError`` in that case : split the composite into individual
        Agama components and call ``MultipolePotentialGPU.from_agama()`` on each.
        """
        return cls([MultipolePotentialGPU.from_agama(pot)])

    def potential(self, xyz, t: float = 0.0) -> cp.ndarray:
        result = self._components[0].potential(xyz, t)
        for c in self._components[1:]:
            result = result + c.potential(xyz, t)
        return result

    def force(self, xyz, t: float = 0.0) -> cp.ndarray:
        result = self._components[0].force(xyz, t)
        for c in self._components[1:]:
            result = result + c.force(xyz, t)
        return result

    def density(self, xyz, t: float = 0.0) -> cp.ndarray:
        result = self._components[0].density(xyz, t)
        for c in self._components[1:]:
            result = result + c.density(xyz, t)
        return result

    def forceDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray]:
        F, dF = self._components[0].forceDeriv(xyz, t)
        for c in self._components[1:]:
            Fi, dFi = c.forceDeriv(xyz, t)
            F  = F  + Fi
            dF = dF + dFi
        return F, dF

    def evalDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        phi, F, dF = self._components[0].evalDeriv(xyz, t)
        for c in self._components[1:]:
            pi, Fi, dFi = c.evalDeriv(xyz, t)
            phi = phi + pi
            F   = F   + Fi
            dF  = dF  + dFi
        return phi, F, dF

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        """Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details."""
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        if der:
            phi_r, force_r, deriv_r = self.evalDeriv(xyz, t)
            results = []
            if pot: results.append(phi_r)
            if acc: results.append(force_r)
            if der: results.append(deriv_r)
        elif acc:
            results = []
            if pot: results.append(self.potential(xyz, t))
            results.append(self.force(xyz, t))
        else:
            results = [self.potential(xyz, t)]
        return results[0] if len(results) == 1 else tuple(results)


# ---------------------------------------------------------------------------
# EvolvingPotentialGPU
# ---------------------------------------------------------------------------

class EvolvingPotentialGPU(_GPUPotBase):
    """
    Time-evolving GPU potential: wraps a sequence of static GPU potentials
    at known snapshot times with linear interpolation.

    Parameters
    ----------
    potentials  : list of GPU potential objects
    times       : array-like of float : snapshot times
    interpolate : bool : True (default) = linear; False = nearest snapshot
    """

    def __init__(self, potentials: list, times, interpolate: bool = True) -> None:
        if len(potentials) != len(times):
            raise ValueError(
                f"len(potentials)={len(potentials)} != len(times)={len(times)}"
            )
        if len(potentials) < 1:
            raise ValueError("Need at least one snapshot.")
        self._pots        = list(potentials)
        self._times       = np.asarray(times, dtype=np.float64)
        self._interpolate = interpolate

    def _bracket(self, t: float):
        times = self._times
        n     = len(times)
        if t <= times[0] or n == 1:
            return 0, 0.0
        if t >= times[-1]:
            return n - 2, 1.0
        lo, hi = 0, n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if times[mid + 1] <= t:
                lo = mid + 1
            else:
                hi = mid
        i     = lo
        alpha = float((t - times[i]) / (times[i + 1] - times[i]))
        return i, alpha

    def _interp(self, method_name: str, xyz, t: float):
        n = len(self._pots)
        if n == 1 or not self._interpolate:
            i = int(np.argmin(np.abs(self._times - t)))
            return getattr(self._pots[i], method_name)(xyz, t)
        i, alpha = self._bracket(t)
        if alpha == 0.0:
            return getattr(self._pots[i], method_name)(xyz, t)
        if alpha == 1.0:
            return getattr(self._pots[i + 1], method_name)(xyz, t)
        v0 = getattr(self._pots[i],     method_name)(xyz, t)
        v1 = getattr(self._pots[i + 1], method_name)(xyz, t)
        return _lerp(v0, v1, alpha)

    def potential(self, xyz, t: float = 0.0):
        return self._interp("potential", xyz, t)

    def force(self, xyz, t: float = 0.0):
        return self._interp("force", xyz, t)

    def density(self, xyz, t: float = 0.0):
        return self._interp("density", xyz, t)

    def forceDeriv(self, xyz, t: float = 0.0):
        if not self._interpolate or len(self._pots) == 1:
            i = int(np.argmin(np.abs(self._times - t)))
            return self._pots[i].forceDeriv(xyz, t)
        i, alpha = self._bracket(t)
        F0, dF0 = self._pots[i].forceDeriv(xyz, t)
        if alpha == 0.0:
            return F0, dF0
        F1, dF1 = self._pots[i + 1].forceDeriv(xyz, t)
        return _lerp(F0, F1, alpha), _lerp(dF0, dF1, alpha)

    def evalDeriv(self, xyz, t: float = 0.0):
        if not self._interpolate or len(self._pots) == 1:
            i = int(np.argmin(np.abs(self._times - t)))
            return self._pots[i].evalDeriv(xyz, t)
        i, alpha = self._bracket(t)
        p0, F0, dF0 = self._pots[i].evalDeriv(xyz, t)
        if alpha == 0.0:
            return p0, F0, dF0
        p1, F1, dF1 = self._pots[i + 1].evalDeriv(xyz, t)
        return _lerp(p0, p1, alpha), _lerp(F0, F1, alpha), _lerp(dF0, dF1, alpha)

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        """Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details."""
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        if der:
            phi_r, force_r, deriv_r = self.evalDeriv(xyz, t)
            results = []
            if pot: results.append(phi_r)
            if acc: results.append(force_r)
            if der: results.append(deriv_r)
        elif acc:
            results = []
            if pot: results.append(self.potential(xyz, t))
            results.append(self.force(xyz, t))
        else:
            results = [self.potential(xyz, t)]
        return results[0] if len(results) == 1 else tuple(results)


def _lerp(a, b, alpha: float):
    """Linear interp a*(1-alpha) + b*alpha."""
    return a * (1.0 - alpha) + b * alpha


# ---------------------------------------------------------------------------
# Modifiers: Shifted and Scaled
# ---------------------------------------------------------------------------

class ShiftedPotentialGPU(_GPUPotBase):
    """
    Shifted modifier: evaluates ``inner`` at ``xyz - center(t)``.

    Parameters
    ----------
    inner  : any GPU potential
    center : array-like, two accepted forms (mirrors Agama):

        * **Static** : shape ``(3,)``: ``[x0, y0, z0]``
        * **Trajectory** : shape ``(T, 4)``: each row is ``[t, x, y, z]``.
          Center is linearly interpolated at the requested time.
          Clamped to the first/last entry outside the time range.

    Examples
    --------
    Static::

        ShiftedPotentialGPU(pot_lmc, center=LMC_xyz_100)

    Trajectory (Agama-style)::

        ShiftedPotentialGPU(pot_lmc, center=LMC_traj[0::100, :4])
    """

    def __init__(self, inner, center) -> None:
        self._inner = inner
        center = np.asarray(center, dtype=np.float64)
        
        if center.ndim == 1 and center.shape == (3,):
            self._is_static = True
            self._center_static = center
        elif center.ndim == 2 and center.shape[1] >= 4:
            self._is_static = False
            
            # Monotonically increasing times are required for interpolation; check and enforce this.
            if not np.all(center[1:, 0] > center[:-1, 0]):
                center = center[center[:, 0].argsort()]

            self._times = center[:, 0]
            pos = center[:, 1:4]
            
            if center.shape[1] >= 7:
                # 7-column: Time, Position, Velocity -> Hermite (Cubic matching velocities)
                vel = center[:, 4:7]
                self._spline = CubicHermiteSpline(self._times, pos, vel)
            else:
                # 4-column: Time, Position -> Standard Cubic Spline
                self._spline = CubicSpline(self._times, pos, bc_type='not-a-knot')

            # Cache Boundary Values for Extrapolation
            self._t0 = self._times[0]
            self._tN = self._times[-1]
            
            # Position at boundaries
            self._pos0 = self._spline(self._t0)
            self._posN = self._spline(self._tN)
            
            # Velocity at boundaries (1st derivative)
            # Even for 4-col data, SciPy calculates the derivative at the edge.
            self._vel0 = self._spline(self._t0, 1)
            self._velN = self._spline(self._tN, 1)       
        else:
            raise ValueError("Center must be (3,) or (T, 4) or (T, 7).")

    def _center_at(self, t: float) -> np.ndarray:
        # Case A: Static Center
        if self._is_static:
            return self._center_static
        
        # Case B: Extrapolate Backward (t < start) 
        if t < self._t0:
            dt = t - self._t0
            return self._pos0 + self._vel0 * dt
        
        # Case C: Extrapolate Forward (t > end)
        if t > self._tN:
            dt = t - self._tN
            return self._posN + self._velN * dt
        
        # Case D: Standard Interpolation (inside domain)
        return self._spline(t)

    def _shift(self, xyz, t: float = 0.0):
        # 1. Evaluate center at time t (CPU side, O(1))
        c_cpu = self._center_at(t)
        # 2. Push 3 floats to GPU
        c_gpu = cp.asarray(c_cpu, dtype=xyz.dtype)
        # 3. CuPy elementwise broadcast (extremely fast, VRAM bandwidth bound)
        return xyz - c_gpu

    # --- Pass-through to inner ---
    def potential(self, xyz, t: float = 0.0):
        return self._inner.potential(self._shift(xyz, t), t)

    def force(self, xyz, t: float = 0.0):
        return self._inner.force(self._shift(xyz, t), t)

    def density(self, xyz, t: float = 0.0):
        return self._inner.density(self._shift(xyz, t), t)

    def forceDeriv(self, xyz, t: float = 0.0):
        return self._inner.forceDeriv(self._shift(xyz, t), t)

    def evalDeriv(self, xyz, t: float = 0.0):
        return self._inner.evalDeriv(self._shift(xyz, t), t)

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        return self._inner.eval(self._shift(xyz, t), pot=pot, acc=acc, der=der, t=t)


class ScaledPotentialGPU(_GPUPotBase):
    """
    Scaled modifier: ``Phi_s(x,t) = a(t)*s(t) * Phi(x*s(t))`` where ``s(t) = 1/scale(t)``.

    From Agama's ``potential_composite.cpp``:

    * Phi      scaled by  ``a * s``
    * Force    scaled by  ``a * s^2``
    * Hessian  scaled by  ``a * s^3``

    Parameters
    ----------
    inner : any GPU potential
    scale : float  **or**  array-like
        * ``float`` : static spatial scale factor.
        * ``(T, 2)`` : time-varying scale: each row is ``[t, scale(t)]``.
          CubicSpline fit; linear extrapolation outside the time range.
          ``ampl`` stays at the provided scalar value.
        * ``(T, 3)`` : time-varying scale *and* amplitude: rows ``[t, ampl(t), scale(t)]``.
          Matches Agama's ``scale=`` file format (K=2 values per row).
    ampl : float : static amplitude multiplier (ignored when scale is (T,3), default 1.0)
    """

    def __init__(self, inner, scale, ampl: float = 1.0) -> None:
        self._inner = inner

        # --- detect shape ---
        if isinstance(scale, (int, float)):
            arr = None
        else:
            arr = np.asarray(scale, dtype=np.float64)
            if arr.ndim == 0:
                arr = None

        if arr is None:
            # Static scalar
            self._is_static  = True
            self._scale_val  = float(scale)
            self._ampl_val   = float(ampl)
            return

        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                "scale must be a float, (T,2) array [t, scale(t)], "
                "or (T,3) array [t, ampl(t), scale(t)]. "
                f"Got shape {arr.shape}."
            )

        # Ensure monotonically increasing times
        times = arr[:, 0]
        if not np.all(times[1:] > times[:-1]):
            idx   = np.argsort(times)
            arr   = arr[idx]
            times = arr[:, 0]

        self._is_static = False
        self._times     = times
        self._t0        = float(times[0])
        self._tN        = float(times[-1])

        # Scale spline
        scales           = arr[:, -1]   # last column is always scale
        self._scale_spl  = CubicSpline(times, scales, bc_type='not-a-knot')
        self._scale0     = float(self._scale_spl(self._t0))
        self._scaleN     = float(self._scale_spl(self._tN))
        self._dscale0    = float(self._scale_spl(self._t0, 1))
        self._dscaleN    = float(self._scale_spl(self._tN, 1))

        # Amplitude spline (only for (T,3))
        if arr.shape[1] == 3:
            ampls           = arr[:, 1]
            self._ampl_spl  = CubicSpline(times, ampls, bc_type='not-a-knot')
            self._ampl0     = float(self._ampl_spl(self._t0))
            self._amplN     = float(self._ampl_spl(self._tN))
            self._dampl0    = float(self._ampl_spl(self._t0, 1))
            self._damplN    = float(self._ampl_spl(self._tN, 1))
        else:
            self._ampl_spl = None

        self._ampl_val = float(ampl)   # scalar fallback / override for (T,2) case

    def _sa(self, t: float):
        """Return (s, a) = (1/scale(t), ampl(t)) with linear extrapolation."""
        if self._is_static:
            return 1.0 / self._scale_val, self._ampl_val

        # Scale
        if t < self._t0:
            dt = t - self._t0
            sc = self._scale0 + self._dscale0 * dt
        elif t > self._tN:
            dt = t - self._tN
            sc = self._scaleN + self._dscaleN * dt
        else:
            sc = float(self._scale_spl(t))
        s = 1.0 / sc

        # Amplitude
        if self._ampl_spl is None:
            a = self._ampl_val
        elif t < self._t0:
            dt = t - self._t0
            a  = self._ampl0 + self._dampl0 * dt
        elif t > self._tN:
            dt = t - self._tN
            a  = self._amplN + self._damplN * dt
        else:
            a = float(self._ampl_spl(t))

        return s, a

    def potential(self, xyz, t: float = 0.0):
        s, a = self._sa(t)
        return a * s * self._inner.potential(cp.asarray(xyz) * s, t)

    def force(self, xyz, t: float = 0.0):
        s, a = self._sa(t)
        return a * s * s * self._inner.force(cp.asarray(xyz) * s, t)

    def density(self, xyz, t: float = 0.0):
        s, a = self._sa(t)
        return a * s * s * s * self._inner.density(cp.asarray(xyz) * s, t)

    def forceDeriv(self, xyz, t: float = 0.0):
        s, a  = self._sa(t)
        xs    = cp.asarray(xyz) * s
        F, dF = self._inner.forceDeriv(xs, t)
        return a * s * s * F, a * s * s * s * dF

    def evalDeriv(self, xyz, t: float = 0.0):
        s, a       = self._sa(t)
        xs         = cp.asarray(xyz) * s
        phi, F, dF = self._inner.evalDeriv(xs, t)
        return a * s * phi, a * s * s * F, a * s * s * s * dF

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        if der:
            phi_r, force_r, deriv_r = self.evalDeriv(xyz, t)
            results = []
            if pot: results.append(phi_r)
            if acc: results.append(force_r)
            if der: results.append(deriv_r)
        elif acc:
            results = []
            if pot: results.append(self.potential(xyz, t))
            results.append(self.force(xyz, t))
        else:
            results = [self.potential(xyz, t)]
        return results[0] if len(results) == 1 else tuple(results)


# ---------------------------------------------------------------------------
# Factory: PotentialGPU  (mirrors agama.Potential API)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agama-CPU-routed factory functions
# These build agama.Potential, export to a temp Multipole file, load as GPU.
# ---------------------------------------------------------------------------

def _kl(kw: dict) -> dict:
    """Return a lowercase-no-underscore key dict for case-insensitive lookup."""
    return {k.lower().replace('_', ''): v for k, v in kw.items()}


def _build_spheroid_gpu(**kw):
    """Spheroid via Agama CPU -> Multipole export -> MultipolePotentialGPU.
    Caller may override with lmax=N (kernel supports up to lmax=32)."""
    import agama
    return MultipolePotentialGPU.from_agama(agama.Potential(type='Spheroid', **kw))


def _build_king_gpu(**kw):
    """King via Agama CPU -> Multipole export -> MultipolePotentialGPU."""
    import agama
    return MultipolePotentialGPU.from_agama(agama.Potential(type='King', **kw))


def _build_dehnen_gpu(**kw):
    """
    Dehnen factory (case-insensitive kwargs):
      - gamma in [0, 2] (inclusive).
      - Spherical + gamma != 2  ->  DehnenSphericalPotentialGPU (GPU kernel, fast).
      - Triaxial (axisRatioY or axisRatioZ != 1) or gamma == 2
                                ->  Agama CPU Spheroid(alpha=1, beta=4) -> MultipolePotentialGPU.
    """
    try:
        from nbody_streams.agama_helper._analytic_potentials import DehnenSphericalPotentialGPU as _DehSph
    except ImportError:
        from _analytic_potentials import DehnenSphericalPotentialGPU as _DehSph
    kl = _kl(kw)

    gamma = float(kl.get('gamma', 1.0))
    yr    = float(kl.get('axisratioy', 1.0))
    zr    = float(kl.get('axisratioz', 1.0))

    if not 0.0 <= gamma <= 2.0:
        raise ValueError(f"Dehnen gamma must be in [0, 2], got {gamma}")

    is_spherical = abs(yr - 1.0) < 1e-10 and abs(zr - 1.0) < 1e-10

    if is_spherical and abs(gamma - 2.0) > 1e-10:
        mass = float(kl.get('mass', 1.0))
        a    = float(kl.get('scaleradius', 1.0))
        return _DehSph(mass=mass, scaleRadius=a, gamma=gamma)

    # Triaxial or gamma=2: route through Agama as Spheroid(alpha=1, beta=4).
    import agama
    return MultipolePotentialGPU.from_agama(
        agama.Potential(type='Spheroid', alpha=1, beta=4, **kw)
    )


def _build_disk_gpu(**kw):
    """
    Full Disk potential via Agama CPU: DiskAnsatz + axisymmetric Multipole(lmax=32).

    Agama builds type='Disk' as a composite of DiskAnsatz + Multipole internally and
    exports a two-section INI.  The DiskAnsatz section contains no parameters (Agama
    limitation), so we reconstruct DiskAnsatzPotentialGPU directly from the input kwargs
    and extract only the Multipole section from the export for MultipolePotentialGPU.

    Accepts all Agama Disk kwargs: surfaceDensity, scaleRadius, scaleHeight,
    innerCutoffRadius, sersicIndex, etc.
    """
    import agama, re as _re
    try:
        from nbody_streams.agama_helper._analytic_potentials import DiskAnsatzPotentialGPU as _DiskAnsatz
    except ImportError:
        from _analytic_potentials import DiskAnsatzPotentialGPU as _DiskAnsatz

    kl = _kl(kw)

    # Build DiskAnsatz GPU directly from the original input params (Agama can't export them).
    disk_gpu = _DiskAnsatz(
        surfaceDensity    = float(kl.get('surfacedensity', kl.get('densitynorm', 1.0))),
        scaleRadius       = float(kl.get('scaleradius', 1.0)),
        scaleHeight       = float(kl.get('scaleheight', 0.1)),
        innerCutoffRadius = float(kl.get('innercutoffradius', 0.0)),
    )

    # Build the Agama Disk and export to extract the Multipole section.
    pot_agama = agama.Potential(type='Disk', **kw)
    shm      = "/dev/shm"
    tmp_dir  = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) \
                   else tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"agama_disk_{uuid.uuid4().hex}.coef")
    try:
        pot_agama.export(tmp_path)
        with open(tmp_path, 'r') as _f:
            export_text = _f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    # Split on [Potential] headers and find the Multipole section with inline Coefficients.
    sections = _re.split(r'(?=^\[Potential)', export_text,
                         flags=_re.MULTILINE | _re.IGNORECASE)
    multipole_text = None
    for sec in sections:
        if (_re.search(r'type\s*=\s*Multipole', sec, _re.IGNORECASE)
                and 'Coefficients' in sec):
            multipole_text = sec
            break

    if multipole_text is None:
        raise RuntimeError(
            "Agama Disk export did not contain a Multipole section with Coefficients.\n"
            "Export preview:\n" + export_text[:500]
        )

    # Load Multipole from raw section text (read_coefs accepts raw strings).
    try:
        from nbody_streams.agama_helper import read_coefs as _read_coefs
    except ImportError:
        from _coefs import read_coefs as _read_coefs

    multipole_gpu = MultipolePotentialGPU(_read_coefs(multipole_text))
    return CompositePotentialGPU([disk_gpu, multipole_gpu])


# Map Agama type strings (lowercase) -> analytic GPU class or factory callable.
# Populated after _analytic_potentials is imported on first call.
_ANALYTIC_TYPE_MAP: dict | None = None


def _get_analytic_map() -> dict:
    global _ANALYTIC_TYPE_MAP
    if _ANALYTIC_TYPE_MAP is not None:
        return _ANALYTIC_TYPE_MAP
    try:
        from nbody_streams.agama_helper._analytic_potentials import (
            NFWPotentialGPU,
            PlummerPotentialGPU,
            HernquistPotentialGPU,
            IsochronePotentialGPU,
            MiyamotoNagaiPotentialGPU,
            LogHaloPotentialGPU,
            DiskAnsatzPotentialGPU,
            UniformAccelerationGPU,
        )
    except ImportError:
        from _analytic_potentials import (  # noqa: E402
            NFWPotentialGPU,
            PlummerPotentialGPU,
            HernquistPotentialGPU,
            IsochronePotentialGPU,
            MiyamotoNagaiPotentialGPU,
            LogHaloPotentialGPU,
            DiskAnsatzPotentialGPU,
            UniformAccelerationGPU,
        )
    _ANALYTIC_TYPE_MAP = {
        "nfw":                  NFWPotentialGPU,
        "plummer":              PlummerPotentialGPU,
        "hernquist":            HernquistPotentialGPU,
        "dehnen":               _build_dehnen_gpu,
        "isochrone":            IsochronePotentialGPU,
        "miyamotonagai":        MiyamotoNagaiPotentialGPU,
        "loghalo":              LogHaloPotentialGPU,
        "logarithmic":          LogHaloPotentialGPU,
        "diskansatz":           DiskAnsatzPotentialGPU,
        "uniformacceleration":  UniformAccelerationGPU,
        # Full Disk (DiskAnsatz + axisymmetric Multipole lmax=32) via Agama CPU export
        "disk":                 _build_disk_gpu,
        "spheroid":             _build_spheroid_gpu,
        "king":                 _build_king_gpu,
    }
    return _ANALYTIC_TYPE_MAP


def _apply_modifiers(pot, center, scale, ampl):
    """Wrap *pot* with Shifted/Scaled modifiers if requested.

    ``scale`` may be:
    * ``None`` or a float : passed through as-is.
    * A string ``"v"`` or ``"a s"`` (from INI parsing) : parsed as a single
      scale factor or as ``"ampl scale"`` pair (Agama's K=2 inline format).
    * A (T,2) or (T,3) ndarray : time-varying; forwarded directly.
    """
    if scale is not None or ampl != 1.0:
        # Handle string values that come from INI key=value parsing
        if isinstance(scale, str):
            parts = scale.strip().split()
            if len(parts) == 1:
                scale = float(parts[0])
            elif len(parts) == 2:
                # Agama inline format: "ampl scale_factor"
                ampl  = float(parts[0])
                scale = float(parts[1])
            else:
                raise ValueError(
                    f"Cannot parse scale= value {scale!r}. "
                    "Expected a float, 'ampl scale' pair, or (T,2)/(T,3) array."
                )
        pot = ScaledPotentialGPU(pot, scale=scale if scale is not None else 1.0, ampl=ampl)
    if center is not None:
        pot = ShiftedPotentialGPU(pot, center=center)
    return pot


def _coerce(v: str):
    """Convert a string value from an INI file to int / float / str."""
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    return v


# Canonical camelCase param names keyed by their lowercased+no-underscore form.
# Used to normalise INI key=value pairs before dispatching to GPU constructors.
_CANONICAL_PARAM: dict[str, str] = {
    # Only GPU-native class constructor params are listed here.
    # Agama-routed types (Spheroid, King, Disk, Dehnen-triaxial) pass kwargs
    # directly to agama.Potential which is case-insensitive — no remapping needed.
    "mass":               "mass",
    "scaleradius":        "scaleRadius",
    "scaleheight":        "scaleHeight",
    "gamma":              "gamma",
    "alpha":              "alpha",
    "beta":               "beta",
    "velocity":           "velocity",
    "v0":                 "velocity",   # Agama Logarithmic uses v0; GPU class uses velocity
    "coreradius":         "coreRadius",
    "axisratioy":         "axisRatioY",
    "axisratioz":         "axisRatioZ",
    "surfacedensity":     "surfaceDensity",
    # NOTE: densityNorm is intentionally NOT aliased here.  For Spheroid/NFW
    # it is a distinct Agama parameter that must reach agama.Potential unchanged.
    # _build_disk_gpu handles densityNorm → surfaceDensity internally via _kl.
    "innercutoffradius":  "innerCutoffRadius",
    "outercutoffradius":  "outerCutoffRadius",
    "cutoffstrength":     "cutoffStrength",
    "w0":                 "W0",
    "trunc":              "trunc",
    "sersicindex":        "sersicIndex",
    "lmax":               "lmax",
    "gridsizer":          "gridSizeR",
    "gridsizez":          "gridSizeZ",
    "ax":                 "ax",
    "ay":                 "ay",
    "az":                 "az",
    "file":               "file",
    "potential":          "potential",
    "interplinear":       "interpLinear",
}


def _normalize_params(d: dict) -> dict:
    """Return *d* with keys mapped to their canonical camelCase equivalents.

    Keys not in ``_CANONICAL_PARAM`` are preserved verbatim so that unknown
    Agama params pass through unchanged.
    """
    out: dict = {}
    for k, v in d.items():
        norm = k.lower().replace("_", "").replace(" ", "")
        out[_CANONICAL_PARAM.get(norm, k)] = v
    return out


def _pop_ci(d: dict, key: str, default=None):
    """Case-insensitive pop from *d* (first matching key wins)."""
    for k in list(d.keys()):
        if k.lower() == key.lower():
            return d.pop(k)
    return default


def _is_potential_ini(p: Path) -> bool:
    """True when *p* looks like an Agama multi-section potential INI file."""
    if not p.exists():
        return False
    if p.suffix.lower() in ('.ini', '.pot'):
        return True
    # Peek at content : coef files start with '[Multipole]' or numeric data
    try:
        head = p.read_text(encoding='utf-8', errors='ignore')[:512]
    except OSError:
        return False
    return bool(__import__('re').search(r'^\[Potential', head, __import__('re').IGNORECASE | __import__('re').MULTILINE))


def _load_potential_ini(p: Path):
    """
    Parse an Agama-style INI potential file (one or more ``[Potential ...]``
    sections) and return a single GPU potential or CompositePotentialGPU.

    Handles:
    * ``type=Multipole`` with **inline Coefficients** block : passes raw section
      text directly to ``read_coefs`` (avoids any temp-file round-trip).
    * ``type=Multipole`` with ``file=`` reference : loads the referenced coef file
      via ``MultipolePotentialGPU.from_file``, resolving relative paths w.r.t. *p*.
    * ``type=Evolving`` with ``Timestamps`` block : parses time/filename pairs and
      builds ``EvolvingPotentialGPU``.
    * ``type=DiskAnsatz`` : **skipped** (Agama exports this without parameters; use
      ``type=Disk`` to get a full DiskAnsatz+Multipole composite via ``_build_disk_gpu``).
    * Everything else : dispatched via ``_build_single(params_dict)``.
    """
    import re as _re
    base = p.parent
    text  = p.read_text(encoding='utf-8')
    lines = text.splitlines()

    # Find all [Potential ...] section start indices
    starts = [i for i, ln in enumerate(lines)
               if _re.match(r'^\s*\[Potential', ln, _re.IGNORECASE)]
    if not starts:
        raise ValueError(f"No [Potential] sections found in {p}.")

    built = []
    for idx, start in enumerate(starts):
        end           = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        section_lines = lines[start:end]

        # Parse key=value params; stop at Coefficients or Timestamps data blocks
        params     = {}
        data_start = None       # line index (within section_lines) of data block
        data_kind  = None       # 'coef' | 'ts'
        for j, ln in enumerate(section_lines[1:], start=1):
            s = ln.strip()
            if not s or s.startswith('#') or s.startswith(';'):
                continue
            if s.lower() == 'coefficients':
                data_start = j
                data_kind  = 'coef'
                break
            if s.lower() == 'timestamps':
                data_start = j
                data_kind  = 'ts'
                break
            if '=' in s:
                k, _, v = s.partition('=')
                k_s = k.strip()
                # Strip inline comments from value
                v_s = v.split('#')[0].strip()
                params[k_s] = _coerce(v_s)

        _type_key = next((k for k in params if k.lower() == 'type'), None)
        type_ = str(params.get(_type_key, '')).lower().replace(' ', '').replace('_', '')

        # ---- DiskAnsatz: no stored params : skip silently ----
        if type_ == 'diskansatz':
            continue

        # ---- Multipole: inline coefficients ----
        if type_ == 'multipole' and data_kind == 'coef':
            section_text = '\n'.join(section_lines)
            try:
                from nbody_streams.agama_helper import read_coefs as _rc
            except ImportError:
                from _coefs import read_coefs as _rc
            built.append(MultipolePotentialGPU(_rc(section_text)))
            continue

        # ---- Multipole: file= reference ----
        if type_ == 'multipole':
            coef_file = str(params.get('file') or params.get('File', ''))
            if not coef_file:
                raise ValueError(
                    f"Multipole section in {p} has neither inline Coefficients "
                    "nor a 'file' key."
                )
            coef_path = Path(coef_file)
            if not coef_path.is_absolute():
                coef_path = base / coef_path
            built.append(MultipolePotentialGPU.from_file(coef_path))
            continue

        # ---- CylSpline: inline Coefficients or file= reference ----
        if type_ == 'cylspline':
            if data_kind == 'coef':
                section_text = '\n'.join(section_lines)
                try:
                    from nbody_streams.agama_helper import read_coefs as _rc
                except ImportError:
                    from _coefs import read_coefs as _rc
                built.append(CylSplinePotentialGPU(_rc(section_text)))
            else:
                coef_file = str(params.get('file') or params.get('File', ''))
                if not coef_file:
                    raise ValueError(
                        f"CylSpline section in {p} has neither inline Coefficients "
                        "nor a 'file' key."
                    )
                coef_path = Path(coef_file)
                if not coef_path.is_absolute():
                    coef_path = base / coef_path
                built.append(CylSplinePotentialGPU.from_file(coef_path))
            continue

        # ---- Evolving: parse Timestamps block ----
        if type_ == 'evolving':
            if data_kind != 'ts':
                raise ValueError(
                    f"Evolving section in {p} is missing a 'Timestamps' block."
                )
            times = []
            pots  = []
            for ln in section_lines[data_start + 1:]:
                s = ln.strip()
                if not s or s.startswith('#') or s.startswith(';'):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                try:
                    t_val = float(parts[0])
                except ValueError:
                    continue
                snap_path = Path(parts[1])
                if not snap_path.is_absolute():
                    snap_path = base / snap_path
                times.append(t_val)
                pots.append(_build_single(snap_path, {}))
            if not pots:
                raise ValueError(
                    f"Evolving section in {p} has no valid Timestamps entries."
                )
            interp_linear = bool(
                params.get('linearInterp', params.get('interpLinear', False))
            )
            built.append(EvolvingPotentialGPU(pots, times, interpolate=interp_linear))
            continue

        # ---- Everything else: dispatch via _build_single(dict) ----
        built.append(_build_single(params, {}))

    if not built:
        raise ValueError(
            f"No buildable components found in {p}. "
            "(All sections may have been DiskAnsatz stubs with no stored params.)"
        )
    return built[0] if len(built) == 1 else CompositePotentialGPU(built)


def _build_single(source, pot_kw: dict):
    """Dispatch a single source to the appropriate GPU potential class."""
    try:
        from nbody_streams.agama_helper._coefs import MultipoleCoefs, CylSplineCoefs
    except ImportError:
        MultipoleCoefs = None
        CylSplineCoefs = None

    if MultipoleCoefs is not None and isinstance(source, MultipoleCoefs):
        return MultipolePotentialGPU(source, **pot_kw)

    if CylSplineCoefs is not None and isinstance(source, CylSplineCoefs):
        return CylSplinePotentialGPU(source, **pot_kw)

    # dict : e.g. dict(type='Disk', surfaceDensity=...) : Agama-style component spec
    if isinstance(source, dict):
        d      = dict(source)
        type_s = _pop_ci(d, 'type')
        if type_s is None:
            raise ValueError("Component dict must include a 'type' key.")
        type_k = str(type_s).lower().replace(' ', '').replace('_', '')
        # modifier keys that belong to PotentialGPU, not the constructor
        cen_  = _pop_ci(d, 'center')
        sc_   = _pop_ci(d, 'scale')
        am_   = float(_pop_ci(d, 'ampl') or 1.0)
        # Do NOT normalize here: Agama-routed factories (Spheroid, King, Disk)
        # pass kwargs directly to agama.Potential which is case-insensitive.
        # GPU-native constructors (NFW, Plummer, …) normalize inside PotentialGPU.
        # Expansion types with a file= reference must bypass PotentialGPU(type=)
        # because that path only handles analytic types.
        if type_k == 'cylspline':
            coef_ref = str(d.pop('file', None) or '')
            if not coef_ref:
                raise ValueError("CylSpline component dict requires a 'file' key.")
            pot = CylSplinePotentialGPU.from_file(coef_ref)
        elif type_k == 'multipole':
            coef_ref = str(d.pop('file', None) or '')
            if not coef_ref:
                raise ValueError("Multipole component dict requires a 'file' key.")
            pot = MultipolePotentialGPU.from_file(coef_ref)
        else:
            pot = PotentialGPU(type=type_s, **d)
        return _apply_modifiers(pot, cen_, sc_, am_)

    if isinstance(source, (str, Path)):
        p = Path(source)
        # Check for Agama-style coefficient/INI file (starts with [Potential])
        if _is_potential_ini(p):
            return _load_potential_ini(p)
        return MultipolePotentialGPU.from_file(source, **pot_kw)

    # Already a GPU potential object (any class) : pass through
    if callable(getattr(source, 'potential', None)) and callable(getattr(source, 'force', None)):
        return source

    if type(source).__name__ == "Potential":
        try:
            n = source.nComponents
        except AttributeError:
            n = 1
        if n > 1:
            return CompositePotentialGPU(
                [_build_single(source[i], pot_kw) for i in range(n)]
            )
        return MultipolePotentialGPU.from_agama(source, **pot_kw)

    raise TypeError(
        f"Cannot build a GPU potential from {type(source).__name__!r}. "
        "Pass a file path, MultipoleCoefs, agama.Potential, GPU potential object, "
        "or use type= for analytic potentials."
    )


def PotentialGPU(*args,
                 type:   str   | None = None,
                 file:   str   | None = None,
                 center        = None,
                 scale:  float | None = None,
                 ampl:   float        = 1.0,
                 **kw):
    """
    GPU potential factory : mirrors the ``agama.Potential`` API.

    Usage
    -----
    Analytic potential::

        PotentialGPU(type='NFW', mass=1e12, scaleRadius=20)
        PotentialGPU(type='Plummer', mass=5e10, scaleRadius=3)

    From file (Multipole coef or HDF5)::

        PotentialGPU(file='path/to/snap.coef_mult')
        PotentialGPU(file='path/to/snap.h5')

    From MultipoleCoefs dataclass or agama.Potential::

        PotentialGPU(coefs_obj)
        PotentialGPU(agama_pot)

    Composite : multiple positional arguments::

        PotentialGPU(pot_mw, pot_disk, pot_lmc)   # any mix of GPU pot objects
        PotentialGPU(coefs_mw, coefs_disk)

    Modifiers applied after building::

        PotentialGPU(coefs_lmc, center=LMC_traj[:, :4])   # time-varying shift
        PotentialGPU(coefs_lmc, center=[x0, y0, z0])       # static shift
        PotentialGPU(coefs_lmc, scale=2.0, ampl=0.5)       # scaled

    Parameters
    ----------
    *args
        One or more sources (file paths, coef objects, GPU pot objects,
        agama.Potential).  Two or more args -> CompositePotentialGPU.
    type : str, optional
        Agama-style type name (case-insensitive): 'NFW', 'Plummer',
        'Hernquist', 'Dehnen', 'Isochrone', 'MiyamotoNagai', 'LogHalo',
        'DiskAnsatz', 'Disk', 'Spheroid', 'King', 'UniformAcceleration'.
        'Disk' builds the full DiskAnsatz+Multipole composite via Agama CPU
        (supports innerCutoffRadius).  Remaining kwargs forwarded to the
        constructor or Agama CPU builder.
    file : str or Path, optional
        Shorthand for ``PotentialGPU('path/to/file')``.  Accepts:

        * A Multipole coefficient file (``.coef_mul_DR``, starts with
          ``[Potential]\\ntype=Multipole``).
        * A multi-section Agama INI file containing any mix of analytic
          types, ``type=Multipole`` (inline or ``file=`` reference), and
          ``type=Evolving`` (with ``Timestamps`` block) : returns a
          ``CompositePotentialGPU`` or ``EvolvingPotentialGPU`` as appropriate.
    center : array-like (3,) or (T, 4) or (T, 7), optional
        Wrap the result in ``ShiftedPotentialGPU``.
        Shape (3,) -> static shift; shape (T, 4) -> time-varying trajectory
        with columns [t, x, y, z] interpolated with regularized cubic spline 
        or [t, x, y, z, vx, vy, vz] inteproplated with Hermite spline.
    scale : float, optional
        Wrap the result in ``ScaledPotentialGPU(scale=scale, ampl=ampl)``.
    ampl : float, optional
        Amplitude multiplier for ``ScaledPotentialGPU`` (default 1.0).
    **kw
        For analytic types: constructor kwargs (mass, scaleRadius, ...).
        For expansion types: forwarded to ``MultipolePotentialGPU``.
    """
    # --- collect modifier kwargs, leave rest for the potential constructor ---
    pot_kw = kw   # forwarded to expansion constructors; analytic uses kw directly

    # --- multiple positional args -> composite ---
    if len(args) > 1:
        components = [_build_single(a, pot_kw) for a in args]
        pot = CompositePotentialGPU(components)
        return _apply_modifiers(pot, center, scale, ampl)

    # --- type= dispatch ---
    if type is not None:
        key = type.lower().replace(" ", "").replace("_", "")
        # Expansion types are not analytic; they require a coefficient file.
        # Route them directly so that file= is honoured instead of being ignored
        # by the analytic map lookup below.
        if key == 'cylspline':
            if file is None:
                raise ValueError(
                    "type='CylSpline' requires file= to specify the coefficient file."
                )
            pot = CylSplinePotentialGPU.from_file(file, **pot_kw)
            return _apply_modifiers(pot, center, scale, ampl)
        if key == 'multipole':
            if file is None:
                raise ValueError(
                    "type='Multipole' requires file= to specify the coefficient file."
                )
            pot = MultipolePotentialGPU.from_file(file, **pot_kw)
            return _apply_modifiers(pot, center, scale, ampl)
        amap = _get_analytic_map()
        cls  = amap.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown GPU potential type {type!r}. "
                f"Available: {sorted(amap.keys())}"
            )
        import inspect as _inspect
        if _inspect.isclass(cls):
            # GPU-native analytic class: normalize kwargs to canonical camelCase
            # so that mixed-case INI params (Mass=, ScaleRadius=) work.
            pot = cls(**_normalize_params(kw))
        else:
            # Agama-routed factory (_build_spheroid_gpu, _build_disk_gpu, …):
            # pass kwargs through unchanged — agama.Potential is case-insensitive
            # and the factory handles its own internal normalization via _kl.
            pot = cls(**kw)
        return _apply_modifiers(pot, center, scale, ampl)

    # --- file= shorthand ---
    if file is not None:
        p = Path(file)
        pot = _load_potential_ini(p) if _is_potential_ini(p) \
              else MultipolePotentialGPU.from_file(file, **pot_kw)
        return _apply_modifiers(pot, center, scale, ampl)

    # --- single positional arg ---
    if len(args) == 1:
        pot = _build_single(args[0], pot_kw)
        return _apply_modifiers(pot, center, scale, ampl)

    raise TypeError(
        "PotentialGPU() requires at least one argument, type=, or file=."
    )
