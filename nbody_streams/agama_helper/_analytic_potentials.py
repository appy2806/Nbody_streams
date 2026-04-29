"""
_analytic_potentials.py
~~~~~~~~~~~~~~~~~~~~~~~
Phase 2: GPU-accelerated analytic potential models using CuPy element-wise kernels.

Each class matches the Agama Python constructor API as closely as possible:
    NFWPotentialGPU(mass, scaleRadius)
    PlummerPotentialGPU(mass, scaleRadius)
    HernquistPotentialGPU(mass, scaleRadius)
    IsochronePotentialGPU(mass, scaleRadius)
    MiyamotoNagaiPotentialGPU(mass, scaleRadius, scaleHeight)
    LogHaloPotentialGPU(velocity, coreRadius, axisRatioY=1, axisRatioZ=1)
    DehnenSphericalPotentialGPU(mass, scaleRadius, gamma=1)
    DiskAnsatzPotentialGPU(surfaceDensity, scaleRadius, scaleHeight, innerCutoffRadius=0)
    UniformAccelerationGPU(ax=0, ay=0, az=0)

All classes expose:
    .potential(xyz, t=0.)   -> (N,) or scalar CuPy float64    [(km/s)^2]
    .force(xyz, t=0.)       -> (N,3) or (3,) CuPy float64     [(km/s)^2/kpc]
    .density(xyz, t=0.)     -> (N,) or scalar CuPy float64    [Msol/kpc^3]
    .forceDeriv(xyz, t=0.)  -> (force, deriv)  deriv: [dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx]
    .evalDeriv(xyz, t=0.)   -> (phi, force, deriv)
    .from_agama(pot)        -> class method: build from agama.Potential object

Units: Agama convention --- mass=Msol, length=kpc, velocity=km/s.
    G = 4.30091727067736e-06  kpc (km/s)^2 / Msol

Note on King and Spheroid:
    These are materialised as Multipole expansions by Agama at construction time.
    Use MultipolePotentialGPU.from_agama(agama.Potential(type='King', ...))
    to load them as GPU Multipole potentials.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np

try:
    import cupy as cp
except ImportError as _err:
    raise ImportError("CuPy is required.") from _err

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_G       = 4.30091727067736e-06 # kpc (km/s)^2 / Msol
_INV_4PIG = 1.0 / (4.0 * math.pi * _G)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prep_xyz(xyz) -> Tuple[cp.ndarray, bool]:
    xyz = cp.asarray(xyz, dtype=cp.float64)
    if xyz.ndim == 1:
        if xyz.shape[0] != 3:
            raise ValueError(f"Single-point xyz must have shape (3,), got {xyz.shape}")
        return xyz.reshape(1, 3), True
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3), got {xyz.shape}")
    return cp.ascontiguousarray(xyz), False


def _squeeze(arr, single):
    return arr[0] if single else arr


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _AnalyticBase:
    """Abstract base --- subclasses implement _phi, _grad, _hess, _rho."""

    def __add__(self, other):
        try:
            from nbody_streams.agama_helper._potential import CompositePotentialGPU
        except ImportError:
            from _potential import CompositePotentialGPU
        from_self  = self._components  if isinstance(self,  CompositePotentialGPU) else [self]
        from_other = other._components if isinstance(other, CompositePotentialGPU) else [other]
        return CompositePotentialGPU(from_self + from_other)

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def potential(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        phi = self._phi(arr[:, 0], arr[:, 1], arr[:, 2])
        return _squeeze(phi, single)

    def force(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        # We pass the shape directly to the grad function
        # and let the subclass return the finished 2D array
        out = -self._grad(arr[:, 0], arr[:, 1], arr[:, 2]) # force = -grad(Phi)
        return _squeeze(out, single)

    def density(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        rho = self._rho(arr[:, 0], arr[:, 1], arr[:, 2])
        return _squeeze(rho, single)

    def forceDeriv(self, xyz, t=0.0):
            arr, single = _prep_xyz(xyz)
            x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
            # Get both in their final stacked forms directly from kernels
            f = - self._grad(x, y, z)
            h = self._hess(x, y, z) 
            if single:
                return f[0], -h[0]
            return f, -h
    
    def evalDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        arr, single = _prep_xyz(xyz)
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        phi = self._phi(x, y, z)
        grad = self._grad(x, y, z)
        h = self._hess(x, y, z)
        # Physics mapping: Force = -Grad, Deriv = -Hessian
        force = -grad
        deriv = -h

        if single:
            return phi[0], force[0], deriv[0]
        return phi, force, deriv

    def eval(self, xyz, pot: bool = False, acc: bool = False,
             der: bool = False, t: float = 0.0):
        """
        Agama-compatible eval --- returns any combination of potential, acceleration,
        and its derivatives in a single call.

        Matches ``agama.Potential.eval(xyz, pot=False, acc=False, der=False, t=0)``.

        Returns a single array when only one quantity is requested, otherwise a
        tuple in the order (phi, force, deriv) for whichever subset is requested.

        Parameters
        ----------
        pot : bool   --- include potential Phi  [(km/s)^2]
        acc : bool   --- include acceleration -gradPhi  [(km/s)^2/kpc]
        der : bool   --- include force derivatives -d2Phi/dxidxj  shape (N,6)
        """
        if not (pot or acc or der):
            raise ValueError("eval(): at least one of pot, acc, der must be True.")
        arr, single = _prep_xyz(xyz)
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        results = []
        if pot:
            results.append(_squeeze(self._phi(x, y, z), single))
        if acc:
            results.append(_squeeze(-self._grad(x, y, z), single))
        if der:
            results.append(_squeeze(-self._hess(x, y, z), single))
        return results[0] if len(results) == 1 else tuple(results)

    # Subclasses must implement:
    def _phi(self, x, y, z): raise NotImplementedError
    def _grad(self, x, y, z): raise NotImplementedError   # returns (dPhi/dx, dPhi/dy, dPhi/dz)
    def _hess(self, x, y, z): raise NotImplementedError   # returns (N,6) [Hxx,Hyy,Hzz,Hxy,Hyz,Hxz]
    def _rho(self, x, y, z): raise NotImplementedError


# ---------------------------------------------------------------------------
# NFW
# ---------------------------------------------------------------------------

# --- Fused Kernels ---
# We define these outside the class so they are compiled only once.

_nfw_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T rs', 'T out',
    '''
    T r_raw = sqrt(x*x + y*y + z*z);
    T r = (r_raw < 1e-300) ? (T)1e-300 : r_raw; 
    out = -GM * log1p(r / rs) / r;
    ''', 'nfw_phi'
)

_nfw_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T M, T rs', 'T out',
    '''
    T r_raw = sqrt(x*x + y*y + z*z);
    // Ternary: (condition) ? value_if_true : value_if_false
    T r = (r_raw < (T)1e-300) ? (T)1e-300 : r_raw; 
    
    T rrs = r + rs;
    // Using a literal for PI is safer than relying on M_PI being defined
    out = M / (12.566370614359172 * r * rrs * rrs); 
    ''', 'nfw_rho'
)

# 'T gx, T gy, T gz' becomes one 2D array 'T out'
# We access it using the index 'i' (the point index)
_nfw_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T rs',
    'raw T out', # 'raw' allows us to index manually
    '''
    T r = sqrt(x*x + y*y + z*z);
    T r_safe = (r < (T)1e-300) ? (T)1e-300 : r;
    T dphi_dr = GM * (log1p(r_safe/rs)/(r_safe*r_safe) - 1.0/(r_safe*(r_safe+rs)));
    
    T inv_r = 1.0 / r_safe;
    // We write directly into the final memory block
    out[i*3 + 0] = dphi_dr * x * inv_r; // Positive dPhi/dx
    out[i*3 + 1] = dphi_dr * y * inv_r; // Positive dPhi/dy
    out[i*3 + 2] = dphi_dr * z * inv_r; // Positive dPhi/dz
    ''', 'nfw_grad'
)

_nfw_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T rs',
    'raw T out', 
    '''
    T r2_raw = x*x + y*y + z*z;
    T r2 = (r2_raw < (T)1e-300) ? (T)1e-300 : r2_raw;
    T r = sqrt(r2);
    T rrs = r + rs;
    T ln_term = log1p(r / rs);
    
    // Physics: dPhi/dr and d2Phi/dr2
    T d1 = GM * (ln_term/r2 - 1.0/(r*rrs));
    T d2 = GM * (-2.0*ln_term/(r2*r) + 1.0/(r2*rrs) + (2.0*r + rs)/(r2*rrs*rrs));
    
    // Geometry: Cartesian Hessian components
    T inv_r2 = 1.0 / r2;
    T d1_r = d1 / r;
    T A = (d2 - d1_r) * inv_r2;
    
    // Standard Layout: [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
    out[i*6 + 0] = A*x*x + d1_r; // Hxx
    out[i*6 + 1] = A*y*y + d1_r; // Hyy
    out[i*6 + 2] = A*z*z + d1_r; // Hzz
    out[i*6 + 3] = A*x*y;        // Hxy
    out[i*6 + 4] = A*y*z;        // Hyz
    out[i*6 + 5] = A*x*z;        // Hxz
    ''', 'nfw_hess'
)

class NFWPotentialGPU(_AnalyticBase):
    """
    Navarro-Frenk-White potential optimized with CuPy ElementwiseKernels.
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._rs = float(scaleRadius)

    @classmethod
    def from_agama(cls, pot) -> "NFWPotentialGPU":
        raise TypeError(
            "NFWPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters (mass, scaleRadius, …).\n"
            "Construct directly:  NFWPotentialGPU(mass=..., scaleRadius=...)"
        )

    def _phi(self, x, y, z):
        return _nfw_phi_kernel(x, y, z, self._GM, self._rs)

    def _rho(self, x, y, z):
        return _nfw_rho_kernel(x, y, z, self._M, self._rs)

    def _grad(self, x, y, z):
            # Allocate the memory once
            out = cp.empty((x.size, 3), dtype=x.dtype)
            # Kernel writes directly into 'out'
            _nfw_grad_kernel(x, y, z, self._GM, self._rs, out)
            return out

    def _hess(self, x, y, z):
            N = x.size
            out = cp.empty((N, 6), dtype=x.dtype)
            # Similar logic for Hessian kernel using 'raw T out'
            _nfw_hess_kernel(x, y, z, self._GM, self._rs, out)
            return out


# ---------------------------------------------------------------------------
# Plummer
# ---------------------------------------------------------------------------

_plummer_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b2', 'T out',
    '''
    T r2 = x*x + y*y + z*z;
    out = -GM / sqrt(r2 + b2);
    ''', 'plummer_phi'
)

_plummer_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T M, T b2', 'T out',
    '''
    T r2 = x*x + y*y + z*z;
    T s2 = r2 + b2;
    // rho = 3M/(4pi) * b² / (r²+b²)^(2.5)
    out = (3.0 * M / 12.566370614359172) * b2 / pow(s2, 2.5);
    ''', 'plummer_rho'
)

_plummer_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b2',
    'raw T out',
    '''
    T r2 = x*x + y*y + z*z;
    T s2 = r2 + b2;
    T c = GM / (s2 * sqrt(s2)); // GM/s^3
    
    out[i*3 + 0] = c * x; // dPhi/dx
    out[i*3 + 1] = c * y; // dPhi/dy
    out[i*3 + 2] = c * z; // dPhi/dz
    ''', 'plummer_grad'
)

_plummer_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b2',
    'raw T out',
    '''
    T r2 = x*x + y*y + z*z;
    T s2 = r2 + b2;
    T s  = sqrt(s2);
    T s3 = s2 * s;
    T s5 = s3 * s2;

    T A = -3.0 * GM / s5;
    T B = GM / s3;

    out[i*6 + 0] = A*x*x + B; // Hxx
    out[i*6 + 1] = A*y*y + B; // Hyy
    out[i*6 + 2] = A*z*z + B; // Hzz
    out[i*6 + 3] = A*x*y;     // Hxy
    out[i*6 + 4] = A*y*z;     // Hyz
    out[i*6 + 5] = A*x*z;     // Hxz
    ''', 'plummer_hess'
)

class PlummerPotentialGPU(_AnalyticBase):
    """
    Plummer sphere: Phi(r) = -GM / sqrt(r^2 + b^2)

    Constructor: PlummerPotentialGPU(mass=1, scaleRadius=1)
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._b2 = float(scaleRadius) * float(scaleRadius)

    @classmethod
    def from_agama(cls, pot) -> "PlummerPotentialGPU":
        raise TypeError(
            "PlummerPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  PlummerPotentialGPU(mass=..., scaleRadius=...)"
        )

    def _phi(self, x, y, z):
            return _plummer_phi_kernel(x, y, z, self._GM, self._b2)

    def _rho(self, x, y, z):
        return _plummer_rho_kernel(x, y, z, self._M, self._b2)

    def _grad(self, x, y, z):
        # Return -grad(Phi) = Force
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _plummer_grad_kernel(x, y, z, self._GM, self._b2, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _plummer_hess_kernel(x, y, z, self._GM, self._b2, out)
        return out

# ---------------------------------------------------------------------------
# Hernquist
# ---------------------------------------------------------------------------

_hernquist_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a', 'T out',
    '''
    T r = sqrt(x*x + y*y + z*z);
    out = -GM / (r + a);
    ''', 'hernquist_phi'
)

_hernquist_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T M, T a', 'T out',
    '''
    T r_raw = sqrt(x*x + y*y + z*z);
    T r = (r_raw < (T)1e-300) ? (T)1e-300 : r_raw;
    // rho = M*a / (2*pi * r * (r+a)^3)
    out = (M * a) / (6.283185307179586 * r * pow(r + a, 3));
    ''', 'hernquist_rho'
)

_hernquist_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a',
    'raw T out',
    '''
    T r_raw = sqrt(x*x + y*y + z*z);
    T r = (r_raw < (T)1e-300) ? (T)1e-300 : r_raw;
    
    // dPhi/dr = GM / (r+a)^2
    T ra = r + a;
    T f_coeff = GM / (ra * ra * r); 
    
    out[i*3 + 0] = f_coeff * x; // Grad X
    out[i*3 + 1] = f_coeff * y; // Grad Y
    out[i*3 + 2] = f_coeff * z; // Grad Z
    ''', 'hernquist_grad'
)

_hernquist_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a',
    'raw T out',
    '''
    T r2_raw = x*x + y*y + z*z;
    T r2 = (r2_raw < (T)1e-300) ? (T)1e-300 : r2_raw;
    T r = sqrt(r2);
    T ra = r + a;
    
    // Physics:
    // d1 = dPhi/dr = GM / (r+a)^2
    // d2 = d2Phi/dr2 = -2GM / (r+a)^3
    T d1 = GM / (ra * ra);
    T d2 = -2.0 * GM / (ra * ra * ra);
    
    // Geometry: Cartesian Hessian
    T d1_r = d1 / r;
    T A = (d2 - d1_r) / r2;
    
    out[i*6 + 0] = A*x*x + d1_r; // Hxx
    out[i*6 + 1] = A*y*y + d1_r; // Hyy
    out[i*6 + 2] = A*z*z + d1_r; // Hzz
    out[i*6 + 3] = A*x*y;        // Hxy
    out[i*6 + 4] = A*y*z;        // Hyz
    out[i*6 + 5] = A*x*z;        // Hxz
    ''', 'hernquist_hess'
)

class HernquistPotentialGPU(_AnalyticBase):
    """
    Hernquist: Phi(r) = -GM / (r + a)

    Constructor: HernquistPotentialGPU(mass=1, scaleRadius=1)
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._a  = float(scaleRadius)

    @classmethod
    def from_agama(cls, pot) -> "HernquistPotentialGPU":
        raise TypeError(
            "HernquistPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  HernquistPotentialGPU(mass=..., scaleRadius=...)"
        )

    def _phi(self, x, y, z):
        return _hernquist_phi_kernel(x, y, z, self._GM, self._a)

    def _rho(self, x, y, z):
        return _hernquist_rho_kernel(x, y, z, self._M, self._a)

    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _hernquist_grad_kernel(x, y, z, self._GM, self._a, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _hernquist_hess_kernel(x, y, z, self._GM, self._a, out)
        return out

# ---------------------------------------------------------------------------
# Dehnen (spherical)
# ---------------------------------------------------------------------------

_dehnen_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T g, T exp', 'T out',
    '''
    T r = sqrt(x*x + y*y + z*z);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T u = safe_r / (safe_r + a);
    out = -(GM / a) * (1.0 - pow(u, exp)) / exp;
    ''', 'dehnen_phi'
)

_dehnen_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T M, T a, T g', 'T out',
    '''
    T r = sqrt(x*x + y*y + z*z);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    // rho = (3-g)*M / (4*pi*a³) * (r/a)^(-g) * (1 + r/a)^(g-4)
    T term1 = (3.0 - g) * M / (12.566370614359172 * a * a * a);
    out = term1 * pow(safe_r / a, -g) * pow(1.0 + safe_r / a, g - 4.0);
    ''', 'dehnen_rho'
)

_dehnen_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T g',
    'raw T out',
    '''
    T r = sqrt(x*x + y*y + z*z);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T ra = safe_r + a;
    
    // dPhi/dr = GM * r^(1-g) * (r+a)^(g-3)
    T d1 = GM * pow(safe_r, 1.0 - g) * pow(ra, g - 3.0);
    T g_coeff = d1 / safe_r;
    
    out[i*3 + 0] = g_coeff * x; // Grad X
    out[i*3 + 1] = g_coeff * y; // Grad Y
    out[i*3 + 2] = g_coeff * z; // Grad Z
    ''', 'dehnen_grad'
)

_dehnen_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T g',
    'raw T out',
    '''
    T r2 = x*x + y*y + z*z;
    T r = sqrt(r2);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T ra = safe_r + a;
    
    // d1 = dPhi/dr
    T d1 = GM * pow(safe_r, 1.0 - g) * pow(ra, g - 3.0);
    // d2 = d2Phi/dr2 = d1 * ((1-g)/r + (g-3)/(r+a))
    T d2 = d1 * ((1.0 - g) / safe_r + (g - 3.0) / ra);
    
    T d1_r = d1 / safe_r;
    T A = (d2 - d1_r) / (safe_r * safe_r);
    
    out[i*6 + 0] = A*x*x + d1_r; // Hxx
    out[i*6 + 1] = A*y*y + d1_r; // Hyy
    out[i*6 + 2] = A*z*z + d1_r; // Hzz
    out[i*6 + 3] = A*x*y;        // Hxy
    out[i*6 + 4] = A*y*z;        // Hyz
    out[i*6 + 5] = A*x*z;        // Hxz
    ''', 'dehnen_hess'
)

class DehnenSphericalPotentialGPU(_AnalyticBase):
    """
    Dehnen spherical: Phi(r) = -(GM/a) * (1 - (r/(r+a))^(2-gamma)) / (2-gamma)
    Special case gamma=2: Phi(r) = -(GM/a) * ln(1 + a/r)

    Constructor: DehnenSphericalPotentialGPU(mass=1, scaleRadius=1, gamma=1)
    Note: gamma must be in [0,3). Triaxial Dehnen is NOT implemented (needs quadrature).
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 gamma: float = 1.0):
        if not 0.0 <= gamma < 2.0:
            raise ValueError(
                f"gamma must be in [0, 2) for the GPU kernel, got {gamma}. "
                "gamma=2 or triaxial cases are handled by PotentialGPU(type='Dehnen', ...) "
                "which routes through Agama CPU."
            )
        
        self._M = float(mass)
        self._GM    = _G * self._M
        self._a     = float(scaleRadius)
        self._gamma = float(gamma)
        self._exp   = 2.0 - self._gamma   # exponent in (r/(r+a))^(2-gamma)

    @classmethod
    def from_agama(cls, pot) -> "DehnenSphericalPotentialGPU":
        raise TypeError(
            "DehnenSphericalPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  DehnenSphericalPotentialGPU(mass=..., scaleRadius=..., gamma=...)"
        )

    def _phi(self, x, y, z):
        return _dehnen_phi_kernel(x, y, z, self._GM, self._a, self._gamma, self._exp)

    def _rho(self, x, y, z):
        return _dehnen_rho_kernel(x, y, z, self._M, self._a, self._gamma)
    
    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _dehnen_grad_kernel(x, y, z, self._GM, self._a, self._gamma, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _dehnen_hess_kernel(x, y, z, self._GM, self._a, self._gamma, out)
        return out

# ---------------------------------------------------------------------------
# Isochrone
# ---------------------------------------------------------------------------

_isochrone_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b', 'T out',
    '''
    T r2 = x*x + y*y + z*z;
    T s = sqrt(r2 + b*b);
    out = -GM / (b + s);
    ''', 'isochrone_phi'
)

_isochrone_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b, T INV_4PIG',
    'T out',
    '''
    T r2_raw = x*x + y*y + z*z;
    T r2 = (r2_raw < (T)1e-300) ? (T)1e-300 : r2_raw;
    T s = sqrt(r2 + b*b);
    T bs = b + s;
    
    // Identical math to the Hessian kernel
    T d1_r = GM / (s * bs * bs); 
    T A = -GM * (b + 3.0*s) / (pow(s, 3) * pow(bs, 3));
    
    // Laplacian = Hxx + Hyy + Hzz
    // Hii = A*xi^2 + d1_r
    // Sum = A*(x^2 + y^2 + z^2) + 3*d1_r
    T lap = A * r2 + 3.0 * d1_r;
    
    out = lap * INV_4PIG;
    ''', 'isochrone_rho'
)

_isochrone_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b',
    'raw T out',
    '''
    T r2 = x*x + y*y + z*z;
    T s = sqrt(r2 + b*b);
    T bs = b + s;
    
    // dPhi/dr = GM / (s * (b+s)^2)
    // Grad = (dPhi/dr) * (x/r). Note: r in dPhi/dr and r in denominator cancel.
    T g_coeff = GM / (s * bs * bs);
    
    out[i*3 + 0] = g_coeff * x; // Grad X
    out[i*3 + 1] = g_coeff * y; // Grad Y
    out[i*3 + 2] = g_coeff * z; // Grad Z
    ''', 'isochrone_grad'
)

_isochrone_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T b',
    'raw T out',
    '''
    T r2_raw = x*x + y*y + z*z;
    T r2 = (r2_raw < (T)1e-300) ? (T)1e-300 : r2_raw;
    T s = sqrt(r2 + b*b);
    T bs = b + s;
    
    // d1 = dPhi/dr / r
    T d1_r = GM / (s * bs * bs); 
    
    // d2 = d2Phi/dr2
    // d/dr [GM*r / (s*bs^2)] = GM * [ (s*bs + r^2/s + 2r^2/bs) ] ... 
    // Simplified Cartesian A = (d2 - d1/r)/r^2:
    T A = -GM * (b + 3.0*s) / (s*s*s * bs * bs * bs);
    
    out[i*6 + 0] = A*x*x + d1_r; // Hxx
    out[i*6 + 1] = A*y*y + d1_r; // Hyy
    out[i*6 + 2] = A*z*z + d1_r; // Hzz
    out[i*6 + 3] = A*x*y;        // Hxy
    out[i*6 + 4] = A*y*z;        // Hyz
    out[i*6 + 5] = A*x*z;        // Hxz
    ''', 'isochrone_hess'
)

class IsochronePotentialGPU(_AnalyticBase):
    """
    Isochrone: Phi(r) = -GM / (b + sqrt(r^2 + b^2))

    Constructor: IsochronePotentialGPU(mass=1, scaleRadius=1)
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._b  = float(scaleRadius)

    @classmethod
    def from_agama(cls, pot) -> "IsochronePotentialGPU":
        raise TypeError(
            "IsochronePotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  IsochronePotentialGPU(mass=..., scaleRadius=...)"
        )

    def _phi(self, x, y, z):
        return _isochrone_phi_kernel(x, y, z, self._GM, self._b)

    def _rho(self, x, y, z):
        # Using the trace of the Hessian for perfect self-consistency
        return _isochrone_rho_kernel(x, y, z, self._GM, self._b, _INV_4PIG)

    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _isochrone_grad_kernel(x, y, z, self._GM, self._b, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _isochrone_hess_kernel(x, y, z, self._GM, self._b, out)
        return out


# ---------------------------------------------------------------------------
# MiyamotoNagai
# ---------------------------------------------------------------------------

_mn_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T b', 'T out',
    '''
    T D = sqrt(z*z + b*b);
    T aD = a + D;
    out = -GM / sqrt(x*x + y*y + aD*aD);
    ''', 'mn_phi'
)

_mn_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T b, T INV_4PI', 'T out',
    '''
    T R2 = x*x + y*y;
    T D = sqrt(z*z + b*b);
    T D3 = D*D*D;
    T aD = a + D;
    T S2 = R2 + aD*aD;
    T S5 = pow(S2, 2.5);
    
    // Laplacian (nabla^2 Phi) from your closed form:
    // b^2 * (a*R^2 + (a+3D)*(a+D)^2) / (D^3 * S^5)
    // rho = (GM*b^2 / 4pi) * (a*R2 + (a+3D)*aD^2) / (D^3 * S^5)
    out = (GM * b*b * INV_4PI) * (a*R2 + (a + 3.0*D)*aD*aD) / (D3 * S5);
    ''', 'mn_rho'
)

_mn_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T b',
    'raw T out',
    '''
    T D = sqrt(z*z + b*b);
    T aD = a + D;
    T S2 = x*x + y*y + aD*aD;
    T S3 = S2 * sqrt(S2);
    
    T common = GM / S3;
    out[i*3 + 0] = common * x;                // dPhi/dx
    out[i*3 + 1] = common * y;                // dPhi/dy
    out[i*3 + 2] = common * aD * z / D;       // dPhi/dz
    ''', 'mn_grad'
)

_mn_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T GM, T a, T b',
    'raw T out',
    '''
    T R2 = x*x + y*y;
    T D = sqrt(z*z + b*b);
    T D2 = D*D;
    T aD = a + D;
    T S2 = R2 + aD*aD;
    T S = sqrt(S2);
    T S3 = S2 * S;
    T S5 = S3 * S2;

    // Hxx, Hyy, Hxy
    out[i*6 + 0] = GM * (1.0/S3 - 3.0*x*x/S5); // Hxx
    out[i*6 + 1] = GM * (1.0/S3 - 3.0*y*y/S5); // Hyy
    out[i*6 + 3] = -3.0 * GM * x * y / S5;     // Hxy

    // Hzz logic matching your explicit terms:
    T daD_dz = z / D;
    // term1 = ((daD_dz*D - aD*z/D) * z + aD*D) / (D2 * S3)
    T term1 = ((daD_dz * D - aD * z / D) * z + aD * D) / (D2 * S3);
    // term2 = 3.0*aD*z * (aD*z/D) / (D * S5)
    T term2 = 3.0 * aD * z * (aD * z / D) / (D * S5);
    out[i*6 + 2] = GM * (term1 - term2);       // Hzz

    // Hxz, Hyz
    T off_z = -3.0 * GM * aD * z / (D * S5);
    out[i*6 + 5] = off_z * x;                  // Hxz
    out[i*6 + 4] = off_z * y;                  // Hyz
    ''', 'mn_hess'
)

class MiyamotoNagaiPotentialGPU(_AnalyticBase):
    """
    Miyamoto-Nagai disk: Phi = -GM / sqrt(R^2 + (sqrt(z^2+b^2)+a)^2)
    where R^2 = x^2+y^2.

    Constructor: MiyamotoNagaiPotentialGPU(mass=1, scaleRadius=1, scaleHeight=0.1)
    """

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 scaleHeight: float = 0.1):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._a  = float(scaleRadius)
        self._b  = float(scaleHeight)
        # Using 1/(4*pi) for the density kernel
        self._inv4pi = 1.0 / (4.0 * math.pi)

    @classmethod
    def from_agama(cls, pot) -> "MiyamotoNagaiPotentialGPU":
        raise TypeError(
            "MiyamotoNagaiPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  MiyamotoNagaiPotentialGPU(mass=..., scaleRadius=..., scaleHeight=...)"
        )

    def _phi(self, x, y, z):
        return _mn_phi_kernel(x, y, z, self._GM, self._a, self._b)

    def _rho(self, x, y, z):
        # Using the global _INV_4PIG for unit consistency
        return _mn_rho_kernel(x, y, z, self._GM, self._a, self._b, _INV_4PIG)

    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _mn_grad_kernel(x, y, z, self._GM, self._a, self._b, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _mn_hess_kernel(x, y, z, self._GM, self._a, self._b, out)
        return out


# ---------------------------------------------------------------------------
# Logarithmic halo
# ---------------------------------------------------------------------------

_log_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T v02, T rc2, T p2, T q2', 'T out',
    '''
    T m2 = rc2 + x*x + y*y/p2 + z*z/q2;
    out = (v02 * 0.5) * log(m2);
    ''', 'log_phi'
)

_log_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T v02, T rc2, T p2, T q2, T INV_4PIG', 'T out',
    '''
    T m2 = rc2 + x*x + y*y/p2 + z*z/q2;
    T m4 = m2 * m2;
    // nabla²Phi = v0² * ((1 + 1/p² + 1/q²)/m² - 2*(x² + y²/p⁴ + z²/q⁴)/m⁴)
    T term1 = (1.0 + 1.0/p2 + 1.0/q2) / m2;
    T term2 = 2.0 * (x*x + y*y/(p2*p2) + z*z/(q2*q2)) / m4;
    out = v02 * (term1 - term2) * INV_4PIG;
    ''', 'log_rho'
)

_log_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T v02, T rc2, T p2, T q2',
    'raw T out',
    '''
    T m2 = rc2 + x*x + y*y/p2 + z*z/q2;
    T coeff = v02 / m2;
    out[i*3 + 0] = coeff * x;           // dPhi/dx
    out[i*3 + 1] = coeff * y / p2;      // dPhi/dy
    out[i*3 + 2] = coeff * z / q2;      // dPhi/dz
    ''', 'log_grad'
)

_log_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T v02, T rc2, T p2, T q2',
    'raw T out',
    '''
    T m2 = rc2 + x*x + y*y/p2 + z*z/q2;
    T m4 = m2 * m2;
    
    // Diagonal components
    out[i*6 + 0] = v02 * (1.0/m2 - 2.0*x*x/m4);            // Hxx
    out[i*6 + 1] = v02 * (1.0/(m2*p2) - 2.0*y*y/(m4*p2*p2)); // Hyy
    out[i*6 + 2] = v02 * (1.0/(m2*q2) - 2.0*z*z/(m4*q2*q2)); // Hzz
    
    // Off-diagonal components
    T common = -2.0 * v02 / m4;
    out[i*6 + 3] = common * x * y / p2;      // Hxy
    out[i*6 + 4] = common * y * z / (p2*q2); // Hyz
    out[i*6 + 5] = common * x * z / q2;      // Hxz
    ''', 'log_hess'
)

class LogHaloPotentialGPU(_AnalyticBase):
    """
    LogHalo: Phi = (v0^2/2) * ln(rc^2 + x^2 + y^2/p^2 + z^2/q^2)

    Constructor matches Agama:
        LogHaloPotentialGPU(velocity=1, coreRadius=0.01, axisRatioY=1, axisRatioZ=1)
    """

    def __init__(self, velocity: float = 1.0, coreRadius: float = 0.01,
                 axisRatioY: float = 1.0, axisRatioZ: float = 1.0,
                 scaleRadius: float | None = None):
        # Agama names its core-radius param 'scaleRadius' for the Logarithmic type;
        # accept it as an alias so INI files and PotentialGPU(type='Logarithmic', ...)
        # work with both names.
        if scaleRadius is not None:
            coreRadius = scaleRadius
        self._v02  = float(velocity)**2
        self._rc2  = float(coreRadius)**2
        self._p2   = float(axisRatioY)**2
        self._q2   = float(axisRatioZ)**2

    @classmethod
    def from_agama(cls, pot) -> "LogHaloPotentialGPU":
        raise TypeError(
            "LogHaloPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  LogHaloPotentialGPU(velocity=..., coreRadius=..., axisRatioY=..., axisRatioZ=...)"
        )

    def _phi(self, x, y, z):
        return _log_phi_kernel(x, y, z, self._v02, self._rc2, self._p2, self._q2)

    def _rho(self, x, y, z):
        return _log_rho_kernel(x, y, z, self._v02, self._rc2, self._p2, self._q2, _INV_4PIG)

    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _log_grad_kernel(x, y, z, self._v02, self._rc2, self._p2, self._q2, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _log_hess_kernel(x, y, z, self._v02, self._rc2, self._p2, self._q2, out)
        return out

# ---------------------------------------------------------------------------
# DiskAnsatz  (formulas from CLAUD.md Q2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DiskAnsatz kernels --- accept innerCutoffRadius (hin >= 0).
#
# Radial function with inner exponential cutoff:
#   f(r) = 4*pi*Sigma * exp(-r/hr - hin/r)
#   f'/f = -1/hr + hin/r^2          (fdo_f)
#   f'   = f * fdo_f
#   f''  = f * (fdo_f^2 - 2*hin/r^3)
#
# When hin=0 these reduce exactly to the original exponential-only formulas.
# H(z) vertical function (exponential scaleHeight > 0 here; sech2/thin not yet split).
# ---------------------------------------------------------------------------

_disk_ansatz_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T Sigma, T hr, T hz, T hin', 'T out',
    '''
    T r     = sqrt(x*x + y*y + z*z);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T fval  = 12.566370614359172 * Sigma * exp(-safe_r / hr - hin / safe_r);

    T abz = abs(z);
    T u   = abz / hz;
    T Hval = (hz / 2.0) * (exp(-u) - 1.0 + u);

    out = fval * Hval;
    ''', 'disk_ansatz_phi'
)

_disk_ansatz_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T Sigma, T hr, T hz, T hin', 'raw T out',
    '''
    T r      = sqrt(x*x + y*y + z*z);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T inv_r  = 1.0 / safe_r;

    T fval  = 12.566370614359172 * Sigma * exp(-safe_r / hr - hin / safe_r);
    T fdo_f = -1.0 / hr + hin * inv_r * inv_r;
    T fder1 = fval * fdo_f;

    T abz  = abs(z);
    T u    = abz / hz;
    T eu   = exp(-u);
    T Hval  = (hz / 2.0) * (eu - 1.0 + u);
    T Hder1 = 0.5 * ((z > 0) - (z < 0)) * (1.0 - eu);

    T common = Hval * fder1 * inv_r;
    out[i*3 + 0] = common * x;
    out[i*3 + 1] = common * y;
    out[i*3 + 2] = common * z + Hder1 * fval;
    ''', 'disk_ansatz_grad'
)

_disk_ansatz_hess_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T Sigma, T hr, T hz, T hin', 'raw T out',
    '''
    T r2     = x*x + y*y + z*z;
    T r      = sqrt(r2);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T inv_r  = 1.0 / safe_r;

    T fval  = 12.566370614359172 * Sigma * exp(-safe_r / hr - hin / safe_r);
    T fdo_f = -1.0 / hr + hin * inv_r * inv_r;
    T fder1 = fval * fdo_f;
    T fder2 = fval * (fdo_f * fdo_f - 2.0 * hin * inv_r * inv_r * inv_r);

    T abz   = abs(z);
    T u     = abz / hz;
    T eu    = exp(-u);
    T Hval  = (hz / 2.0) * (eu - 1.0 + u);
    T Hder1 = 0.5 * ((z > 0) - (z < 0)) * (1.0 - eu);
    T Hder2 = 0.5 * eu / hz;

    T xr = x * inv_r; T yr = y * inv_r; T zr = z * inv_r;
    T A  = Hval * (fder2 - fder1 * inv_r);
    T B  = Hval * fder1 * inv_r;

    out[i*6 + 0] = A * xr * xr + B;                                              // Hxx
    out[i*6 + 1] = A * yr * yr + B;                                              // Hyy
    out[i*6 + 2] = (Hval * (fder2 * zr * zr + fder1 * (1.0 - zr * zr) * inv_r)
                    + 2.0 * fder1 * Hder1 * zr + fval * Hder2);                 // Hzz
    out[i*6 + 3] = A * xr * yr;                                                  // Hxy
    out[i*6 + 4] = A * yr * zr + fder1 * Hder1 * yr;                            // Hyz
    out[i*6 + 5] = A * xr * zr + fder1 * Hder1 * xr;                            // Hxz
    ''', 'disk_ansatz_hess'
)

# Density = Laplacian(Phi) / (4*pi*G).
# Laplacian = Hxx + Hyy + Hzz
#           = Hval*(f'' + 2*f'*inv_r) + 2*f'*H'*z/r + f*H''
_disk_ansatz_rho_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T Sigma, T hr, T hz, T hin, T INV_4PIG', 'T out',
    '''
    T r2     = x*x + y*y + z*z;
    T r      = sqrt(r2);
    T safe_r = (r < (T)1e-300) ? (T)1e-300 : r;
    T inv_r  = 1.0 / safe_r;

    T fval  = 12.566370614359172 * Sigma * exp(-safe_r / hr - hin / safe_r);
    T fdo_f = -1.0 / hr + hin * inv_r * inv_r;
    T fder1 = fval * fdo_f;
    T fder2 = fval * (fdo_f * fdo_f - 2.0 * hin * inv_r * inv_r * inv_r);

    T abz   = abs(z);
    T u     = abz / hz;
    T eu    = exp(-u);
    T Hval  = (hz / 2.0) * (eu - 1.0 + u);
    T Hder1 = 0.5 * ((z > 0) - (z < 0)) * (1.0 - eu);
    T Hder2 = 0.5 * eu / hz;

    T zr  = z * inv_r;
    T lap = Hval * (fder2 + 2.0 * fder1 * inv_r)
            + 2.0 * fder1 * Hder1 * zr
            + fval * Hder2;
    out = lap * INV_4PIG;
    ''', 'disk_ansatz_rho'
)


class DiskAnsatzPotentialGPU(_AnalyticBase):
    """
    DiskAnsatz separable disk potential: Phi(R,z) = f(r) * H(z)
    where r = sqrt(R^2+z^2) is the 3D spherical radius.

    f(r) = 4*pi * surfaceDensity * exp(-r/scaleRadius - innerCutoffRadius/r)
    H(z) = exponential (scaleHeight > 0), isothermal sech^2 (scaleHeight < 0), or thin (=0)

    Constructor:
        DiskAnsatzPotentialGPU(surfaceDensity=1, scaleRadius=1, scaleHeight=0.1,
                               innerCutoffRadius=0)
    """

    def __init__(self, surfaceDensity=1.0, scaleRadius=1.0, scaleHeight=0.1,
                 innerCutoffRadius=0.0):
        self._Sigma = float(surfaceDensity) * _G  # absorb G so kernel output is in (km/s)^2
        self._hr    = float(scaleRadius)
        self._hz    = float(abs(scaleHeight))   # kernels use |hz|; sign handled below
        self._hin   = float(innerCutoffRadius)

        if self._hz < 1e-10:
            self._mode = "thin"
        elif scaleHeight > 0:
            self._mode = "exp"
        else:
            self._mode = "sech2"

    @classmethod
    def from_agama(cls, pot) -> "DiskAnsatzPotentialGPU":
        raise TypeError(
            "DiskAnsatzPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  DiskAnsatzPotentialGPU(surfaceDensity=..., scaleRadius=..., "
            "scaleHeight=..., innerCutoffRadius=...)"
        )

    def _phi(self, x, y, z):
        return _disk_ansatz_phi_kernel(x, y, z, self._Sigma, self._hr, self._hz, self._hin)

    def _grad(self, x, y, z):
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _disk_ansatz_grad_kernel(x, y, z, self._Sigma, self._hr, self._hz, self._hin, out)
        return out

    def _hess(self, x, y, z):
        out = cp.empty((x.size, 6), dtype=x.dtype)
        _disk_ansatz_hess_kernel(x, y, z, self._Sigma, self._hr, self._hz, self._hin, out)
        return out

    def _rho(self, x, y, z):
        return _disk_ansatz_rho_kernel(
            x, y, z, self._Sigma, self._hr, self._hz, self._hin, _INV_4PIG)

# ---------------------------------------------------------------------------
# UniformAcceleration
# ---------------------------------------------------------------------------

_uniform_phi_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T ax, T ay, T az', 'T out',
    'out = -(ax*x + ay*y + az*z);', 
    'uniform_phi'
)

_uniform_grad_kernel = cp.ElementwiseKernel(
    'T x, T y, T z, T ax, T ay, T az',
    'raw T out',
    '''
    out[i*3 + 0] = -ax; // dPhi/dx
    out[i*3 + 1] = -ay; // dPhi/dy
    out[i*3 + 2] = -az; // dPhi/dz
    ''', 'uniform_grad'
)

class UniformAccelerationGPU(_AnalyticBase):
    """
    Uniform (spatially constant) acceleration: F = (ax, ay, az).
    Phi(x,y,z) = -ax*x - ay*y - az*z  (Phi = x·(-a))

    Constructor: UniformAccelerationGPU(ax=0, ay=0, az=0)
    Time-varying acceleration: pass ax/ay/az as scalars evaluated at the desired time.
    """

    def __init__(self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0):
        self._ax = float(ax)
        self._ay = float(ay)
        self._az = float(az)

    @classmethod
    def from_agama(cls, pot) -> "UniformAccelerationGPU":
        raise TypeError(
            "UniformAccelerationGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  UniformAccelerationGPU(ax=..., ay=..., az=...)"
        )

    def _phi(self, x, y, z):
        return _uniform_phi_kernel(x, y, z, self._ax, self._ay, self._az)

    def _grad(self, x, y, z):
        # dPhi/dx = -ax, etc. (constant)
        out = cp.empty((x.size, 3), dtype=x.dtype)
        _uniform_grad_kernel(x, y, z, self._ax, self._ay, self._az, out)
        return out

    def _hess(self, x, y, z):
        # Always zero for a linear potential
        return cp.zeros((x.size, 6), dtype=x.dtype)

    def _rho(self, x, y, z):
        # Laplacian of linear function = 0
        return cp.zeros(x.size, dtype=x.dtype)



# ---------------------------------------------------------------------------
# Factory function  (mirrors agama.Potential interface)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    'nfw':                 NFWPotentialGPU,
    'plummer':             PlummerPotentialGPU,
    'hernquist':           HernquistPotentialGPU,
    'isochrone':           IsochronePotentialGPU,
    'miyamotonagai':       MiyamotoNagaiPotentialGPU,
    'logarithmic':         LogHaloPotentialGPU,
    'loghalo':             LogHaloPotentialGPU,
    'dehnen':              DehnenSphericalPotentialGPU,
    'dehnensph':           DehnenSphericalPotentialGPU,
    'diskansatz':          DiskAnsatzPotentialGPU,
    'uniformacceleration': UniformAccelerationGPU,
}


def AnalyticPotentialGPU(type: str, **kwargs) -> _AnalyticBase:
    """
    Factory function matching Agama's constructor syntax::

        pot = AnalyticPotentialGPU(type='NFW', mass=1e12, scaleRadius=20)

    Supported types: NFW, Plummer, Hernquist, Isochrone, MiyamotoNagai,
    Logarithmic/LogHalo, Dehnen (spherical only), DiskAnsatz, UniformAcceleration.

    For King and Spheroid, materialise as Multipole first::

        agama_pot = agama.Potential(type='King', ...)
        from gpu_potential import MultipolePotentialGPU
        gpu_pot   = MultipolePotentialGPU.from_agama(agama_pot)
    """
    key = type.lower().replace('_', '').replace(' ', '')
    if key not in _TYPE_MAP:
        raise ValueError(
            f"Unknown analytic potential type '{type}'. "
            f"Supported: {sorted(_TYPE_MAP.keys())}"
        )
    cls = _TYPE_MAP[key]
    # Map common Agama keyword aliases
    kw = {k.lower(): v for k, v in kwargs.items()}
    # Rename common Agama-style keys to our constructor names
    _renames = {
        'v0': 'velocity', 'coreRadius': 'coreRadius',
        'axisRatioY': 'axisRatioY', 'axisRatioZ': 'axisRatioZ',
    }
    return cls(**{_renames.get(k, k): v for k, v in kw.items()})
