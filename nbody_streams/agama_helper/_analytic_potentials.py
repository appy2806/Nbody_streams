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

CuPy is an optional dependency of the package.  This module imports without it;
constructing or evaluating any potential below then raises a descriptive
ImportError pointing at ``pip install 'nbody_streams[cuda]'``.
"""

from __future__ import annotations

import inspect
import math
from bisect import bisect_right
from pathlib import Path
from typing import Tuple, Union

import numpy as np

# CuPy is optional (see _cupy.py): without it the kernels below become
# placeholders that raise a descriptive ImportError when first called, so the
# module still imports on CPU-only installs.
try:
    from ._cupy import cp
except ImportError:                             # direct script execution
    from _cupy import cp                        # noqa: E402

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


def _precision_dtype(precision: str):
    if precision not in ('float64', 'float32'):
        raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}.")
    return np.float32 if precision == 'float32' else np.float64


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------
#
# Every kernel is templated on T (float32 or float64) and has the signature
#     (raw float64 pos, raw T par) -> raw float64 out,   launched with size=N.
# ``pos`` is the C-contiguous float64 (N,3) array from _prep_xyz; each thread
# converts its own x, y, z to T in registers (as the tree code's recentring
# does), so there is no separate cast pass.  ``par`` holds the model constants
# in T; being an array it also lets CuPy resolve T.  Outputs are float64.
#
# The kernels return what the API returns: the force -grad(Phi) and the force
# derivatives -d2Phi/dxi dxj, so no negation pass follows.  Literals are
# wrapped as T(...) so float32 never silently promotes to double arithmetic.

_PREAMBLE = r"""
template <typename T> __device__ __forceinline__ T tiny_() { return sizeof(T) == 4 ? T(1e-30) : T(1e-300); }

// Spherical model, dPhi/dr = c r:  force = -c x
#define SPH_FORCE(c)                                                         \
    out[3*i] = -(c) * x; out[3*i + 1] = -(c) * y; out[3*i + 2] = -(c) * z;

// Spherical model, Hessian H_ij = A x_i x_j + B delta_ij:  writes -H as
// [xx, yy, zz, xy, yz, xz]
#define SPH_DERIV(A, B)                                                      \
    out[6*i]     = -((A)*x*x + (B)); out[6*i + 1] = -((A)*y*y + (B));        \
    out[6*i + 2] = -((A)*z*z + (B)); out[6*i + 3] = -(A)*x*y;                \
    out[6*i + 4] = -(A)*y*z;         out[6*i + 5] = -(A)*x*z;

// Same, from d2Phi/dr2 = D2 and (dPhi/dr)/r = B:  H = D2 n n + B (1 - n n).
// For cusps B >> D2 at small r, and A x x + B would cancel in the radial
// component; here every diagonal term is a sum of positive weights.
#define SPH_DERIV_R(D2, B, r)                                                \
    { T nx = x / (r), ny = y / (r), nz = z / (r), C = (D2) - (B);           \
      T xx = nx*nx, yy = ny*ny, zz = nz*nz;                                 \
      out[6*i]     = -((D2)*xx + (B)*(yy + zz));                            \
      out[6*i + 1] = -((D2)*yy + (B)*(xx + zz));                            \
      out[6*i + 2] = -((D2)*zz + (B)*(xx + yy));                            \
      out[6*i + 3] = -C*nx*ny; out[6*i + 4] = -C*ny*nz; out[6*i + 5] = -C*nx*nz; }
"""

_LOAD = 'T x = (T)pos[3*i], y = (T)pos[3*i + 1], z = (T)pos[3*i + 2];\n'


def _make_kernels(name: str, params: str, phi: str, force: str, deriv: str,
                  rho: str = None, preamble: str = ''):
    """Build the (phi, force, deriv, rho) kernels of one model.  *params*
    unpacks ``par`` into named T constants and is shared by all four."""
    def k(body, suffix):
        if body is None:
            return None
        return cp.ElementwiseKernel(
            'raw float64 pos, raw T par', 'raw float64 out',
            _LOAD + params + body, f'{name}_{suffix}',
            preamble=_PREAMBLE + preamble)
    return k(phi, 'phi'), k(force, 'force'), k(deriv, 'deriv'), k(rho, 'rho')


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

# Import _GPUPotBase so that all analytic potentials share the same
# isinstance-detectable base class as the BFE potentials.
# _potential.py only imports _analytic_potentials lazily (inside functions),
# so this top-level import is safe — no circular dependency.
# ``_AgamaTimeSpline`` lives alongside the other spline helpers in _potential.py
# because the Shifted/Scaled modifiers use it too.
try:
    from nbody_streams.agama_helper._potential import (
        _GPUPotBase as _GPUPotBase,
        _AgamaTimeSpline as _AgamaTimeSpline,
    )
except ImportError:
    from _potential import (  # noqa: E402
        _GPUPotBase as _GPUPotBase,
        _AgamaTimeSpline as _AgamaTimeSpline,
    )


class _AnalyticBase(_GPUPotBase):
    """Abstract base for analytic GPU potentials.

    Inherits ``__add__`` / ``__radd__`` composition from ``_GPUPotBase`` so that
    ``isinstance(pot, _GPUPotBase)`` returns True for every GPU potential class —
    analytic or BFE — enabling a single ``_is_gpu_potential()`` check in run.py
    and tree_gpu without needing to enumerate both base classes.

    Subclasses either set ``_kernels = (phi, force, deriv, rho)`` (from
    ``_make_kernels``) and call ``_set_params``, or override the hooks
    ``_phi/_force/_deriv/_rho(arr, t)``, which take the float64 (N,3) array
    from ``_prep_xyz`` and return float64 results.
    """

    _dt = np.float64
    _kernels = (None, None, None, None)

    @property
    def precision(self) -> str:
        """Arithmetic precision of the kernels, ``'float64'`` or ``'float32'``."""
        return np.dtype(self._dt).name

    def _set_params(self, precision: str, *values):
        self._dt  = _precision_dtype(precision)
        self._par = cp.asarray(values, dtype=self._dt)

    def _launch(self, kernel, arr, ncomp: int, par=None):
        n   = arr.shape[0]
        out = cp.empty((n,) if ncomp == 1 else (n, ncomp), dtype=cp.float64)
        if n:
            kernel(arr, self._par if par is None else par, out, size=n)
        return out

    # --- kernel hooks -------------------------------------------------------
    def _phi(self, arr, t: float = 0.0):
        return self._launch(self._kernels[0], arr, 1)

    def _force(self, arr, t: float = 0.0):     # -grad(Phi)
        return self._launch(self._kernels[1], arr, 3)

    def _deriv(self, arr, t: float = 0.0):     # -Hessian [xx,yy,zz,xy,yz,xz]
        return self._launch(self._kernels[2], arr, 6)

    def _rho(self, arr, t: float = 0.0):
        return self._launch(self._kernels[3], arr, 1)

    # --- public API ---------------------------------------------------------
    def potential(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        return _squeeze(self._phi(arr, t), single)

    def force(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        return _squeeze(self._force(arr, t), single)

    def density(self, xyz, t: float = 0.0) -> cp.ndarray:
        arr, single = _prep_xyz(xyz)
        return _squeeze(self._rho(arr, t), single)

    def forceDeriv(self, xyz, t: float = 0.0):
        arr, single = _prep_xyz(xyz)
        f, d = self._force(arr, t), self._deriv(arr, t)
        return (f[0], d[0]) if single else (f, d)

    def evalDeriv(self, xyz, t: float = 0.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        arr, single = _prep_xyz(xyz)
        phi, f, d = self._phi(arr, t), self._force(arr, t), self._deriv(arr, t)
        return (phi[0], f[0], d[0]) if single else (phi, f, d)

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
        results = []
        if pot:
            results.append(_squeeze(self._phi(arr, t), single))
        if acc:
            results.append(_squeeze(self._force(arr, t), single))
        if der:
            results.append(_squeeze(self._deriv(arr, t), single))
        return results[0] if len(results) == 1 else tuple(results)


_PRECISION_DOC = """\
precision : {'float64', 'float32'}
    Arithmetic inside the kernels (default ``'float64'``).  Positions are
    read and results written as float64 either way.  ``'float32'`` is good
    to ~1e-7 relative and is much faster on GPUs with slow FP64 (consumer
    and L40-class cards run it at 1/64 rate)."""


# ---------------------------------------------------------------------------
# NFW
# ---------------------------------------------------------------------------

# g(s) = [ln(1+s) - s/(1+s)] / s^2 with s = r/rs, so dPhi/dr = GM g / rs^2
# and d2Phi/dr2 = GM g'(s) / rs^3.  The closed form cancels for small s (both
# terms of the bracket go as s, and g' = q^2/s - 2g/s with q = 1/(1+s)).
# There we use w = s/(1+s) = 1 - q, for which ln(1+s) = -ln(1-w) gives
#     g = q^2 G(w),   g' = q^3 (q G'(w) - 2 G(w)),   G(w) = sum_j w^j / (j+2),
# a series of positive terms with only a mild (x1.5) cancellation in g'.
# 24 terms reach rounding level below s0 = 1 (float) / 0.25 (double); above
# s0 the closed form loses at most ~1.5/s0^2 eps.
_NFW_PRE = r"""
// 1/(j+2), j = 0..23, in double and float: converting double -> float runs
// on the FP64 units, which are slow on consumer GPUs
__constant__ double nfw_c_[24] = {
    0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285,
    0.125, 0.1111111111111111, 0.1, 0.09090909090909091, 0.08333333333333333, 0.07692307692307693,
    0.07142857142857142, 0.06666666666666667, 0.0625, 0.058823529411764705, 0.05555555555555555, 0.05263157894736842,
    0.05, 0.047619047619047616, 0.045454545454545456, 0.043478260869565216, 0.041666666666666664, 0.04};
__constant__ float nfw_cf_[24] = {
    0.5f, 0.3333333333333333f, 0.25f, 0.2f, 0.16666666666666666f, 0.14285714285714285f,
    0.125f, 0.1111111111111111f, 0.1f, 0.09090909090909091f, 0.08333333333333333f, 0.07692307692307693f,
    0.07142857142857142f, 0.06666666666666667f, 0.0625f, 0.058823529411764705f, 0.05555555555555555f, 0.05263157894736842f,
    0.05f, 0.047619047619047616f, 0.045454545454545456f, 0.043478260869565216f, 0.041666666666666664f, 0.04f};
template <typename T> __device__ __forceinline__ T nfw_coef_(int j)
{ return sizeof(T) == 4 ? T(nfw_cf_[j]) : T(nfw_c_[j]); }

template <typename T> __device__ void nfw_g_(T s, T* g, T* dg)
{
    T q = T(1) / (T(1) + s);
    if (s < (sizeof(T) == 4 ? T(1) : T(0.25))) {
        T w = s * q, G = nfw_coef_<T>(23), dG = T(0);
        for (int j = 22; j >= 0; --j) { dG = dG * w + G; G = G * w + nfw_coef_<T>(j); }
        *g  = q * q * G;
        *dg = q * q * q * (q * dG - T(2) * G);
    } else {
        T is = T(1) / s;
        *g  = (log1p(s) - s * q) * is * is;
        *dg = q * q * is - T(2) * (*g) * is;
    }
}
"""

_NFW = _make_kernels(
    'nfw', 'T GM = par[0], rs = par[1], M = par[2];\n'
           'T r = sqrt(x*x + y*y + z*z); if (r < tiny_<T>()) r = tiny_<T>();\n',
    phi='out[i] = -GM * log1p(r / rs) / r;',
    force='''
    T g, dg; nfw_g_(r / rs, &g, &dg);
    T c = GM * g / (rs * rs * r);
    SPH_FORCE(c)
    ''',
    deriv='''
    T g, dg; nfw_g_(r / rs, &g, &dg);
    T B  = GM * g / (rs * rs * r);                      // (dPhi/dr) / r
    T D2 = GM * dg / (rs * rs * rs);                    // d2Phi/dr2
    SPH_DERIV_R(D2, B, r)
    ''',
    rho='T rrs = r + rs; out[i] = M / (T(12.566370614359172) * r * rrs * rrs);',
    preamble=_NFW_PRE,
)


class NFWPotentialGPU(_AnalyticBase):
    """
    Navarro-Frenk-White potential optimized with CuPy ElementwiseKernels.

    Constructor: NFWPotentialGPU(mass=1, scaleRadius=1, precision='float64')
    """

    _kernels = _NFW

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 precision: str = 'float64'):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._rs = float(scaleRadius)
        self._set_params(precision, self._GM, self._rs, self._M)

    @classmethod
    def from_agama(cls, pot) -> "NFWPotentialGPU":
        raise TypeError(
            "NFWPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters (mass, scaleRadius, …).\n"
            "Construct directly:  NFWPotentialGPU(mass=..., scaleRadius=...)"
        )


# ---------------------------------------------------------------------------
# Plummer
# ---------------------------------------------------------------------------

_PLUMMER = _make_kernels(
    'plummer', 'T GM = par[0], b2 = par[1], M = par[2];\n'
               'T s2 = x*x + y*y + z*z + b2, s = sqrt(s2);\n',
    phi='out[i] = -GM / s;',
    force='T c = GM / (s2 * s); SPH_FORCE(c)',
    deriv='T B = GM / (s2 * s); T A = T(-3) * B / s2; SPH_DERIV(A, B)',
    # rho = 3M/(4pi) b^2 / (r^2+b^2)^(5/2)
    rho='out[i] = (T(3) * M / T(12.566370614359172)) * b2 / (s2 * s2 * s);',
)


class PlummerPotentialGPU(_AnalyticBase):
    """
    Plummer sphere: Phi(r) = -GM / sqrt(r^2 + b^2)

    Constructor: PlummerPotentialGPU(mass=1, scaleRadius=1, precision='float64')
    """

    _kernels = _PLUMMER

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 precision: str = 'float64'):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._b2 = float(scaleRadius) * float(scaleRadius)
        self._set_params(precision, self._GM, self._b2, self._M)

    @classmethod
    def from_agama(cls, pot) -> "PlummerPotentialGPU":
        raise TypeError(
            "PlummerPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  PlummerPotentialGPU(mass=..., scaleRadius=...)"
        )


# ---------------------------------------------------------------------------
# Hernquist
# ---------------------------------------------------------------------------

_HERNQUIST = _make_kernels(
    'hernquist', 'T GM = par[0], a = par[1], M = par[2];\n'
                 'T r = sqrt(x*x + y*y + z*z); if (r < tiny_<T>()) r = tiny_<T>();\n'
                 'T ra = r + a;\n',
    phi='out[i] = -GM / ra;',
    # dPhi/dr = GM/(r+a)^2,  d2Phi/dr2 = -2GM/(r+a)^3
    force='T c = GM / (ra * ra * r); SPH_FORCE(c)',
    deriv='''
    T B = GM / (ra * ra * r);
    SPH_DERIV_R(T(-2) * GM / (ra * ra * ra), B, r)
    ''',
    # rho = M a / (2 pi r (r+a)^3)
    rho='out[i] = (M * a) / (T(6.283185307179586) * r * ra * ra * ra);',
)


class HernquistPotentialGPU(_AnalyticBase):
    """
    Hernquist: Phi(r) = -GM / (r + a)

    Constructor: HernquistPotentialGPU(mass=1, scaleRadius=1, precision='float64')
    """

    _kernels = _HERNQUIST

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 precision: str = 'float64'):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._a  = float(scaleRadius)
        self._set_params(precision, self._GM, self._a, self._M)

    @classmethod
    def from_agama(cls, pot) -> "HernquistPotentialGPU":
        raise TypeError(
            "HernquistPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  HernquistPotentialGPU(mass=..., scaleRadius=...)"
        )


# ---------------------------------------------------------------------------
# Dehnen (spherical)
# ---------------------------------------------------------------------------

_DEHNEN = _make_kernels(
    'dehnen', 'T GM = par[0], a = par[1], g = par[2], e = par[3], M = par[4];\n'
              'T r = sqrt(x*x + y*y + z*z); if (r < tiny_<T>()) r = tiny_<T>();\n'
              'T ra = r + a;\n',
    # 1 - (r/(r+a))^e written as -expm1(-e ln(1 + a/r)): no cancellation at r >> a
    phi='out[i] = (GM / a) * expm1(-e * log1p(a / r)) / e;',
    # dPhi/dr = GM r^(1-g) (r+a)^(g-3)
    force='T c = GM * pow(r, T(1) - g) * pow(ra, g - T(3)) / r; SPH_FORCE(c)',
    deriv='''
    T d1 = GM * pow(r, T(1) - g) * pow(ra, g - T(3));
    T d2 = d1 * ((T(1) - g) / r + (g - T(3)) / ra);
    SPH_DERIV_R(d2, d1 / r, r)
    ''',
    # rho = (3-g) M / (4 pi a^3) (r/a)^(-g) (1 + r/a)^(g-4)
    rho='''
    out[i] = (T(3) - g) * M / (T(12.566370614359172) * a * a * a)
             * pow(r / a, -g) * pow(T(1) + r / a, g - T(4));
    ''',
)


class DehnenSphericalPotentialGPU(_AnalyticBase):
    """
    Dehnen spherical: Phi(r) = -(GM/a) * (1 - (r/(r+a))^(2-gamma)) / (2-gamma)
    Special case gamma=2: Phi(r) = -(GM/a) * ln(1 + a/r)

    Constructor: DehnenSphericalPotentialGPU(mass=1, scaleRadius=1, gamma=1, precision='float64')
    Note: gamma must be in [0,3). Triaxial Dehnen is NOT implemented (needs quadrature).
    """

    _kernels = _DEHNEN

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 gamma: float = 1.0, precision: str = 'float64'):
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
        self._set_params(precision, self._GM, self._a, self._gamma, self._exp, self._M)

    @classmethod
    def from_agama(cls, pot) -> "DehnenSphericalPotentialGPU":
        raise TypeError(
            "DehnenSphericalPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  DehnenSphericalPotentialGPU(mass=..., scaleRadius=..., gamma=...)"
        )


# ---------------------------------------------------------------------------
# Isochrone
# ---------------------------------------------------------------------------

_ISOCHRONE = _make_kernels(
    'isochrone', 'T GM = par[0], b = par[1], M4pi = par[2];\n'
                 'T r2 = x*x + y*y + z*z, s = sqrt(r2 + b*b), bs = b + s;\n',
    phi='out[i] = -GM / bs;',
    # dPhi/dr / r = GM / (s (b+s)^2)
    force='T c = GM / (s * bs * bs); SPH_FORCE(c)',
    deriv='''
    T B = GM / (s * bs * bs);
    T A = -GM * (b + T(3) * s) / (s * s * s * bs * bs * bs);
    SPH_DERIV(A, B)
    ''',
    # rho = M [3(b+s)s^2 - r^2(b+3s)] / (4 pi (b+s)^3 s^3); the bracket
    # simplifies to b (3b^2 + 3sb + 2r^2), which has no cancellation at r >> b
    rho='out[i] = M4pi * b * (T(3) * b * b + T(3) * s * b + T(2) * r2) / (bs * bs * bs * s * s * s);',
)


class IsochronePotentialGPU(_AnalyticBase):
    """
    Isochrone: Phi(r) = -GM / (b + sqrt(r^2 + b^2))

    Constructor: IsochronePotentialGPU(mass=1, scaleRadius=1, precision='float64')
    """

    _kernels = _ISOCHRONE

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 precision: str = 'float64'):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._b  = float(scaleRadius)
        self._set_params(precision, self._GM, self._b, self._M / (4.0 * math.pi))

    @classmethod
    def from_agama(cls, pot) -> "IsochronePotentialGPU":
        raise TypeError(
            "IsochronePotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  IsochronePotentialGPU(mass=..., scaleRadius=...)"
        )


# ---------------------------------------------------------------------------
# MiyamotoNagai
# ---------------------------------------------------------------------------

_MN = _make_kernels(
    'mn', 'T GM = par[0], a = par[1], b = par[2], rho_n = par[3];\n'
          'T R2 = x*x + y*y, D = sqrt(z*z + b*b), aD = a + D;\n'
          'T S2 = R2 + aD*aD, S = sqrt(S2), S3 = S2 * S;\n',
    phi='out[i] = -GM / S;',
    force='''
    T c = GM / S3;
    out[3*i] = -c * x; out[3*i + 1] = -c * y; out[3*i + 2] = -c * aD * z / D;
    ''',
    deriv='''
    T S5 = S3 * S2;
    out[6*i]     = -GM * (T(1) / S3 - T(3) * x * x / S5);           // -Hxx
    out[6*i + 1] = -GM * (T(1) / S3 - T(3) * y * y / S5);           // -Hyy
    // Hzz = GM [ (aD D - a z^2/D) / (D^2 S^3) - 3 (aD z / D)^2 / S^5 ]
    T zD = aD * z / D;
    out[6*i + 2] = -GM * ((aD * D - a * z * z / D) / (D * D * S3) - T(3) * zD * zD / S5);
    out[6*i + 3] = T(3) * GM * x * y / S5;                          // -Hxy
    T off_z = T(3) * GM * zD / S5;
    out[6*i + 4] = off_z * y;                                       // -Hyz
    out[6*i + 5] = off_z * x;                                       // -Hxz
    ''',
    # rho = (M b^2 / 4pi) (a R^2 + (a+3D)(a+D)^2) / (D^3 S^5)
    rho='out[i] = rho_n * (a * R2 + (a + T(3) * D) * aD * aD) / (D * D * D * S3 * S2);',
)


class MiyamotoNagaiPotentialGPU(_AnalyticBase):
    """
    Miyamoto-Nagai disk: Phi = -GM / sqrt(R^2 + (sqrt(z^2+b^2)+a)^2)
    where R^2 = x^2+y^2.

    Constructor: MiyamotoNagaiPotentialGPU(mass=1, scaleRadius=1, scaleHeight=0.1,
                                           precision='float64')
    """

    _kernels = _MN

    def __init__(self, mass: float = 1.0, scaleRadius: float = 1.0,
                 scaleHeight: float = 0.1, precision: str = 'float64'):
        self._M  = float(mass)
        self._GM = _G * self._M
        self._a  = float(scaleRadius)
        self._b  = float(scaleHeight)
        self._set_params(precision, self._GM, self._a, self._b,
                         self._M * self._b ** 2 / (4.0 * math.pi))

    @classmethod
    def from_agama(cls, pot) -> "MiyamotoNagaiPotentialGPU":
        raise TypeError(
            "MiyamotoNagaiPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  MiyamotoNagaiPotentialGPU(mass=..., scaleRadius=..., scaleHeight=...)"
        )


# ---------------------------------------------------------------------------
# Logarithmic halo
# ---------------------------------------------------------------------------

_LOG = _make_kernels(
    'log', 'T v02 = par[0], rc2 = par[1], ip2 = par[2], iq2 = par[3], rho_n = par[4];\n'
           'T m2 = rc2 + x*x + y*y*ip2 + z*z*iq2;\n',
    phi='out[i] = T(0.5) * v02 * log(m2);',
    force='''
    T c = v02 / m2;
    out[3*i] = -c * x; out[3*i + 1] = -c * y * ip2; out[3*i + 2] = -c * z * iq2;
    ''',
    deriv='''
    T im2 = T(1) / m2, im4 = im2 * im2;
    T X = x, Y = y * ip2, Z = z * iq2;                  // grad(m2) / 2
    out[6*i]     = -v02 * (im2       - T(2) * X * X * im4);
    out[6*i + 1] = -v02 * (im2 * ip2 - T(2) * Y * Y * im4);
    out[6*i + 2] = -v02 * (im2 * iq2 - T(2) * Z * Z * im4);
    T c = T(2) * v02 * im4;
    out[6*i + 3] = c * X * Y;
    out[6*i + 4] = c * Y * Z;
    out[6*i + 5] = c * X * Z;
    ''',
    # nabla^2 Phi = v0^2 ((1 + 1/p^2 + 1/q^2)/m^2 - 2 (x^2 + y^2/p^4 + z^2/q^4)/m^4)
    rho='''
    out[i] = rho_n * ((T(1) + ip2 + iq2) / m2
                      - T(2) * (x*x + y*y*ip2*ip2 + z*z*iq2*iq2) / (m2 * m2));
    ''',
)


class LogHaloPotentialGPU(_AnalyticBase):
    """
    LogHalo: Phi = (v0^2/2) * ln(rc^2 + x^2 + y^2/p^2 + z^2/q^2)

    Constructor matches Agama:
        LogHaloPotentialGPU(velocity=1, coreRadius=0.01, axisRatioY=1, axisRatioZ=1,
                            precision='float64')
    """

    _kernels = _LOG

    def __init__(self, velocity: float = 1.0, coreRadius: float = 0.01,
                 axisRatioY: float = 1.0, axisRatioZ: float = 1.0,
                 scaleRadius: float | None = None, precision: str = 'float64'):
        # Agama names its core-radius param 'scaleRadius' for the Logarithmic type;
        # accept it as an alias so INI files and PotentialGPU(type='Logarithmic', ...)
        # work with both names.
        if scaleRadius is not None:
            coreRadius = scaleRadius
        self._v02  = float(velocity)**2
        self._rc2  = float(coreRadius)**2
        self._p2   = float(axisRatioY)**2
        self._q2   = float(axisRatioZ)**2
        self._set_params(precision, self._v02, self._rc2, 1.0 / self._p2, 1.0 / self._q2,
                         self._v02 * _INV_4PIG)

    @classmethod
    def from_agama(cls, pot) -> "LogHaloPotentialGPU":
        raise TypeError(
            "LogHaloPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  LogHaloPotentialGPU(velocity=..., coreRadius=..., axisRatioY=..., axisRatioZ=...)"
        )


# ---------------------------------------------------------------------------
# DiskAnsatz
# ---------------------------------------------------------------------------
#
# Phi(R,z) = f(r) H(z), r = sqrt(R^2+z^2), with an inner exponential cutoff:
#   f(r) = 4*pi*G*Sigma * exp(-r/hr - hin/r)
#   f'/f = -1/hr + hin/r^2          (fdo_f)
#   f'   = f * fdo_f
#   f''  = f * (fdo_f^2 - 2*hin/r^3)
# When hin=0 these reduce exactly to the exponential-only formulas.
# H(z) is the exponential vertical function (scaleHeight > 0; sech2/thin not
# yet split):
#   H   = (hz/2) (e^-u - 1 + u),  u = |z|/hz
#   H'  = sign(z)/2 (1 - e^-u)
#   H'' = e^-u / (2 hz)
# e^-u - 1 + u ~ u^2/2 near the midplane, so it is summed as a series there.

_DISK_PRE = r"""
// 1/(j+2)!, j = 0..16, in double and float (see nfw_c_)
__constant__ double disk_ifact_[17] = {
    0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333,
    0.001388888888888889, 0.0001984126984126984, 2.48015873015873e-05,
    2.7557319223985893e-06, 2.755731922398589e-07, 2.505210838544172e-08,
    2.08767569878681e-09, 1.6059043836821613e-10, 1.1470745597729725e-11,
    7.647163731819816e-13, 4.779477332387385e-14, 2.8114572543455206e-15,
    1.5619206968586225e-16};
__constant__ float disk_ifactf_[17] = {
    0.5f, 0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f,
    0.001388888888888889f, 0.0001984126984126984f, 2.48015873015873e-05f,
    2.7557319223985893e-06f, 2.755731922398589e-07f, 2.505210838544172e-08f,
    2.08767569878681e-09f, 1.6059043836821613e-10f, 1.1470745597729725e-11f,
    7.647163731819816e-13f, 4.779477332387385e-14f, 2.8114572543455206e-15f,
    1.5619206968586225e-16f};
template <typename T> __device__ __forceinline__ T disk_coef_(int j)
{ return sizeof(T) == 4 ? T(disk_ifactf_[j]) : T(disk_ifact_[j]); }

template <typename T> __device__ void disk_h_(T z, T hz, T* H, T* dH, T* d2H)
{
    T u = fabs(z) / hz, em1 = expm1(-u), h;
    if (u < T(1)) {
        T p = disk_coef_<T>(16);
        for (int j = 15; j >= 0; --j) p = p * (-u) + disk_coef_<T>(j);
        h = u * u * p;
    } else h = em1 + u;
    *H   = T(0.5) * hz * h;
    *dH  = T(-0.5) * T((z > T(0)) - (z < T(0))) * em1;
    *d2H = T(0.5) * (em1 + T(1)) / hz;
}
"""

_DISK_PARAMS = ('T Sigma4pi = par[0], hr = par[1], hz = par[2], hin = par[3], inv4piG = par[4];\n'
                'T r = sqrt(x*x + y*y + z*z); if (r < tiny_<T>()) r = tiny_<T>();\n'
                'T inv_r = T(1) / r;\n'
                'T fval = Sigma4pi * exp(-r / hr - hin * inv_r);\n'
                'T fdo_f = -T(1) / hr + hin * inv_r * inv_r;\n'
                'T fder1 = fval * fdo_f;\n'
                'T Hval, Hder1, Hder2; disk_h_(z, hz, &Hval, &Hder1, &Hder2);\n')

_DISK = _make_kernels(
    'disk_ansatz', _DISK_PARAMS,
    phi='out[i] = fval * Hval;',
    force='''
    T common = Hval * fder1 * inv_r;
    out[3*i]     = -common * x;
    out[3*i + 1] = -common * y;
    out[3*i + 2] = -(common * z + Hder1 * fval);
    ''',
    deriv='''
    T fder2 = fval * (fdo_f * fdo_f - T(2) * hin * inv_r * inv_r * inv_r);
    T xr = x * inv_r, yr = y * inv_r, zr = z * inv_r;
    T A  = Hval * (fder2 - fder1 * inv_r);
    T B  = Hval * fder1 * inv_r;
    out[6*i]     = -(A * xr * xr + B);
    out[6*i + 1] = -(A * yr * yr + B);
    out[6*i + 2] = -(Hval * (fder2 * zr * zr + fder1 * (T(1) - zr * zr) * inv_r)
                     + T(2) * fder1 * Hder1 * zr + fval * Hder2);
    out[6*i + 3] = -(A * xr * yr);
    out[6*i + 4] = -(A * yr * zr + fder1 * Hder1 * yr);
    out[6*i + 5] = -(A * xr * zr + fder1 * Hder1 * xr);
    ''',
    # Density = Laplacian(Phi) / (4 pi G)
    #         = [H (f'' + 2 f'/r) + 2 f' H' z/r + f H''] / (4 pi G)
    rho='''
    T fder2 = fval * (fdo_f * fdo_f - T(2) * hin * inv_r * inv_r * inv_r);
    out[i] = (Hval * (fder2 + T(2) * fder1 * inv_r) + T(2) * fder1 * Hder1 * z * inv_r
              + fval * Hder2) * inv4piG;
    ''',
    preamble=_DISK_PRE,
)


class DiskAnsatzPotentialGPU(_AnalyticBase):
    """
    DiskAnsatz separable disk potential: Phi(R,z) = f(r) * H(z)
    where r = sqrt(R^2+z^2) is the 3D spherical radius.

    f(r) = 4*pi * surfaceDensity * exp(-r/scaleRadius - innerCutoffRadius/r)
    H(z) = exponential (scaleHeight > 0), isothermal sech^2 (scaleHeight < 0), or thin (=0)

    Constructor:
        DiskAnsatzPotentialGPU(surfaceDensity=1, scaleRadius=1, scaleHeight=0.1,
                               innerCutoffRadius=0, precision='float64')
    """

    _kernels = _DISK

    def __init__(self, surfaceDensity=1.0, scaleRadius=1.0, scaleHeight=0.1,
                 innerCutoffRadius=0.0, precision: str = 'float64'):
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
        self._set_params(precision, 4.0 * math.pi * self._Sigma, self._hr, self._hz,
                         self._hin, _INV_4PIG)

    @classmethod
    def from_agama(cls, pot) -> "DiskAnsatzPotentialGPU":
        raise TypeError(
            "DiskAnsatzPotentialGPU.from_agama() is not supported: Agama does not export "
            "analytic potential parameters.\n"
            "Construct directly:  DiskAnsatzPotentialGPU(surfaceDensity=..., scaleRadius=..., "
            "scaleHeight=..., innerCutoffRadius=...)"
        )

# ---------------------------------------------------------------------------
# UniformAcceleration
# ---------------------------------------------------------------------------

def _read_accel_table(src) -> _AgamaTimeSpline:
    """Build the acceleration spline from Agama's ``file=`` argument.

    Accepts what ``agama.Potential(type='UniformAcceleration', file=...)``
    accepts: a path to a whitespace-separated text file, or the equivalent
    array already loaded in memory.

    * ``(T, 4)`` --- ``[t, ax, ay, az]``, regularized natural cubic spline.
    * ``(T, 7)`` --- ``[t, ax, ay, az, dax/dt, day/dt, daz/dt]``, Hermite spline.
    """
    if isinstance(src, _AgamaTimeSpline):
        return src

    if isinstance(src, (str, Path)):
        arr = np.loadtxt(str(src), comments=('#', ';'))
    else:
        arr = np.asarray(src, dtype=np.float64)

    arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[1] not in (4, 7):
        raise ValueError(
            "UniformAcceleration file= must be a (T,4) table [t, ax, ay, az] "
            "or a (T,7) table [t, ax, ay, az, dax/dt, day/dt, daz/dt]; "
            f"got shape {arr.shape}."
        )
    if arr.shape[1] == 7:
        return _AgamaTimeSpline(arr[:, 0], arr[:, 1:4], arr[:, 4:7])
    return _AgamaTimeSpline(arr[:, 0], arr[:, 1:4])



# Phi = -a.x;  force = +a everywhere
_UNIFORM = _make_kernels(
    'uniform', 'T ax = par[0], ay = par[1], az = par[2];\n',
    phi='out[i] = -(ax * x + ay * y + az * z);',
    force='out[3*i] = ax; out[3*i + 1] = ay; out[3*i + 2] = az;',
    deriv=None,
)


class UniformAccelerationGPU(_AnalyticBase):
    """
    Spatially uniform, optionally time-dependent acceleration ``a(t)``, arising
    from working in a non-inertial reference frame (e.g. the MW disc frame
    accelerating towards an infalling LMC).

    ``Phi(x, t) = -a(t)·x``, so the force is ``+a(t)`` everywhere and both the
    Hessian and the density are identically zero.

    Matches Agama's ``UniformAcceleration`` (``potential_composite.h``).

    Constructors
    ------------
    Constant acceleration::

        UniformAccelerationGPU(ax=0.01, ay=-0.02, az=0.005)

    Time-dependent acceleration, mirroring
    ``agama.Potential(type='UniformAcceleration', file=...)``::

        acc = np.loadtxt('accMW')            # (T, 4): t, ax, ay, az
        UniformAccelerationGPU(file=acc)
        UniformAccelerationGPU(file='path/to/accMW')

    Parameters
    ----------
    ax, ay, az : float, optional
        Constant acceleration components [(km/s)^2/kpc]. Ignored when
        ``file`` is given.
    file : str, Path or array-like, optional
        Time-dependent acceleration table, either a path to a whitespace-
        separated text file or the equivalent array:

        * ``(T, 4)`` --- ``[t, ax, ay, az]``, interpolated with a **regularized
          natural cubic spline** (Agama's ``math::CubicSpline(..., reg=true)``).
        * ``(T, 7)`` --- ``[t, ax, ay, az, dax/dt, day/dt, daz/dt]``,
          interpolated with a cubic Hermite spline.

        Outside the tabulated range the acceleration is extrapolated linearly
        from the endpoint value and derivative, as Agama does.

    Notes
    -----
    Time interpolation happens on the CPU once per call (O(log T)); the GPU
    kernels see three plain floats, so a time-dependent instance costs the same
    per particle as a constant one.
    """

    _kernels = _UNIFORM

    def __init__(self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0,
                 file=None, precision: str = 'float64'):
        if file is not None:
            self._spline = _read_accel_table(file)
            a0 = self._spline(float(self._spline.times[0]))
            self._ax, self._ay, self._az = (float(a0[0]), float(a0[1]), float(a0[2]))
        else:
            self._spline = None
            self._ax = float(ax)
            self._ay = float(ay)
            self._az = float(az)
        self._set_params(precision, self._ax, self._ay, self._az)
        self._par_key = (self._ax, self._ay, self._az)

    @property
    def is_time_dependent(self) -> bool:
        return self._spline is not None and self._spline.times.size > 1

    @classmethod
    def from_agama(cls, pot) -> "UniformAccelerationGPU":
        raise TypeError(
            "UniformAccelerationGPU.from_agama() is not supported: Agama does not export "
            "the acceleration table of a UniformAcceleration potential.\n"
            "Rebuild it from the same source you passed to Agama:\n"
            "    UniformAccelerationGPU(file=acc_table)        # (T,4) array or path\n"
            "    UniformAccelerationGPU(ax=..., ay=..., az=...)  # constant"
        )

    def _a_at(self, t: float) -> Tuple[float, float, float]:
        """Acceleration components at time *t*."""
        if self._spline is None:
            return self._ax, self._ay, self._az
        a = self._spline(float(t))
        return float(a[0]), float(a[1]), float(a[2])

    def _par_at(self, t: float):
        """Device copy of a(t), re-uploaded only when the value changes."""
        a = self._a_at(t)
        if a != self._par_key:
            self._par, self._par_key = cp.asarray(a, dtype=self._dt), a
        return self._par

    # --- kernel hooks -------------------------------------------------------
    def _phi(self, arr, t: float = 0.0):
        return self._launch(self._kernels[0], arr, 1, self._par_at(t))

    def _force(self, arr, t: float = 0.0):
        return self._launch(self._kernels[1], arr, 3, self._par_at(t))

    def _deriv(self, arr, t: float = 0.0):
        # Always zero for a potential linear in x --- no time dependence
        return cp.zeros((arr.shape[0], 6), dtype=cp.float64)

    def _rho(self, arr, t: float = 0.0):
        # Laplacian of a linear function = 0 (Agama returns 0 here too)
        return cp.zeros(arr.shape[0], dtype=cp.float64)

    def __repr__(self) -> str:
        if self.is_time_dependent:
            tk = self._spline.times
            return (f"UniformAccelerationGPU(file=<{tk.size} rows, "
                    f"t=[{tk[0]:g}, {tk[-1]:g}]>)")
        return (f"UniformAccelerationGPU(ax={self._ax:g}, "
                f"ay={self._ay:g}, az={self._az:g})")


# Append the shared ``precision`` entry to each class docstring.
for _cls in (NFWPotentialGPU, PlummerPotentialGPU, HernquistPotentialGPU,
             DehnenSphericalPotentialGPU, IsochronePotentialGPU,
             MiyamotoNagaiPotentialGPU, LogHaloPotentialGPU, DiskAnsatzPotentialGPU):
    _cls.__doc__ = inspect.cleandoc(_cls.__doc__) + "\n\n" + _PRECISION_DOC
_doc = inspect.cleandoc(UniformAccelerationGPU.__doc__)
UniformAccelerationGPU.__doc__ = _doc.replace("\n\nNotes\n", "\n" + _PRECISION_DOC + "\n\nNotes\n", 1)


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
