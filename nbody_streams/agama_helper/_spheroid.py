"""
_spheroid.py
~~~~~~~~~~~~
Triaxial double-power-law ("Spheroid") potential evaluated directly from its
density, with axis ratios and normalisation that may vary in time.

This is *not* a replacement for ``PotentialGPU(type='Spheroid', ...)``, which
still builds the potential with Agama and evaluates its Multipole expansion.
``SpheroidPotentialGPU`` exists for the case Agama cannot express: a halo whose
shape changes during the integration.

Density (Agama's ``SpheroidDensityFnc``, same parameter names)::

    rho(m) = rho0 (m/a)^-gamma [1 + (m/a)^alpha]^((gamma-beta)/alpha)
                  * exp[-(m/r_cut)^xi]
    m^2    = x^2 + y^2/p^2 + z^2/q^2          p = axisRatioY, q = axisRatioZ

Potential and forces come from the homoeoid theorem (Chandrasekhar 1969;
Binney & Tremaine 2008 eqs. 2.140, 2.141), written with tau = 1/s^2 - 1 and
D_y = 1 + (p^2-1) s^2, D_z = 1 + (q^2-1) s^2::

    dPhi/dx_i = 4 pi G p q x_i  int_0^1 rho(m(s)) s^2 / (D_i sqrt(D_y D_z)) ds
    Phi       = -2 pi G p q     int_0^1 dpsi(m(s))    / sqrt(D_y D_z)      ds
    m(s)^2    = s^2 (x^2 + y^2/D_y + z^2/D_z),   D_x = 1

where ``dpsi(m) = 2 int_m^inf rho(m') m' dm'``.  The force integral needs only
rho, which is closed form, so the shape can change at every step without
rebuilding anything.  ``dpsi`` depends on the radial profile alone, not on the
shape, so it is tabulated once at construction and only the potential uses it.

The integrals use Gauss-Legendre quadrature in v with s = v^2.  The
substitution removes the s^(2-gamma) cusp singularity and concentrates nodes
where the far-field integrand lives.  The default 32 nodes give relative force
errors of ~1e-14 for integer-slope profiles (NFW, Hernquist) and <~3e-9 for
general ones, out to 40 scale radii; force derivatives (Hessian) reach
~1e-7 there.  64 nodes bring both to ~1e-12 at twice the cost.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    from ._cupy import cp
except ImportError:                             # direct script execution
    from _cupy import cp                        # noqa: E402

try:
    from nbody_streams.agama_helper._potential import _AgamaTimeSpline
    from nbody_streams.agama_helper._analytic_potentials import (
        _AnalyticBase, _prep_xyz, _squeeze, _G,
    )
except ImportError:
    from _potential import _AgamaTimeSpline                         # noqa: E402
    from _analytic_potentials import _AnalyticBase, _prep_xyz, _squeeze, _G  # noqa: E402


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

# Dimensionless profile rho/rho0 at u^2 = (m/a)^2 and its log-slope
# dln(rho)/dln(u).  Consumer GPUs run FP64 at 1/32-1/64 rate, so transcendental
# calls dominate; two exact fast paths avoid them:
#   ibg >= 0 : alpha == 1 with integer gamma and beta-gamma (NFW, Hernquist,
#              Jaffe) -> rho = 1 / (u^gamma (1+u)^(beta-gamma)), no pow/exp
#   xi == 2  : cutoff exp(-(u cut)^2) needs no pow
# ``cut`` = a / outerCutoffRadius (0 = no cutoff).
#
# Every kernel is templated on T (float32 or float64).  Literals are wrapped as
# T(...) so float32 never silently promotes to double arithmetic.
#
# Positions are read straight from the C-contiguous float64 (N,3) array and
# outputs written as float64; only the arithmetic runs in T.  As in the tree
# code (_recentre.py: ``r = (float)(p - c)``) the float64 -> T conversion
# happens in registers, after the float64 scaling, with no separate cast pass.
# All kernels are all-raw and launched with ``size=N``.
_PREAMBLE = r"""
template <typename T> __device__ T tiny_() { return sizeof(T) == 4 ? T(1e-30) : T(1e-300); }

template <typename T> __device__ T ipow_(T x, int n) {
    T r = T(1);
    for (int k = 0; k < n; ++k) r *= x;
    return r;
}

template <typename T>
__device__ T sph_rho(T u2, int ig, int ibg, T alpha, T beta, T gamma, T cut, T xi, T* dln)
{
    if (u2 < tiny_<T>()) u2 = tiny_<T>();
    T rho, frac;                // frac = u^alpha / (1 + u^alpha)
    T lu = T(0);
    if (ibg >= 0) {
        T u  = sqrt(u2);
        rho  = T(1) / (ipow_(u, ig) * ipow_(T(1) + u, ibg));
        frac = u / (T(1) + u);
    } else {
        lu = T(0.5) * log(u2);
        T ua = exp(alpha * lu);
        rho  = exp(-gamma * lu + (gamma - beta) / alpha * log1p(ua));
        frac = T(1) / (T(1) + T(1) / ua);   // finite for ua = 0 and ua = inf
    }
    T d = -gamma + (gamma - beta) * frac;
    if (cut > T(0)) {
        T c;
        if (xi == T(2)) c = cut * cut * u2;
        else {
            if (ibg >= 0) lu = T(0.5) * log(u2);
            c = exp(xi * (lu + log(cut)));
        }
        rho *= exp(-c);
        d   -= xi * c;
    }
    *dln = d;
    return rho;
}
"""

_PROFILE_ARGS = 'int32 ig, int32 ibg, T alpha, T beta, T gamma, T cut, T xi'
_POS_ARGS = 'raw float64 pos, float64 inv_a'
_LOAD_XYZ = ('T X = (T)(pos[3*i] * inv_a), Y = (T)(pos[3*i + 1] * inv_a), '
             'Z = (T)(pos[3*i + 2] * inv_a);')

_spheroid_rho_kernel = cp.ElementwiseKernel(
    # pr = [p^2, q^2, rho0, alpha, beta, gamma, cut, xi]: an array (not scalars)
    # so that CuPy can resolve T for this kernel, which has no other T array
    f'{_POS_ARGS}, raw T pr, int32 ig, int32 ibg', 'raw float64 out',
    """
    _LOAD_XYZ
    T dln;
    out[i] = (double)(pr[2] * sph_rho(X*X + Y*Y/pr[0] + Z*Z/pr[1], ig, ibg,
                                      pr[3], pr[4], pr[5], pr[6], pr[7], &dln));
    """.replace('_LOAD_XYZ', _LOAD_XYZ), 'spheroid_rho', preamble=_PREAMBLE
)

# Per-node tables (depend on the node and on p, q only, so they are built once
# per call on the host): nd[k] = s^2, nd[nq+k] = 1/D_y, nd[2nq+k] = 1/D_z,
# nd[3nq+k] = W = GL weight * ds/dv / sqrt(D_y D_z).
_NODE_ARGS = 'raw T nd, int32 nq'

# amp = -4 pi G rho0 p q a, so the kernel writes the force -dPhi/dx_i.
_spheroid_force_kernel = cp.ElementwiseKernel(
    f'{_POS_ARGS}, {_NODE_ARGS}, {_PROFILE_ARGS}, T amp',
    'raw float64 out',
    """
    _LOAD_XYZ
    T X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    T J0 = T(0), J1 = T(0), J2 = T(0), dln;
    for (int k = 0; k < nq; ++k) {
        T s2 = nd[k], iDy = nd[nq + k], iDz = nd[2*nq + k];
        T f  = nd[3*nq + k] * s2
             * sph_rho(s2 * (X2 + Y2*iDy + Z2*iDz), ig, ibg, alpha, beta, gamma, cut, xi, &dln);
        J0 += f;  J1 += f * iDy;  J2 += f * iDz;
    }
    out[i*3 + 0] = (double)(amp * X * J0);
    out[i*3 + 1] = (double)(amp * Y * J1);
    out[i*3 + 2] = (double)(amp * Z * J2);
    """.replace('_LOAD_XYZ', _LOAD_XYZ), 'spheroid_force', preamble=_PREAMBLE
)

# H_ij = 4 pi G rho0 p q [delta_ij J_i + 2 X_i X_j K_ij],
# K_ij = int drho/d(u^2) s^4 / (D_i D_j sqrt(D_y D_z)) ds.
# amp = -4 pi G rho0 p q, so the kernel writes the force derivatives -H_ij.
_spheroid_deriv_kernel = cp.ElementwiseKernel(
    f'{_POS_ARGS}, {_NODE_ARGS}, {_PROFILE_ARGS}, T amp',
    'raw float64 out',
    """
    _LOAD_XYZ
    T X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    T J0 = T(0), J1 = T(0), J2 = T(0);
    T K00 = T(0), K11 = T(0), K22 = T(0), K01 = T(0), K12 = T(0), K02 = T(0);
    T dln;
    for (int k = 0; k < nq; ++k) {
        T s2 = nd[k], iDy = nd[nq + k], iDz = nd[2*nq + k];
        T u2  = s2 * (X2 + Y2*iDy + Z2*iDz);
        T rho = sph_rho(u2, ig, ibg, alpha, beta, gamma, cut, xi, &dln);
        T f   = nd[3*nq + k] * s2 * rho;
        J0 += f;  J1 += f * iDy;  J2 += f * iDz;
        // only reached at the origin, where every X_i X_j factor is zero anyway
        T g = (u2 > tiny_<T>()) ? f * s2 * dln / (T(2) * u2) : T(0);
        K00 += g;              K11 += g * iDy * iDy;  K22 += g * iDz * iDz;
        K01 += g * iDy;        K12 += g * iDy * iDz;  K02 += g * iDz;
    }
    out[i*6 + 0] = (double)(amp * (J0 + T(2) * X2 * K00));   // xx
    out[i*6 + 1] = (double)(amp * (J1 + T(2) * Y2 * K11));   // yy
    out[i*6 + 2] = (double)(amp * (J2 + T(2) * Z2 * K22));   // zz
    out[i*6 + 3] = (double)(amp * T(2) * X * Y * K01);       // xy
    out[i*6 + 4] = (double)(amp * T(2) * Y * Z * K12);       // yz
    out[i*6 + 5] = (double)(amp * T(2) * X * Z * K02);       // xz
    """.replace('_LOAD_XYZ', _LOAD_XYZ), 'spheroid_deriv', preamble=_PREAMBLE
)

# dpsi(u) from a uniform ln(u) table: cubic Hermite in (ln u, ln dpsi) with
# exact slopes; power-law (log-log linear) extrapolation off either end.
_spheroid_phi_kernel = cp.ElementwiseKernel(
    f'{_POS_ARGS}, {_NODE_ARGS}, raw T tl, raw T ts, int32 nt, T lu0, T h, T amp',
    'raw float64 out',
    """
    _LOAD_XYZ
    T X2 = X*X, Y2 = Y*Y, Z2 = Z*Z;
    T acc = T(0);
    for (int k = 0; k < nq; ++k) {
        T s2 = nd[k];
        T u2 = s2 * (X2 + Y2*nd[nq + k] + Z2*nd[2*nq + k]);
        T lu = T(0.5) * log(u2 < tiny_<T>() ? tiny_<T>() : u2);
        T f  = (lu - lu0) / h;
        T lp;
        if (f <= T(0)) {
            lp = tl[0] + ts[0] * (lu - lu0);
        } else if (f >= T(nt - 1)) {
            lp = tl[nt-1] + ts[nt-1] * (lu - lu0 - h * T(nt - 1));
        } else {
            int j = (int)f;
            T t = f - T(j), t2 = t*t, t3 = t2*t;
            lp = (T(2)*t3 - T(3)*t2 + T(1)) * tl[j]   + (t3 - T(2)*t2 + t) * h * ts[j]
               + (T(3)*t2 - T(2)*t3)        * tl[j+1] + (t3 - t2)          * h * ts[j+1];
        }
        acc += nd[3*nq + k] * exp(lp);
    }
    out[i] = (double)(-amp * acc);
    """.replace('_LOAD_XYZ', _LOAD_XYZ), 'spheroid_phi', preamble=_PREAMBLE
)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------

def _profile_np(u, alpha, beta, gamma, cut, xi):
    """NumPy twin of the device ``sph_rho``: (rho/rho0, dln rho / dln u)."""
    ua  = u ** alpha
    rho = u ** -gamma * (1.0 + ua) ** ((gamma - beta) / alpha)
    dln = -gamma + (gamma - beta) * ua / (1.0 + ua)
    if cut > 0.0:
        c = (cut * u) ** xi
        rho = rho * np.exp(-c)
        dln = dln - xi * c
    return rho, dln


def _ln_grid_integral(f, lo, hi, n, order=8):
    """Integrals of f(u) d(ln u) over each cell of a uniform ln(u) grid."""
    lu = np.linspace(lo, hi, n)
    h = lu[1] - lu[0]
    v, w = np.polynomial.legendre.leggauss(order)
    u = np.exp(lu[:-1, None] + 0.5 * h * (v + 1.0))
    return lu, (f(u) * (0.5 * h * w)).sum(axis=1)


def _dpsi_table(alpha, beta, gamma, cut, xi, n: int = 8192):
    """Tabulate ln dpsi(u) and its log-slope on a uniform ln(u) grid.

    ``dpsi(u) = 2 int_u^inf rho(u') u' du'`` (units of rho0 a^2) depends on
    the radial profile only, so this runs once per instance.  The range is
    wide enough that neither end's power-law continuation (the tail integral
    above ``hi``, the log-log extrapolation below ``lo``) is visible at 1e-12.
    """
    lo, hi = -30.0, 25.0
    if cut > 0.0:
        # stop where the cutoff has removed everything (exp(-200) ~ 1e-87)
        hi = min(hi, math.log(200.0 ** (1.0 / xi) / cut))
    lu, cell = _ln_grid_integral(
        lambda u: 2.0 * _profile_np(u, alpha, beta, gamma, cut, xi)[0] * u * u,
        lo, hi, n)
    u_hi = math.exp(hi)
    rho_hi, dln_hi = _profile_np(u_hi, alpha, beta, gamma, cut, xi)
    if not -dln_hi > 2.0:
        raise ValueError("Spheroid potential diverges: need beta > 2 or an outer cutoff.")
    tail = 2.0 * rho_hi * u_hi ** 2 / (-dln_hi - 2.0)   # local power law beyond hi
    dpsi = tail + np.append(np.cumsum(cell[::-1])[::-1], 0.0)
    u = np.exp(lu)
    rho, _ = _profile_np(u, alpha, beta, gamma, cut, xi)
    slope = -2.0 * rho * u * u / dpsi
    return np.log(dpsi), slope, float(lo), float(lu[1] - lu[0])


def _mass_integral(alpha, beta, gamma, cut, xi) -> float:
    """``int_0^inf rho(u)/rho0 u^2 du``, or NaN when the mass is infinite."""
    if cut == 0.0:
        if beta <= 3.0:
            return math.nan
        # Agama's closed form (SpheroidParam::mass)
        return math.exp(math.lgamma((beta - 3.0) / alpha) + math.lgamma((3.0 - gamma) / alpha)
                        - math.lgamma((beta - gamma) / alpha)) / alpha
    hi = math.log(200.0 ** (1.0 / xi) / cut)
    _, cell = _ln_grid_integral(
        lambda u: _profile_np(u, alpha, beta, gamma, cut, xi)[0] * u ** 3,
        -30.0, hi, 4096)
    return float(cell.sum())


def _time_param(name: str, value) -> Tuple[float, _AgamaTimeSpline | None]:
    """A scalar, or a ``(T,2)`` ``[t, v]`` / ``(T,3)`` ``[t, v, dv/dt]`` table."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr), None
    arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"{name} must be a scalar, a (T,2) [t, value] table or a (T,3) "
            f"[t, value, dvalue/dt] table; got shape {arr.shape}.")
    if arr.shape[1] == 3:
        spl = _AgamaTimeSpline(arr[:, 0], arr[:, 1:2], arr[:, 2:3])
    else:
        spl = _AgamaTimeSpline(arr[:, 0], arr[:, 1:2])
    return float(arr[0, 1]), spl


def _gl_nodes(n: int):
    """Nodes s^2 and weights for int_0^1 g(s) ds with s = v^2."""
    v, w = np.polynomial.legendre.leggauss(n)
    v = 0.5 * (v + 1.0)
    w = 0.5 * w
    return v ** 4, w * 2.0 * v


def _int_slope(v: float) -> int:
    """``v`` as a small non-negative int, or -1."""
    return int(round(v)) if v >= 0 and abs(v - round(v)) < 1e-12 and v <= 8 else -1


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class SpheroidPotentialGPU(_AnalyticBase):
    """
    Triaxial double-power-law potential with time-dependent shape.

    Same density and parameter names as Agama's ``Spheroid``, but evaluated
    directly from the density (no Multipole expansion), so the axis ratios
    and normalisation can follow a time table.

    Parameters
    ----------
    densityNorm : float or table, optional
        Central density normalisation ``rho0`` [Msol/kpc^3].  Mutually
        exclusive with ``mass``.
    mass : float or table, optional
        Total mass [Msol] (default 1, as Agama).  ``rho0`` is derived at
        every time as ``mass / (4 pi a^3 p q I)`` with ``I`` the profile's
        mass integral, so the mass inside *every* ellipsoid ``m`` stays fixed
        while the shape changes.  For profiles of infinite mass (``beta <= 3``,
        no cutoff; Agama raises there) ``I = 1``, i.e. ``mass`` is the NFW-style
        normalisation ``4 pi rho0 a^3 p q``.
    scaleRadius, alpha, beta, gamma, outerCutoffRadius, cutoffStrength : float
        Radial profile, as Agama (defaults 1, 1, 4, 1, inf, 2).
    axisRatioY, axisRatioZ : float or table
        ``p = y/x`` and ``q = z/x``.  A table is ``(T,2)`` ``[t, value]``
        (regularized natural cubic spline, Agama's rule for ``center=`` and
        ``scale=``) or ``(T,3)`` ``[t, value, dvalue/dt]`` (cubic Hermite).
        Outside the table the value is extrapolated linearly.
    nodes : int
        Quadrature nodes (default 32).
    precision : {'float64', 'float32'}
        Arithmetic inside the kernels (default ``'float64'``).  ``'float32'``
        is 35-70x faster on consumer GPUs (FP64 runs at 1/64 rate there) and
        good to ~1e-6 relative, comparable to the float32 self-gravity
        kernels.  Inputs and outputs stay float64 either way.

    Notes
    -----
    Static ``scaleRadius``: for a time-dependent size wrap the potential with
    ``scale=`` (``ScaledPotentialGPU``), which also scales the cutoff radius.
    Orientation and pattern rotation come from ``orientation=``/``rotation=``.
    """

    def __init__(self, densityNorm=None, mass=None, scaleRadius: float = 1.0,
                 alpha: float = 1.0, beta: float = 4.0, gamma: float = 1.0,
                 outerCutoffRadius: float = math.inf, cutoffStrength: float = 2.0,
                 axisRatioY=1.0, axisRatioZ=1.0, nodes: int = 32,
                 precision: str = 'float64'):
        if precision not in ('float64', 'float32'):
            raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}.")
        self._dt = np.float32 if precision == 'float32' else np.float64
        if densityNorm is not None and mass is not None:
            raise ValueError("Parameters 'mass' and 'densityNorm' are mutually exclusive.")
        self._a     = float(scaleRadius)
        self._alpha = float(alpha)
        self._beta  = float(beta)
        self._gamma = float(gamma)
        self._rcut  = float(outerCutoffRadius)
        self._xi    = float(cutoffStrength)
        if not self._a > 0:
            raise ValueError("Spheroid scale radius must be positive.")
        if not self._alpha > 0:
            raise ValueError("Spheroid parameter alpha must be positive.")
        if not self._gamma < 3:
            raise ValueError("Spheroid inner slope gamma must be less than 3.")
        if not self._gamma <= self._beta:
            raise ValueError("Spheroid inner slope gamma should not exceed the outer slope beta.")
        if not (self._beta > 2 or math.isfinite(self._rcut)):
            raise ValueError("Spheroid outer slope beta must be greater than 2, "
                             "or a cutoff radius must be provided.")
        self._cut = self._a / self._rcut if math.isfinite(self._rcut) else 0.0

        self._p0, self._p_spl = _time_param("axisRatioY", axisRatioY)
        self._q0, self._q_spl = _time_param("axisRatioZ", axisRatioZ)
        for nm, v0, spl in (("axisRatioY", self._p0, self._p_spl),
                            ("axisRatioZ", self._q0, self._q_spl)):
            vals = spl._f if spl is not None else np.array([v0])
            if not np.all(vals > 0):
                raise ValueError(f"Spheroid {nm} must be positive.")

        self._by_mass = densityNorm is None
        self._n0, self._n_spl = _time_param(
            "mass" if self._by_mass else "densityNorm",
            1.0 if (self._by_mass and mass is None) else (mass if self._by_mass else densityNorm))
        I = _mass_integral(self._alpha, self._beta, self._gamma, self._cut, self._xi)
        self._mass_int = 1.0 if math.isnan(I) else I

        ig, ibg = _int_slope(self._gamma), _int_slope(self._beta - self._gamma)
        if self._alpha != 1.0 or ig < 0 or ibg < 0:
            ig, ibg = -1, -1                     # general pow/exp path
        dt = self._dt
        self._prof = (np.int32(ig), np.int32(ibg), dt(self._alpha), dt(self._beta),
                      dt(self._gamma), dt(self._cut), dt(self._xi))
        self._nq = np.int32(nodes)
        self._s2, self._wn = _gl_nodes(int(nodes))
        self._nd_key, self._nd = None, None
        tl, ts, lu0, h = _dpsi_table(self._alpha, self._beta, self._gamma, self._cut, self._xi)
        self._tl, self._ts = cp.asarray(tl, dtype=dt), cp.asarray(ts, dtype=dt)
        self._nt, self._lu0, self._h = np.int32(tl.size), dt(lu0), dt(h)

    @property
    def is_time_dependent(self) -> bool:
        return any(s is not None and s.times.size > 1
                   for s in (self._p_spl, self._q_spl, self._n_spl))

    @classmethod
    def from_agama(cls, pot) -> "SpheroidPotentialGPU":
        raise TypeError(
            "SpheroidPotentialGPU.from_agama() is not supported: Agama does not export "
            "Spheroid parameters.  Construct directly with the same keywords, or use "
            "PotentialGPU(type='Spheroid', ...) for the Agama Multipole route.")

    # --- parameters at time t -----------------------------------------------

    def shape(self, t: float = 0.0) -> Tuple[float, float]:
        """Axis ratios ``(p, q)`` at time *t*."""
        p = self._p_spl(float(t))[0] if self._p_spl is not None else self._p0
        q = self._q_spl(float(t))[0] if self._q_spl is not None else self._q0
        if not (p > 0 and q > 0):
            raise ValueError(f"Spheroid axis ratios non-positive at t={t}: p={p}, q={q}.")
        return p, q

    def rho0(self, t: float = 0.0) -> float:
        """Density normalisation at time *t* [Msol/kpc^3]."""
        n = self._n_spl(float(t))[0] if self._n_spl is not None else self._n0
        if not self._by_mass:
            return n
        p, q = self.shape(t)
        return n / (4.0 * math.pi * self._a ** 3 * p * q * self._mass_int)

    def _state(self, t):
        """Node table for the shape at *t*, and the amplitude 4 pi G rho0 p q
        (Hessian; times a for the gradient, a^2/2 for the potential)."""
        p, q = self.shape(t)
        if self._nd_key != (p, q):
            s2 = self._s2
            iDy = 1.0 / (1.0 + (p * p - 1.0) * s2)
            iDz = 1.0 / (1.0 + (q * q - 1.0) * s2)
            self._nd = cp.asarray(np.concatenate([s2, iDy, iDz, self._wn * np.sqrt(iDy * iDz)]),
                                  dtype=self._dt)
            self._nd_key = (p, q)
        return self._nd, p, q, 4.0 * math.pi * _G * self.rho0(t) * p * q

    # --- kernel hooks -------------------------------------------------------
    # ``arr`` is the C-contiguous float64 (N,3) array from _prep_xyz.  Kernels
    # read it directly, compute in self._dt and write float64: no cast passes.

    def _phi(self, arr, t: float = 0.0):
        nd, _, _, amp = self._state(t)
        dt = self._dt
        out = cp.empty(arr.shape[0], dtype=cp.float64)
        _spheroid_phi_kernel(arr, 1.0 / self._a, nd, self._nq, self._tl, self._ts,
                             self._nt, self._lu0, self._h, dt(0.5 * amp * self._a ** 2),
                             out, size=arr.shape[0])
        return out

    def _force(self, arr, t: float = 0.0):
        nd, _, _, amp = self._state(t)
        out = cp.empty((arr.shape[0], 3), dtype=cp.float64)
        _spheroid_force_kernel(arr, 1.0 / self._a, nd, self._nq, *self._prof,
                               self._dt(-amp * self._a), out, size=arr.shape[0])
        return out

    def _deriv(self, arr, t: float = 0.0):
        nd, _, _, amp = self._state(t)
        out = cp.empty((arr.shape[0], 6), dtype=cp.float64)
        _spheroid_deriv_kernel(arr, 1.0 / self._a, nd, self._nq, *self._prof,
                               self._dt(-amp), out, size=arr.shape[0])
        return out

    def _rho(self, arr, t: float = 0.0):
        p, q = self.shape(t)
        ig, ibg = self._prof[:2]
        pr = cp.asarray([p * p, q * q, self.rho0(t), *self._prof[2:]], dtype=self._dt)
        out = cp.empty(arr.shape[0], dtype=cp.float64)
        _spheroid_rho_kernel(arr, 1.0 / self._a, pr, ig, ibg, out, size=arr.shape[0])
        return out

    def __repr__(self) -> str:
        def fmt(v0, spl):
            if spl is not None and spl.times.size > 1:
                return f"<{spl.times.size} rows, t=[{spl.times[0]:g}, {spl.times[-1]:g}]>"
            return f"{v0:g}"
        norm = "mass" if self._by_mass else "densityNorm"
        return (f"SpheroidPotentialGPU({norm}={fmt(self._n0, self._n_spl)}, "
                f"scaleRadius={self._a:g}, alpha={self._alpha:g}, beta={self._beta:g}, "
                f"gamma={self._gamma:g}, axisRatioY={fmt(self._p0, self._p_spl)}, "
                f"axisRatioZ={fmt(self._q0, self._q_spl)})")
