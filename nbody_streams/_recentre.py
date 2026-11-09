"""
Float64 -> float32 position conversion relative to a nearby centre.

float32 keeps ~7 significant digits relative to the magnitude of a number, so
a position at 40 kpc is rounded to ~4e-3 pc before any float32 force code sees
it.  For compact structure far from the origin (a 1-pc cluster at 40 kpc)
that rounding is a few percent of the separations between close neighbours:
the 99th-percentile force error is ~2e-2 for both the GPU tree and the
float32 direct sum, against ~5e-4 for the same cluster at the origin.
Rescaling the units does not help (the rounding is relative); translating
does.  Self-gravity (forces and potential) is translation invariant, so
subtracting a centre in float64 before the cast changes nothing but the
precision.

Used by the float32 paths of ``compute_nbody_forces_gpu``,
``compute_nbody_potential_gpu``, the CPU direct functions (when float32 is
requested), the GPU direct integrator and ``tree_gravity_gpu``.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None
    _CUPY_OK = False

# Number of particles (strided subsample) used to estimate the centre.  The
# centre only has to lie near the system: the subtraction itself is exact in
# float64, so a coarse estimate gives the full benefit (two kernel launches).
_N_CENTRE_SAMPLE = 4096

_sub_cast_kernel = None
_centre_module = None

# One-block reduction over the strided subsample: replaces ~8 small CuPy
# launches (slice, cast, weights, sums) whose launch latency was ~0.25 ms per
# call -- about 3% of a 10k-particle direct-sum step.  Accumulates in double;
# falls back to the unweighted mean when the sampled mass is zero.
_CENTRE_SRC = r"""
template<typename MT, bool HASMASS>
__device__ void centre_impl(const double* __restrict__ pos, const MT* __restrict__ mass,
                            long long n, long long step, int mass_sub, double* __restrict__ c)
{
    __shared__ double sh[7][256];
    double v[7] = {0, 0, 0, 0, 0, 0, 0};   /* w*x, w*y, w*z, w, x, y, z */
    long long cnt = 0;
    for (long long i = (long long)threadIdx.x * step; i < n; i += (long long)blockDim.x * step) {
        const double x = pos[3*i], y = pos[3*i+1], z = pos[3*i+2];
        const double w = HASMASS ? (double)mass[mass_sub ? i / step : i] : 1.0;
        v[0] += w*x; v[1] += w*y; v[2] += w*z; v[3] += w;
        v[4] += x;   v[5] += y;   v[6] += z;   cnt++;
    }
    __shared__ long long shc[256];
    for (int k = 0; k < 7; k++) sh[k][threadIdx.x] = v[k];
    shc[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int k = 0; k < 7; k++) sh[k][threadIdx.x] += sh[k][threadIdx.x + s];
            shc[threadIdx.x] += shc[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (sh[3][0] > 0.0) {
            c[0] = sh[0][0] / sh[3][0]; c[1] = sh[1][0] / sh[3][0]; c[2] = sh[2][0] / sh[3][0];
        } else {
            const double m = (double)(shc[0] > 0 ? shc[0] : 1);
            c[0] = sh[4][0] / m; c[1] = sh[5][0] / m; c[2] = sh[6][0] / m;
        }
    }
}
extern "C" __global__ void centre_nomass(const double* pos, const double* mass, long long n, long long step, int ms, double* c)
{ centre_impl<double, false>(pos, mass, n, step, ms, c); }
extern "C" __global__ void centre_m64(const double* pos, const double* mass, long long n, long long step, int ms, double* c)
{ centre_impl<double, true>(pos, mass, n, step, ms, c); }
extern "C" __global__ void centre_m32(const double* pos, const float* mass, long long n, long long step, int ms, double* c)
{ centre_impl<float, true>(pos, mass, n, step, ms, c); }
"""


def _gpu_sub_cast(pos, centre):
    """Fused (pos - centre).astype(float32) for a C-contiguous (N, 3) float64 array."""
    global _sub_cast_kernel
    if _sub_cast_kernel is None:
        _sub_cast_kernel = cp.ElementwiseKernel(
            "float64 p, raw float64 c", "float32 r",
            "r = (float)(p - c[i % 3])", "nbody_streams_sub_cast")
    return _sub_cast_kernel(pos, centre)


def _gpu_centre(pos, mass):
    """Single-launch subsample centre for a C-contiguous (N, 3) float64 CuPy array."""
    global _centre_module
    if _centre_module is None:
        _centre_module = cp.RawModule(code=_CENTRE_SRC, options=("-std=c++14",))
    n = pos.shape[0]
    step = max(1, n // _N_CENTRE_SAMPLE)
    c = cp.empty(3, dtype=cp.float64)
    mass_sub = 0
    if mass is None or np.isscalar(mass):
        name, m = "centre_nomass", pos
    else:
        if not isinstance(mass, cp.ndarray):
            m = cp.asarray(np.asarray(mass, dtype=np.float64)[::step])   # transfer the subsample only
            mass_sub = 1
        elif mass.dtype in (cp.float32, cp.float64):
            m = cp.ascontiguousarray(mass)
        else:
            m = mass.astype(cp.float64)
        name = "centre_m32" if m.dtype == cp.float32 else "centre_m64"
    _centre_module.get_function(name)(
        (1,), (256,), (pos, m, np.int64(n), np.int64(step), np.int32(mass_sub), c))
    return c


def estimate_centre(pos, mass=None):
    """Mass-weighted mean of a strided subsample of ``pos`` (float64, shape (3,)).

    Returns an array on the same device as ``pos``.  ``mass`` may be None, a
    scalar, or an array on either device.  Falls back to the unweighted mean
    when the sampled mass is zero.
    """
    if _CUPY_OK and isinstance(pos, cp.ndarray):
        return _gpu_centre(cp.ascontiguousarray(pos, dtype=cp.float64), mass)
    n = pos.shape[0]
    step = max(1, n // _N_CENTRE_SAMPLE)
    ps = np.asarray(pos[::step], dtype=np.float64)
    if mass is None or np.isscalar(mass):
        return ps.mean(axis=0)
    ms = mass[::step] if hasattr(mass, "shape") else np.asarray(mass)[::step]
    if _CUPY_OK and isinstance(ms, cp.ndarray):
        ms = cp.asnumpy(ms)
    ms = np.asarray(ms, dtype=np.float64)
    if ms.sum() <= 0:
        return ps.mean(axis=0)
    return (ms[:, None] * ps).sum(axis=0) / ms.sum()


def to_float32_relative(pos, mass=None, center=None):
    """Return float32 positions relative to a centre, and that centre.

    Parameters
    ----------
    pos : ndarray or cupy.ndarray, shape (N, 3)
        Positions.  float32 input is returned unchanged with ``centre=None``:
        its precision was already set when it was rounded to float32.
    mass : scalar or array, optional
        Weights for the centre estimate.
    center : array_like, shape (3,), optional
        Centre to subtract.  Default: :func:`estimate_centre`.

    Returns
    -------
    rel : same array type as ``pos``, float32, shape (N, 3), C-contiguous
    centre : float64 array of shape (3,) on the same device, or None
    """
    on_gpu = _CUPY_OK and isinstance(pos, cp.ndarray)
    xp = cp if on_gpu else np
    if not on_gpu:
        pos = np.asarray(pos)
    if pos.dtype == xp.float32:
        return xp.ascontiguousarray(pos), None
    pos = xp.ascontiguousarray(pos, dtype=xp.float64)
    if center is None:
        c = estimate_centre(pos, mass)
    else:
        c = xp.asarray(center, dtype=xp.float64).reshape(3)
    if on_gpu:
        return _gpu_sub_cast(pos, c), c
    return np.ascontiguousarray((pos - c).astype(np.float32)), c
