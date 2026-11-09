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
# float64, so a coarse estimate gives the full benefit at ~0.2 ms for N = 1e6.
_N_CENTRE_SAMPLE = 4096

_sub_cast_kernel = None


def _gpu_sub_cast(pos, centre):
    """Fused (pos - centre).astype(float32) for a C-contiguous (N, 3) float64 array."""
    global _sub_cast_kernel
    if _sub_cast_kernel is None:
        _sub_cast_kernel = cp.ElementwiseKernel(
            "float64 p, raw float64 c", "float32 r",
            "r = (float)(p - c[i % 3])", "nbody_streams_sub_cast")
    return _sub_cast_kernel(cp.ascontiguousarray(pos), centre)


def estimate_centre(pos, mass=None):
    """Mass-weighted mean of a strided subsample of ``pos`` (float64, shape (3,)).

    Returns an array on the same device as ``pos``.  ``mass`` may be None, a
    scalar, or an array on either device.
    """
    xp = cp if (_CUPY_OK and isinstance(pos, cp.ndarray)) else np
    n = pos.shape[0]
    step = max(1, n // _N_CENTRE_SAMPLE)
    ps = pos[::step].astype(xp.float64, copy=False)
    if mass is None or np.isscalar(mass):
        return ps.mean(axis=0)
    if not hasattr(mass, "shape"):
        mass = np.asarray(mass)
    ms = mass[::step]                                  # subsample before any transfer
    if xp is np and _CUPY_OK and isinstance(ms, cp.ndarray):
        ms = cp.asnumpy(ms)
    ms = xp.asarray(ms).astype(xp.float64, copy=False)
    msum = ms.sum()
    if xp is np and msum <= 0:
        return ps.mean(axis=0)
    # all-massless subsample on the GPU: fall back to the plain mean without a sync
    w = xp.where(msum > 0, ms, xp.ones_like(ms))
    return (w[:, None] * ps).sum(axis=0) / w.sum()


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
    pos = pos.astype(xp.float64, copy=False)
    if center is None:
        c = estimate_centre(pos, mass)
    else:
        c = xp.asarray(center, dtype=xp.float64).reshape(3)
    if on_gpu:
        return _gpu_sub_cast(pos, c), c
    return np.ascontiguousarray((pos - c).astype(np.float32)), c
