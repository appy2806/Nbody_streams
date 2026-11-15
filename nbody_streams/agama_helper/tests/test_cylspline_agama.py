"""
test_cylspline_agama.py
~~~~~~~~~~~~~~~~~~~~~~~
Pointwise agreement of ``CylSplinePotentialGPU`` with Agama's CylSpline.

Each point is compared with its own error, not normalised by the largest
force in the sample (which hid 0.5% errors in the inner disk before).

Covers
------
1. Normalised associated Legendre functions vs scipy (the outer-asymptote
   recurrence used the stale P_{l-3} instead of P_{l-2}).
2. Rscale = -Mtot/Phi0, with Mtot from Agama's asymptote on the z axis.
3. Potential, force and Hessian vs Agama at grid nodes, cell midpoints and
   in and near the disk plane.
4. Off the grid: Agama evaluates only the m = 0, even-l asymptote terms; the
   GPU keeps all of them.  Without the extra terms the GPU equals Agama, with
   them the potential is closer to continuous across the grid edge, and
   force / Hessian are the derivatives of the potential.
5. The z axis: values equal the limit approached along +x.
"""

import os

import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp
from scipy.special import lpmv, factorial

from nbody_streams.agama_helper import read_coefs
from nbody_streams.agama_helper._potential import (
    CylSplinePotentialGPU, _build_cylspline_data, _sph_harm_agama_vect,
    _CS_MUL0, _CS_MUL1,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(_HERE, "600.bar.none_8.coef_cylsp_DR")


@pytest.fixture(scope="module")
def pots():
    return agama.Potential(FILE), CylSplinePotentialGPU.from_file(FILE)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1:
        return np.abs(a / b - 1)
    return np.abs(a - b).max(1) / np.abs(b).max(1)


# ---------------------------------------------------------------------------
# 1. Legendre functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [0, 1, 2, 5])
def test_sph_harm_vs_scipy(m):
    ct = np.linspace(-0.99, 0.99, 41)
    st = np.sqrt(1 - ct**2)
    got = _sph_harm_agama_vect(8, m, ct, st)
    for l in range(m, 9):
        norm = np.sqrt((2*l + 1) / (4*np.pi) * factorial(l - m) / factorial(l + m))
        np.testing.assert_allclose(got[:, l - m], norm * lpmv(m, l, ct), rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Rscale
# ---------------------------------------------------------------------------

def test_rscale_reproduces_agama_node_derivatives(pots):
    """Agama's force at the nodes fixes its spline coordinates: the GPU's
    Rscale must give the same d(log -Phi0)/dz there."""
    ref, _ = pots
    d = _build_cylspline_data(read_coefs(FILE))
    Rs = d["Rscale"]
    R = np.sinh(d["lR_grid"]) * Rs
    z = np.sinh(d["lz_grid"]) * Rs
    RR, ZZ = np.meshgrid(R[1:-1], z[1:-1], indexing="ij")
    x = np.c_[RR.ravel(), 0 * RR.ravel(), ZZ.ravel()]
    gpu = CylSplinePotentialGPU.from_file(FILE)
    assert _rel(cp.asnumpy(gpu.force(cp.asarray(x))), ref.force(x)).max() < 1e-10


# ---------------------------------------------------------------------------
# 3. Pointwise potential, force, Hessian
# ---------------------------------------------------------------------------

def _region(name, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    lo, hi, flat = {"inner": (0.01, 1, 0.05), "disk": (1, 30, 0.05),
                    "halo": (1, 400, 1.0), "off_grid": (710, 3000, 1.0)}[name]  # grid: R, |z| <= 500
    x = v * 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)[:, None]
    x[:, 2] *= flat
    return x


@pytest.mark.parametrize("region", ["inner", "disk", "halo"])
def test_pointwise_vs_agama(pots, region):
    ref, gpu = pots
    x = _region(region)
    xc = cp.asarray(x)
    assert _rel(cp.asnumpy(gpu.potential(xc)), ref.potential(x)).max() < 1e-10
    assert _rel(cp.asnumpy(gpu.force(xc)), ref.force(x)).max() < 1e-9
    der = ref.eval(x, der=True)
    assert _rel(cp.asnumpy(gpu.forceDeriv(xc)[1]), der).max() < 1e-7


def test_grid_nodes_and_midpoints(pots):
    ref, gpu = pots
    c = read_coefs(FILE)
    R = np.asarray(c.R_grid)[1:]
    z = np.asarray(c.z_grid)
    Rm = np.sqrt(R[:-1] * R[1:])
    zm = 0.5 * (z[:-1] + z[1:])
    for RRg, ZZg in ((R, z), (Rm, zm)):
        RR, ZZ = np.meshgrid(RRg, ZZg, indexing="ij")
        x = np.c_[0.6 * RR.ravel(), 0.8 * RR.ravel(), ZZ.ravel()]
        assert _rel(cp.asnumpy(gpu.force(cp.asarray(x))), ref.force(x)).max() < 1e-10


# ---------------------------------------------------------------------------
# 4. Off the grid
# ---------------------------------------------------------------------------

def _extra_terms(x):
    """The asymptote terms Agama does not evaluate (odd l, or m != 0)."""
    d = _build_cylspline_data(read_coefs(FILE))
    W, r0, L, M = d["W_outer"], d["r0_outer"], d["lmax_outer"], d["mmax_outer"]
    r = np.linalg.norm(x, axis=1)
    ct, st = x[:, 2] / r, np.hypot(x[:, 0], x[:, 1]) / r
    ph = np.arctan2(x[:, 1], x[:, 0])
    out = np.zeros(len(r))
    for m in range(M + 1):
        P = _sph_harm_agama_vect(L, m, ct, st)
        for l in range(m, L + 1):
            if m == 0 and l % 2 == 0:
                continue
            p = (r0 / r) ** (l + 1) * P[:, l - m]
            if m == 0:
                out += _CS_MUL0 * W[l*(l+1)] * p
            else:
                out += _CS_MUL1 * p * (W[l*(l+1)+m] * np.cos(m*ph) + W[l*(l+1)-m] * np.sin(m*ph))
    return out


def test_off_grid_is_agama_plus_kept_terms(pots):
    ref, gpu = pots
    x = _region("off_grid")
    phi = cp.asnumpy(gpu.potential(cp.asarray(x)))
    assert _rel(phi - _extra_terms(x), ref.potential(x)).max() < 1e-10
    assert np.median(_rel(phi, ref.potential(x))) > 1e-4      # the terms matter


def test_off_grid_derivatives_consistent(pots):
    _, gpu = pots
    x = _region("off_grid", n=500)
    h = 1e-5 * np.linalg.norm(x, axis=1)[:, None]
    fd_f, fd_h = np.zeros((len(x), 3)), np.zeros((len(x), 3, 3))
    for k in range(3):
        e = np.zeros(3); e[k] = 1
        xp, xm = cp.asarray(x + h * e), cp.asarray(x - h * e)
        fd_f[:, k] = -cp.asnumpy(gpu.potential(xp) - gpu.potential(xm)) / (2 * h[:, 0])
        fd_h[:, :, k] = cp.asnumpy(gpu.force(xp) - gpu.force(xm)) / (2 * h)
    fd6 = np.c_[fd_h[:, 0, 0], fd_h[:, 1, 1], fd_h[:, 2, 2], fd_h[:, 0, 1], fd_h[:, 1, 2], fd_h[:, 0, 2]]
    xc = cp.asarray(x)
    assert np.median(_rel(cp.asnumpy(gpu.force(xc)), fd_f)) < 1e-8
    assert np.median(_rel(cp.asnumpy(gpu.forceDeriv(xc)[1]), fd6)) < 1e-8


def test_potential_closer_to_continuous_at_grid_edge(pots):
    ref, gpu = pots
    rng = np.random.default_rng(5)
    n = 2000
    ph = rng.uniform(0, 2*np.pi, n)
    z = rng.uniform(-499, 499, n)
    def at(R):
        return np.c_[R*np.cos(ph), R*np.sin(ph), z]
    xin, xout = at(500 - 5e-4), at(500 + 5e-4)
    jump_gpu = _rel(cp.asnumpy(gpu.potential(cp.asarray(xout))), cp.asnumpy(gpu.potential(cp.asarray(xin))))
    jump_agama = _rel(ref.potential(xout), ref.potential(xin))
    assert np.median(jump_gpu) < 0.2 * np.median(jump_agama)


# ---------------------------------------------------------------------------
# 5. The z axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", [30.0, -3.0, 0.5, 800.0, -1500.0])
def test_axis_is_limit_along_x(pots, z):
    _, gpu = pots
    on = cp.asarray([[0.0, 0.0, z]])
    near = cp.asarray([[1e-6, 0.0, z]])
    f_on, d_on = (cp.asnumpy(a)[0] for a in gpu.forceDeriv(on))
    f_nr, d_nr = (cp.asnumpy(a)[0] for a in gpu.forceDeriv(near))
    assert np.abs(d_on - d_nr).max() < 3e-5 * np.abs(d_nr).max()
    assert np.abs(f_on - f_nr).max() < 3e-6 * np.abs(f_nr).max()
    np.testing.assert_allclose(cp.asnumpy(gpu.force(on))[0], f_on, rtol=1e-12, atol=1e-12)
