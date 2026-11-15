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
2. Rscale = -Mtot/Phi0, with Mtot from the full asymptote on the z axis.
3. Potential, force and Hessian vs Agama at grid nodes, cell midpoints, in
   and near the disk plane, and off the grid (outer asymptote, which Agama
   evaluates with its m = 0, even-l terms only).
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
                    "halo": (1, 400, 1.0), "off_grid": (600, 3000, 1.0)}[name]
    x = v * 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)[:, None]
    x[:, 2] *= flat
    return x


@pytest.mark.parametrize("region", ["inner", "disk", "halo", "off_grid"])
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
