"""
Tests for cubic-spline resampling of a coefficient time series.

``spline_resample_coefs`` is the user-side half of the time-axis contract: the
package never interpolates coefficients itself, but this helper shows the
intended shape of a resampling and hands the result back through ``with_times``.

The key invariant is **node preservation**: a cubic spline passes through every
input node, so resampling onto a grid that contains the original times must
reproduce those coefficients exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from nbody_streams.agama_helper import (
    CylSplineCoefs,
    MultipoleCoefs,
    read_coefs,
    refine_times,
    spline_resample_coefs,
    stack_coefs,
)

DATA = Path(__file__).parent
MULT_FILE = DATA / "600.dark.none_8.coef_mul_DR"
CYLSP_THREE = DATA / "three_section.coef_cylsp"

try:
    import agama  # noqa: F401
    HAS_AGAMA = True
except ImportError:
    HAS_AGAMA = False

#: Deliberately uneven, like a real FIRE snapshot cadence.
TIMES = [0.0, 0.5, 2.0, 2.25, 5.0]
SCALES = [1.0, 2.0, 0.5, 1.5, 0.75]

pytestmark = pytest.mark.skipif(not HAS_AGAMA, reason="agama not installed")


@pytest.fixture(scope="module")
def mult_series():
    base = read_coefs(MULT_FILE)
    snaps = []
    for k in SCALES:
        c = base.copy()
        c.phi = c.phi * k
        c.dphi_dr = c.dphi_dr * k
        snaps.append(c)
    return stack_coefs(snaps, TIMES)


@pytest.fixture(scope="module")
def cylsp_series():
    base = read_coefs(CYLSP_THREE)
    snaps = []
    for k in SCALES:
        c = base.copy()
        c.phi = {m: v * k for m, v in c.phi.items()}
        c.dphi_dR = {m: v * k for m, v in c.dphi_dR.items()}
        c.dphi_dz = {m: v * k for m, v in c.dphi_dz.items()}
        snaps.append(c)
    return stack_coefs(snaps, TIMES)


# ---------------------------------------------------------------------------
# refine_times
# ---------------------------------------------------------------------------

def test_refine_times_preserves_original_nodes():
    t = np.array(TIMES)
    fine = refine_times(t, factor=10)
    assert len(fine) == 10 * (len(t) - 1) + 1
    np.testing.assert_allclose(fine[::10], t)
    assert np.all(np.diff(fine) > 0)


def test_refine_times_handles_uneven_spacing():
    """A plain linspace would NOT contain the original nodes when dt varies."""
    t = np.array([0.0, 0.1, 5.0])
    fine = refine_times(t, factor=4)
    np.testing.assert_allclose(fine[::4], t)
    naive = np.linspace(t[0], t[-1], len(fine))
    assert not np.allclose(naive[::4], t)


@pytest.mark.parametrize("factor", [1, 2, 3, 7])
def test_refine_times_factor(factor):
    t = np.array(TIMES)
    fine = refine_times(t, factor=factor)
    assert len(fine) == factor * (len(t) - 1) + 1
    np.testing.assert_allclose(fine[::factor], t)


def test_refine_times_rejects_bad_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        refine_times([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least 2 samples"):
        refine_times([1.0])
    with pytest.raises(ValueError, match="factor"):
        refine_times([0.0, 1.0], factor=0)


# ---------------------------------------------------------------------------
# Node preservation -- the core invariant
# ---------------------------------------------------------------------------

def test_mult_resample_preserves_nodes_exactly(mult_series):
    fine = spline_resample_coefs(mult_series, refine_times(mult_series.times, 10))
    assert fine.n_times == 10 * (mult_series.n_times - 1) + 1
    fine.validate()
    np.testing.assert_allclose(fine.phi[..., ::10], mult_series.phi, rtol=1e-12)
    np.testing.assert_allclose(fine.dphi_dr[..., ::10], mult_series.dphi_dr, rtol=1e-12)
    np.testing.assert_allclose(fine.times[::10], mult_series.times)


def test_cylsp_resample_preserves_nodes_exactly(cylsp_series):
    fine = spline_resample_coefs(cylsp_series, refine_times(cylsp_series.times, 10))
    fine.validate()
    for m in cylsp_series.m_values:
        np.testing.assert_allclose(fine.phi[m][..., ::10], cylsp_series.phi[m], rtol=1e-12)
        np.testing.assert_allclose(fine.dphi_dR[m][..., ::10], cylsp_series.dphi_dR[m], rtol=1e-12)
        np.testing.assert_allclose(fine.dphi_dz[m][..., ::10], cylsp_series.dphi_dz[m], rtol=1e-12)


def test_resample_leaves_grids_and_labels_untouched(mult_series):
    fine = spline_resample_coefs(mult_series, refine_times(mult_series.times, 3))
    np.testing.assert_array_equal(fine.R_grid, mult_series.R_grid)
    assert fine.lm_labels == mult_series.lm_labels
    assert fine.metadata == mult_series.metadata


def test_cylsp_phi_only_resample_keeps_derivatives_none():
    """A #Phi-only CylSpline (like the FIRE archives) must not gain fake derivatives."""
    base = read_coefs(DATA / "600.bar.none_8.coef_cylsp_DR")
    assert base.dphi_dR is None
    ser = stack_coefs([base.copy(), base.copy()], [0.0, 1.0])
    fine = spline_resample_coefs(ser, refine_times(ser.times, 4))
    assert fine.dphi_dR is None and fine.dphi_dz is None
    fine.validate()


# ---------------------------------------------------------------------------
# Downsampling and arbitrary grids
# ---------------------------------------------------------------------------

def test_resample_onto_a_coarser_grid(mult_series):
    coarse = np.linspace(TIMES[0], TIMES[-1], 3)
    out = spline_resample_coefs(mult_series, coarse)
    assert out.n_times == 3
    out.validate()
    # endpoints are original nodes, so they are reproduced exactly
    np.testing.assert_allclose(out.phi[..., 0], mult_series.phi[..., 0], rtol=1e-12)
    np.testing.assert_allclose(out.phi[..., -1], mult_series.phi[..., -1], rtol=1e-12)


def test_resample_to_1000_steps(mult_series):
    out = spline_resample_coefs(mult_series, np.linspace(TIMES[0], TIMES[-1], 1000))
    assert out.n_times == 1000
    out.validate()


def test_resample_is_linear_in_the_coefficients(mult_series):
    """Each series here is base * SCALES[k], so the spline of a column tracks it."""
    fine = spline_resample_coefs(mult_series, refine_times(mult_series.times, 5))
    col = mult_series.column(0, 0)
    for k, t_idx in enumerate(range(0, fine.n_times, 5)):
        ratio = fine.phi[:, col, t_idx] / mult_series.phi[:, col, 0]
        np.testing.assert_allclose(ratio, SCALES[k], rtol=1e-10)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_resample_rejects_a_timeless_object():
    base = read_coefs(MULT_FILE)
    with pytest.raises(ValueError, match="no time axis"):
        spline_resample_coefs(base, [0.0, 1.0])


def test_resample_rejects_a_non_coef_object():
    with pytest.raises(TypeError, match="MultipoleCoefs or CylSplineCoefs"):
        spline_resample_coefs(np.zeros((3, 3)), [0.0, 1.0])


def test_resample_rejects_2d_times(mult_series):
    with pytest.raises(ValueError, match="must be 1-D"):
        spline_resample_coefs(mult_series, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# agama.Spline is a NATURAL cubic spline
# ---------------------------------------------------------------------------

def test_agama_spline_matches_scipy_natural_not_the_scipy_default():
    """
    Pins why this helper requires agama instead of falling back to scipy.

    ``agama.Spline(x, y)`` uses natural boundary conditions.  SciPy's default is
    ``"not-a-knot"``, which gives a visibly different spline near the endpoints.
    """
    scipy_interp = pytest.importorskip("scipy.interpolate")

    rng = np.random.default_rng(0)
    t = np.array(TIMES)
    y = rng.normal(size=t.size) * 100.0
    fine = refine_times(t, 10)

    a = agama.Spline(t, y)(fine)
    natural = scipy_interp.CubicSpline(t, y, bc_type="natural")(fine)
    default = scipy_interp.CubicSpline(t, y)(fine)          # not-a-knot

    scale = np.abs(y).max()
    np.testing.assert_allclose(a, natural, atol=1e-10 * scale)
    assert np.max(np.abs(a - default)) / scale > 1e-4, (
        "scipy's default bc_type used to differ from agama; if this now matches, "
        "the no-scipy-fallback rationale in spline_resample_coefs needs revisiting."
    )


# ---------------------------------------------------------------------------
# The resampled object is a first-class citizen
# ---------------------------------------------------------------------------

def test_resampled_series_materializes_and_matches_at_nodes(mult_series):
    fine = spline_resample_coefs(mult_series, refine_times(mult_series.times, 10))
    pot_orig = mult_series.materialize_potential()
    pot_fine = fine.materialize_potential()

    xyz = np.array([[8.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    for t in mult_series.times:
        a = np.asarray(pot_orig.potential(xyz, t=float(t)))
        b = np.asarray(pot_fine.potential(xyz, t=float(t)))
        np.testing.assert_allclose(b, a, rtol=1e-9)


def test_resampled_series_round_trips_through_h5(mult_series, tmp_path):
    fine = spline_resample_coefs(mult_series, refine_times(mult_series.times, 4))
    path = fine.to_h5(tmp_path / "fine.h5")
    back = read_coefs(path, group_name="all")
    np.testing.assert_allclose(back.phi, fine.phi, rtol=1e-12)
    np.testing.assert_allclose(back.times, fine.times)
