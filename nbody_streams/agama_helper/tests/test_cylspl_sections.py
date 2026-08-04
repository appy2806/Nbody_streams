"""
Regression tests for section-aware CylSpline coefficient parsing.

An Agama ``CylSpline`` export written by ``Potential.export()`` carries three
sections — ``#Phi``, ``#dPhi/dR`` and ``#dPhi/dz`` — each repeating the *full*
set of ``m`` blocks.  A section-blind scan of ``\\t#m`` lines therefore lands on
the last section and silently returns ``dPhi/dz`` as ``phi``, with a duplicated
``m_values`` list.  These tests pin the fixed behaviour.

Fixtures
--------
``three_section.coef_cylsp``
    Real agama 1.0.160 export (Ferrers bar, mmax=4, 10x11 grid), three sections.
``600.bar.none_8.coef_cylsp_DR``
    Legacy single-section file — must keep parsing exactly as before.
"""

from pathlib import Path

import numpy as np
import pytest

from nbody_streams.agama_helper import CylSplineCoefs, read_coefs
from nbody_streams.agama_helper._coefs import read_cylspl_coefs

DATA = Path(__file__).parent
THREE_SECTION = DATA / "three_section.coef_cylsp"
ONE_SECTION = DATA / "600.bar.none_8.coef_cylsp_DR"

try:
    import agama  # noqa: F401
    HAS_AGAMA = True
except ImportError:
    HAS_AGAMA = False


# ---------------------------------------------------------------------------
# Three-section parsing
# ---------------------------------------------------------------------------

def test_three_section_m_values_are_deduplicated():
    """The pre-fix parser returned [0, 0, 0, 2, 2, 2, 4, 4, 4]."""
    cc = read_cylspl_coefs(THREE_SECTION)
    assert cc.m_values == [0, 2, 4]


def test_three_section_populates_all_derivatives():
    cc = read_cylspl_coefs(THREE_SECTION)
    assert cc.dphi_dR is not None
    assert cc.dphi_dz is not None
    assert sorted(cc.phi) == sorted(cc.dphi_dR) == sorted(cc.dphi_dz) == [0, 2, 4]
    for m in cc.m_values:
        assert cc.phi[m].shape == (len(cc.R_grid), len(cc.z_grid))
        assert cc.dphi_dR[m].shape == cc.phi[m].shape
        assert cc.dphi_dz[m].shape == cc.phi[m].shape


def test_phi_is_not_dphi_dz():
    """The core bug: ``phi`` used to hold the ``#dPhi/dz`` table."""
    cc = read_cylspl_coefs(THREE_SECTION)
    assert not np.allclose(cc.phi[0], cc.dphi_dz[0])
    assert not np.allclose(cc.phi[0], cc.dphi_dR[0])
    # A bound monopole potential is negative everywhere it is resolved.
    assert cc.phi[0].max() <= 0.0


def test_phi_matches_the_phi_section_of_the_raw_file():
    """Cross-check the first data row of the m=0 block against the raw text."""
    lines = THREE_SECTION.read_text().splitlines()
    phi_idx = lines.index("#Phi")
    m0_idx = next(i for i in range(phi_idx, len(lines)) if lines[i].startswith("0\t#m"))
    raw_row = [float(v) for v in lines[m0_idx + 2].split()]

    cc = read_cylspl_coefs(THREE_SECTION)
    assert cc.R_grid[0] == pytest.approx(raw_row[0])
    np.testing.assert_allclose(cc.phi[0][0], raw_row[1:])


def test_read_coefs_dispatches_to_the_section_aware_parser():
    cc = read_coefs(THREE_SECTION)
    assert isinstance(cc, CylSplineCoefs)
    assert cc.m_values == [0, 2, 4]
    assert cc.dphi_dz is not None


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------

def test_three_section_string_round_trip_is_lossless():
    cc = read_cylspl_coefs(THREE_SECTION)
    rt = read_cylspl_coefs(cc.to_coef_string())

    assert rt.m_values == cc.m_values
    np.testing.assert_allclose(rt.R_grid, cc.R_grid)
    np.testing.assert_allclose(rt.z_grid, cc.z_grid)
    for m in cc.m_values:
        np.testing.assert_allclose(rt.phi[m], cc.phi[m])
        np.testing.assert_allclose(rt.dphi_dR[m], cc.dphi_dR[m])
        np.testing.assert_allclose(rt.dphi_dz[m], cc.dphi_dz[m])


def test_to_coef_string_emits_all_three_markers():
    cc = read_cylspl_coefs(THREE_SECTION)
    text = cc.to_coef_string()
    assert text.count("\n#Phi\n") == 1
    assert text.count("\n#dPhi/dR\n") == 1
    assert text.count("\n#dPhi/dz\n") == 1
    # Three sections x three m blocks.
    assert sum(1 for ln in text.splitlines() if "\t#m" in ln) == 9


def test_derivative_sections_are_omitted_when_absent():
    cc = read_cylspl_coefs(ONE_SECTION)
    text = cc.to_coef_string()
    assert "#dPhi/dR" not in text
    assert "#dPhi/dz" not in text


# ---------------------------------------------------------------------------
# Backward compatibility with the legacy single-section fixture
# ---------------------------------------------------------------------------

def test_single_section_file_is_unchanged():
    cc = read_cylspl_coefs(ONE_SECTION)
    assert cc.dphi_dR is None
    assert cc.dphi_dz is None
    assert cc.m_values == list(range(-8, 9))
    assert cc.phi[0].shape == (25, 49)
    assert len(cc.R_grid) == 25
    assert len(cc.z_grid) == 49


def test_single_section_round_trip_is_byte_identical():
    """``to_coef_string`` output for a Phi-only file must not have changed."""
    cc = read_cylspl_coefs(ONE_SECTION)
    once = cc.to_coef_string()
    twice = read_cylspl_coefs(once).to_coef_string()
    assert once == twice


def test_zeroed_propagates_derivative_sections():
    cc = read_cylspl_coefs(THREE_SECTION)
    z = cc.zeroed([0])
    assert z.dphi_dR is not None and z.dphi_dz is not None
    for m in (2, 4):
        assert np.all(z.phi[m] == 0.0)
        assert np.all(z.dphi_dR[m] == 0.0)
        assert np.all(z.dphi_dz[m] == 0.0)
    np.testing.assert_allclose(z.phi[0], cc.phi[0])
    np.testing.assert_allclose(z.dphi_dz[0], cc.dphi_dz[0])


def test_mismatched_derivative_section_raises():
    """A #dPhi/dz section over a different m set must not be silently accepted."""
    text = read_cylspl_coefs(THREE_SECTION).to_coef_string()
    head, _, tail = text.partition("#dPhi/dz\n")
    # Relabel the m=4 block of the #dPhi/dz section as m=6.
    tail = tail.replace("4\t#m", "6\t#m")
    with pytest.raises(ValueError, match="differs from the #Phi section"):
        read_cylspl_coefs(head + "#dPhi/dz\n" + tail)


# ---------------------------------------------------------------------------
# Agama round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_AGAMA, reason="agama not installed")
def test_agama_reload_matches_the_source_file():
    """``to_coef_string`` -> agama must reproduce the source potential."""
    import agama

    from nbody_streams.agama_helper import load_agama_potential

    pot_src = agama.Potential(file=str(THREE_SECTION))
    pot_rt = load_agama_potential(read_cylspl_coefs(THREE_SECTION))

    xyz = np.array([[1.0, 0.0, 0.3], [3.0, 2.0, -1.0], [8.0, -4.0, 2.0]])
    np.testing.assert_allclose(pot_rt.potential(xyz), pot_src.potential(xyz), rtol=1e-12)
    np.testing.assert_allclose(pot_rt.force(xyz), pot_src.force(xyz), rtol=1e-10, atol=1e-12)
