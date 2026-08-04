"""
test_series.py
~~~~~~~~~~~~~~~
Tests for the optional trailing time axis on :class:`MultipoleCoefs` and
:class:`CylSplineCoefs` (``_coefs.py``): ``times``, ``has_time_axis``,
``n_times``, ``snapshot()``/``__getitem__``, ``with_times()``, ``stack_coefs()``,
and the time-aware forms of ``zeroed``, ``radial_power``/``total_power``,
``to_coef_string``/``to_coef_strings``, ``to_h5``, ``to_evolving_ini`` and
``materialize_potential``.

Fixtures
--------
``600.dark.none_8.coef_mul_DR``
    Multipole, lmax=8, 25 radii x 81 (l, m) columns, carries ``dphi_dr``.
``600.bar.none_8.coef_cylsp_DR``
    CylSpline, single ``#Phi`` section, m=-8..8, 25x49 grid, no derivatives.
``three_section.coef_cylsp``
    CylSpline with ``#Phi`` + ``#dPhi/dR`` + ``#dPhi/dz``, m=0/2/4, 10x11 grid.

Time series are built the way the feature is meant to be used: read a
time-less fixture once, scale it by a handful of known factors, then
``stack_coefs()`` the results.  Because the scaling is linear, the resulting
series has an exactly known relationship to the base snapshot at every
sampled time -- this is what the ``materialize_potential`` scaling checks
lean on.
"""

from pathlib import Path

import numpy as np
import pytest

from nbody_streams.agama_helper import (
    CylSplineCoefs,
    MultipoleCoefs,
    read_coefs,
    read_cylspl_coefs,
    read_mult_coefs,
    stack_coefs,
)

try:
    import agama  # noqa: F401
    HAS_AGAMA = True
except ImportError:
    HAS_AGAMA = False


DATA = Path(__file__).parent
MULT_FILE = DATA / "600.dark.none_8.coef_mul_DR"
CYLSP_ONE = DATA / "600.bar.none_8.coef_cylsp_DR"
CYLSP_THREE = DATA / "three_section.coef_cylsp"

#: Per-snapshot scaling used to build every synthetic time series in this
#: file: snapshot k is the base fixture multiplied by SCALES[k], stamped at
#: TIMES[k].  Because BFE coefficients enter the reconstructed potential
#: linearly, the materialized potential at TIMES[k] must equal SCALES[k]
#: times the base potential -- this is exploited by the agama-backed tests.
TIMES = [0.0, 1.0, 2.0]
SCALES = [1.0, 2.0, 0.5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_mult(base: MultipoleCoefs, k: float) -> MultipoleCoefs:
    c = base.copy()
    c.phi = c.phi * k
    if c.dphi_dr is not None:
        c.dphi_dr = c.dphi_dr * k
    return c


def _scale_cylsp(base: CylSplineCoefs, k: float) -> CylSplineCoefs:
    c = base.copy()
    c.phi = {m: v * k for m, v in c.phi.items()}
    if c.dphi_dR is not None:
        c.dphi_dR = {m: v * k for m, v in c.dphi_dR.items()}
    if c.dphi_dz is not None:
        c.dphi_dz = {m: v * k for m, v in c.dphi_dz.items()}
    return c


# ---------------------------------------------------------------------------
# Fixtures -- parsed once per module, series built once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mult_base() -> MultipoleCoefs:
    return read_mult_coefs(MULT_FILE)


@pytest.fixture(scope="module")
def cylsp_base() -> CylSplineCoefs:
    return read_cylspl_coefs(CYLSP_ONE)


@pytest.fixture(scope="module")
def cylsp3_base() -> CylSplineCoefs:
    return read_cylspl_coefs(CYLSP_THREE)


@pytest.fixture(scope="module")
def mult_snaps(mult_base) -> list:
    return [_scale_mult(mult_base, k) for k in SCALES]


@pytest.fixture(scope="module")
def mult_series(mult_snaps) -> MultipoleCoefs:
    return stack_coefs(mult_snaps, TIMES)


@pytest.fixture(scope="module")
def cylsp_snaps(cylsp_base) -> list:
    return [_scale_cylsp(cylsp_base, k) for k in SCALES]


@pytest.fixture(scope="module")
def cylsp_series(cylsp_snaps) -> CylSplineCoefs:
    return stack_coefs(cylsp_snaps, TIMES)


@pytest.fixture(scope="module")
def cylsp3_snaps(cylsp3_base) -> list:
    return [_scale_cylsp(cylsp3_base, k) for k in SCALES]


@pytest.fixture(scope="module")
def cylsp3_series(cylsp3_snaps) -> CylSplineCoefs:
    return stack_coefs(cylsp3_snaps, TIMES)


# ---------------------------------------------------------------------------
# 1. Single-snapshot read is unchanged
# ---------------------------------------------------------------------------

def test_single_snapshot_mult_has_no_time_axis(mult_base):
    assert mult_base.times is None
    assert mult_base.has_time_axis is False
    assert mult_base.n_times is None
    assert mult_base.phi.ndim == 2
    assert mult_base.phi.shape == (25, 81)


def test_single_snapshot_cylsp_one_section_has_no_time_axis(cylsp_base):
    assert cylsp_base.times is None
    assert cylsp_base.has_time_axis is False
    assert cylsp_base.n_times is None
    for m, table in cylsp_base.phi.items():
        assert table.ndim == 2


def test_single_snapshot_cylsp_three_section_has_no_time_axis(cylsp3_base):
    assert cylsp3_base.times is None
    assert cylsp3_base.has_time_axis is False
    assert cylsp3_base.n_times is None
    for m, table in cylsp3_base.phi.items():
        assert table.ndim == 2


# ---------------------------------------------------------------------------
# 2. Stack -> snapshot(i) / obj[i] reproduce the inputs exactly
# ---------------------------------------------------------------------------

def test_mult_series_has_expected_time_axis(mult_series):
    assert mult_series.has_time_axis
    assert mult_series.n_times == len(TIMES)
    np.testing.assert_array_equal(mult_series.times, TIMES)
    assert mult_series.phi.shape == (25, 81, len(TIMES))


def test_mult_snapshot_reproduces_input_exactly(mult_series, mult_snaps):
    for i, snap in enumerate(mult_snaps):
        s = mult_series.snapshot(i)
        assert np.array_equal(s.phi, snap.phi)
        assert np.array_equal(s.dphi_dr, snap.dphi_dr)
        assert s.times is None


def test_mult_getitem_matches_snapshot(mult_series):
    for i in range(mult_series.n_times):
        a = mult_series[i]
        b = mult_series.snapshot(i)
        assert np.array_equal(a.phi, b.phi)
        assert np.array_equal(a.dphi_dr, b.dphi_dr)


def test_cylsp_series_has_expected_time_axis(cylsp3_series):
    assert cylsp3_series.has_time_axis
    assert cylsp3_series.n_times == len(TIMES)
    np.testing.assert_array_equal(cylsp3_series.times, TIMES)
    for m in cylsp3_series.m_values:
        assert cylsp3_series.phi[m].shape[-1] == len(TIMES)


def test_cylsp_snapshot_reproduces_input_exactly(cylsp3_series, cylsp3_snaps):
    for i, snap in enumerate(cylsp3_snaps):
        s = cylsp3_series.snapshot(i)
        for m in snap.m_values:
            assert np.array_equal(s.phi[m], snap.phi[m])
            assert np.array_equal(s.dphi_dR[m], snap.dphi_dR[m])
            assert np.array_equal(s.dphi_dz[m], snap.dphi_dz[m])
        assert s.times is None


def test_cylsp_getitem_matches_snapshot(cylsp3_series):
    for i in range(cylsp3_series.n_times):
        a = cylsp3_series[i]
        b = cylsp3_series.snapshot(i)
        for m in a.m_values:
            assert np.array_equal(a.phi[m], b.phi[m])


# ---------------------------------------------------------------------------
# 3. to_coef_string -> re-read round trip (identity, both types)
# ---------------------------------------------------------------------------

def test_mult_string_round_trip_timeless(mult_base):
    rt = read_mult_coefs(mult_base.to_coef_string())
    np.testing.assert_allclose(rt.R_grid, mult_base.R_grid, rtol=1e-10)
    np.testing.assert_allclose(rt.phi, mult_base.phi, rtol=1e-10)
    np.testing.assert_allclose(rt.dphi_dr, mult_base.dphi_dr, rtol=1e-10)
    assert rt.lm_labels == mult_base.lm_labels


def test_mult_string_round_trip_per_snapshot_of_series(mult_series, mult_snaps):
    for i, snap in enumerate(mult_snaps):
        rt = read_mult_coefs(mult_series.to_coef_string(t=i))
        np.testing.assert_allclose(rt.phi, snap.phi, rtol=1e-10)
        np.testing.assert_allclose(rt.dphi_dr, snap.dphi_dr, rtol=1e-10)


def test_cylsp_string_round_trip_timeless(cylsp3_base):
    rt = read_cylspl_coefs(cylsp3_base.to_coef_string())
    for m in cylsp3_base.m_values:
        np.testing.assert_allclose(rt.phi[m], cylsp3_base.phi[m], rtol=1e-10)
        np.testing.assert_allclose(rt.dphi_dR[m], cylsp3_base.dphi_dR[m], rtol=1e-10)
        np.testing.assert_allclose(rt.dphi_dz[m], cylsp3_base.dphi_dz[m], rtol=1e-10)


def test_cylsp_string_round_trip_per_snapshot_of_series(cylsp3_series, cylsp3_snaps):
    for i, snap in enumerate(cylsp3_snaps):
        rt = read_cylspl_coefs(cylsp3_series.to_coef_string(t=i))
        for m in snap.m_values:
            np.testing.assert_allclose(rt.phi[m], snap.phi[m], rtol=1e-10)


def test_to_coef_strings_returns_one_string_per_time(mult_series):
    strings = mult_series.to_coef_strings()
    assert len(strings) == mult_series.n_times
    for i, s in enumerate(strings):
        rt = read_mult_coefs(s)
        np.testing.assert_allclose(rt.phi, mult_series.snapshot(i).phi, rtol=1e-10)


def test_to_coef_strings_single_element_without_time_axis(mult_base):
    strings = mult_base.to_coef_strings()
    assert len(strings) == 1
    np.testing.assert_allclose(read_mult_coefs(strings[0]).phi, mult_base.phi, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. to_h5 -> read_coefs(group_name=...) round trip
# ---------------------------------------------------------------------------

def test_h5_round_trip_group_all_mult(mult_series, tmp_path):
    path = tmp_path / "mult_series.h5"
    mult_series.to_h5(path)
    back = read_coefs(path, group_name="all")
    assert back.has_time_axis
    np.testing.assert_allclose(back.times, mult_series.times)
    np.testing.assert_allclose(back.phi, mult_series.phi, rtol=1e-10)
    np.testing.assert_allclose(back.dphi_dr, mult_series.dphi_dr, rtol=1e-10)


def test_h5_round_trip_group_all_cylsp(cylsp3_series, tmp_path):
    path = tmp_path / "cylsp_series.h5"
    cylsp3_series.to_h5(path)
    back = read_coefs(path, group_name="all")
    assert back.has_time_axis
    np.testing.assert_allclose(back.times, cylsp3_series.times)
    for m in cylsp3_series.m_values:
        np.testing.assert_allclose(back.phi[m], cylsp3_series.phi[m], rtol=1e-10)


def test_h5_explicit_group_name_ordering(mult_series, tmp_path):
    path = tmp_path / "mult_series_ordered.h5"
    mult_series.to_h5(path)

    # Select groups out of storage order, with explicit times matching the
    # chosen order -- an explicit times= argument always overrides whatever
    # is embedded, so this is the documented, unambiguous way to reorder.
    order = [2, 0, 1]
    groups = [f"snap_{i:04d}" for i in order]
    times = [TIMES[i] for i in order]
    back = read_coefs(path, group_name=groups, times=times)

    np.testing.assert_allclose(back.times, times)
    for j, orig_i in enumerate(order):
        np.testing.assert_allclose(
            back.snapshot(j).phi, mult_series.snapshot(orig_i).phi, rtol=1e-10
        )


@pytest.mark.parametrize(
    "order",
    [[2, 0, 1], [1, 2], [0], [2, 1, 0]],
    ids=["reordered", "subset", "single", "reversed"],
)
def test_h5_explicit_group_name_pairs_stored_times_correctly(mult_series, tmp_path, order):
    """
    Regression: an explicit *group_name* must take each group's time from the
    archive's canonical (numerically sorted) position, not from the caller's
    ordinal.  Reordering used to hand back the stored times in file order, so a
    same-length reordering silently paired every snapshot with the wrong time.
    """
    path = tmp_path / f"ordered_{'_'.join(map(str, order))}.h5"
    mult_series.to_h5(path)

    groups = [f"snap_{i:04d}" for i in order]
    back = read_coefs(path, group_name=groups)          # note: no times= argument

    np.testing.assert_allclose(back.times, [TIMES[i] for i in order])
    for j, orig_i in enumerate(order):
        np.testing.assert_allclose(
            back.snapshot(j).phi, mult_series.snapshot(orig_i).phi, rtol=1e-10
        )


def test_h5_write_times_false_and_no_times_arg_raises(mult_series, tmp_path):
    path = tmp_path / "no_times.h5"
    mult_series.to_h5(path, write_times=False)
    with pytest.raises(ValueError):
        read_coefs(path, group_name="all")


def test_h5_write_times_false_but_explicit_times_still_works(mult_series, tmp_path):
    path = tmp_path / "no_times_explicit.h5"
    mult_series.to_h5(path, write_times=False)
    back = read_coefs(path, group_name="all", times=mult_series.times)
    np.testing.assert_allclose(back.times, mult_series.times)


# ---------------------------------------------------------------------------
# 5. radial_power / total_power shapes (Multipole only -- CylSpline has no
#    l-indexed power spectrum)
# ---------------------------------------------------------------------------

def test_radial_power_shape_no_time_axis(mult_base):
    rp = mult_base.radial_power(0)
    assert rp.shape == (len(mult_base.R_grid),)
    tp = mult_base.total_power(0)
    assert isinstance(tp, float)


def test_radial_power_shape_with_time_axis_matches_snapshots(mult_series, mult_snaps):
    rp = mult_series.radial_power(2)
    assert rp.shape == (len(mult_series.R_grid), mult_series.n_times)
    tp = mult_series.total_power(2)
    assert tp.shape == (mult_series.n_times,)
    for i, snap in enumerate(mult_snaps):
        np.testing.assert_allclose(rp[:, i], snap.radial_power(2))
        assert tp[i] == pytest.approx(snap.total_power(2))


def test_radial_power_missing_l_no_time_axis(mult_base):
    rp = mult_base.radial_power(99)
    assert rp.shape == (len(mult_base.R_grid),)
    assert np.all(rp == 0.0)
    assert mult_base.total_power(99) == 0.0


def test_radial_power_missing_l_with_time_axis(mult_series):
    rp = mult_series.radial_power(99)
    assert rp.shape == (len(mult_series.R_grid), mult_series.n_times)
    assert np.all(rp == 0.0)
    tp = mult_series.total_power(99)
    assert tp.shape == (mult_series.n_times,)
    assert np.all(tp == 0.0)


# ---------------------------------------------------------------------------
# 6. zeroed with a time axis
# ---------------------------------------------------------------------------

def test_zeroed_mult_with_time_axis_zeroes_every_time(mult_series):
    z = mult_series.zeroed([0])
    assert z.has_time_axis
    np.testing.assert_array_equal(z.times, mult_series.times)

    keep_cols = [i for i, (l, m) in enumerate(z.lm_labels) if l == 0]
    other_cols = [i for i, (l, m) in enumerate(z.lm_labels) if l != 0]
    np.testing.assert_array_equal(z.phi[:, keep_cols], mult_series.phi[:, keep_cols])
    assert np.all(z.phi[:, other_cols] == 0.0)
    assert np.all(z.dphi_dr[:, other_cols] == 0.0)


def test_zeroed_cylsp_with_time_axis_zeroes_every_time(cylsp3_series):
    z = cylsp3_series.zeroed([0])
    assert z.has_time_axis
    np.testing.assert_array_equal(z.times, cylsp3_series.times)

    np.testing.assert_array_equal(z.phi[0], cylsp3_series.phi[0])
    np.testing.assert_array_equal(z.dphi_dR[0], cylsp3_series.dphi_dR[0])
    np.testing.assert_array_equal(z.dphi_dz[0], cylsp3_series.dphi_dz[0])
    for m in (2, 4):
        assert np.all(z.phi[m] == 0.0)
        assert np.all(z.dphi_dR[m] == 0.0)
        assert np.all(z.dphi_dz[m] == 0.0)


# ---------------------------------------------------------------------------
# 7. with_times: attach / relabel / reject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nt", [10, 1000])
def test_with_times_attach_mult(mult_base, nt):
    times = np.linspace(0.0, 1.0, nt)
    phi = np.stack([mult_base.phi] * nt, axis=-1)
    dphi_dr = np.stack([mult_base.dphi_dr] * nt, axis=-1)
    out = mult_base.with_times(times, phi=phi, dphi_dr=dphi_dr)
    assert out.n_times == nt
    assert out.phi.shape == (len(mult_base.R_grid), len(mult_base.lm_labels), nt)
    np.testing.assert_array_equal(out.times, times)


def test_with_times_rejects_wrong_leading_axes(mult_base):
    bad_phi = np.stack([mult_base.phi[:-1]] * 3, axis=-1)  # wrong nR
    with pytest.raises(ValueError):
        mult_base.with_times([0.0, 1.0, 2.0], phi=bad_phi)


def test_with_times_rejects_len_times_mismatch_phi_trailing_axis(mult_base):
    phi = np.stack([mult_base.phi] * 3, axis=-1)
    with pytest.raises(ValueError):
        mult_base.with_times([0.0, 1.0], phi=phi)  # len 2 != phi.shape[-1] == 3


def test_with_times_no_phi_relabels_only(mult_series):
    new_times = np.asarray(mult_series.times) + 100.0
    out = mult_series.with_times(new_times)
    np.testing.assert_array_equal(out.phi, mult_series.phi)
    np.testing.assert_array_equal(out.dphi_dr, mult_series.dphi_dr)
    np.testing.assert_array_equal(out.times, new_times)


def test_with_times_no_phi_rejects_wrong_length(mult_series):
    with pytest.raises(ValueError):
        mult_series.with_times(mult_series.times[:-1])


def test_with_times_no_phi_on_timeless_object_raises(mult_base):
    with pytest.raises(ValueError):
        mult_base.with_times([0.0])


def test_with_times_cylsp_attach(cylsp_base):
    nt = 4
    times = np.linspace(0.0, 1.0, nt)
    phi = {m: np.stack([v] * nt, axis=-1) for m, v in cylsp_base.phi.items()}
    out = cylsp_base.with_times(times, phi=phi)
    assert out.n_times == nt
    for m in cylsp_base.m_values:
        assert out.phi[m].shape == cylsp_base.phi[m].shape + (nt,)


def test_with_times_cylsp_rejects_wrong_m_keys(cylsp_base):
    nt = 2
    phi = {m: np.stack([v] * nt, axis=-1) for m, v in cylsp_base.phi.items()}
    del phi[list(phi)[0]]  # drop one m -- keys no longer match m_values
    with pytest.raises(ValueError):
        cylsp_base.with_times([0.0, 1.0], phi=phi)


# ---------------------------------------------------------------------------
# 8. stack_coefs error conditions
# ---------------------------------------------------------------------------

def test_stack_grid_mismatch_raises(mult_base):
    a = mult_base.copy()
    b = mult_base.copy()
    b.R_grid = b.R_grid[:-1]
    b.phi = b.phi[:-1]
    b.dphi_dr = b.dphi_dr[:-1]
    with pytest.raises(ValueError):
        stack_coefs([a, b], [0.0, 1.0])


def test_stack_label_mismatch_raises(mult_base):
    a = mult_base.copy()
    b = mult_base.copy()
    b.lm_labels = list(b.lm_labels)
    b.lm_labels[0] = (99, 0)
    with pytest.raises(ValueError):
        stack_coefs([a, b], [0.0, 1.0])


def test_stack_metadata_mismatch_raises(mult_base):
    a = mult_base.copy()
    b = mult_base.copy()
    b.metadata = dict(b.metadata)
    b.metadata["symmetry"] = "not-the-same-symmetry"
    with pytest.raises(ValueError):
        stack_coefs([a, b], [0.0, 1.0])


def test_stack_mixed_types_raises(mult_base, cylsp_base):
    with pytest.raises(TypeError):
        stack_coefs([mult_base.copy(), cylsp_base.copy()], [0.0, 1.0])


def test_stack_item_already_has_time_axis_raises(mult_series, mult_base):
    with pytest.raises(ValueError):
        stack_coefs([mult_series, mult_base.copy()], [0.0, 1.0])


def test_stack_cylsp_grid_and_m_values_mismatch_raise(cylsp_base):
    a = cylsp_base.copy()
    b = cylsp_base.copy()
    b.m_values = list(b.m_values)[:-1]
    b.phi = {m: v for m, v in b.phi.items() if m in b.m_values}
    with pytest.raises(ValueError):
        stack_coefs([a, b], [0.0, 1.0])


# ---------------------------------------------------------------------------
# 9. CylSpline three-section round trip through a time series
# ---------------------------------------------------------------------------

def test_cylsp_three_section_series_h5_round_trip(cylsp3_series, tmp_path):
    path = tmp_path / "cyl3_series.h5"
    cylsp3_series.to_h5(path)
    back = read_coefs(path, group_name="all")

    assert back.dphi_dR is not None
    assert back.dphi_dz is not None
    for m in cylsp3_series.m_values:
        np.testing.assert_allclose(back.phi[m], cylsp3_series.phi[m], rtol=1e-10)
        np.testing.assert_allclose(back.dphi_dR[m], cylsp3_series.dphi_dR[m], rtol=1e-10)
        np.testing.assert_allclose(back.dphi_dz[m], cylsp3_series.dphi_dz[m], rtol=1e-10)
        assert not np.allclose(back.phi[m], back.dphi_dz[m])
        assert not np.allclose(back.phi[m], back.dphi_dR[m])


# ---------------------------------------------------------------------------
# 10. Multipole to_coef_string requires dphi_dr
# ---------------------------------------------------------------------------

def test_multipole_to_coef_string_raises_without_dphi_dr(mult_base):
    m = mult_base.copy()
    m.dphi_dr = None
    with pytest.raises(ValueError, match="dphi_dr"):
        m.to_coef_string()


# ---------------------------------------------------------------------------
# 11. validate() catches damage from direct field assignment
# ---------------------------------------------------------------------------

def test_validate_catches_mult_phi_column_damage(mult_base):
    m = mult_base.copy()
    m.phi = m.phi[:, :-1]
    with pytest.raises(ValueError, match="MultipoleCoefs.phi"):
        m.validate()


def test_validate_catches_mult_times_truncation_damage(mult_series):
    m = mult_series.copy()
    m.times = m.times[:-1]
    with pytest.raises(ValueError, match="phi"):
        m.validate()


def test_validate_invoked_by_to_coef_string_mult(mult_base):
    m = mult_base.copy()
    m.phi = m.phi[:, :-1]
    with pytest.raises(ValueError):
        m.to_coef_string()


def test_validate_invoked_by_to_h5_mult(mult_series, tmp_path):
    m = mult_series.copy()
    m.times = m.times[:-1]
    with pytest.raises(ValueError):
        m.to_h5(tmp_path / "damaged.h5")


def test_validate_catches_cylsp_z_grid_damage(cylsp_base):
    c = cylsp_base.copy()
    c.z_grid = c.z_grid[:-1]
    with pytest.raises(ValueError, match="CylSplineCoefs"):
        c.validate()


def test_validate_invoked_by_to_coef_string_cylsp(cylsp_base):
    c = cylsp_base.copy()
    c.z_grid = c.z_grid[:-1]
    with pytest.raises(ValueError):
        c.to_coef_string()


# ---------------------------------------------------------------------------
# 12. Sequence-of-paths and Evolving .ini sources
# ---------------------------------------------------------------------------

def test_sequence_of_paths_reads_back_same_arrays(mult_snaps, tmp_path):
    paths = []
    for i, snap in enumerate(mult_snaps):
        p = tmp_path / f"snap_{i:04d}.coef_mult"
        p.write_text(snap.to_coef_string())
        paths.append(str(p))

    back = read_coefs(paths, times=TIMES)
    assert back.has_time_axis
    np.testing.assert_allclose(back.times, TIMES)
    for i, snap in enumerate(mult_snaps):
        np.testing.assert_allclose(back.snapshot(i).phi, snap.phi, rtol=1e-10)


def test_evolving_ini_round_trip_times_come_from_file(mult_series, tmp_path):
    ini_path = tmp_path / "series.ini"
    mult_series.to_evolving_ini(ini_path)

    # No times= argument -- must be recovered from the .ini's Timestamps.
    back = read_coefs(ini_path)
    assert back.has_time_axis
    np.testing.assert_allclose(back.times, mult_series.times)
    for i in range(mult_series.n_times):
        np.testing.assert_allclose(
            back.snapshot(i).phi, mult_series.snapshot(i).phi, rtol=1e-10
        )


def test_sequence_of_paths_and_ini_agree(mult_snaps, tmp_path):
    paths = []
    for i, snap in enumerate(mult_snaps):
        p = tmp_path / f"seq_{i:04d}.coef_mult"
        p.write_text(snap.to_coef_string())
        paths.append(str(p))
    from_seq = read_coefs(paths, times=TIMES)

    series = stack_coefs(mult_snaps, TIMES)
    ini_path = tmp_path / "cmp.ini"
    series.to_evolving_ini(ini_path)
    from_ini = read_coefs(ini_path)

    np.testing.assert_allclose(from_seq.times, from_ini.times)
    np.testing.assert_allclose(from_seq.phi, from_ini.phi, rtol=1e-10)


# ---------------------------------------------------------------------------
# 13. copy() is a deep copy
# ---------------------------------------------------------------------------

def test_copy_is_deep_mult(mult_base):
    c = mult_base.copy()
    original_value = mult_base.phi[0, 0]
    c.phi[0, 0] = original_value + 12345.0
    assert mult_base.phi[0, 0] == original_value
    assert c.phi[0, 0] != mult_base.phi[0, 0]


def test_copy_is_deep_mult_series(mult_series):
    c = mult_series.copy()
    original_value = mult_series.times[0]
    c.times[0] = original_value + 999.0
    assert mult_series.times[0] == original_value


def test_copy_is_deep_cylsp(cylsp_base):
    c = cylsp_base.copy()
    m0 = cylsp_base.m_values[0]
    original_value = cylsp_base.phi[m0][0, 0]
    c.phi[m0][0, 0] = original_value + 12345.0
    assert cylsp_base.phi[m0][0, 0] == original_value


# ---------------------------------------------------------------------------
# 14. column(l, m) -- Multipole only
# ---------------------------------------------------------------------------

def test_column_returns_correct_index(mult_base):
    for expected_idx, (l, m) in enumerate(mult_base.lm_labels):
        assert mult_base.column(l, m) == expected_idx


def test_column_missing_pair_raises_keyerror(mult_base):
    with pytest.raises(KeyError):
        mult_base.column(99, 0)


# ---------------------------------------------------------------------------
# 15. agama-backed materialize_potential -- series scaling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_AGAMA, reason="agama not installed")
def test_materialize_potential_timeless_mult(mult_base):
    pot = mult_base.materialize_potential()
    xyz = np.array([[3.0, 1.0, 0.5], [10.0, -2.0, 4.0]])
    assert np.all(np.isfinite(pot.potential(xyz)))


@pytest.mark.skipif(not HAS_AGAMA, reason="agama not installed")
def test_materialize_potential_series_scaling_mult(mult_base, mult_series):
    base_pot = mult_base.materialize_potential()
    series_pot = mult_series.materialize_potential()

    xyz = np.array([[3.0, 1.0, 0.5], [10.0, -2.0, 4.0], [-6.0, 5.0, -2.0]])
    base_phi = base_pot.potential(xyz)

    for k, t in zip(SCALES, TIMES):
        got = series_pot.potential(xyz, t=t)
        np.testing.assert_allclose(got, k * base_phi, rtol=1e-6)


@pytest.mark.skipif(not HAS_AGAMA, reason="agama not installed")
def test_materialize_potential_series_scaling_cylsp(cylsp3_base, cylsp3_series):
    base_pot = cylsp3_base.materialize_potential()
    series_pot = cylsp3_series.materialize_potential()

    xyz = np.array([[1.0, 0.0, 0.3], [3.0, 2.0, -1.0], [8.0, -4.0, 2.0]])
    base_phi = base_pot.potential(xyz)

    for k, t in zip(SCALES, TIMES):
        got = series_pot.potential(xyz, t=t)
        np.testing.assert_allclose(got, k * base_phi, rtol=1e-6)


# ---------------------------------------------------------------------------
# 16. Hardening: source/group dispatch and template collisions
# ---------------------------------------------------------------------------

def test_zeroed_does_not_share_times_with_parent(mult_series):
    """A caller relabelling the filtered copy must not rewrite the parent."""
    z = mult_series.zeroed([0])
    assert z.times is not mult_series.times
    z.times[0] = -999.0
    assert mult_series.times[0] == TIMES[0]


def test_zeroed_does_not_share_times_with_parent_cylsp(cylsp3_series):
    z = cylsp3_series.zeroed([0])
    assert z.times is not cylsp3_series.times
    z.times[0] = -999.0
    assert cylsp3_series.times[0] == TIMES[0]


def test_validate_names_the_field_when_cylsp_phi_is_not_a_dict(cylsp3_series):
    """A stacked array in place of the per-m dict used to raise a bare NumPy error."""
    damaged = cylsp3_series.copy()
    damaged.phi = list(damaged.phi.values())
    with pytest.raises(TypeError, match=r"CylSplineCoefs\.phi must be a dict"):
        damaged.validate()


@pytest.mark.parametrize("wrap", [np.array, iter, list, tuple], ids=["ndarray", "iterator", "list", "tuple"])
def test_sequence_sources_accept_any_ordered_container(mult_series, tmp_path, wrap):
    """np.sort(glob(...)) and generators are natural ways to build an ordered list."""
    files = mult_series.to_coef_files(tmp_path / "cf")
    back = read_coefs(wrap(files), times=TIMES)
    assert back.n_times == len(TIMES)
    np.testing.assert_allclose(back.phi, mult_series.phi, rtol=1e-10)


@pytest.mark.parametrize("wrap", [np.array, list, tuple], ids=["ndarray", "list", "tuple"])
def test_group_name_accepts_any_ordered_container(mult_series, tmp_path, wrap):
    path = tmp_path / f"groups_{wrap.__name__}.h5"
    mult_series.to_h5(path)
    back = read_coefs(path, group_name=wrap(["snap_0001", "snap_0000"]))
    np.testing.assert_allclose(back.times, [TIMES[1], TIMES[0]])
    np.testing.assert_allclose(
        back.snapshot(0).phi, mult_series.snapshot(1).phi, rtol=1e-10
    )


def test_group_name_rejected_for_sequence_source(mult_series, tmp_path):
    files = mult_series.to_coef_files(tmp_path / "cf")
    with pytest.raises(ValueError, match="has no meaning for a sequence"):
        read_coefs(files, group_name="all", times=TIMES)


def test_group_name_rejected_for_ini_source(mult_series, tmp_path):
    ini = mult_series.to_evolving_ini(tmp_path / "s.ini")
    with pytest.raises(ValueError, match="has no meaning for an Evolving .ini"):
        read_coefs(ini, group_name="all")


def test_to_coef_files_rejects_a_name_fmt_without_the_index(mult_series, tmp_path):
    """Without {i} every time would overwrite the last -- a silent data loss."""
    with pytest.raises(ValueError, match=r"name_fmt.*distinct name"):
        mult_series.to_coef_files(tmp_path / "collide", name_fmt="snap{ext}")


def test_to_h5_rejects_a_group_fmt_without_the_index(mult_series, tmp_path):
    with pytest.raises(ValueError, match=r"group_fmt.*distinct name"):
        mult_series.to_h5(tmp_path / "collide.h5", group_fmt="snap")


def test_timeless_to_coef_files_still_accepts_a_constant_name(mult_base, tmp_path):
    """One time sample cannot collide, so a fixed name stays legal."""
    paths = mult_base.to_coef_files(tmp_path / "one", name_fmt="only{ext}")
    assert len(paths) == 1
    assert Path(paths[0]).name == "only.coef_mult"
