"""
tests/test_multi_species.py
Tests for the multi-species simulation pipeline introduced in feat-multi_spec.

Covers:
  - Species dataclass validation and helpers
  - Smart mass / softening HDF5 storage
  - Backward-compatible ParticleReader for old-format files
  - New multi-species ParticleReader for new-format files
  - Restart file species metadata (save & load)
  - run_simulation CPU path (single- and two-species)
  - Performance / future warnings
  - Backward compat: old run_nbody_cpu still works without species kwarg
"""
from __future__ import annotations

import warnings
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import nbody_streams as nb
from nbody_streams.species import (
    Species,
    PerformanceWarning,
    _build_particle_arrays,
    _validate_species,
    _split_by_species,
    _emit_performance_warnings,
)
from nbody_streams.nbody_io import (
    _save_snapshot,
    _save_restart,
    _load_restart,
    _is_uniform,
)

try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pyfalcon  # noqa: F401
    PYFALCON_AVAILABLE = True
except ImportError:
    PYFALCON_AVAILABLE = False

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_xv(N: int) -> np.ndarray:
    """Random (N, 6) phase space  - tiny, just for IO tests."""
    return RNG.standard_normal((N, 6)).astype(np.float64)


def _plummer_like(N: int) -> np.ndarray:
    """Very small Plummer-like ICs for integration tests."""
    pos = RNG.standard_normal((N, 3)) * 0.5
    vel = RNG.standard_normal((N, 3)) * 0.1
    return np.hstack([pos, vel])


# ===========================================================================
# 1. Species dataclass
# ===========================================================================

class TestSpeciesDataclass:
    def test_scalar_mass_softening(self):
        s = Species(name="dark", N=100, mass=1e6, softening=0.1)
        assert s.N == 100
        assert s.mass == 1e6
        assert s.softening == 0.1

    def test_array_mass(self):
        m = np.ones(50) * 2e5
        s = Species(name="star", N=50, mass=m, softening=0.05)
        np.testing.assert_array_equal(s.mass, m)

    def test_convenience_dark(self):
        s = Species.dark(200, 1e6, 0.2)
        assert s.name == "dark"
        assert s.N == 200

    def test_convenience_star(self):
        s = Species.star(100, 1e5, 0.05)
        assert s.name == "star"

    def test_negative_N_raises(self):
        with pytest.raises(ValueError, match="N must be > 0"):
            Species(name="dark", N=0, mass=1.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Species(name="", N=10, mass=1.0)

    def test_mass_array_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="mass array shape"):
            Species(name="dark", N=10, mass=np.ones(5))

    def test_softening_array_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="softening array shape"):
            Species(name="dark", N=10, mass=1.0, softening=np.ones(5))


# ===========================================================================
# 2. _build_particle_arrays
# ===========================================================================

class TestBuildParticleArrays:
    def test_uniform_masses_expand(self):
        sp = [
            Species.dark(N=10, mass=1e6, softening=0.1),
            Species.star(N=5,  mass=1e5, softening=0.05),
        ]
        m, h = _build_particle_arrays(sp)
        assert m.shape == (15,)
        assert h.shape == (15,)
        np.testing.assert_allclose(m[:10], 1e6)
        np.testing.assert_allclose(m[10:], 1e5)
        np.testing.assert_allclose(h[:10], 0.1)
        np.testing.assert_allclose(h[10:], 0.05)

    def test_per_particle_mass_passthrough(self):
        m_arr = np.arange(1, 11, dtype=float)
        sp = [Species(name="bh", N=10, mass=m_arr, softening=0.0)]
        m, _ = _build_particle_arrays(sp)
        np.testing.assert_array_equal(m, m_arr)

    def test_single_species(self):
        sp = [Species.dark(N=3, mass=2.0, softening=0.0)]
        m, h = _build_particle_arrays(sp)
        np.testing.assert_array_equal(m, [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(h, [0.0, 0.0, 0.0])


# ===========================================================================
# 3. _validate_species
# ===========================================================================

class TestValidateSpecies:
    def test_valid(self):
        xv = _random_xv(15)
        sp = [Species.dark(10, 1e6, 0.1), Species.star(5, 1e5, 0.05)]
        _validate_species(xv, sp)  # should not raise

    def test_n_mismatch_raises(self):
        xv = _random_xv(10)
        sp = [Species.dark(8, 1e6, 0.1), Species.star(5, 1e5, 0.05)]
        with pytest.raises(ValueError, match="does not match"):
            _validate_species(xv, sp)

    def test_duplicate_name_raises(self):
        xv = _random_xv(20)
        sp = [Species.dark(10, 1e6, 0.1), Species.dark(10, 1e6, 0.1)]
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_species(xv, sp)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_species(_random_xv(10), [])


# ===========================================================================
# 4. _split_by_species
# ===========================================================================

class TestSplitBySpecies:
    def test_two_species(self):
        xv = _random_xv(15)
        sp = [Species.dark(10, 1.0), Species.star(5, 1.0)]
        out = _split_by_species(xv, sp)
        assert set(out.keys()) == {"dark", "star"}
        np.testing.assert_array_equal(out["dark"], xv[:10])
        np.testing.assert_array_equal(out["star"], xv[10:])

    def test_three_species(self):
        xv = _random_xv(30)
        sp = [
            Species("dark", 10, 1.0),
            Species("star", 15, 1.0),
            Species("bh",    5, 1.0),
        ]
        out = _split_by_species(xv, sp)
        assert out["dark"].shape == (10, 6)
        assert out["star"].shape == (15, 6)
        assert out["bh"].shape   == (5,  6)


# ===========================================================================
# 5. Performance / Future warnings
# ===========================================================================

class TestPerformanceWarnings:
    def test_cpu_direct_large_n(self):
        with pytest.warns(PerformanceWarning, match="CPU direct"):
            _emit_performance_warnings(25_000, "cpu", "direct")

    def test_cpu_direct_small_n_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _emit_performance_warnings(5_000, "cpu", "direct")  # no warning

    def test_gpu_direct_large_n(self):
        with pytest.warns(PerformanceWarning, match="GPU direct"):
            _emit_performance_warnings(600_000, "gpu", "direct")

    def test_perf_warning_2m(self):
        with pytest.warns(PerformanceWarning, match="method='tree'"):
            _emit_performance_warnings(2_100_000, "cpu", "direct")

    def test_cpu_tree_no_warning(self):
        """Tree method on CPU never warns for moderate N."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerformanceWarning)
            _emit_performance_warnings(50_000, "cpu", "tree")


# ===========================================================================
# 6. _is_uniform
# ===========================================================================

class TestIsUniform:
    def test_uniform(self):
        arr = np.full(100, 1e6)
        ok, v = _is_uniform(arr)
        assert ok is True
        assert v == pytest.approx(1e6)

    def test_non_uniform(self):
        arr = np.arange(1, 101, dtype=float)
        ok, _ = _is_uniform(arr)
        assert ok is False

    def test_single_element(self):
        ok, v = _is_uniform(np.array([3.14]))
        assert ok is True
        assert v == pytest.approx(3.14)


# ===========================================================================
# 7. Smart IO  - _save_snapshot (new multi-species path)
# ===========================================================================

class TestSmartSnapshotIO:
    def test_uniform_mass_stored_as_scalar(self, tmp_path):
        sp = [Species.dark(10, 1e6, 0.1), Species.star(5, 1e5, 0.05)]
        xv = _random_xv(15)
        _save_snapshot(xv, 0, 0.0, tmp_path, species=sp, time_step=0.01)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            props = f["properties"]
            assert "n_species" in props.attrs
            assert "dark" in props
            assert "m" in props["dark"]           # scalar stored
            assert "m_array" not in props["dark"]  # no per-particle array

    def test_per_particle_mass_stored_as_array(self, tmp_path):
        m_arr = np.linspace(1e5, 2e5, 10)
        sp = [Species(name="dark", N=10, mass=m_arr, softening=0.1)]
        xv = _random_xv(10)
        _save_snapshot(xv, 0, 0.0, tmp_path, species=sp, time_step=0.01)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            props = f["properties"]
            assert "m_array" in props["dark"]   # per-particle stored
            assert "m" not in props["dark"]

    def test_uniform_softening_scalar(self, tmp_path):
        sp = [Species.dark(8, 1e6, 0.15)]
        xv = _random_xv(8)
        _save_snapshot(xv, 0, 0.0, tmp_path, species=sp)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            assert "eps" in f["properties"]["dark"]
            assert "eps_array" not in f["properties"]["dark"]

    def test_per_particle_softening_array(self, tmp_path):
        h_arr = np.linspace(0.05, 0.15, 12)
        sp = [Species(name="dark", N=12, mass=1e6, softening=h_arr)]
        xv = _random_xv(12)
        _save_snapshot(xv, 0, 0.0, tmp_path, species=sp)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            assert "eps_array" in f["properties"]["dark"]

    def test_species_names_attribute(self, tmp_path):
        sp = [Species.dark(5, 1e6), Species.star(3, 1e5)]
        _save_snapshot(_random_xv(8), 0, 0.0, tmp_path, species=sp)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            raw = f["properties"].attrs["species_names"]
            names = [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n)
                     for n in raw]
        assert names == ["dark", "star"]

    def test_three_species(self, tmp_path):
        sp = [
            Species.dark(10, 1e6, 0.1),
            Species.star(5,  1e5, 0.05),
            Species("bh", 1, 1e9, 0.001),
        ]
        xv = _random_xv(16)
        _save_snapshot(xv, 0, 0.0, tmp_path, species=sp)

        with h5py.File(tmp_path / "snapshot.h5", "r") as f:
            props = f["properties"]
            assert int(props.attrs["n_species"]) == 3
            assert "bh" in props


# ===========================================================================
# 8. ParticleReader  - backward compat (old dark/star format)
# ===========================================================================

class TestParticleReaderBackwardCompat:
    def _write_old_format(self, tmp_path, N_dark=10, N_star=5):
        """Write an HDF5 file in the old two-species format."""
        xv = _random_xv(N_dark + N_star)
        _save_snapshot(
            xv, 0, 1.5, tmp_path,
            num_dark=N_dark, num_star=N_star,
            mass_dark=1e6, mass_star=1e5,
            eps_dark=0.1, eps_star=0.05,
            time_step=0.01,
        )
        return xv

    def test_reads_num_dark_star(self, tmp_path):
        self._write_old_format(tmp_path, N_dark=10, N_star=5)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        assert r.num_dark == 10
        assert r.num_star == 5

    def test_reads_mass_dark_star(self, tmp_path):
        self._write_old_format(tmp_path)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        assert r.mass_dark == pytest.approx(1e6)
        assert r.mass_star == pytest.approx(1e5)

    def test_read_snapshot_dark_star_shape(self, tmp_path):
        self._write_old_format(tmp_path, N_dark=8, N_star=4)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        part = r.read_snapshot(0)
        assert part.dark["posvel"].shape == (8, 6)
        assert part.star["posvel"].shape == (4, 6)

    def test_read_snapshot_part_species_dict(self, tmp_path):
        self._write_old_format(tmp_path, N_dark=6, N_star=3)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        part = r.read_snapshot(0)
        assert "dark" in part.species
        assert "star" in part.species

    def test_species_list_populated(self, tmp_path):
        self._write_old_format(tmp_path, N_dark=7, N_star=2)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        names = [s.name for s in r.species_list]
        assert "dark" in names
        assert "star" in names

    def test_dark_only_no_star_group(self, tmp_path):
        """Old file with only dark particles (no star group)."""
        xv = _random_xv(12)
        _save_snapshot(xv, 0, 0.0, tmp_path,
                       num_dark=12, num_star=0,
                       mass_dark=1e6, eps_dark=0.1)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        assert r.num_dark == 12
        assert r.num_star == 0


# ===========================================================================
# 9. ParticleReader  - new multi-species format
# ===========================================================================

class TestParticleReaderMultiSpecies:
    def _write_new_format(self, tmp_path, species):
        xv = _random_xv(sum(s.N for s in species))
        _save_snapshot(xv, 0, 0.0, tmp_path, species=species)
        return xv

    def test_two_species_reader(self, tmp_path):
        sp = [Species.dark(10, 1e6, 0.1), Species.star(5, 1e5, 0.05)]
        self._write_new_format(tmp_path, sp)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        assert r.num_dark == 10
        assert r.num_star == 5

    def test_three_species_reader(self, tmp_path):
        sp = [
            Species.dark(8,  1e6, 0.1),
            Species.star(4,  1e5, 0.05),
            Species("bh", 1, 1e9, 0.001),
        ]
        self._write_new_format(tmp_path, sp)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        names = [s.name for s in r.species_list]
        assert "dark" in names
        assert "star" in names
        assert "bh" in names

    def test_read_snapshot_species_key(self, tmp_path):
        sp = [Species.dark(6, 1e6), Species.star(3, 1e5)]
        orig_xv = self._write_new_format(tmp_path, sp)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        part = r.read_snapshot(0)
        assert "dark" in part.species
        assert "star" in part.species
        assert part.species["dark"]["posvel"].shape == (6, 6)
        np.testing.assert_allclose(part.species["dark"]["posvel"],
                                   orig_xv[:6], rtol=1e-6)

    def test_per_particle_mass_round_trips(self, tmp_path):
        m_arr = np.linspace(1e5, 2e5, 10)
        sp = [Species(name="dark", N=10, mass=m_arr, softening=0.1)]
        self._write_new_format(tmp_path, sp)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        # species_list should have the full array
        dark_sp = next(s for s in r.species_list if s.name == "dark")
        assert not np.isscalar(dark_sp.mass)
        np.testing.assert_allclose(dark_sp.mass, m_arr)

    def test_backward_compat_dark_attr(self, tmp_path):
        sp = [Species.dark(7, 2e6, 0.2), Species.star(3, 2e5, 0.1)]
        self._write_new_format(tmp_path, sp)
        r = nb.ParticleReader(str(tmp_path / "snapshot.h5"))
        # Legacy attributes must still work
        assert r.num_dark == 7
        assert r.mass_dark == pytest.approx(2e6)


# ===========================================================================
# 10. Restart file  - species metadata
# ===========================================================================

class TestRestartFileSpecies:
    def test_save_and_load_with_species(self, tmp_path):
        N = 15
        xv = _random_xv(N)
        m = np.ones(N) * 1e6
        h = np.ones(N) * 0.1
        names = ["dark", "star"]
        Ns = [10, 5]

        _save_restart(xv, 1.5, 100, tmp_path, 3,
                      mass_arr=m, softening_arr=h,
                      species_names=names, species_N=Ns)

        result = _load_restart(tmp_path)
        assert result is not None
        (xv_r, t, step, snap_c,
         m_r, h_r, sp_names_r, sp_N_r) = result

        np.testing.assert_array_equal(xv_r, xv)
        assert t == pytest.approx(1.5)
        assert step == 100
        assert snap_c == 3
        np.testing.assert_array_equal(m_r, m)
        np.testing.assert_array_equal(h_r, h)
        assert sp_names_r == ["dark", "star"]
        assert sp_N_r == [10, 5]

    def test_load_old_restart_backward_compat(self, tmp_path):
        """Old restart files without species fields load without error."""
        xv = _random_xv(10)
        # Save using old API (no species kwargs)
        _save_restart(xv, 0.5, 50, tmp_path, 2)

        result = _load_restart(tmp_path)
        assert result is not None
        (xv_r, t, step, snap_c,
         m_r, h_r, sp_names_r, sp_N_r) = result

        np.testing.assert_array_equal(xv_r, xv)
        assert t == pytest.approx(0.5)
        assert step == 50
        assert snap_c == 2
        # Old file: these are None
        assert m_r is None
        assert h_r is None
        assert sp_names_r is None
        assert sp_N_r is None

    def test_no_restart_returns_none(self, tmp_path):
        assert _load_restart(tmp_path) is None


# ===========================================================================
# 11. run_simulation  - CPU integration tests (no GPU needed)
# ===========================================================================

class TestRunSimulationCPU:
    N_DARK = 50
    N_STAR = 30
    DT     = 1e-3
    T_END  = 5e-3   # 5 steps  - just check it runs and returns

    def test_single_species_returns_dict(self, tmp_path):
        xv = _plummer_like(self.N_DARK)
        dm = Species.dark(self.N_DARK, mass=1.0, softening=0.05)
        result = nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="direct",
            output_dir=str(tmp_path / "out_single"),
            save_snapshots=False, verbose=False,
        )
        assert isinstance(result, dict)
        assert "dark" in result
        assert result["dark"].shape == (self.N_DARK, 6)

    def test_two_species_returns_correct_shapes(self, tmp_path):
        xv_dm = _plummer_like(self.N_DARK)
        xv_st = _plummer_like(self.N_STAR)
        xv = np.vstack([xv_dm, xv_st])
        dm   = Species.dark(self.N_DARK, mass=1.0, softening=0.05)
        star = Species.star(self.N_STAR, mass=0.1, softening=0.02)
        result = nb.run_simulation(
            xv, [dm, star],
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="direct",
            output_dir=str(tmp_path / "out_two"),
            save_snapshots=False, verbose=False,
        )
        assert result["dark"].shape  == (self.N_DARK, 6)
        assert result["star"].shape  == (self.N_STAR, 6)

    def test_three_species_dict_keys(self, tmp_path):
        N_BH = 1
        xv = _plummer_like(self.N_DARK + self.N_STAR + N_BH)
        sp = [
            Species.dark(self.N_DARK, 1.0, 0.05),
            Species.star(self.N_STAR, 0.1, 0.02),
            Species("bh", N_BH, 1e3, 0.001),
        ]
        result = nb.run_simulation(
            xv, sp,
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="direct",
            output_dir=str(tmp_path / "out_three"),
            save_snapshots=False, verbose=False,
        )
        assert set(result.keys()) == {"dark", "star", "bh"}

    def test_snapshots_written_with_species_metadata(self, tmp_path):
        xv = _plummer_like(self.N_DARK)
        dm = Species.dark(self.N_DARK, mass=1.0, softening=0.05)
        out = tmp_path / "out_snap"
        nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="direct",
            output_dir=str(out),
            save_snapshots=True, snapshots=2, verbose=False,
        )
        # Check HDF5 has new multi-species schema
        hdf5_files = list(out.glob("*.h5"))
        assert hdf5_files, "No HDF5 file written"
        with h5py.File(hdf5_files[0], "r") as f:
            assert "n_species" in f["properties"].attrs

    def test_reader_reads_simulation_output(self, tmp_path):
        xv = _plummer_like(self.N_DARK)
        dm = Species.dark(self.N_DARK, mass=1.0, softening=0.05)
        out = tmp_path / "out_reader"
        nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="direct",
            output_dir=str(out),
            save_snapshots=True, snapshots=2, verbose=False,
        )
        r = nb.ParticleReader(str(out / "snapshot.h5"))
        assert r.num_dark == self.N_DARK
        part = r.read_snapshot(0)
        assert part.dark["posvel"].shape == (self.N_DARK, 6)

    @pytest.mark.skipif(not PYFALCON_AVAILABLE, reason="pyfalcon not installed")
    def test_cpu_tree_runs(self, tmp_path):
        xv = _plummer_like(self.N_DARK)
        dm = Species.dark(self.N_DARK, mass=1.0, softening=0.05)
        result = nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T_END, dt=self.DT,
            G=1.0,
            architecture="cpu", method="tree",
            output_dir=str(tmp_path / "out_tree"),
            save_snapshots=False, verbose=False,
        )
        assert result["dark"].shape == (self.N_DARK, 6)


# ===========================================================================
# 12. run_simulation  - validation / error paths
# ===========================================================================

class TestRunSimulationValidation:
    def test_bad_architecture_raises(self):
        xv = _random_xv(10)
        sp = [Species.dark(10, 1.0)]
        with pytest.raises(ValueError, match="architecture"):
            nb.run_simulation(xv, sp, 0.0, 0.01, 0.001,
                              architecture="tpu")

    def test_bad_method_raises(self):
        xv = _random_xv(10)
        sp = [Species.dark(10, 1.0)]
        with pytest.raises(ValueError, match="method"):
            nb.run_simulation(xv, sp, 0.0, 0.01, 0.001,
                              method="fmm")

    def test_species_n_mismatch_raises(self):
        xv = _random_xv(10)
        sp = [Species.dark(8, 1.0)]   # only 8 but xv has 10
        with pytest.raises(ValueError, match="does not match"):
            nb.run_simulation(xv, sp, 0.0, 0.01, 0.001,
                              architecture="cpu")

    def test_wrong_phase_space_shape_raises(self):
        xv = _random_xv(10)[:, :5]   # (10, 5)  - wrong
        sp = [Species.dark(10, 1.0)]
        with pytest.raises(ValueError, match="shape"):
            nb.run_simulation(xv, sp, 0.0, 0.01, 0.001,
                              architecture="cpu")

    def test_performance_warning_via_run_simulation(self):
        """run_simulation emits PerformanceWarning for large CPU direct N."""
        N = 25_000
        xv = np.zeros((N, 6))
        sp = [Species.dark(N, 1.0, 0.1)]
        with pytest.warns(PerformanceWarning):
            # Raises ImportError or actually runs  - we only care about warning
            try:
                nb.run_simulation(xv, sp, 0.0, 1e-10, 1e-10,
                                  architecture="cpu", method="direct",
                                  save_snapshots=False, verbose=False)
            except Exception:
                pass


# ===========================================================================
# 13. Backward compat  - old run_nbody_cpu / run_nbody_gpu work unchanged
# ===========================================================================

class TestOldAPIBackwardCompat:
    def test_run_nbody_cpu_no_species_kwarg(self, tmp_path):
        N = 20
        xv = _plummer_like(N)
        masses = np.ones(N)
        final = nb.run_nbody_cpu(
            xv, masses,
            time_start=0.0, time_end=1e-3, dt=1e-4,
            softening=0.05, G=1.0,
            method="direct", kernel="plummer",
            output_dir=str(tmp_path / "old_cpu"),
            save_snapshots=False, verbose=False,
        )
        assert final.shape == (N, 6)

    def test_run_nbody_cpu_snapshot_old_format(self, tmp_path):
        """Without species kwarg, old dark/star HDF5 schema is written."""
        N = 20
        xv = _plummer_like(N)
        masses = np.ones(N)
        out = tmp_path / "old_format_snap"
        nb.run_nbody_cpu(
            xv, masses,
            time_start=0.0, time_end=1e-3, dt=5e-4,
            softening=0.05, G=1.0,
            method="direct", kernel="plummer",
            output_dir=str(out),
            save_snapshots=True, snapshots=2, verbose=False,
        )
        hdf5_files = list(out.glob("*.h5"))
        assert hdf5_files
        with h5py.File(hdf5_files[0], "r") as f:
            # Old format: no n_species attribute
            assert "n_species" not in f["properties"].attrs
            assert "dark" in f["properties"]

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
    def test_run_nbody_gpu_no_species_kwarg(self, tmp_path):
        N = 20
        xv = _plummer_like(N)
        masses = np.ones(N)
        final = nb.run_nbody_gpu(
            xv, masses,
            time_start=0.0, time_end=1e-3, dt=1e-4,
            softening=0.05, G=1.0,
            output_dir=str(tmp_path / "old_gpu"),
            save_snapshots=False, verbose=False,
        )
        assert final.shape == (N, 6)


# ===========================================================================
# 14. debug_energy -- dispatcher coverage for all CPU paths
# ===========================================================================

class TestDebugEnergy:
    """
    Verify that debug_energy=True works without errors on all dispatcher paths
    that are available without a GPU (CPU direct and CPU tree).
    The test only checks that the run completes and returns correct shapes --
    energy correctness is covered by test_physics.py.
    """

    N    = 20
    DT   = 1e-3
    T    = 2e-3   # 2 steps

    def test_debug_energy_cpu_direct(self, tmp_path, capsys):
        xv = _plummer_like(self.N)
        dm = Species.dark(self.N, mass=1.0, softening=0.05)
        result = nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T, dt=self.DT,
            G=1.0, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "de_cpu_direct"),
            save_snapshots=False, verbose=True,
            debug_energy=True,
        )
        assert result["dark"].shape == (self.N, 6)
        captured = capsys.readouterr()
        assert "Energy" in captured.out or "dE" in captured.out, (
            "debug_energy=True should print energy info"
        )

    @pytest.mark.skipif(not PYFALCON_AVAILABLE, reason="pyfalcon not installed")
    def test_debug_energy_cpu_tree(self, tmp_path, capsys):
        xv = _plummer_like(self.N)
        dm = Species.dark(self.N, mass=1.0, softening=0.05)
        result = nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T, dt=self.DT,
            G=1.0, architecture="cpu", method="tree",
            output_dir=str(tmp_path / "de_cpu_tree"),
            save_snapshots=False, verbose=True,
            debug_energy=True,
        )
        assert result["dark"].shape == (self.N, 6)
        captured = capsys.readouterr()
        assert "Energy" in captured.out or "dE" in captured.out

    def test_debug_energy_false_no_output(self, tmp_path, capsys):
        """debug_energy=False (default) must not print energy lines."""
        xv = _plummer_like(self.N)
        dm = Species.dark(self.N, mass=1.0, softening=0.05)
        nb.run_simulation(
            xv, [dm],
            time_start=0.0, time_end=self.T, dt=self.DT,
            G=1.0, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "de_off"),
            save_snapshots=False, verbose=False,
            debug_energy=False,
        )
        captured = capsys.readouterr()
        assert "dE/E" not in captured.out

    def test_unexpected_kwarg_cpu_raises(self):
        """run_simulation must raise TypeError on unknown kwargs for CPU path."""
        xv = _plummer_like(self.N)
        dm = Species.dark(self.N, mass=1.0, softening=0.05)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            nb.run_simulation(
                xv, [dm],
                time_start=0.0, time_end=self.T, dt=self.DT,
                G=1.0, architecture="cpu", method="direct",
                save_snapshots=False, verbose=False,
                totally_unknown_kwarg=42,
            )

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
    def test_unexpected_kwarg_gpu_direct_raises(self):
        """run_simulation must raise TypeError on unknown kwargs for GPU direct path."""
        xv = _plummer_like(self.N)
        dm = Species.dark(self.N, mass=1.0, softening=0.05)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            nb.run_simulation(
                xv, [dm],
                time_start=0.0, time_end=self.T, dt=self.DT,
                G=1.0, architecture="gpu", method="direct",
                save_snapshots=False, verbose=False,
                totally_unknown_kwarg=42,
            )
