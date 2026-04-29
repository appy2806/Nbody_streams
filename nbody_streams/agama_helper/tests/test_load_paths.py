"""
test_load_paths.py
~~~~~~~~~~~~~~~~~~
Tests for load_agama_potential / load_agama_evolving_potential with gpu=True,
multi-section INI parsing, and GPU paths for composite/analytic potential types.

Covers
------
1. load_agama_potential(gpu=True) — Multipole coef file, CylSpline coef file,
   MultipoleCoefs dataclass, with keep_lm_mult filtering, with center=
2. load_agama_evolving_potential(gpu=True) — from HDF5 (built in-test)
3. Multi-section INI: [Potential halo] + [Potential disk] — arbitrary headers
4. Case-insensitive params: Mass=, ScaleRadius= etc.
5. Disk GPU path — CompositePotentialGPU(DiskAnsatzGPU + MultipolePotentialGPU)
   accuracy vs Agama CPU  (tolerances from tech_err.md: phi~5e-5, force~3e-3)
6. King GPU path — MultipolePotentialGPU built via Agama CPU export
   accuracy vs Agama CPU inside tidal radius
"""

import os, tempfile
import numpy as np
import pytest

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp
from nbody_streams.agama_helper import (
    PotentialGPU,
    load_agama_potential,
    load_agama_evolving_potential,
    read_coefs,
    write_snapshot_coefs_to_h5,
)
from nbody_streams.agama_helper._potential import (
    MultipolePotentialGPU,
    CylSplinePotentialGPU,
    CompositePotentialGPU,
    EvolvingPotentialGPU,
    ShiftedPotentialGPU,
)
from nbody_streams.agama_helper._analytic_potentials import (
    DiskAnsatzPotentialGPU,
    NFWPotentialGPU,
    MiyamotoNagaiPotentialGPU,
)


# ---------------------------------------------------------------------------
# Paths to test coef files (present in the same directory)
# ---------------------------------------------------------------------------

_HERE           = os.path.dirname(__file__)
_COEF_MULT      = os.path.join(_HERE, "600.dark.none_8.coef_mul_DR")
_COEF_CYLSP     = os.path.join(_HERE, "600.bar.none_8.coef_cylsp_DR")
_COEF_LMC       = os.path.join(_HERE, "100.LMC.none_8.coef_mult")

_HAS_MULT       = os.path.exists(_COEF_MULT)
_HAS_CYLSP      = os.path.exists(_COEF_CYLSP)
_HAS_LMC        = os.path.exists(_COEF_LMC)

skip_no_mult    = pytest.mark.skipif(not _HAS_MULT,  reason=f"Missing {_COEF_MULT}")
skip_no_cylsp   = pytest.mark.skipif(not _HAS_CYLSP, reason=f"Missing {_COEF_CYLSP}")
skip_no_lmc     = pytest.mark.skipif(not _HAS_LMC,   reason=f"Missing {_COEF_LMC}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(N, seed=42, lo=-30, hi=30):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lo, hi, (N, 3)).astype(np.float64)
    xyz[:, :2] += 0.1
    return xyz


def _rel_phi(gpu, cpu):
    return float(np.max(np.abs(np.asarray(gpu) - cpu) / (np.abs(cpu) + 1e-30)))


def _rel_force(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1))) + 1e-30
    return float(np.max(np.abs(np.asarray(gpu_f) - cpu_f)) / fmag)


# ===========================================================================
# 1. load_agama_potential(gpu=True)
# ===========================================================================

class TestLoadPotentialGPU:

    @skip_no_mult
    def test_multipole_coef_file(self):
        """gpu=True from a Multipole coef file → MultipolePotentialGPU."""
        pot = load_agama_potential(_COEF_MULT, gpu=True)
        assert isinstance(pot, MultipolePotentialGPU)

        xyz = _pts(200)
        phi = pot.potential(cp.asarray(xyz))
        assert phi.shape == (200,)
        assert cp.all(cp.isfinite(phi))

    @skip_no_cylsp
    def test_cylspline_coef_file(self):
        """gpu=True from a CylSpline coef file → CylSplinePotentialGPU."""
        pot = load_agama_potential(_COEF_CYLSP, gpu=True)
        assert isinstance(pot, CylSplinePotentialGPU)

        xyz = _pts(200, lo=1, hi=60)
        phi = pot.potential(cp.asarray(xyz))
        assert phi.shape == (200,)
        assert cp.all(cp.isfinite(phi))

    @skip_no_mult
    def test_multipole_coefs_dataclass(self):
        """gpu=True from a MultipoleCoefs dataclass → MultipolePotentialGPU."""
        mc  = read_coefs(_COEF_MULT)
        pot = load_agama_potential(mc, gpu=True)
        assert isinstance(pot, MultipolePotentialGPU)

    @skip_no_mult
    def test_filtered_multipole_gpu_matches_cpu(self):
        """gpu=True + keep_lm_mult: GPU and CPU should agree after filtering."""
        keep = [(0, 0), (2, 0), (2, 2)]
        pot_gpu = load_agama_potential(_COEF_MULT, keep_lm_mult=keep, gpu=True)
        pot_cpu = load_agama_potential(_COEF_MULT, keep_lm_mult=keep, gpu=False)
        assert isinstance(pot_gpu, MultipolePotentialGPU)

        xyz = _pts(500)
        phi_gpu = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz)))
        phi_cpu = pot_cpu.potential(xyz)
        assert _rel_phi(phi_gpu, phi_cpu) < 1e-3

    @skip_no_mult
    def test_gpu_true_with_center(self):
        """gpu=True with center= should return ShiftedPotentialGPU."""
        center = np.array([1.0, 2.0, 0.5])
        pot = load_agama_potential(_COEF_MULT, gpu=True, center=center)
        assert isinstance(pot, ShiftedPotentialGPU)

        xyz_cp = cp.asarray(_pts(100))
        phi    = pot.potential(xyz_cp)
        assert phi.shape == (100,)
        assert cp.all(cp.isfinite(phi))

    @skip_no_mult
    def test_gpu_false_returns_agama(self):
        """Default gpu=False returns agama.Potential, not a GPU object."""
        pot = load_agama_potential(_COEF_MULT, gpu=False)
        assert type(pot).__name__ == "Potential"   # agama.Potential


# ===========================================================================
# 2. load_agama_evolving_potential(gpu=True) — tested with in-test HDF5
# ===========================================================================

class TestLoadEvolvingGPU:

    @skip_no_mult
    def test_evolving_from_h5_gpu(self, tmp_path):
        """Build a 3-snapshot HDF5, load with gpu=True → EvolvingPotentialGPU."""
        h5_path = str(tmp_path / "test_evolving.h5")
        times   = np.array([0.0, 1.0, 2.0])
        write_snapshot_coefs_to_h5(
            snapshot_ids=[0, 1, 2],
            coef_file_patterns=[_COEF_MULT],
            h5_output_paths=[h5_path],
            times=times,
        )

        pot = load_agama_evolving_potential(h5_path, times=times, gpu=True)
        assert isinstance(pot, EvolvingPotentialGPU)
        assert len(pot._pots) == 3

        xyz_cp = cp.asarray(_pts(100))
        phi    = pot.potential(xyz_cp, t=1.0)
        assert phi.shape == (100,)
        assert cp.all(cp.isfinite(phi))

    @skip_no_mult
    def test_evolving_gpu_with_center(self, tmp_path):
        """Evolving gpu=True with center= → EvolvingPotentialGPU inside ShiftedPotentialGPU."""
        h5_path = str(tmp_path / "test_evolving_shift.h5")
        times   = np.array([0.0, 1.0])
        write_snapshot_coefs_to_h5(
            snapshot_ids=[0, 1],
            coef_file_patterns=[_COEF_MULT],
            h5_output_paths=[h5_path],
            times=times,
        )
        center = np.array([0.0, 0.0, 0.0])
        pot    = load_agama_evolving_potential(h5_path, times=times, gpu=True, center=center)
        assert isinstance(pot, ShiftedPotentialGPU)


# ===========================================================================
# 3. Multi-section INI parsing
# ===========================================================================

class TestMultiSectionINI:

    def _write_pot(self, tmp_path, content, suffix=".ini"):
        """Write a potential config file with the given suffix (any extension works)."""
        p = tmp_path / f"pot{suffix}"
        p.write_text(content)
        return str(p)

    def test_content_detection_arbitrary_extension(self, tmp_path):
        """_is_potential_ini detects multi-section content regardless of extension.

        Files that start with [Potential ...] are treated as potential configs
        even when they carry a non-.ini extension (.pot, .dat, .txt, no extension).
        """
        ini = "[Potential]\ntype = NFW\nmass = 1e12\nscaleRadius = 20\n"
        for suffix in (".dat", ".txt", ".pot", ".cfg", ""):
            p = self._write_pot(tmp_path, ini, suffix=suffix)
            pot = PotentialGPU(file=p)
            assert isinstance(pot, NFWPotentialGPU), f"Failed for suffix {suffix!r}"

    def test_two_section_arbitrary_headers(self, tmp_path):
        """[Potential halo] + [Potential disk] headers with a .dat extension."""
        ini = (
            "[Potential halo]\n"
            "type = NFW\n"
            "mass = 1e12\n"
            "scaleRadius = 20\n\n"
            "[Potential disk]\n"
            "type = MiyamotoNagai\n"
            "mass = 5e10\n"
            "scaleRadius = 3\n"
            "scaleHeight = 0.3\n"
        )
        # Use a non-.ini extension to prove detection is content-based
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini, suffix=".dat"))
        assert isinstance(pot, CompositePotentialGPU)
        assert len(pot._components) == 2
        assert isinstance(pot._components[0], NFWPotentialGPU)
        assert isinstance(pot._components[1], MiyamotoNagaiPotentialGPU)

        # Values should be finite
        xyz_cp = cp.asarray(_pts(100))
        assert cp.all(cp.isfinite(pot.potential(xyz_cp)))

    def test_three_sections_numeric(self, tmp_path):
        """[Potential 0] + [Potential 1] + [Potential 2] (numbered headers)."""
        ini = (
            "[Potential 0]\n"
            "type = NFW\n"
            "mass = 1e12\n"
            "scaleRadius = 20\n\n"
            "[Potential 1]\n"
            "type = Plummer\n"
            "mass = 5e11\n"
            "scaleRadius = 10\n\n"
            "[Potential 2]\n"
            "type = Isochrone\n"
            "mass = 1e11\n"
            "scaleRadius = 2\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, CompositePotentialGPU)
        assert len(pot._components) == 3

    def test_single_section_no_name(self, tmp_path):
        """Plain [Potential] header → single GPU pot, not composite."""
        ini = (
            "[Potential]\n"
            "type = NFW\n"
            "mass = 1e12\n"
            "scaleRadius = 20\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, NFWPotentialGPU)

    def test_case_insensitive_type_key(self, tmp_path):
        """TYPE = NFW (uppercase) should be recognised."""
        ini = (
            "[Potential]\n"
            "TYPE = NFW\n"
            "mass = 1e12\n"
            "scaleRadius = 20\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, NFWPotentialGPU)

    def test_case_insensitive_param_keys(self, tmp_path):
        """Mixed-case param keys: Mass=, ScaleRadius= should work."""
        ini = (
            "[Potential]\n"
            "type = NFW\n"
            "Mass = 1e12\n"
            "ScaleRadius = 20\n"
        )
        pot_gpu = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot_gpu, NFWPotentialGPU)

        pot_cpu = agama.Potential(type='NFW', mass=1e12, scaleRadius=20)
        xyz = _pts(200)
        phi_gpu = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz)))
        phi_cpu = pot_cpu.potential(xyz)
        assert _rel_phi(phi_gpu, phi_cpu) < 1e-10

    @skip_no_mult
    def test_section_with_multipole_file_ref(self, tmp_path):
        """[Potential] type=Multipole file=<path> → MultipolePotentialGPU."""
        ini = (
            "[Potential halo]\n"
            f"type = Multipole\n"
            f"file = {_COEF_MULT}\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, MultipolePotentialGPU)

    @skip_no_cylsp
    def test_section_with_cylspline_file_ref(self, tmp_path):
        """[Potential bar] type=CylSpline file=<path> → CylSplinePotentialGPU."""
        ini = (
            "[Potential bar]\n"
            f"type = CylSpline\n"
            f"file = {_COEF_CYLSP}\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, CylSplinePotentialGPU)

    @skip_no_mult
    @skip_no_cylsp
    def test_mixed_composite_bfe_plus_analytic(self, tmp_path):
        """Composite: Multipole file + analytic NFW section."""
        ini = (
            "[Potential halo]\n"
            f"type = Multipole\n"
            f"file = {_COEF_MULT}\n\n"
            "[Potential outer]\n"
            "type = NFW\n"
            "mass = 5e11\n"
            "scaleRadius = 50\n"
        )
        pot = PotentialGPU(file=self._write_pot(tmp_path, ini))
        assert isinstance(pot, CompositePotentialGPU)
        assert isinstance(pot._components[0], MultipolePotentialGPU)
        assert isinstance(pot._components[1], NFWPotentialGPU)


# ===========================================================================
# 4. Disk GPU accuracy
# ===========================================================================

class TestDiskGPUAccuracy:
    """
    Disk is a CompositePotentialGPU of DiskAnsatzPotentialGPU + MultipolePotentialGPU.
    Agama does not export DiskAnsatz parameters in its .export() call, so the GPU
    path reconstructs DiskAnsatzPotentialGPU directly from the input kwargs.

    Accuracy (from tech_err.md): phi ~5e-5, force ~3e-3 vs Agama CPU.
    In practice the agreement is much tighter (~1e-6) because DiskAnsatz is an
    analytic kernel and the Multipole export lmax=32 is very high.
    """

    _KW = dict(surfaceDensity=5e8, scaleRadius=3.0, scaleHeight=0.3)

    @pytest.fixture(scope="class")
    def pots(self):
        gpu = PotentialGPU(type='Disk', **self._KW)
        cpu = agama.Potential(type='Disk', **self._KW)
        return gpu, cpu

    def test_returns_composite(self, pots):
        gpu, _ = pots
        assert isinstance(gpu, CompositePotentialGPU)
        comps = gpu._components
        assert any(isinstance(c, DiskAnsatzPotentialGPU) for c in comps)
        assert any(isinstance(c, MultipolePotentialGPU)  for c in comps)

    def test_phi_accuracy(self, pots):
        gpu, cpu = pots
        xyz = _pts(500, lo=-10, hi=10)
        phi_gpu = cp.asnumpy(gpu.potential(cp.asarray(xyz)))
        phi_cpu = cpu.potential(xyz)
        assert _rel_phi(phi_gpu, phi_cpu) < 1e-4, (
            f"Disk phi rel err {_rel_phi(phi_gpu, phi_cpu):.2e} > 1e-4"
        )

    def test_force_accuracy(self, pots):
        gpu, cpu = pots
        xyz = _pts(500, lo=-10, hi=10)
        f_gpu = cp.asnumpy(gpu.force(cp.asarray(xyz)))
        f_cpu = cpu.force(xyz)
        assert _rel_force(f_gpu, f_cpu) < 5e-3, (
            f"Disk force rel err {_rel_force(f_gpu, f_cpu):.2e} > 5e-3"
        )

    def test_disk_with_inner_cutoff(self):
        """Disk with innerCutoffRadius should still build without error."""
        gpu = PotentialGPU(
            type='Disk',
            surfaceDensity=5e8,
            scaleRadius=3.0,
            scaleHeight=0.3,
            innerCutoffRadius=1.0,
        )
        assert isinstance(gpu, CompositePotentialGPU)
        xyz_cp = cp.asarray(_pts(50, lo=-10, hi=10))
        assert cp.all(cp.isfinite(gpu.potential(xyz_cp)))

    def test_disk_via_load_agama_potential_gpu(self):
        """load_agama_potential(gpu=True) on a Disk coef file (exported by Agama)
        should produce a working BFE potential (Agama exports Disk as Multipole)."""
        import uuid, os as _os
        kw = dict(surfaceDensity=5e8, scaleRadius=3.0, scaleHeight=0.3)
        disk_agama = agama.Potential(type='Disk', **kw)
        shm = "/dev/shm"
        tmp_dir  = shm if (_os.path.isdir(shm) and _os.access(shm, _os.W_OK)) \
                       else tempfile.gettempdir()
        tmp_path = _os.path.join(tmp_dir, f"disk_test_{uuid.uuid4().hex}.coef")
        try:
            disk_agama.export(tmp_path)
            pot = load_agama_potential(tmp_path, gpu=True)
        finally:
            try: _os.unlink(tmp_path)
            except FileNotFoundError: pass
        # Agama exports Disk as Multipole+DiskAnsatz; our loader skips DiskAnsatz
        # (no stored params) and returns only the Multipole BFE part.
        assert isinstance(pot, MultipolePotentialGPU)
        xyz_cp = cp.asarray(_pts(100, lo=-10, hi=10))
        assert cp.all(cp.isfinite(pot.potential(xyz_cp)))


# ===========================================================================
# 5. King GPU accuracy
# ===========================================================================

class TestKingGPUAccuracy:
    """
    King is exported as a spherical (lmax=0) Multipole BFE by Agama.
    Agreement vs Agama CPU is ~1e-5 inside the tidal radius.
    Test points must stay within the tidal radius (~2*scaleRadius for trunc=2).
    """

    _KW = dict(mass=1e5, scaleRadius=0.01, W0=7.0, trunc=2.0)

    @pytest.fixture(scope="class")
    def pots(self):
        gpu = PotentialGPU(type='King', **self._KW)
        cpu = agama.Potential(type='King', **self._KW)
        return gpu, cpu

    def test_returns_multipole(self, pots):
        gpu, _ = pots
        assert isinstance(gpu, MultipolePotentialGPU)

    def test_phi_accuracy_inside_tidal_radius(self, pots):
        gpu, cpu = pots
        rng = np.random.default_rng(42)
        xyz = rng.uniform(-0.015, 0.015, (300, 3)).astype(np.float64)
        xyz[:, :2] += 1e-4   # avoid exact z-axis

        phi_gpu = cp.asnumpy(gpu.potential(cp.asarray(xyz)))
        phi_cpu = cpu.potential(xyz)
        rel = _rel_phi(phi_gpu, phi_cpu)
        assert rel < 1e-3, f"King phi rel err {rel:.2e} > 1e-3"

    def test_force_accuracy_inside_tidal_radius(self, pots):
        gpu, cpu = pots
        rng = np.random.default_rng(7)
        xyz = rng.uniform(-0.015, 0.015, (300, 3)).astype(np.float64)
        xyz[:, :2] += 1e-4

        f_gpu = cp.asnumpy(gpu.force(cp.asarray(xyz)))
        f_cpu = cpu.force(xyz)
        assert _rel_force(f_gpu, f_cpu) < 1e-2


# ===========================================================================
# 6. PotentialGPU(type=...) case-insensitive dispatch
# ===========================================================================

class TestPotentialGPUCaseInsensitive:
    """PotentialGPU factory should accept type= and kwargs in any case."""

    def test_type_case_insensitive(self):
        """type='nfw', 'NFW', 'Nfw' should all work."""
        kw = dict(mass=1e12, scaleRadius=20.)
        p1 = PotentialGPU(type='nfw',    **kw)
        p2 = PotentialGPU(type='NFW',    **kw)
        p3 = PotentialGPU(type='Nfw',    **kw)
        xyz_cp = cp.asarray(_pts(10))
        ref = p1.potential(xyz_cp)
        assert float(cp.max(cp.abs(p2.potential(xyz_cp) - ref))) < 1e-10
        assert float(cp.max(cp.abs(p3.potential(xyz_cp) - ref))) < 1e-10

    def test_param_keys_uppercase(self):
        """NFW built with Mass= and ScaleRadius= (uppercase) matches agama.Potential."""
        pot_gpu = PotentialGPU(type='NFW', Mass=1e12, ScaleRadius=20.)
        pot_cpu = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.)
        xyz = _pts(100)
        phi_gpu = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz)))
        phi_cpu = pot_cpu.potential(xyz)
        assert _rel_phi(phi_gpu, phi_cpu) < 1e-10

    def test_v0_alias_for_loghalo(self):
        """Logarithmic potential: v0= (Agama alias) maps to velocity= (GPU param)."""
        # In _CANONICAL_PARAM: 'v0' → 'velocity'
        ini_txt = (
            "[Potential]\n"
            "type = Logarithmic\n"
            "v0 = 200\n"
            "scaleRadius = 0.1\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".ini", mode="w", delete=False) as f:
            f.write(ini_txt); tmp = f.name
        try:
            pot = PotentialGPU(file=tmp)
        finally:
            os.unlink(tmp)
        from nbody_streams.agama_helper._analytic_potentials import LogHaloPotentialGPU
        assert isinstance(pot, LogHaloPotentialGPU)
        xyz_cp = cp.asarray(_pts(50))
        assert cp.all(cp.isfinite(pot.potential(xyz_cp)))
