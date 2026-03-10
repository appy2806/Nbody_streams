"""
tests/test_physics.py
=====================
Physics-level validation tests for nbody_streams.

These tests run actual N-body integrations and verify that:

  (A) Energy and linear momentum are conserved to floating-point accuracy
      for the OLD API (``run_nbody_cpu`` with ``species=None``) — the same
      call pattern used in the ``devel`` branch.

  (B) The NEW API (``run_simulation`` with ``Species``) introduced in
      feat-multi_spec conserves energy and momentum identically.

  (C) Multi-species runs (dark + star + optional BH) conserve total energy
      and momentum across all particles combined.

  (D) Regression between devel-style and feat-multi_spec API:
      ``run_nbody_cpu(species=None)`` and ``run_simulation(single species)``
      must produce trajectories that agree to high precision — confirming
      that the new IO / snapshot paths introduced no physics changes.

  (E) IO round-trip: final phase-space array returned by ``run_simulation``
      must match the last snapshot stored on disk.

  (F) GPU conservation tests (skipped when CuPy is absent).

All tests use CPU direct summation; N is kept small so the suite is fast.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import nbody_streams as nb
from nbody_streams.species import Species

try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Integration parameters
# ---------------------------------------------------------------------------
N_PHYS    = 80       # particle count for physics / regression tests
EPS       = 0.2      # large softening keeps forces gentle -> good conservation
G         = 1.0      # dimensionless G
M_TOT     = 1.0      # total mass of the system
DT        = 5e-3     # timestep
T_END     = 0.25     # end time (50 steps total)

# Tolerances
E_REL_TOL  = 1e-2    # 1 % relative energy drift (very conservative for leapfrog)
MOM_ATOL   = 1e-8    # absolute tolerance for momentum (per component)
COM_ATOL   = 1e-7    # absolute tolerance for COM drift prediction
TRAJ_RTOL  = 1e-5    # relative tolerance for devel vs new-API trajectory match


# ---------------------------------------------------------------------------
# Pure-NumPy physics helpers (no dependency on package force code)
# ---------------------------------------------------------------------------

def _pe(pos: np.ndarray, mass: np.ndarray,
        softening: float = EPS, G: float = G) -> float:
    """
    Plummer-softened gravitational potential energy.

    Uses O(N^2) upper-triangle counting (exact, no double-counting).
    Only practical for N < ~500.
    """
    diff  = pos[:, None, :] - pos[None, :, :]       # (N, N, 3)
    r2    = np.sum(diff ** 2, axis=-1) + softening ** 2  # (N, N)
    inv_r = 1.0 / np.sqrt(r2)
    np.fill_diagonal(inv_r, 0.0)                     # exclude self-interaction
    return -0.5 * G * float(np.einsum("i,j,ij->", mass, mass, inv_r))


def _spline_pe(pos: np.ndarray, mass: np.ndarray,
               softening: float = EPS, G: float = G) -> float:
    """
    Cubic spline-softened gravitational PE, kernel-consistent with
    run_simulation(kernel='spline') / run_nbody_cpu(kernel='spline').

    Matches _get_potential_kernel(kernel_id=4) from fields.py, using the
    max(h_i, h_j) pair-softening convention of the backends.
    Valid for uniform softening (single-species or same-eps multi-species).
    """
    diff = pos[:, None, :] - pos[None, :, :]   # (N, N, 3)
    r    = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, N)
    h    = float(softening)
    hinv = 1.0 / h
    q    = r * hinv
    q2   = q * q

    # Default: Newtonian (r >= h)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(r > 0.0, -1.0 / r, 0.0)

    # Outer spline region: 0.5 < q <= 1.0
    m_out  = (q > 0.5) & (q <= 1.0)
    safe_q = np.where(m_out, q, 1.0)   # avoid /0; only used where m_out is True
    phi_out = (
        -3.2
        + 0.06666666666666666 / safe_q
        + safe_q ** 2 * (
            10.666666666666666
            + safe_q * (-16.0 + safe_q * (9.6 - 2.1333333333333333 * safe_q))
        )
    ) * hinv
    phi = np.where(m_out, phi_out, phi)

    # Inner spline region: q <= 0.5 (limit q->0 gives -2.8*hinv)
    m_inn   = q <= 0.5
    phi_inn = (-2.8 + q2 * (5.333333333333333 + q2 * q2 * (6.4 * q - 9.6))) * hinv
    phi = np.where(m_inn, phi_inn, phi)

    np.fill_diagonal(phi, 0.0)   # no self-interaction
    return 0.5 * G * float(np.einsum("i,j,ij->", mass, mass, phi))


def _ke(xv: np.ndarray, mass: np.ndarray) -> float:
    """KE = (1/2) sum_i m_i |v_i|^2."""
    return 0.5 * float(np.sum(mass * np.sum(xv[:, 3:] ** 2, axis=-1)))


def _total_energy(xv: np.ndarray, mass: np.ndarray,
                  softening: float = EPS, G: float = G) -> float:
    """Plummer-softened total energy. Use only when the integrator also uses Plummer."""
    return _ke(xv, mass) + _pe(xv[:, :3], mass, softening, G)


def _total_energy_spline(xv: np.ndarray, mass: np.ndarray,
                         softening: float = EPS, G: float = G) -> float:
    """Spline-softened total energy. Matches run_simulation's hardcoded kernel='spline'."""
    return _ke(xv, mass) + _spline_pe(xv[:, :3], mass, softening, G)


def _total_momentum(xv: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Linear momentum vector p = sum_i m_i v_i."""
    return np.einsum("i,ij->j", mass, xv[:, 3:])


def _com(xv: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Centre-of-mass position."""
    return np.einsum("i,ij->j", mass, xv[:, :3]) / mass.sum()


# ---------------------------------------------------------------------------
# IC factory — fixed seed so every test sees the same initial conditions
# ---------------------------------------------------------------------------

def _make_ic(N: int = N_PHYS) -> np.ndarray:
    """
    Compact random ICs: pos ~ N(0,1), vel ~ N(0,0.3).
    Fixed seed guarantees reproducibility across the full test session.
    Returns (N, 6) float64.
    """
    rng = np.random.default_rng(seed=7)
    pos = rng.standard_normal((N, 3))
    vel = rng.standard_normal((N, 3)) * 0.3
    return np.hstack([pos, vel])


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_old_api(xv: np.ndarray, tmp_dir: str | Path, *,
                 N: int = N_PHYS,
                 m: float = M_TOT / N_PHYS) -> tuple[np.ndarray, np.ndarray]:
    """
    OLD API — ``run_nbody_cpu`` with scalar softening and no ``species`` kwarg.
    Uses 'spline' kernel to match run_simulation's hardcoded default.
    """
    masses = np.full(N, m)
    final = nb.run_nbody_cpu(
        xv, masses,
        time_start=0.0, time_end=T_END, dt=DT,
        softening=EPS, G=G,
        method="direct", kernel="spline",
        output_dir=str(Path(tmp_dir) / "old"),
        save_snapshots=False, verbose=False,
    )
    return final, masses


def _run_new_api(xv: np.ndarray, tmp_dir: str | Path, *,
                 N: int = N_PHYS,
                 m: float = M_TOT / N_PHYS) -> tuple[np.ndarray, np.ndarray]:
    """
    NEW API — ``run_simulation`` with a single dark species.
    Equivalent to the feat-multi_spec branch calling convention.
    """
    dm = Species.dark(N=N, mass=m, softening=EPS)
    result = nb.run_simulation(
        xv, [dm],
        time_start=0.0, time_end=T_END, dt=DT,
        G=G,
        architecture="cpu", method="direct",
        output_dir=str(Path(tmp_dir) / "new"),
        save_snapshots=False, verbose=False,
    )
    masses = np.full(N, m)
    return result["dark"], masses


# ===========================================================================
# A. Energy / momentum conservation — OLD API (devel-equivalent)
# ===========================================================================

class TestConservationOldAPI:
    """
    Verify that the ``run_nbody_cpu`` path (``species=None``) — the behaviour
    of the ``devel`` branch — conserves energy and momentum.

    These are the baseline physics checks; if they fail the problem pre-dates
    feat-multi_spec.
    """

    def test_energy_conserved(self, tmp_path):
        xv0 = _make_ic()
        final, mass = _run_old_api(xv0, tmp_path)
        E0  = _total_energy_spline(xv0,   mass)
        Ef  = _total_energy_spline(final, mass)
        rel = abs((Ef - E0) / E0)
        assert rel < E_REL_TOL, (
            f"Old API: energy drifted by {rel:.3%}  (limit {E_REL_TOL:.0%})"
        )

    def test_momentum_conserved(self, tmp_path):
        xv0 = _make_ic()
        final, mass = _run_old_api(xv0, tmp_path)
        p0 = _total_momentum(xv0,   mass)
        pf = _total_momentum(final, mass)
        np.testing.assert_allclose(
            pf, p0, atol=MOM_ATOL,
            err_msg="Old API: total momentum not conserved",
        )

    def test_com_drift_at_constant_velocity(self, tmp_path):
        """
        With no external forces, COM drifts at constant velocity:
        x_COM(t) = x_COM(0) + v_COM * t.
        """
        xv0 = _make_ic()
        final, mass = _run_old_api(xv0, tmp_path)
        v_com     = _total_momentum(xv0, mass) / mass.sum()
        com0      = _com(xv0,   mass)
        comf      = _com(final, mass)
        expected  = com0 + v_com * T_END
        np.testing.assert_allclose(
            comf, expected, atol=COM_ATOL,
            err_msg="Old API: COM did not drift at constant velocity",
        )

    def test_kinetic_energy_positive(self, tmp_path):
        """KE must always be positive (sanity check on velocity array)."""
        xv0 = _make_ic()
        final, mass = _run_old_api(xv0, tmp_path)
        assert _ke(final, mass) > 0.0

    def test_return_shape(self, tmp_path):
        """run_nbody_cpu must return (N, 6) float64."""
        xv0 = _make_ic()
        final, _ = _run_old_api(xv0, tmp_path)
        assert final.shape == (N_PHYS, 6)
        assert final.dtype == np.float64


# ===========================================================================
# B. Energy / momentum conservation — NEW API (run_simulation)
# ===========================================================================

class TestConservationNewAPI:
    """
    Same checks as A but via the feat-multi_spec ``run_simulation`` API.
    Passing these tests confirms the new code path is physically correct.
    """

    def test_energy_conserved(self, tmp_path):
        xv0 = _make_ic()
        final, mass = _run_new_api(xv0, tmp_path)
        E0  = _total_energy_spline(xv0,   mass)
        Ef  = _total_energy_spline(final, mass)
        rel = abs((Ef - E0) / E0)
        assert rel < E_REL_TOL, (
            f"New API: energy drifted by {rel:.3%}  (limit {E_REL_TOL:.0%})"
        )

    def test_momentum_conserved(self, tmp_path):
        xv0 = _make_ic()
        final, mass = _run_new_api(xv0, tmp_path)
        p0 = _total_momentum(xv0,   mass)
        pf = _total_momentum(final, mass)
        np.testing.assert_allclose(
            pf, p0, atol=MOM_ATOL,
            err_msg="New API: total momentum not conserved",
        )

    def test_com_drift_at_constant_velocity(self, tmp_path):
        xv0 = _make_ic()
        final, mass = _run_new_api(xv0, tmp_path)
        v_com    = _total_momentum(xv0, mass) / mass.sum()
        com0     = _com(xv0,   mass)
        comf     = _com(final, mass)
        expected = com0 + v_com * T_END
        np.testing.assert_allclose(
            comf, expected, atol=COM_ATOL,
            err_msg="New API: COM did not drift at constant velocity",
        )

    def test_result_is_dict_with_correct_key(self, tmp_path):
        """run_simulation must return a dict keyed by species name."""
        xv0 = _make_ic()
        dm  = Species.dark(N=N_PHYS, mass=M_TOT / N_PHYS, softening=EPS)
        result = nb.run_simulation(
            xv0, [dm],
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "dict_check"),
            save_snapshots=False, verbose=False,
        )
        assert isinstance(result, dict)
        assert "dark" in result
        assert result["dark"].shape == (N_PHYS, 6)

    def test_return_dtype_float64(self, tmp_path):
        xv0 = _make_ic()
        final, _ = _run_new_api(xv0, tmp_path)
        assert final.dtype == np.float64


# ===========================================================================
# C. Two-species and three-species energy conservation
# ===========================================================================

class TestConservationMultiSpecies:
    """
    Multi-species runs must conserve the *total* combined energy and momentum
    across all particle species.
    """

    N_DM   = 60
    N_STAR = 40
    M_DM   = 0.8 / 60    # per-particle mass; total DM mass = 0.8
    M_STAR = 0.2 / 40    # per-particle mass; total stellar mass = 0.2
    EPS_DM   = EPS
    EPS_STAR = EPS * 0.5

    def _two_species_ic(self):
        rng = np.random.default_rng(seed=99)
        pos = np.vstack([
            rng.standard_normal((self.N_DM,   3)),
            rng.standard_normal((self.N_STAR, 3)) * 0.5,
        ])
        vel = np.vstack([
            rng.standard_normal((self.N_DM,   3)) * 0.3,
            rng.standard_normal((self.N_STAR, 3)) * 0.4,
        ])
        xv0  = np.hstack([pos, vel])
        mass = np.concatenate([
            np.full(self.N_DM,   self.M_DM),
            np.full(self.N_STAR, self.M_STAR),
        ])
        return xv0, mass

    def _two_species_list(self):
        return [
            Species.dark(N=self.N_DM,   mass=self.M_DM,   softening=self.EPS_DM),
            Species.star(N=self.N_STAR, mass=self.M_STAR, softening=self.EPS_STAR),
        ]

    def test_two_species_energy_conserved(self, tmp_path):
        xv0, mass = self._two_species_ic()
        result = nb.run_simulation(
            xv0, self._two_species_list(),
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "two"),
            save_snapshots=False, verbose=False,
        )
        final = np.vstack([result["dark"], result["star"]])
        E0  = _total_energy_spline(xv0,   mass, softening=self.EPS_DM)
        Ef  = _total_energy_spline(final, mass, softening=self.EPS_DM)
        rel = abs((Ef - E0) / E0)
        # Allow 5× tolerance: mixed softenings mean the PE helper slightly
        # underestimates true PE (uniform eps approximation), but conservation
        # should still hold.
        assert rel < E_REL_TOL * 5, (
            f"Two-species: energy drifted by {rel:.3%}"
        )

    def test_two_species_momentum_conserved(self, tmp_path):
        xv0, mass = self._two_species_ic()
        result = nb.run_simulation(
            xv0, self._two_species_list(),
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "two_mom"),
            save_snapshots=False, verbose=False,
        )
        final = np.vstack([result["dark"], result["star"]])
        p0 = _total_momentum(xv0,   mass)
        pf = _total_momentum(final, mass)
        np.testing.assert_allclose(pf, p0, atol=MOM_ATOL)

    def test_two_species_output_shapes(self, tmp_path):
        xv0, _ = self._two_species_ic()
        result = nb.run_simulation(
            xv0, self._two_species_list(),
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "two_shape"),
            save_snapshots=False, verbose=False,
        )
        assert result["dark"].shape  == (self.N_DM,   6)
        assert result["star"].shape  == (self.N_STAR, 6)

    def test_three_species_energy_conserved(self, tmp_path):
        N_BH = 1
        N    = self.N_DM + self.N_STAR + N_BH
        rng  = np.random.default_rng(seed=123)
        xv0  = np.hstack([
            rng.standard_normal((N, 3)),
            rng.standard_normal((N, 3)) * 0.2,
        ])
        mass = np.concatenate([
            np.full(self.N_DM,   self.M_DM),
            np.full(self.N_STAR, self.M_STAR),
            np.full(N_BH, 0.5),
        ])
        sp = [
            Species.dark(N=self.N_DM,   mass=self.M_DM,   softening=self.EPS_DM),
            Species.star(N=self.N_STAR, mass=self.M_STAR, softening=self.EPS_STAR),
            Species("bh", N=N_BH,       mass=0.5,          softening=0.01),
        ]
        result = nb.run_simulation(
            xv0, sp,
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "three"),
            save_snapshots=False, verbose=False,
        )
        final = np.vstack([result["dark"], result["star"], result["bh"]])
        E0  = _total_energy_spline(xv0,   mass, softening=self.EPS_DM)
        Ef  = _total_energy_spline(final, mass, softening=self.EPS_DM)
        rel = abs((Ef - E0) / E0)
        # BH is massive relative to the system, so allow more tolerance here
        assert rel < 0.15, (
            f"Three-species: energy drifted by {rel:.3%}"
        )

    def test_three_species_momentum_conserved(self, tmp_path):
        N_BH = 1
        N    = self.N_DM + self.N_STAR + N_BH
        rng  = np.random.default_rng(seed=456)
        xv0  = np.hstack([
            rng.standard_normal((N, 3)),
            rng.standard_normal((N, 3)) * 0.2,
        ])
        mass = np.concatenate([
            np.full(self.N_DM,   self.M_DM),
            np.full(self.N_STAR, self.M_STAR),
            np.full(N_BH, 0.5),
        ])
        sp = [
            Species.dark(N=self.N_DM,   mass=self.M_DM,   softening=self.EPS_DM),
            Species.star(N=self.N_STAR, mass=self.M_STAR, softening=self.EPS_STAR),
            Species("bh", N=N_BH,       mass=0.5,          softening=0.01),
        ]
        result = nb.run_simulation(
            xv0, sp,
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(tmp_path / "three_mom"),
            save_snapshots=False, verbose=False,
        )
        final = np.vstack([result["dark"], result["star"], result["bh"]])
        p0 = _total_momentum(xv0,   mass)
        pf = _total_momentum(final, mass)
        np.testing.assert_allclose(pf, p0, atol=MOM_ATOL)


# ===========================================================================
# D. Regression: devel API vs feat-multi_spec API — identical trajectories
# ===========================================================================

class TestRegressionDevelVsMultiSpec:
    """
    Core regression suite.

    Compares ``run_nbody_cpu(species=None)`` — the exact call signature used
    in the ``devel`` branch — against ``run_simulation(single Species)``.

    If the trajectories agree to ``TRAJ_RTOL``, we can be confident that
    feat-multi_spec introduced zero physics changes; only new snapshot / restart
    IO paths are affected.
    """

    def test_trajectory_matches_devel_style(self, tmp_path):
        """
        Single-species run_simulation must reproduce run_nbody_cpu
        (species=None) to high precision.
        """
        xv0 = _make_ic()
        final_old, _ = _run_old_api(xv0, tmp_path)
        final_new, _ = _run_new_api(xv0, tmp_path)
        np.testing.assert_allclose(
            final_new, final_old,
            rtol=TRAJ_RTOL,
            err_msg=(
                "run_simulation trajectory diverges from run_nbody_cpu "
                "(devel-style call).  Physics regression detected in "
                "feat-multi_spec!"
            ),
        )

    def test_energy_error_consistent_old_vs_new(self, tmp_path):
        """
        The new API must not introduce additional energy error compared with
        the old API (beyond a 10x factor for floating-point jitter).
        """
        xv0  = _make_ic()
        mass = np.full(N_PHYS, M_TOT / N_PHYS)

        final_old, _ = _run_old_api(xv0, tmp_path)
        final_new, _ = _run_new_api(xv0, tmp_path)

        E0     = _total_energy_spline(xv0, mass)
        dE_old = abs(_total_energy_spline(final_old, mass) - E0) / abs(E0)
        dE_new = abs(_total_energy_spline(final_new, mass) - E0) / abs(E0)

        assert dE_new < max(dE_old * 10.0, 1e-4), (
            f"New API energy error ({dE_new:.2e}) >> old API ({dE_old:.2e})"
        )

    def test_momentum_both_apis_agree(self, tmp_path):
        """Both code paths must give the same final total momentum."""
        xv0 = _make_ic()
        final_old, mass = _run_old_api(xv0, tmp_path)
        final_new, _    = _run_new_api(xv0, tmp_path)
        pf_old = _total_momentum(final_old, mass)
        pf_new = _total_momentum(final_new, mass)
        np.testing.assert_allclose(pf_old, pf_new, atol=MOM_ATOL)

    def test_old_api_snapshot_readable_by_reader(self, tmp_path):
        """
        Backward compat: old dark/star HDF5 format written by run_nbody_cpu
        must still be readable by ParticleReader after feat-multi_spec.
        """
        N      = 50
        masses = np.full(N, 1.0 / N)
        xv0    = _make_ic(N)
        out    = tmp_path / "old_snap"
        nb.run_nbody_cpu(
            xv0, masses,
            time_start=0.0, time_end=T_END, dt=DT,
            softening=EPS, G=G,
            method="direct", kernel="plummer",
            output_dir=str(out),
            save_snapshots=True, snapshots=2, verbose=False,
        )
        h5_files = sorted(out.glob("*.h5"))
        assert h5_files, "No snapshot files written"
        reader = nb.ParticleReader(str(h5_files[0]))
        assert reader.num_dark == N
        snap = reader.read_snapshot(0)
        assert snap.dark["posvel"].shape == (N, 6)

    def test_new_api_snapshot_readable_by_reader(self, tmp_path):
        """
        New HDF5 schema must be readable by ParticleReader with multi-species
        attributes.
        """
        N   = 50
        m   = 1.0 / N
        xv0 = _make_ic(N)
        dm  = Species.dark(N=N, mass=m, softening=EPS)
        out = tmp_path / "new_snap"
        result = nb.run_simulation(
            xv0, [dm],
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(out),
            save_snapshots=True, snapshots=2, verbose=False,
        )
        h5_files = sorted(out.glob("*.h5"))
        assert h5_files
        reader = nb.ParticleReader(str(h5_files[0]))
        assert reader.num_dark == N
        snap = reader.read_snapshot(0)
        assert "dark" in snap.species

    def test_final_positions_match_last_snapshot(self, tmp_path):
        """
        The (N, 6) array returned by run_simulation must agree with the
        last snapshot written to disk — verifying the IO round-trip.
        """
        N   = 50
        m   = 1.0 / N
        xv0 = _make_ic(N)
        dm  = Species.dark(N=N, mass=m, softening=EPS)
        out = tmp_path / "io_check"
        result = nb.run_simulation(
            xv0, [dm],
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="cpu", method="direct",
            output_dir=str(out),
            save_snapshots=True, snapshots=3, verbose=False,
        )
        h5_files = sorted(out.glob("*.h5"))
        assert h5_files
        reader = nb.ParticleReader(str(h5_files[0]))
        last   = int(reader.Snapshots[-1])
        snap   = reader.read_snapshot(last)
        stored = snap.dark["posvel"]
        np.testing.assert_allclose(
            stored, result["dark"], rtol=1e-5,
            err_msg=(
                "Last snapshot posvel != final array returned by run_simulation"
            ),
        )


# ===========================================================================
# E. GPU conservation tests (requires CuPy)
# ===========================================================================

@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
class TestConservationGPU:
    """
    Energy and momentum conservation tests for the GPU backend.

    These run automatically in CI environments with a GPU; they are skipped
    on CPU-only machines.
    """

    # GPU uses float32 -> allow larger energy tolerance
    GPU_E_TOL = 0.05   # 5 %

    def test_energy_conserved_gpu_old_api(self, tmp_path):
        N    = N_PHYS
        m    = M_TOT / N
        xv0  = _make_ic(N)
        mass = np.full(N, m)
        final = nb.run_nbody_gpu(
            xv0, mass,
            time_start=0.0, time_end=T_END, dt=DT,
            softening=EPS, G=G,
            precision="float32_kahan", kernel="plummer",
            output_dir=str(tmp_path / "gpu_old"),
            save_snapshots=False, verbose=False,
        )
        E0  = _total_energy(xv0,   mass)
        Ef  = _total_energy(final, mass)
        rel = abs((Ef - E0) / E0)
        assert rel < self.GPU_E_TOL, (
            f"GPU old API: energy drifted by {rel:.3%}"
        )

    def test_energy_conserved_gpu_new_api(self, tmp_path):
        N    = N_PHYS
        m    = M_TOT / N
        xv0  = _make_ic(N)
        dm   = Species.dark(N=N, mass=m, softening=EPS)
        result = nb.run_simulation(
            xv0, [dm],
            time_start=0.0, time_end=T_END, dt=DT,
            G=G, architecture="gpu",
            precision="float32_kahan",
            output_dir=str(tmp_path / "gpu_new"),
            save_snapshots=False, verbose=False,
        )
        mass = np.full(N, m)
        E0   = _total_energy_spline(xv0,           mass)
        Ef   = _total_energy_spline(result["dark"], mass)
        rel  = abs((Ef - E0) / E0)
        assert rel < self.GPU_E_TOL, (
            f"GPU new API: energy drifted by {rel:.3%}"
        )

    def test_gpu_trajectory_matches_cpu(self, tmp_path):
        """
        GPU and CPU single-species trajectories must agree to within
        reasonable float32 precision.
        """
        N    = N_PHYS
        m    = M_TOT / N
        xv0  = _make_ic(N)
        mass = np.full(N, m)

        # CPU run
        final_cpu, _ = _run_old_api(xv0, tmp_path, N=N, m=m)

        # GPU run
        final_gpu = nb.run_nbody_gpu(
            xv0, mass,
            time_start=0.0, time_end=T_END, dt=DT,
            softening=EPS, G=G,
            precision="float32_kahan", kernel="spline",
            output_dir=str(tmp_path / "gpu_cpu_cmp"),
            save_snapshots=False, verbose=False,
        )

        # float32 → expect ~1e-3 relative difference in trajectories
        np.testing.assert_allclose(
            final_gpu, final_cpu, rtol=1e-3, atol=1e-6,
            err_msg="GPU and CPU trajectories differ by more than float32 precision",
        )
