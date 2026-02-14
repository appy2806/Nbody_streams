"""Tests for nbody_streams.fast_sims — particle spray & restricted N-body.

Uses the MWPotential22 from data/potentials/ with:
    prog_mass      = 20_000  Msun
    prog_scaleradius = 0.01  kpc
    sat_posvel     = [-10.93, -3.36, -22.2, 70.4, 188.58, 95.84]  (AAU coords)
    time_total     = 3.0  Gyr
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import agama

    agama.setUnits(mass=1, length=1, velocity=1)
    AGAMA_AVAILABLE = True
except ImportError:
    AGAMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AGAMA_AVAILABLE, reason="Agama not installed",
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "nbody_streams" / "data" / "potentials"
MW_POT_FILE = DATA_DIR / "MWPotential22.ini"

PROG_MASS = 20_000          # Msun
PROG_SCALE = 0.01           # kpc
SAT_XV = np.array([-10.93, -3.36, -22.2, 70.4, 188.58, 95.84])
TIME_TOTAL = 3.0            # Gyr
TIME_END = 13.78            # Gyr  (particle spray default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pot_host():
    return agama.Potential(str(MW_POT_FILE))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Particle spray                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class TestCreateParticleSprayStream:

    def test_basic_run(self, pot_host):
        """Smoke test: spray stream runs and returns expected keys."""
        from nbody_streams.fast_sims import create_particle_spray_stream

        result = create_particle_spray_stream(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            scaleradius=PROG_SCALE,
            num_particles=200,
            time_total=TIME_TOTAL,
            time_end=TIME_END,
            save_rate=1,
            prog_pot_kind='Plummer',
        )
        assert isinstance(result, dict)
        assert 'times' in result
        assert 'prog_xv' in result
        assert 'part_xv' in result
        assert result['part_xv'].ndim >= 2

    def test_multi_snapshot(self, pot_host):
        """save_rate > 1 should give interpolated snapshots."""
        from nbody_streams.fast_sims import create_particle_spray_stream

        result = create_particle_spray_stream(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            scaleradius=PROG_SCALE,
            num_particles=100,
            time_total=TIME_TOTAL,
            time_end=TIME_END,
            save_rate=5,
            prog_pot_kind='Plummer',
        )
        assert result['prog_xv'].shape == (5, 6)
        assert len(result['times']) == 5

    def test_fardal_method(self, pot_host):
        """Check Fardal+2015 IC method runs."""
        from nbody_streams.fast_sims import (
            create_ic_particle_spray_fardal2015,
            create_particle_spray_stream,
        )

        result = create_particle_spray_stream(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            scaleradius=PROG_SCALE,
            num_particles=100,
            time_total=TIME_TOTAL,
            time_end=TIME_END,
            save_rate=1,
            prog_pot_kind='Plummer',
            create_ic_method=create_ic_particle_spray_fardal2015,
        )
        assert result['part_xv'].shape[-1] == 6

    def test_custom_time_stripping(self, pot_host):
        """Custom stripping times must match N = num_particles//2 + 1."""
        from nbody_streams.fast_sims import create_particle_spray_stream

        num_particles = 100
        N = num_particles // 2 + 1
        t_strip = np.linspace(TIME_END - TIME_TOTAL, TIME_END, N)

        result = create_particle_spray_stream(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            scaleradius=PROG_SCALE,
            num_particles=num_particles,
            time_total=TIME_TOTAL,
            time_end=TIME_END,
            time_stripping=t_strip,
            save_rate=1,
            prog_pot_kind='Plummer',
        )
        assert result['part_xv'].shape[0] == 2 * N - 2

    def test_bad_time_stripping_length_raises(self, pot_host):
        """Wrong-length time_stripping should raise ValueError."""
        from nbody_streams.fast_sims import create_particle_spray_stream

        with pytest.raises(ValueError, match="time_stripping"):
            create_particle_spray_stream(
                pot_host=pot_host,
                initmass=PROG_MASS,
                sat_cen_present=SAT_XV,
                scaleradius=PROG_SCALE,
                num_particles=100,
                time_total=TIME_TOTAL,
                time_end=TIME_END,
                time_stripping=np.linspace(0, 1, 10),  # wrong length
                prog_pot_kind='Plummer',
            )

    def test_bad_initmass_raises(self, pot_host):
        from nbody_streams.fast_sims import create_particle_spray_stream

        with pytest.raises(ValueError, match="initmass"):
            create_particle_spray_stream(
                pot_host=pot_host,
                initmass=-1,
                sat_cen_present=SAT_XV,
                scaleradius=PROG_SCALE,
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Restricted N-body                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class TestRunRestrictedNbody:

    def test_basic_run(self, pot_host):
        """Smoke test: restricted N-body returns expected keys."""
        from nbody_streams.fast_sims import run_restricted_nbody

        result = run_restricted_nbody(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            scaleradius=PROG_SCALE,
            num_particles=200,
            time_total=TIME_TOTAL,
            time_end=TIME_END,
            step_size=50,
            save_rate=5,
            trajsize_each_step=5,
            prog_pot_kind='Plummer',
        )
        assert isinstance(result, dict)
        assert 'times' in result
        assert 'prog_xv' in result
        assert 'part_xv' in result
        assert 'bound_mass' in result

    def test_xv_init_mode(self, pot_host):
        """When xv_init is provided, no rewinding should occur."""
        from nbody_streams.fast_sims import run_restricted_nbody

        rng = np.random.default_rng(42)
        xv_init = SAT_XV + rng.standard_normal((50, 6)) * np.array([0.01]*3 + [1.0]*3)

        result = run_restricted_nbody(
            pot_host=pot_host,
            initmass=PROG_MASS,
            sat_cen_present=SAT_XV,
            xv_init=xv_init,
            time_total=0.5,
            time_end=TIME_END,
            step_size=50,
            save_rate=3,
            trajsize_each_step=5,
        )
        assert isinstance(result, dict)
        assert 'part_xv' in result

    def test_bad_initmass_raises(self, pot_host):
        from nbody_streams.fast_sims import run_restricted_nbody

        with pytest.raises(ValueError, match="initmass"):
            run_restricted_nbody(
                pot_host=pot_host,
                initmass=-1,
                sat_cen_present=SAT_XV,
                scaleradius=PROG_SCALE,
            )

    def test_bad_xv_init_shape_raises(self, pot_host):
        from nbody_streams.fast_sims import run_restricted_nbody

        with pytest.raises(ValueError, match="shape"):
            run_restricted_nbody(
                pot_host=pot_host,
                initmass=PROG_MASS,
                sat_cen_present=SAT_XV,
                xv_init=np.ones((10, 3)),  # wrong shape
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IC generators (unit-level)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class TestICGenerators:

    @pytest.fixture()
    def orbit_and_tidal(self, pot_host):
        """Short orbit + Jacobi quantities for IC tests."""
        from nbody_streams.fast_sims.spray import _get_jacobi_rad_vel_mtx

        N = 20
        time_sat, orbit_sat = agama.orbit(
            ic=SAT_XV, potential=pot_host,
            time=-1.0, timestart=TIME_END, trajsize=N,
        )
        time_sat = time_sat[::-1]
        orbit_sat = orbit_sat[::-1]

        rj, vj, R = _get_jacobi_rad_vel_mtx(
            pot_host, orbit_sat, PROG_MASS, t=time_sat,
        )
        return orbit_sat, rj, vj, R

    def test_chen2025_shape(self, orbit_and_tidal):
        from nbody_streams.fast_sims import create_ic_particle_spray_chen2025

        orbit_sat, rj, vj, R = orbit_and_tidal
        ic = create_ic_particle_spray_chen2025(orbit_sat, PROG_MASS, rj, R)
        assert ic.shape == (2 * len(orbit_sat), 6)
        assert np.all(np.isfinite(ic))

    def test_fardal2015_shape(self, orbit_and_tidal):
        from nbody_streams.fast_sims import create_ic_particle_spray_fardal2015

        orbit_sat, rj, vj, R = orbit_and_tidal
        ic = create_ic_particle_spray_fardal2015(orbit_sat, rj, vj, R)
        assert ic.shape == (2 * len(orbit_sat), 6)
        assert np.all(np.isfinite(ic))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
