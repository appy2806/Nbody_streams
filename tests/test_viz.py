"""Tests for nbody_streams.viz — visualization functions.

All tests use the Agg backend so no display is needed.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from nbody_streams.viz import plot_density, plot_stream_evolution


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture()
def random_particles():
    rng = np.random.default_rng(42)
    pos = rng.standard_normal((500, 3)) * 50
    mass = np.ones(500)
    return pos, mass


@pytest.fixture()
def orbit_data():
    """Synthetic circular-ish orbit for evolution plots."""
    rng = np.random.default_rng(99)
    T = 80
    times = np.linspace(0, 4, T)
    prog_xv = np.zeros((T, 6))
    prog_xv[:, 0] = 20 * np.cos(times * 2)
    prog_xv[:, 1] = 20 * np.sin(times * 2)
    prog_xv[:, 3] = -40 * np.sin(times * 2)
    prog_xv[:, 4] = 40 * np.cos(times * 2)

    # Particles at final snapshot
    part_pos = rng.standard_normal((30, 3)) * 2 + prog_xv[-1, :3]
    return prog_xv, times, part_pos


# =====================================================================
# plot_density
# =====================================================================
class TestPlotDensity:

    def test_basic(self, random_particles):
        pos, mass = random_particles
        result = plot_density(pos=pos, mass=mass, grid_len=80, no_bins=32)
        assert result is None

    def test_return_dens(self, random_particles):
        pos, mass = random_particles
        im, dens = plot_density(
            pos=pos, mass=mass, grid_len=80, no_bins=32,
            return_dens=True, gauss_convol=False,
        )
        assert dens.shape == (32, 32)
        assert hasattr(im, "get_clim")

    def test_gauss_convol(self, random_particles):
        pos, mass = random_particles
        im, dens = plot_density(
            pos=pos, mass=mass, grid_len=80, no_bins=32,
            return_dens=True, gauss_convol=True,
        )
        assert dens.shape == (32, 32)
        assert hasattr(im, "get_clim")

    def test_volume_density(self, random_particles):
        pos, mass = random_particles
        _, dens = plot_density(
            pos=pos, mass=mass, grid_len=80, no_bins=32,
            return_dens=True, slice_width=20, density_kind="volume",
        )
        assert dens.shape == (32, 32)

    def test_custom_axes(self, random_particles):
        pos, mass = random_particles
        fig, ax = plt.subplots()
        result = plot_density(pos=pos, mass=mass, grid_len=80, no_bins=32, ax=ax)
        assert result is None

    def test_colorbar_true(self, random_particles):
        pos, mass = random_particles
        plot_density(
            pos=pos, mass=mass, grid_len=80, no_bins=32, colorbar_ax=True,
        )

    def test_projection_axes(self, random_particles):
        pos, mass = random_particles
        # XY projection
        plot_density(pos=pos, mass=mass, grid_len=80, no_bins=32, xval="X", yval="Y")

    def test_scalar_mass(self, random_particles):
        pos, _ = random_particles
        plot_density(pos=pos, mass=1.0, grid_len=80, no_bins=32)

    def test_no_pos_raises(self):
        with pytest.raises(ValueError, match="pos"):
            plot_density()

    def test_bad_pos_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            plot_density(pos=np.ones((10, 2)))


# =====================================================================
# plot_stream_evolution
# =====================================================================
class TestPlotStreamEvolution:

    def test_basic_arrays(self, orbit_data):
        prog_xv, times, part_pos = orbit_data
        fig, ax = plot_stream_evolution(prog_xv, times, part_xv=part_pos)
        assert isinstance(fig, Figure)
        assert len(ax) == 3

    def test_dict_input(self, orbit_data):
        prog_xv, times, part_pos = orbit_data
        out = {"prog_xv": prog_xv, "times": times, "part_xv": part_pos}
        fig, ax = plot_stream_evolution(out)
        assert isinstance(fig, Figure)

    def test_with_bound_mass(self, orbit_data):
        prog_xv, times, part_pos = orbit_data
        bound_mass = np.linspace(1.0, 0.3, len(times))
        fig, ax = plot_stream_evolution(
            prog_xv, times, part_xv=part_pos, bound_mass=bound_mass,
        )
        assert isinstance(fig, Figure)

    def test_no_particles(self, orbit_data):
        """Should work even without particle data."""
        prog_xv, times, _ = orbit_data
        fig, ax = plot_stream_evolution(prog_xv, times)
        assert isinstance(fig, Figure)

    def test_3d_mode(self, orbit_data):
        prog_xv, times, part_pos = orbit_data
        fig, ax = plot_stream_evolution(
            prog_xv, times, part_xv=part_pos, three_d_plot=True,
        )
        assert isinstance(fig, Figure)

    def test_multi_snapshot_part_xv(self, orbit_data):
        """part_xv with shape (N, T, 6)."""
        prog_xv, times, _ = orbit_data
        rng = np.random.default_rng(0)
        T = len(times)
        part_xv = rng.standard_normal((20, T, 6)) * 5
        fig, ax = plot_stream_evolution(prog_xv, times, part_xv=part_xv)
        assert isinstance(fig, Figure)

    def test_lmc_traj(self, orbit_data):
        prog_xv, times, part_pos = orbit_data
        T = len(times)
        lmc = np.column_stack([times, 50 * np.ones(T), np.zeros(T), np.zeros(T)])
        fig, ax = plot_stream_evolution(
            prog_xv, times, part_xv=part_pos, LMC_traj=lmc,
        )
        assert isinstance(fig, Figure)

    def test_interactive_without_3d_raises(self, orbit_data):
        prog_xv, times, _ = orbit_data
        with pytest.raises(ValueError, match="three_d_plot"):
            plot_stream_evolution(prog_xv, times, interactive=True)


# =====================================================================
# plot_mollweide (healpy-dependent, skip if unavailable)
# =====================================================================
class TestPlotMollweide:

    @pytest.fixture(autouse=True)
    def _check_healpy(self):
        pytest.importorskip("healpy")

    def test_basic(self):
        from nbody_streams.viz import plot_mollweide

        rng = np.random.default_rng(7)
        pos = rng.standard_normal((2000, 3)) * 100
        result = plot_mollweide(pos, initial_nside=8, return_map=True)
        assert isinstance(result, np.ndarray)

    def test_no_return(self):
        from nbody_streams.viz import plot_mollweide

        rng = np.random.default_rng(7)
        pos = rng.standard_normal((500, 3)) * 100
        result = plot_mollweide(pos, initial_nside=8, return_map=False)
        assert result is None


# =====================================================================
# Standalone runner
# =====================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
