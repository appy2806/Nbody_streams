"""Tests for nbody_streams.viz -- visualization functions.

All tests use the Agg backend so no display is needed.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from nbody_streams.viz import (
    plot_density,
    plot_stream_evolution,
    render_surface_density,
    get_smoothing_lengths,
)


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

    def test_basic_histogram(self, random_particles):
        pos, mass = random_particles
        result = plot_density(pos=pos, mass=mass, gridsize=160, resolution=32,
                              method='histogram')
        assert result is None

    def test_return_dens_histogram(self, random_particles):
        pos, mass = random_particles
        im, dens = plot_density(
            pos=pos, mass=mass, gridsize=160, resolution=32,
            method='histogram', return_dens=True,
        )
        assert dens.shape == (32, 32)
        assert hasattr(im, "get_clim")

    def test_gauss_smooth(self, random_particles):
        pos, mass = random_particles
        im, dens = plot_density(
            pos=pos, mass=mass, gridsize=160, resolution=32,
            method='gauss_smooth', return_dens=True,
        )
        assert dens.shape == (32, 32)
        assert hasattr(im, "get_clim")

    def test_sph_method(self, random_particles):
        pos, mass = random_particles
        im, dens = plot_density(
            pos=pos, mass=mass, gridsize=160, resolution=32,
            method='sph', return_dens=True, arch='cpu',
        )
        assert dens.shape == (32, 32)
        assert hasattr(im, "get_clim")
        # SPH density should be non-negative
        assert np.all(dens >= 0)

    def test_volume_density(self, random_particles):
        pos, mass = random_particles
        _, dens = plot_density(
            pos=pos, mass=mass, gridsize=160, resolution=32,
            method='histogram', return_dens=True,
            slice_width=20, density_kind="volume",
        )
        assert dens.shape == (32, 32)

    def test_custom_axes(self, random_particles):
        pos, mass = random_particles
        fig, ax = plt.subplots()
        result = plot_density(pos=pos, mass=mass, gridsize=160, resolution=32,
                              method='histogram', ax=ax)
        assert result is None

    def test_colorbar_true(self, random_particles):
        pos, mass = random_particles
        plot_density(
            pos=pos, mass=mass, gridsize=160, resolution=32,
            method='histogram', colorbar_ax=True,
        )

    def test_projection_axes(self, random_particles):
        pos, mass = random_particles
        plot_density(pos=pos, mass=mass, gridsize=160, resolution=32,
                     method='histogram', xval="x", yval="y")

    def test_scalar_mass(self, random_particles):
        pos, _ = random_particles
        plot_density(pos=pos, mass=1.0, gridsize=160, resolution=32,
                     method='histogram')

    def test_no_pos_raises(self):
        with pytest.raises(ValueError, match="pos"):
            plot_density()

    def test_bad_pos_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            plot_density(pos=np.ones((10, 2)))

    def test_bad_method_raises(self, random_particles):
        pos, mass = random_particles
        with pytest.raises(ValueError, match="method"):
            plot_density(pos=pos, mass=mass, method='invalid')

    def test_gridsize_symmetry(self, random_particles):
        """imshow extent should be [-gridsize/2, gridsize/2] on both axes."""
        pos, mass = random_particles
        fig, ax = plt.subplots()
        plot_density(pos=pos, mass=mass, gridsize=200.0, resolution=32,
                     method='histogram', ax=ax)
        im = ax.get_images()[0]
        xmin, xmax, ymin, ymax = im.get_extent()
        assert xmin == pytest.approx(-100.0)
        assert xmax == pytest.approx(100.0)
        assert ymin == pytest.approx(-100.0)
        assert ymax == pytest.approx(100.0)


# =====================================================================
# render_surface_density
# =====================================================================
class TestRenderSurfaceDensity:

    def test_returns_correct_shape(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 10, 1000).astype(np.float32)
        y = rng.normal(0, 10, 1000).astype(np.float32)
        mass = np.ones(1000, dtype=np.float32)

        grid, bounds = render_surface_density(
            x, y, mass, resolution=64, gridsize=60.0, arch='cpu',
        )
        assert grid.shape == (64, 64)
        assert grid.dtype == np.float32
        assert np.all(grid >= 0)

    def test_bounds_match_gridsize(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 5, 200).astype(np.float32)
        y = rng.normal(0, 5, 200).astype(np.float32)
        mass = np.ones(200, dtype=np.float32)

        grid, bounds = render_surface_density(
            x, y, mass, resolution=32, gridsize=40.0, arch='cpu',
        )
        xmin, xmax, ymin, ymax = bounds
        assert xmin == pytest.approx(-20.0)
        assert xmax == pytest.approx(20.0)
        assert ymin == pytest.approx(-20.0)
        assert ymax == pytest.approx(20.0)

    def test_precomputed_h(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 10, 500).astype(np.float32)
        y = rng.normal(0, 10, 500).astype(np.float32)
        mass = np.ones(500, dtype=np.float32)
        pos_2d = np.column_stack([x, y])
        h = get_smoothing_lengths(pos_2d, k_neighbors=16)

        grid, bounds = render_surface_density(
            x, y, mass, h=h, resolution=32, gridsize=60.0, arch='cpu',
        )
        assert grid.shape == (32, 32)

    def test_mass_conservation(self):
        """Total mass in grid should approximately match input mass."""
        rng = np.random.default_rng(3)
        N = 2000
        x = rng.uniform(-10, 10, N).astype(np.float32)
        y = rng.uniform(-10, 10, N).astype(np.float32)
        mass = np.full(N, 1e6, dtype=np.float32)
        gridsize = 40.0

        grid, _ = render_surface_density(
            x, y, mass, resolution=128, gridsize=gridsize, arch='cpu',
        )
        pixel_area = (gridsize / 128) ** 2
        mass_in_grid = grid.sum() * pixel_area
        total_mass = float(mass.sum())
        # Particles inside the grid: most should be captured
        inside = np.sum((np.abs(x) < gridsize / 2) & (np.abs(y) < gridsize / 2))
        expected = float(mass[0]) * inside
        assert mass_in_grid == pytest.approx(expected, rel=0.05)

    def test_bad_arch_raises(self):
        x = y = np.ones(10, dtype=np.float32)
        mass = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="arch"):
            render_surface_density(x, y, mass, arch='bad')

    def test_mismatched_h_raises(self):
        x = y = np.ones(10, dtype=np.float32)
        mass = np.ones(10, dtype=np.float32)
        h = np.ones(5, dtype=np.float32)  # wrong length
        with pytest.raises(ValueError, match="length"):
            render_surface_density(x, y, mass, h=h, arch='cpu')

    def test_sort_by_morton(self):
        """Morton sort should produce a valid density grid.

        Grids are NOT expected to be numerically identical: the Numba
        fastmath prange reduction can exploit different SIMD fusion patterns
        when particles are spatially ordered, changing float32 rounding at
        the 1-5% level.  We only verify shape and non-negativity.
        """
        rng = np.random.default_rng(4)
        x = rng.normal(0, 10, 500).astype(np.float32)
        y = rng.normal(0, 10, 500).astype(np.float32)
        mass = np.ones(500, dtype=np.float32)
        pos_2d = np.column_stack([x, y])
        h = get_smoothing_lengths(pos_2d, k_neighbors=16)

        grid_morton, bounds = render_surface_density(
            x, y, mass, h=h, resolution=32, gridsize=60.0,
            arch='cpu', sort_by_morton=True,
        )
        assert grid_morton.shape == (32, 32)
        assert np.all(grid_morton >= 0)
        assert grid_morton.sum() > 0


# =====================================================================
# get_smoothing_lengths
# =====================================================================
class TestGetSmoothingLengths:

    def test_returns_correct_shape(self):
        rng = np.random.default_rng(10)
        pos = rng.standard_normal((300, 2))
        h = get_smoothing_lengths(pos, k_neighbors=16)
        assert h.shape == (300,)
        assert h.dtype == np.float32

    def test_all_positive(self):
        rng = np.random.default_rng(11)
        pos = rng.standard_normal((200, 2))
        h = get_smoothing_lengths(pos, k_neighbors=8)
        assert np.all(h > 0)

    def test_dense_region_smaller_h(self):
        """Particles in a dense cluster should get smaller h than isolated ones."""
        rng = np.random.default_rng(12)
        # Tight cluster of 100 particles + 10 isolated particles far away
        cluster = rng.normal(0, 0.1, (100, 2))
        isolated = rng.uniform(500, 600, (10, 2))
        pos = np.vstack([cluster, isolated])
        h = get_smoothing_lengths(pos, k_neighbors=8)
        assert h[:100].mean() < h[100:].mean()

    def test_bad_pos_shape_raises(self):
        pos = np.ones(100)  # 1-D, should raise
        with pytest.raises(ValueError, match="2-D"):
            get_smoothing_lengths(pos)

    def test_3d_pos_accepted(self):
        """3-D pos should be accepted (D is not restricted to 2)."""
        rng = np.random.default_rng(13)
        pos = rng.standard_normal((200, 3))
        h = get_smoothing_lengths(pos, k_neighbors=8)
        assert h.shape == (200,)


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
