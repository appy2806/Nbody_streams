"""nbody_streams.viz - visualization and plotting.

Projected density maps, Mollweide sky projections, and stream
diagnostic plots.

Usage
-----
>>> from nbody_streams import viz
>>> viz.plot_density(pos=pos, mass=mass)           # SPH by default
>>> viz.plot_density(pos=pos, mass=mass, method='histogram')
>>> grid, bounds = viz.render_surface_density(x, y, mass, resolution=512)
>>> h = viz.get_smoothing_lengths(pos_2d, k_neighbors=32)
>>> viz.plot_mollweide(pos)
>>> viz.plot_stream_evolution(prog_xv, times, part_xv=part_xv)
"""

from .plots import (
    plot_density,
    plot_mollweide,
    plot_stream_sky,
    plot_stream_evolution,
)
from .sph_kernels import render_surface_density, get_smoothing_lengths

__all__ = [
    "plot_density",
    "plot_mollweide",
    "plot_stream_sky",
    "plot_stream_evolution",
    "render_surface_density",
    "get_smoothing_lengths",
]
