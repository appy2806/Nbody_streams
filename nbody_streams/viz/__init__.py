"""nbody_streams.viz — visualization and plotting.

Projected density maps, Mollweide sky projections, and stream
diagnostic plots.

Usage
-----
>>> from nbody_streams import viz
>>> viz.plot_density(pos=pos, mass=mass)
>>> viz.plot_mollweide(pos)
>>> viz.plot_stream_evolution(prog_xv, times, part_xv=part_xv)
"""

from .plots import (
    plot_density,
    plot_mollweide,
    plot_stream_sky,
    plot_stream_evolution,
)

__all__ = [
    "plot_density",
    "plot_mollweide",
    "plot_stream_sky",
    "plot_stream_evolution",
]
