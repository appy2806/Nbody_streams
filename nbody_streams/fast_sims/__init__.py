"""nbody_streams.fast_sims — fast stream simulation methods.

Lightweight alternatives to full N-body integration for generating
stellar streams: particle spray, restricted N-body, and related
techniques.  All methods use Agama for the host potential.

Usage
-----
>>> from nbody_streams import fast_sims
>>> fast_sims.create_particle_spray_stream(...)
>>> fast_sims.run_restricted_nbody(...)
"""

from .spray import (
    create_ic_particle_spray_chen2025,
    create_ic_particle_spray_fardal2015,
    create_particle_spray_stream,
)
from .restricted import run_restricted_nbody

__all__ = [
    "create_ic_particle_spray_chen2025",
    "create_ic_particle_spray_fardal2015",
    "create_particle_spray_stream",
    "run_restricted_nbody",
]
