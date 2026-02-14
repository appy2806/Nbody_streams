"""nbody_streams.fast_sims — fast stream simulation methods.

Lightweight alternatives to full N-body integration for generating
stellar streams: particle spray, restricted N-body, and related
techniques.  All methods use Agama for the host potential.

Usage
-----
>>> from nbody_streams import fast_sims
>>> fast_sims.particle_spray(...)
>>> fast_sims.restricted_nbody(...)
"""

# Re-export public API so users get a flat namespace:
#   nbody_streams.fast_sims.particle_spray(...)
# rather than:
#   nbody_streams.fast_sims.spray.particle_spray(...)

# from .spray import ...        # TODO: uncomment once implemented
# from .restricted import ...   # TODO: uncomment once implemented

__all__ = []  # TODO: populate as functions are added
