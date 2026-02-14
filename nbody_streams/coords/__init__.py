"""nbody_streams.coords — coordinate transformations.

Cartesian/spherical/cylindrical conversions, vector field transforms,
stream coordinate generation, and conversion to observable sky coordinates.

Usage
-----
>>> from nbody_streams import coords
>>> coords.convert_coords(pos, 'cart', 'sph')
>>> coords.convert_vectors(pos, vel, 'cart', 'cyl')
>>> coords.convert_to_vel_los(xv)
>>> coords.generate_stream_coords(xv)
"""

from .transforms import convert_coords, convert_vectors, convert_to_vel_los
from .streams import generate_stream_coords, get_observed_stream_coords

__all__ = [
    "convert_coords",
    "convert_vectors",
    "convert_to_vel_los",
    "generate_stream_coords",
    "get_observed_stream_coords",
]
