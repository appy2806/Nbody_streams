"""nbody_streams.utils — analysis and diagnostic utilities."""

from .main import (
    make_uneven_grid,
    # Empirical radial profiles
    empirical_density_profile,
    empirical_circular_velocity_profile,
    empirical_velocity_dispersion_profile,
    empirical_velocity_rms_profile,
    empirical_velocity_anisotropy_profile,
    # Density fitting
    fit_double_spheroid_profile,
    # Morphological diagnostics
    fit_iterative_ellipsoid,
    # Grid generators
    uniform_spherical_grid,
    spherical_spiral_grid,
    # Centre finding
    find_center_position,
    # Boundness
    compute_iterative_boundness,
    iterative_unbinding,
)

__all__ = [
    "make_uneven_grid",
    "empirical_density_profile",
    "empirical_circular_velocity_profile",
    "empirical_velocity_dispersion_profile",
    "empirical_velocity_rms_profile",
    "empirical_velocity_anisotropy_profile",
    "fit_double_spheroid_profile",
    "fit_iterative_ellipsoid",
    "uniform_spherical_grid",
    "spherical_spiral_grid",
    "find_center_position",
    "compute_iterative_boundness",
    "iterative_unbinding",
]
