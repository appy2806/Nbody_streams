"""nbody_streams — lightweight direct N-body utilities."""

from importlib.metadata import version as _version_lookup, PackageNotFoundError

# --- Versioning ---
try:
    # This works if the package was installed via 'pip install .'
    __version__ = _version_lookup("nbody_streams")
except PackageNotFoundError:
    # Fallback for local development
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "unknown"

# --- Public API ---

# From .io
from .io import ParticleReader

# From .run
from .run import (
    run_nbody_gpu,
    run_nbody_cpu,
    make_plummer_sphere,
    G_DEFAULT,
    NBODY_UNITS,
)

# From .fields
from .fields import (
    compute_nbody_forces_gpu,
    compute_nbody_forces_cpu,
    compute_nbody_potential_gpu,
    compute_nbody_potential_cpu,
    get_gpu_info,
)

# Define what "from nbody_streams import *" does
__all__ = [
    "__version__",
    "ParticleReader",
    "run_nbody_gpu",
    "run_nbody_cpu",
    "G_DEFAULT",
    "NBODY_UNITS",
    "compute_nbody_forces_gpu",
    "compute_nbody_forces_cpu",
    "compute_nbody_potential_gpu",
    "compute_nbody_potential_cpu",
    "get_gpu_info",
]