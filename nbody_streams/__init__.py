"""nbody_streams - lightweight direct N-body utilities."""

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

# Multi-species types and unified simulation entry point
from .species import Species, PerformanceWarning
from .sim import run_simulation

# From .io
from .nbody_io import ParticleReader

# From .run
from .run import (
    run_nbody_gpu,
    run_nbody_cpu,
    make_plummer_sphere,
    G_DEFAULT,
    NBODY_UNITS,
)

# Subpackages
from . import utils
from . import coords
from . import fast_sims
from . import viz

# GPU tree-code (optional - requires libtreeGPU.so; built separately with make)
try:
    from . import tree_gpu
    from .tree_gpu import tree_gravity_gpu, TreeGPU, cuda_alive, run_nbody_gpu_tree
    _TREE_GPU_AVAILABLE = True
except ImportError:
    _TREE_GPU_AVAILABLE = False

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
    # Multi-species
    "Species",
    "PerformanceWarning",
    "run_simulation",
    # I/O
    "ParticleReader",
    # Low-level integration (backward compat)
    "run_nbody_gpu",
    "run_nbody_cpu",
    # Utilities
    "make_plummer_sphere",
    "G_DEFAULT",
    "NBODY_UNITS",
    # Force / potential computation
    "compute_nbody_forces_gpu",
    "compute_nbody_forces_cpu",
    "compute_nbody_potential_gpu",
    "compute_nbody_potential_cpu",
    "get_gpu_info",
    # Subpackages
    "utils",
    "coords",
    "fast_sims",
    "viz",
    # GPU tree-code (present only when libtreeGPU.so is built)
    "tree_gpu",
    "tree_gravity_gpu",
    "TreeGPU",
    "cuda_alive",
    "run_nbody_gpu_tree",
]