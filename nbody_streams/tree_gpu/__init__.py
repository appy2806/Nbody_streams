"""
nbody_streams.tree_gpu
======================
GPU Barnes-Hut tree code for N-body gravity (O(N log N) force evaluation).

Build the shared library before first use::

    cd nbody_streams/tree_gpu
    make -j$(nproc)          # builds libtreeGPU.so alongside the source files

Public API
----------
tree_gravity_gpu
    One-shot force + potential computation.  Allocates/frees a tree per call.
    Use for single evaluations or when N changes between calls.

TreeGPU
    Pre-allocated tree handle for time-stepping loops.  Saves ~27 ms of
    GPU malloc/free overhead per step.  Must be closed explicitly or used
    as a context manager.

cuda_alive
    Lightweight CUDA context health check (reads thread-local error flag via
    cudaGetLastError; no synchronisation).

run_nbody_gpu_tree
    KDK leapfrog integrator using the GPU tree code.  Matches the signature
    of ``run_nbody_gpu`` (direct-sum), with ``theta`` / tree-specific params
    replacing ``precision`` / ``kernel``.  Includes a watchdog thread that
    raises KeyboardInterrupt if a CUDA kernel deadlocks.

Notes
-----
All force evaluations inside the tree code operate in **float32**.  Positions
and velocities in ``run_nbody_gpu_tree`` are also kept in float32 on the GPU
(consistent with the precision of the tree code itself).  Energy diagnostics
are computed in float64.

The softening convention matches nbody_streams direct-sum:
    direct:  ``eps2_ij = max(eps_i^2, eps_j^2)``
    approx:  ``eps2_ij = max(eps_i^2, eps_cell_max^2)``  (max convention)
"""

try:
    from ._force import tree_gravity_gpu, TreeGPU, cuda_alive, G_DEFAULT
    from .run_gpu_tree import run_nbody_gpu_tree
    __all__ = ["tree_gravity_gpu", "TreeGPU", "cuda_alive", "G_DEFAULT",
               "run_nbody_gpu_tree"]
except OSError as _e:
    raise ImportError(
        "nbody_streams.tree_gpu: could not load libtreeGPU.so. "
        "Build it first:\n\n"
        "    cd nbody_streams/tree_gpu && make -j$(nproc)\n\n"
        f"Original error: {_e}"
    ) from _e
