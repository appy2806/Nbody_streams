# nbody_streams

Direct N-body simulator and utilities for collisionless systems (single particle type).
Designed for research and prototyping with a minimal API and optional GPU acceleration.

Core points
- Direct N-body code with both CPU and optional GPU implementations.
- GPU backend uses custom CUDA kernels (CuPy) and Kahan-style corrections for improved float32 accuracy.
- CPU fallback uses NumPy/Numba where available.
- Optional tree/FMM backend (pyfalcon) and optional external potentials via Agama.
- Intended for collisionless systems with up to ~100k particles (benchmarks depend on hardware).

Highlights / performance
- GPU kernels: F32 and Kahan-corrected F32 provide a good speed/accuracy tradeoff.
- On consumer GPUs (RTX3080 example): F32_KAHAN ~6 ms for 10k particles; F64 ~23 ms (benchmarks are hardware-dependent).

Features
- `nbody_streams.fields` — direct-force kernels (GPU and CPU).
- `nbody_streams.io` — `ParticleReader` for HDF5-based snapshots and small helpers (`_save_snapshot`, `_load_restart`).
- `nbody_streams.run` — simple leap-frog (KDK) integrator with optional external potentials.
- `nbody_streams.cuda_kernels` — kernel templates used by cupy.

Quick start (editable install)
```bash
pip install -r requirements.txt
# editable (development) install
pip install -e .


Optional GPU support
- Install CuPy matching your CUDA version (see https://docs.cupy.dev):

Usage example:
from nbody_streams.io import ParticleReader
r = ParticleReader("path/to/sim.*.h5", verbose=True)
snap = r.read_snapshot(10)  # returns dict with pos, vel, mass, time

