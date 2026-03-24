# nbody_streams — Documentation

Source code: https://github.com/appy2806/Nbody_streams

> This directory contains module-level reference documentation in Markdown
> (MyST-compatible).  A full Sphinx build with readthedocs integration is
> planned for a future release.

## Module reference

| Module | File | Description |
|---|---|---|
| Quick start | [quickstart.md](quickstart.md) | Installation, units, hello-world run |
| `nbody_streams` (main) | [main.md](main.md) | `run_simulation`, `Species`, `make_plummer_sphere` |
| `nbody_streams.fields` | [fields.md](fields.md) | Force/potential kernels: GPU and CPU backends |
| `nbody_streams.run` | [run.md](run.md) | Low-level KDK leapfrog integrators |
| `nbody_streams.tree_gpu` | [tree_gpu.md](tree_gpu.md) | GPU Barnes-Hut tree code |
| `nbody_streams.nbody_io` | [io.md](io.md) | HDF5 snapshot I/O, `ParticleReader` |
| `nbody_streams.fast_sims` | [fast_sims.md](fast_sims.md) | Particle spray and restricted N-body stream generation |
| `nbody_streams.viz` | [viz.md](viz.md) | Density maps, SPH renderer, sky plots |
| `nbody_streams.utils` | [utils.md](utils.md) | Profile fitting, shape, centre finding, unbinding |
| `nbody_streams.coords` | [coords.md](coords.md) | Coordinate and vector field transforms |
| `nbody_streams.agama_helper` | [agama_helper.md](agama_helper.md) | Fit, store, modify, and load Agama BFE potentials |
| Dynamical friction | [dynamical_friction.md](dynamical_friction.md) | Chandrasekhar DF: theory, usage, df_* kwargs, caveats |

## Building docs (future)

When the Sphinx build is set up, install the docs dependencies and run:

```bash
pip install sphinx myst-parser sphinx-rtd-theme
sphinx-build -b html docs/ docs/_build/html
```
