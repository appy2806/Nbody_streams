# nbody_streams — Documentation

Source code: https://github.com/appy2806/Nbody_streams

> This directory contains module-level reference documentation in Markdown
> (MyST-compatible).  A full Sphinx build with readthedocs integration is
> planned for a future release.

## Module reference

| Module | Description |
|---|---|
| [`agama_helper`](agama_helper.md) | Fit, store, modify, and load Agama Multipole/CylSpline BFE potentials |

## Building docs (future)

When the Sphinx build is set up, install the docs dependencies and run:

```bash
pip install sphinx myst-parser sphinx-rtd-theme
sphinx-build -b html docs/ docs/_build/html
```
