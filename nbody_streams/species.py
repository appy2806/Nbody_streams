"""
nbody_streams.species
Species definitions and helper functions for multi-species N-body simulations.

Public API
----------
Species         : dataclass describing one particle species
PerformanceWarning : warning class for large-N threshold messages
"""
from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Union


class PerformanceWarning(UserWarning):
    """Warning emitted when particle count exceeds recommended thresholds."""
    pass


@dataclass
class Species:
    """
    Definition of a particle species in an N-body simulation.

    Parameters
    ----------
    name : str
        Species identifier.  Conventional names: ``'dark'``, ``'star'``,
        ``'bh'``.  Arbitrary names such as ``'spec0'``, ``'spec1'`` are
        also supported.
    N : int
        Number of particles belonging to this species.
    mass : float or array_like of shape (N,)
        Particle masses.  A scalar means all particles share the same mass;
        an array gives per-particle masses.
    softening : float or array_like of shape (N,), optional
        Gravitational softening length(s).  Same scalar / array semantics as
        *mass*.  Default: ``0.0``.

    Examples
    --------
    >>> dm    = Species.dark(N=10_000, mass=1e6, softening=0.1)
    >>> stars = Species.star(N=5_000,  mass=1e5, softening=0.05)
    >>> bh    = Species(name='bh', N=1, mass=1e9, softening=0.001)
    """

    name: str
    N: int
    mass: Union[float, NDArray]
    softening: Union[float, NDArray] = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Species name must be a non-empty string")
        if self.N <= 0:
            raise ValueError(f"Species '{self.name}': N must be > 0, got {self.N}")
        if not np.isscalar(self.mass):
            m = np.asarray(self.mass)
            if m.shape != (self.N,):
                raise ValueError(
                    f"Species '{self.name}': mass array shape {m.shape} != ({self.N},)"
                )
        if not np.isscalar(self.softening):
            h = np.asarray(self.softening)
            if h.shape != (self.N,):
                raise ValueError(
                    f"Species '{self.name}': softening array shape {h.shape} != ({self.N},)"
                )

    @staticmethod
    def dark(N: int,
             mass: float | NDArray,
             softening: float | NDArray = 0.0) -> Species:
        """Convenience constructor for dark-matter particles."""
        return Species(name="dark", N=N, mass=mass, softening=softening)

    @staticmethod
    def star(N: int,
             mass: float | NDArray,
             softening: float | NDArray = 0.0) -> Species:
        """Convenience constructor for stellar particles."""
        return Species(name="star", N=N, mass=mass, softening=softening)


# ---------------------------------------------------------------------------
# Internal helpers — not exported in __all__ but importable if needed
# ---------------------------------------------------------------------------

def _build_particle_arrays(
    species: list[Species],
) -> tuple[NDArray, NDArray]:
    """
    Build combined per-particle mass and softening arrays from a species list.

    Parameters
    ----------
    species : list[Species]
        Ordered list of species definitions (determines concatenation order).

    Returns
    -------
    mass_arr : ndarray, shape (N_total,)
        Concatenated mass array.
    softening_arr : ndarray, shape (N_total,)
        Concatenated softening array.
    """
    masses: list[NDArray] = []
    softenings: list[NDArray] = []
    for s in species:
        m = (np.full(s.N, float(s.mass), dtype=np.float64)
             if np.isscalar(s.mass)
             else np.asarray(s.mass, dtype=np.float64))
        h = (np.full(s.N, float(s.softening), dtype=np.float64)
             if np.isscalar(s.softening)
             else np.asarray(s.softening, dtype=np.float64))
        masses.append(m)
        softenings.append(h)
    return np.concatenate(masses), np.concatenate(softenings)


def _validate_species(phase_space: NDArray, species: list[Species]) -> None:
    """
    Validate *species* list against *phase_space* shape.

    Checks
    ------
    * ``sum(s.N for s in species) == phase_space.shape[0]``
    * No duplicate names.
    * All ``N > 0`` (enforced by :class:`Species.__post_init__`).

    Raises
    ------
    ValueError
    """
    if not species:
        raise ValueError("species list must not be empty")

    names = [s.name for s in species]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate species names: {dupes}")

    total_N = sum(s.N for s in species)
    xv_N = phase_space.shape[0]
    if total_N != xv_N:
        raise ValueError(
            f"sum(s.N for s in species) = {total_N} does not match "
            f"phase_space.shape[0] = {xv_N}"
        )


def _split_by_species(xv: NDArray, species: list[Species]) -> dict[str, NDArray]:
    """
    Split a combined ``(N_total, 6)`` phase-space array into per-species slices.

    Parameters
    ----------
    xv : ndarray, shape (N_total, 6)
    species : list[Species]

    Returns
    -------
    dict mapping species name -> ndarray of shape (N_k, 6)
    """
    result: dict[str, NDArray] = {}
    idx = 0
    for s in species:
        result[s.name] = xv[idx : idx + s.N]
        idx += s.N
    return result


def _emit_performance_warnings(
    N_total: int,
    architecture: str,
    method: str,
) -> None:
    """
    Emit :class:`PerformanceWarning` or :class:`FutureWarning` when *N_total*
    exceeds recommended thresholds for the chosen architecture / method.
    """
    if N_total > 2_000_000:
        warnings.warn(
            f"{N_total:,} particles: this scale requires GPU+Tree methods. "
            "GPU+Tree support is not yet implemented in this version.",
            FutureWarning,
            stacklevel=4,
        )
    elif architecture == "cpu" and method == "direct" and N_total > 20_000:
        warnings.warn(
            f"{N_total:,} particles with CPU direct summation is O(N^2) and "
            "will be very slow. "
            "Consider architecture='cpu' + method='tree', or architecture='gpu'.",
            PerformanceWarning,
            stacklevel=4,
        )
    elif architecture == "gpu" and method == "direct" and N_total > 500_000:
        warnings.warn(
            f"{N_total:,} particles with GPU direct summation may be slow at "
            "this scale. "
            "GPU+Tree support is coming in a future version.",
            PerformanceWarning,
            stacklevel=4,
        )
