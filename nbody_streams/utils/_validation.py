"""
nbody_streams.utils._validation
================================

Shared input-validation helpers used across utility functions.

All validators raise ``ValueError`` on invalid input and return sanitised
NumPy arrays ready for downstream computation.
"""
from __future__ import annotations

import numpy as np

__all__: list[str] = []  # nothing public — internal helpers only


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def validate_positions(
    pos,
    *,
    allow_batched: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a position array and compute radial distances.

    Parameters
    ----------
    pos : array_like
        Particle positions. Accepted shapes:

        * ``(N,)``   — pre-computed radii (returned as-is for *radii*).
        * ``(N, 3)`` — Cartesian coordinates.
        * ``(..., 3)`` with ``allow_batched=True`` — batched coordinates
          (e.g. ``(T, N, 3)`` for multiple snapshots).

    allow_batched : bool, optional
        If *True*, any leading dimensions are permitted as long as the last
        axis has size 3.  Default is *False* (only ``(N,)`` or ``(N, 3)``).

    Returns
    -------
    pos : np.ndarray
        The validated position array (at least ``float64``).
    radii : np.ndarray
        Euclidean distances from the origin.  Shape is ``(N,)`` when
        *pos* is ``(N, 3)`` (or ``(N,)``), and ``(..., N)`` in the
        batched case.
    """
    pos = np.asarray(pos, dtype=float)

    if pos.ndim == 1:
        # Already radii
        return pos, pos.copy()

    if allow_batched:
        if pos.ndim < 2 or pos.shape[-1] != 3:
            raise ValueError(
                f"With allow_batched=True, pos must have shape (..., 3), "
                f"got {pos.shape}"
            )
    else:
        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(
                f"pos must have shape (N, 3) or (N,), got {pos.shape}"
            )

    radii = np.linalg.norm(pos, axis=-1)
    return pos, radii


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------

def validate_masses(
    mass,
    n_particles: int,
    *,
    non_negative: bool = False,
) -> np.ndarray:
    """Validate a mass argument and broadcast scalars.

    Parameters
    ----------
    mass : scalar, array_like, or None
        Particle masses.  A scalar is broadcast to shape ``(n_particles,)``.
        *None* is treated as unit mass for every particle.
    n_particles : int
        Expected number of particles.
    non_negative : bool, optional
        If *True*, raise if any mass is negative.

    Returns
    -------
    mass : np.ndarray, shape ``(n_particles,)``
    """
    if mass is None:
        return np.ones(n_particles, dtype=float)

    mass = np.asarray(mass, dtype=float)
    if mass.ndim == 0:
        mass = np.full(n_particles, mass, dtype=float)
    elif mass.shape[0] != n_particles:
        raise ValueError(
            f"mass length ({mass.shape[0]}) does not match number of "
            f"particles ({n_particles})"
        )

    if non_negative and np.any(mass < 0):
        raise ValueError("Particle masses must be non-negative.")

    return mass


# ---------------------------------------------------------------------------
# Velocities
# ---------------------------------------------------------------------------

def validate_velocities(
    vel,
    n_particles: int,
    *,
    allow_batched: bool = False,
) -> np.ndarray:
    """Validate a velocity array.

    Parameters
    ----------
    vel : array_like or None
        Particle velocities.  *None* returns a zero array of shape
        ``(n_particles, 3)``.  Otherwise accepted shapes mirror
        :func:`validate_positions`.
    n_particles : int
        Expected number of particles (ignored in batched mode).
    allow_batched : bool, optional
        If *True*, any leading dimensions are permitted as long as the last
        axis has size 3.

    Returns
    -------
    vel : np.ndarray
        Validated velocity array (at least ``float64``).
    """
    if vel is None:
        return np.zeros((n_particles, 3), dtype=float)

    vel = np.asarray(vel, dtype=float)

    if allow_batched:
        if vel.ndim < 2 or vel.shape[-1] != 3:
            raise ValueError(
                f"With allow_batched=True, vel must have shape (..., 3), "
                f"got {vel.shape}"
            )
    else:
        if vel.ndim == 1:
            if vel.shape[0] != n_particles:
                raise ValueError(
                    f"vel length ({vel.shape[0]}) does not match number "
                    f"of particles ({n_particles})"
                )
        elif vel.ndim == 2:
            if vel.shape[1] != 3:
                raise ValueError(
                    f"vel must have shape (N, 3) or (N,), got {vel.shape}"
                )
            if vel.shape[0] != n_particles:
                raise ValueError(
                    f"vel length ({vel.shape[0]}) does not match number "
                    f"of particles ({n_particles})"
                )
        else:
            raise ValueError(
                f"vel must have shape (N, 3) or (N,), got {vel.shape}"
            )

    return vel


# ---------------------------------------------------------------------------
# Scalar parameters
# ---------------------------------------------------------------------------

def validate_nbins(nbins: int) -> None:
    """Raise ``ValueError`` if *nbins* is not a positive integer."""
    if not isinstance(nbins, (int, np.integer)) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")
