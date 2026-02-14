"""Stream coordinate generation and observable conversions.

Generate stream-aligned coordinate frames from galactocentric phase-space
data, and convert to sky-observable quantities (RA, Dec, v_los) via Agama.
"""

from __future__ import annotations
import numpy as np

# =====================================================================
# Stream coordinate generation
# =====================================================================
def generate_stream_coords(
    xv: np.ndarray,
    xv_prog: np.ndarray | list | None = None,
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert galactocentric phase space into stream-aligned coordinates
    (phi1, phi2) for single or multiple streams.

    The stream frame is defined by the angular momentum vector of the
    progenitor orbit.  Particles are projected into a rotated frame where
    phi1 runs along the stream and phi2 measures the perpendicular offset.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6) or (S, N, 6)
        Galactocentric phase-space coordinates [x, y, z, vx, vy, vz].
        S = number of streams / time steps, N = number of particles.
    xv_prog : np.ndarray or None, optional
        Progenitor phase-space vector(s), shape (6,) or (S, 6).
        If None, the particle closest to the median position of each
        stream is used as the progenitor.
    degrees : bool, default True
        Return angles in degrees; otherwise radians.
    optimizer_fit : bool, default False
        If True, apply a scipy.optimize rotation in the phi1-phi2 plane
        to minimise the spread in phi2 (aligns the stream along phi1).
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``scipy.optimize.minimize``
        when *optimizer_fit* is True.

    Returns
    -------
    phi1 : np.ndarray
        Stream longitude.  Shape (N,) or (S, N).
    phi2 : np.ndarray
        Stream latitude.   Shape (N,) or (S, N).
    """
    xv = np.asarray(xv)

    # Normalise to 3-D: (S, N, 6)
    if xv.ndim == 2:
        xv = xv[None, ...]
        was_single = True
    elif xv.ndim == 3:
        was_single = False
    else:
        raise ValueError(f"xv must be 2-D (N, 6) or 3-D (S, N, 6), got shape {xv.shape}")

    S, N, D = xv.shape
    if D != 6:
        raise ValueError(f"Last dimension must be 6, got {D}")

    # --- Resolve progenitor(s) ---
    _prog_empty = (
        xv_prog is None
        or (isinstance(xv_prog, (list, tuple)) and len(xv_prog) == 0)
        or (isinstance(xv_prog, np.ndarray) and xv_prog.size == 0)
    )
    if _prog_empty:
        # Auto-detect: particle closest to median position per stream
        med = np.median(xv[:, :, :3], axis=1)                       # (S, 3)
        dists = np.linalg.norm(xv[:, :, :3] - med[:, None, :], axis=2)  # (S, N)
        idxs = np.argmin(dists, axis=1)                              # (S,)
        xv_prog = np.array([xv[s, idxs[s]] for s in range(S)])      # (S, 6)
    else:
        xv_prog = np.asarray(xv_prog)
        if xv_prog.ndim == 1:
            if was_single:
                xv_prog = xv_prog[None, :]                          # (1, 6)
            else:
                import warnings
                warnings.warn(
                    f"Single progenitor provided for {S} streams — "
                    f"broadcasting to all streams.",
                    UserWarning, stacklevel=2,
                )
                xv_prog = np.tile(xv_prog[None, :], (S, 1))        # (S, 6)
        elif xv_prog.ndim == 2:
            if xv_prog.shape[0] != S:
                raise ValueError(
                    f"Number of progenitors ({xv_prog.shape[0]}) "
                    f"must match number of streams ({S})"
                )
        else:
            raise ValueError(
                f"xv_prog must be 1-D (6,) or 2-D (S, 6), got shape {xv_prog.shape}"
            )

    # --- Build stream basis from progenitor angular momentum ---
    # L = r x v  (angular momentum direction defines the stream pole)
    L = np.cross(xv_prog[:, :3], xv_prog[:, 3:])                    # (S, 3)
    L /= np.linalg.norm(L, axis=1)[:, None]

    xhat = xv_prog[:, :3] / np.linalg.norm(xv_prog[:, :3], axis=1)[:, None]
    zhat = L
    yhat = np.cross(zhat, xhat)

    # Rotation matrices: columns are the new basis vectors
    R = np.stack([xhat, yhat, zhat], axis=-1)                        # (S, 3, 3)

    # --- Project particles into stream frame ---
    coords = xv[:, :, :3] @ R                                       # (S, N, 3)
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    phi1 = np.arctan2(ys, xs)
    phi2 = np.arcsin(zs / rs)

    # --- Optional in-plane rotation to minimise phi2 scatter ---
    if optimizer_fit:
        from scipy.optimize import minimize
        for s in range(S):
            def _cost(theta, _p1=phi1[s], _p2=phi2[s]):
                c, s_ = np.cos(theta), np.sin(theta)
                return np.sum((s_ * _p1 + c * _p2) ** 2)

            res = minimize(_cost, x0=0.0, **(fit_kwargs or {}))
            angle = res.x.item()
            c, s_ = np.cos(angle), np.sin(angle)
            phi1[s], phi2[s] = (
                c * phi1[s] - s_ * phi2[s],
                s_ * phi1[s] + c * phi2[s],
            )

    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)

    # Squeeze back if input was a single stream
    if was_single:
        phi1, phi2 = phi1[0], phi2[0]

    return phi1, phi2


# =====================================================================
# Observed (sky) stream coordinates — requires Agama
# =====================================================================
def get_observed_stream_coords(
    xv: np.ndarray,
    xv_prog: np.ndarray | list | None = None,
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
    galcen_distance: float = 8.122,
    galcen_v_sun: tuple = (12.9, 245.6, 7.78),
    z_sun: float = 0.0208,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert galactocentric phase-space coordinates to observed sky
    coordinates (RA, Dec, v_los) and stream coordinates (phi1, phi2).

    Uses Agama for the galactocentric -> galactic -> ICRS transformation,
    and ``generate_stream_coords`` for the stream-aligned frame.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6) or (S, N, 6)
        Galactocentric Cartesian phase-space [x, y, z, vx, vy, vz].
        Positions in kpc, velocities in km/s.
    xv_prog : np.ndarray or None, optional
        Progenitor phase-space vector(s).  See ``generate_stream_coords``.
    degrees : bool, default True
        Return angles in degrees; otherwise radians.
    optimizer_fit : bool, default False
        Optimise the stream-frame rotation (see ``generate_stream_coords``).
    fit_kwargs : dict or None, optional
        Extra kwargs for the optimiser.
    galcen_distance : float, default 8.122
        Sun–Galactic-centre distance in kpc.
    galcen_v_sun : tuple, default (12.9, 245.6, 7.78)
        Solar motion (U, V, W) in km/s.
    z_sun : float, default 0.0208
        Height of the Sun above the midplane in kpc.

    Returns
    -------
    ra, dec : np.ndarray
        Right Ascension and Declination.
    v_los : np.ndarray
        Line-of-sight velocity in km/s.
    phi1, phi2 : np.ndarray
        Stream-aligned coordinates.

    Raises
    ------
    ImportError
        If Agama is not installed.
    """
    try:
        import agama
        agama.setUnits(mass=1.0, length=1.0, velocity=1.0)
    except ImportError:
        raise ImportError(
            "The 'agama' library is required for observed coordinate "
            "transformations.  Install via 'pip install agama'."
        )

    xv = np.asarray(xv)
    is_batch = (xv.ndim == 3)
    if not is_batch:
        xv = xv[None, ...]  # (1, N, 6)

    S, N, _ = xv.shape

    # --- Resolve progenitors ---
    # Normalise xv_prog into shape (S, 6) before passing downstream.
    # generate_stream_coords handles the heavy lifting; here we just
    # need a concrete array (or None) to pass along.
    _prog_empty = (
        xv_prog is None
        or (isinstance(xv_prog, (list, tuple)) and len(xv_prog) == 0)
        or (isinstance(xv_prog, np.ndarray) and xv_prog.size == 0)
    )
    if _prog_empty:
        # Let generate_stream_coords auto-detect
        xv_prog_arr = None
    else:
        xv_prog_arr = np.asarray(xv_prog)
        if xv_prog_arr.ndim == 1:
            xv_prog_arr = np.broadcast_to(xv_prog_arr, (S, 6)).copy()
        elif xv_prog_arr.ndim == 2 and xv_prog_arr.shape[0] == 1:
            xv_prog_arr = np.broadcast_to(xv_prog_arr, (S, 6)).copy()

    # --- Galactocentric -> Galactic -> ICRS via Agama ---
    flat = xv.reshape(-1, 6)
    x, y, z, vx, vy, vz = flat.T

    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(
        x, y, z, vx, vy, vz,
        galcen_distance=galcen_distance,
        galcen_v_sun=galcen_v_sun,
        z_sun=z_sun,
    )
    ra, dec = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b)

    if degrees:
        ra  = np.degrees(ra)
        dec = np.degrees(dec)

    ra   = ra.reshape(S, N)
    dec  = dec.reshape(S, N)
    vlos = vlos.reshape(S, N)

    # --- Stream coordinates ---
    phi1, phi2 = generate_stream_coords(
        xv, xv_prog_arr,
        degrees=degrees,
        optimizer_fit=optimizer_fit,
        fit_kwargs=fit_kwargs,
    )

    if not is_batch:
        return ra[0], dec[0], vlos[0], phi1[0], phi2[0]
    return ra, dec, vlos, phi1, phi2