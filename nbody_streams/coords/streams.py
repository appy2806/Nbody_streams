"""Stream coordinate generation and observable conversions.

Generate stream-aligned coordinate frames from galactocentric phase-space
data, and convert to sky-observable quantities (RA, Dec, v_los) via Agama.
"""

from __future__ import annotations
import numpy as np

_KMS_KPC_TO_MASYR = 1.0 / 4.74047   # [mas/yr] per [km/s/kpc]
_RHO_GUARD        = 1e-6             # kpc, guards against pole singularity
# =====================================================================
# Stream coordinate generation
# =====================================================================
def generate_stream_coords(
    xv: np.ndarray,
    xv_prog: np.ndarray | list | None = None,
    return_rotation: bool = False,
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
    return_rotation : bool, default False
        If True, also return the rotation matrices used to define the
        stream frame.  Shape (3, 3) or  (S, 3, 3), where columns are the basis vectors.
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
    R : np.ndarray, optional
        Rotation matrix defining the stream frame.  Returned if *return_rotation* is True.
        Shape (3, 3) or (S, 3, 3), where columns are the basis vectors [xhat, yhat, zhat].
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
                    f"Single progenitor provided for {S} streams - "
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
    r_prog = xv_prog[:, :3]
    v_prog = xv_prog[:, 3:]
    L = np.cross(r_prog, v_prog)                    # (S, 3)
    L /= np.linalg.norm(L, axis=1)[:, None]

    xhat = r_prog / np.linalg.norm(r_prog, axis=1)[:, None]
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

    # --- Optional pole-tilt rotation to minimise phi2 scatter ---
    if optimizer_fit:
        from scipy.optimize import minimize

        for s in range(S):
            r_prog_s = xv_prog[s, :3]
            pos_s    = xv[s, :, :3]
            xhat_0   = R[s, :, 0].copy()
            yhat_0   = R[s, :, 1].copy()
            zhat_0   = R[s, :, 2].copy()

            def _build_R_s(params, _xh=xhat_0, _yh=yhat_0, _zh=zhat_0,
                           _rp=r_prog_s):
                alpha, beta = params
                # Tilt zhat linearly in the (xhat, yhat) directions
                # (exact for small angles; sufficient for stream alignment)
                zhat_new  = _zh + alpha * _xh + beta * _yh
                zhat_new /= np.linalg.norm(zhat_new)

                # Repin xhat: project progenitor onto new equatorial plane
                r_n      = _rp / np.linalg.norm(_rp)
                xhat_new = r_n - np.dot(r_n, zhat_new) * zhat_new
                norm_x   = np.linalg.norm(xhat_new)
                if norm_x < 1e-10:
                    # Progenitor is too close to new pole — fall back
                    xhat_new = _xh - np.dot(_xh, zhat_new) * zhat_new
                    norm_x   = np.linalg.norm(xhat_new)
                xhat_new /= norm_x
                yhat_new  = np.cross(zhat_new, xhat_new)

                return np.stack([xhat_new, yhat_new, zhat_new], axis=-1)

            def _cost(params, _pos=pos_s):
                R_new   = _build_R_s(params)
                c       = _pos @ R_new
                r_new   = np.linalg.norm(c, axis=1)
                phi2_new = np.arcsin(np.clip(c[:, 2] / r_new, -1.0, 1.0))
                return np.sum(phi2_new ** 2)

            res    = minimize(_cost, x0=[0.0, 0.0], **(fit_kwargs or {}))
            R[s]   = _build_R_s(res.x)

        # Recompute phi1, phi2 exactly from updated R — guaranteed consistent
        # with to_stream_coords using the same R
        coords = xv[:, :, :3] @ R
        xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
        rs     = np.sqrt(xs**2 + ys**2 + zs**2)
        phi1   = np.arctan2(ys, xs)
        phi2   = np.arcsin(np.clip(zs / rs, -1.0, 1.0))

        # R = R @ M # (S,3,3): absorb in-plane rotation into R so that returned R is consistent with final (phi1, phi2).

    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)

    # Squeeze back if input was a single stream
    if was_single:
        phi1, phi2, R = phi1[0], phi2[0], R[0]
        
    return (phi1, phi2, R) if return_rotation else (phi1, phi2)

def to_stream_coords(
    xv: np.ndarray,
    R: np.ndarray,
    degrees: bool = True,
    return_proper_motions: bool = False,
    mas_yr: bool = True,
) -> tuple:
    """
    Project galactocentric positions or phase-space vectors into a
    pre-computed stream frame.

    Parameters
    ----------
    xv : np.ndarray, shape (..., 3) or (..., 6)
        Galactocentric input. Pure positions (..., 3) are accepted when
        return_proper_motions=False; full phase-space (..., 6) is required
        for proper motions. Any number of leading batch dimensions is
        supported. Units: kpc for positions, km/s for velocities.
    R : np.ndarray, shape (3, 3) or (S, 3, 3)
        Rotation matrix from generate_stream_coords(return_rotation=True).
        Columns are [xhat, yhat, zhat].
        - (3, 3)    — the same frame is applied to every batch.
        - (S, 3, 3) — per-batch frames; S must equal xv.shape[0].
    degrees : bool, default True
        Return phi1, phi2 in degrees; otherwise radians.
    return_proper_motions : bool, default False
        If True, also return mu_phi1*cos(phi2) and mu_phi2. Requires
        6-column input.
    mas_yr : bool, default True
        Convert angular velocities to mas/yr (assumes kpc, km/s input).
        Otherwise return in km/s/kpc (= rad per [kpc/km·s]).
        NOTE: uses galactocentric r, not heliocentric distance — these are
        NOT directly comparable to Gaia proper motions.

    Returns
    -------
    phi1 : np.ndarray
        Stream longitude. Same leading shape as xv.
    phi2 : np.ndarray
        Stream latitude. Same leading shape as xv.
    mu_phi1_cosphi2 : np.ndarray, optional
        d(phi1)/dt * cos(phi2). Returned only when return_proper_motions=True.
    mu_phi2 : np.ndarray, optional
        d(phi2)/dt. Returned only when return_proper_motions=True.

    Raises
    ------
    ValueError
        If R.shape is not (3,3) or (S,3,3), if S mismatches xv.shape[0],
        or if return_proper_motions=True with position-only input.
    """
    xv = np.asarray(xv, dtype=float)
    R  = np.asarray(R,  dtype=float)

    def _apply_R(arr, R, batched_R):
        """
        Project (..., 3) array into stream frame.
        Handles both a single (3,3) R and batched (S,3,3) R correctly.
        """
        if not batched_R:
            return arr @ R                             # broadcasts over any leading dims
        if arr.ndim == 2:                              # (S, 3) — one vector per frame
            return (arr[:, None, :] @ R)[:, 0, :]      # -> (S, 3)
        return arr @ R                                 # (S, N, 3) @ (S, 3, 3) → (S, N, 3)

    # --- Input validation ---
    last_dim = xv.shape[-1]
    if last_dim not in (3, 6):
        raise ValueError(f"Last axis of xv must be 3 or 6, got {last_dim}.")
    if return_proper_motions and last_dim != 6:
        raise ValueError(
            "return_proper_motions=True requires 6-column xv, got 3. "
            "Pass full phase-space [x, y, z, vx, vy, vz]."
        )

    # --- Resolve R shape: broadcast (3,3) or validate (S,3,3) ---
    if R.ndim == 2:
        if R.shape != (3, 3):
            raise ValueError(f"R must be (3,3) or (S,3,3), got {R.shape}.")
        batched_R = False

    elif R.ndim == 3:
        if R.shape[1:] != (3, 3):
            raise ValueError(f"R must be (3,3) or (S,3,3), got {R.shape}.")
        S_R = R.shape[0]
        if xv.ndim < 1 or xv.shape[0] != S_R:
            raise ValueError(
                f"R has S={S_R} frames but xv.shape[0]={xv.shape[0]}. "
                "First dimension of xv must match S."
            )
        batched_R = True

    else:
        raise ValueError(f"R must be 2-D (3,3) or 3-D (S,3,3), got {R.ndim}-D.")

    # --- Project positions ---
    pos = xv[..., :3]                              # (..., 3)

    # (... , 3) @ (3, 3) → (..., 3), broadcasts over any leading dims
    coords = _apply_R(pos, R, batched_R)

    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    r   = np.sqrt(xs**2 + ys**2 + zs**2)
    rho = np.sqrt(xs**2 + ys**2)

    phi1 = np.arctan2(ys, xs)
    phi2 = np.arcsin(np.clip(zs / np.where(r > 0, r, 1.0), -1.0, 1.0))

    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)

    if not return_proper_motions:
        return phi1, phi2

    # --- Angular velocities in stream frame ---
    vel = xv[..., 3:]
    # (... , 3) @ (3, 3) → (..., 3), broadcasts over any leading dims
    vs = _apply_R(vel, R, batched_R)

    vxs, vys, vzs = vs[..., 0], vs[..., 1], vs[..., 2]

    # Radial velocity component
    vr = (xs*vxs + ys*vys + zs*vzs) / np.where(r > 0, r, 1.0)

    # Guard against stream pole (phi2 → ±90°, rho → 0)
    safe_rho_r = np.where(rho > _RHO_GUARD, rho * r, np.nan)

    # d(phi1)/dt * cos(phi2) = (xs*vys - ys*vxs) / (rho * r)
    mu_phi1_cosphi2 = (xs * vys - ys * vxs) / safe_rho_r

    # d(phi2)/dt = (r*vzs - zs*vr) / (rho * r)
    mu_phi2 = (r * vzs - zs * vr) / safe_rho_r

    if mas_yr:
        mu_phi1_cosphi2 = mu_phi1_cosphi2 * _KMS_KPC_TO_MASYR
        mu_phi2         = mu_phi2         * _KMS_KPC_TO_MASYR

    return phi1, phi2, mu_phi1_cosphi2, mu_phi2


# =====================================================================
# Observed (sky) stream coordinates - requires Agama
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
        Sun-Galactic-centre distance in kpc.
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