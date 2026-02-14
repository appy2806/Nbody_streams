"""Coordinate system and vector field conversions.

Cartesian <-> spherical <-> cylindrical position and vector transforms.
All functions accept arbitrary batch dimensions: input shape (..., 3).

Coordinate conventions
----------------------
- Cartesian : (x, y, z)
- Spherical : (rho, theta, phi)
    rho   = radial distance
    theta = polar angle (colatitude) from +z, in [0, pi]
    phi   = azimuthal angle, [0, 2*pi) or (-pi, pi] with mollweide=True
- Cylindrical : (R, phi, z)
    R   = sqrt(x^2 + y^2)
    phi = azimuthal angle, [0, 2*pi)
    z   = height (same as Cartesian z)
"""

from __future__ import annotations
import numpy as np

# =====================================================================
# Input validation
# =====================================================================
def _as_3vec(arr: np.ndarray) -> np.ndarray:
    """Validate and return float array with last dimension == 3."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected array with last dimension == 3, got shape {arr.shape}."
        )
    return arr


def _propagate_nans(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """If any component of a point is NaN, set the entire output point to NaN."""
    invalid = np.any(np.isnan(inp), axis=-1)
    if np.any(invalid):
        out[invalid] = np.nan
    return out


# =====================================================================
# Private coordinate transforms — all work with (..., 3)
# =====================================================================
def _cart_to_sph(xyz: np.ndarray, mollweide: bool = False) -> np.ndarray:
    """Cartesian -> Spherical.  See `convert_coords` for details."""
    xyz = _as_3vec(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    xy_sq = x**2 + y**2
    rho   = np.sqrt(xy_sq + z**2)
    theta = np.arctan2(np.sqrt(xy_sq), z)   # colatitude from +z
    phi   = np.arctan2(y, x)
    phi   = np.mod(phi, 2 * np.pi)          # [0, 2*pi)

    if mollweide:
        # Healpy expects phi in (-pi, pi]
        phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)

    out = np.empty_like(xyz)
    out[..., 0] = rho
    out[..., 1] = theta
    out[..., 2] = phi
    return _propagate_nans(xyz, out)


def _sph_to_cart(sph: np.ndarray, mollweide: bool = False) -> np.ndarray:
    """Spherical -> Cartesian.  See `convert_coords` for details."""
    sph = _as_3vec(sph)
    rho, theta, phi = sph[..., 0], sph[..., 1], sph[..., 2]

    if mollweide:
        # Map (-pi, pi] back to [0, 2*pi) before trig
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    sin_theta = np.sin(theta)
    out = np.empty_like(sph)
    out[..., 0] = rho * sin_theta * np.cos(phi)
    out[..., 1] = rho * sin_theta * np.sin(phi)
    out[..., 2] = rho * np.cos(theta)
    return _propagate_nans(sph, out)


def _cart_to_cyl(xyz: np.ndarray) -> np.ndarray:
    """Cartesian -> Cylindrical."""
    xyz = _as_3vec(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    R   = np.sqrt(x**2 + y**2)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    out = np.empty_like(xyz)
    out[..., 0] = R
    out[..., 1] = phi
    out[..., 2] = z
    return _propagate_nans(xyz, out)


def _cyl_to_cart(cyl: np.ndarray) -> np.ndarray:
    """Cylindrical -> Cartesian."""
    cyl = _as_3vec(cyl)
    R, phi, z = cyl[..., 0], cyl[..., 1], cyl[..., 2]

    out = np.empty_like(cyl)
    out[..., 0] = R * np.cos(phi)
    out[..., 1] = R * np.sin(phi)
    out[..., 2] = z
    return _propagate_nans(cyl, out)


def _sph_to_cyl(sph: np.ndarray) -> np.ndarray:
    """Spherical -> Cylindrical (direct, no Cartesian intermediate)."""
    sph = _as_3vec(sph)
    rho, theta, phi = sph[..., 0], sph[..., 1], sph[..., 2]

    out = np.empty_like(sph)
    out[..., 0] = rho * np.sin(theta)   # R
    out[..., 1] = phi                    # phi (same)
    out[..., 2] = rho * np.cos(theta)   # z
    return _propagate_nans(sph, out)


def _cyl_to_sph(cyl: np.ndarray) -> np.ndarray:
    """Cylindrical -> Spherical (direct, no Cartesian intermediate)."""
    cyl = _as_3vec(cyl)
    R, phi, z = cyl[..., 0], cyl[..., 1], cyl[..., 2]

    out = np.empty_like(cyl)
    out[..., 0] = np.sqrt(R**2 + z**2)  # rho
    out[..., 1] = np.arctan2(R, z)      # theta (colatitude)
    out[..., 2] = phi                    # phi (same)
    return _propagate_nans(cyl, out)


# Dispatch table: (from_sys, to_sys) -> callable
_COORD_DISPATCH = {
    ("cart", "sph"): _cart_to_sph,
    ("sph", "cart"): _sph_to_cart,
    ("cart", "cyl"): _cart_to_cyl,
    ("cyl", "cart"): _cyl_to_cart,
    ("sph", "cyl"): _sph_to_cyl,
    ("cyl", "sph"): _cyl_to_sph,
}

_VALID_SYSTEMS = ("cart", "sph", "cyl")


# =====================================================================
# Public API: convert_coords
# =====================================================================
def convert_coords(
    data: np.ndarray,
    from_sys: str,
    to_sys: str,
    *,
    mollweide: bool = False,
) -> np.ndarray:
    """
    Convert positions between coordinate systems.

    Parameters
    ----------
    data : array_like, shape (..., 3)
        Input coordinates.  Arbitrary leading batch dimensions are supported.
    from_sys, to_sys : {'cart', 'sph', 'cyl'}
        Source and target coordinate systems.
    mollweide : bool, default False
        Only relevant for cart<->sph conversions.
        If True, phi wraps to (-pi, pi] (Healpy convention).

    Returns
    -------
    np.ndarray, shape (..., 3)
        Converted coordinates.

    Examples
    --------
    >>> pos_sph = convert_coords(pos_cart, 'cart', 'sph')
    >>> pos_cart = convert_coords(pos_sph, 'sph', 'cart', mollweide=True)
    """
    from_sys = from_sys.lower()
    to_sys = to_sys.lower()

    if from_sys not in _VALID_SYSTEMS:
        raise ValueError(f"from_sys must be one of {_VALID_SYSTEMS}, got '{from_sys}'")
    if to_sys not in _VALID_SYSTEMS:
        raise ValueError(f"to_sys must be one of {_VALID_SYSTEMS}, got '{to_sys}'")
    if from_sys == to_sys:
        return np.array(_as_3vec(data))  # copy

    fn = _COORD_DISPATCH[(from_sys, to_sys)]

    # Only cart<->sph transforms accept mollweide
    if mollweide and (from_sys, to_sys) in (("cart", "sph"), ("sph", "cart")):
        return fn(data, mollweide=True)
    return fn(data)


# =====================================================================
# Private vector field rotation matrices — all (..., 3, 3)
# =====================================================================
def _rotation_cart_to_sph(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Build the Cartesian->Spherical rotation matrix for each point.

    The Jacobian matrix J maps (vx, vy, vz) -> (v_rho, v_theta, v_phi):
        | sin(theta)cos(phi)   sin(theta)sin(phi)   cos(theta) |
        | cos(theta)cos(phi)   cos(theta)sin(phi)  -sin(theta) |
        | -sin(phi)            cos(phi)              0          |

    Parameters
    ----------
    theta, phi : array_like, shape (...)
        Spherical angles at each point.

    Returns
    -------
    R : np.ndarray, shape (..., 3, 3)
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi),   np.cos(phi)
    z      = np.zeros_like(theta)

    # Build (..., 3, 3)
    R = np.stack([
        np.stack([st * cp, st * sp,  ct], axis=-1),
        np.stack([ct * cp, ct * sp, -st], axis=-1),
        np.stack([-sp,     cp,        z], axis=-1),
    ], axis=-2)
    return R


def _rotation_cart_to_cyl(phi: np.ndarray) -> np.ndarray:
    """
    Build the Cartesian->Cylindrical rotation matrix for each point.

        |  cos(phi)  sin(phi)  0 |
        | -sin(phi)  cos(phi)  0 |
        |  0         0         1 |

    Parameters
    ----------
    phi : array_like, shape (...)

    Returns
    -------
    R : np.ndarray, shape (..., 3, 3)
    """
    cp, sp = np.cos(phi), np.sin(phi)
    z = np.zeros_like(phi)
    o = np.ones_like(phi)

    R = np.stack([
        np.stack([ cp, sp, z], axis=-1),
        np.stack([-sp, cp, z], axis=-1),
        np.stack([  z,  z, o], axis=-1),
    ], axis=-2)
    return R


# =====================================================================
# Public API: convert_vectors
# =====================================================================
def convert_vectors(
    pos: np.ndarray,
    vec: np.ndarray,
    from_sys: str,
    to_sys: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate vector fields between coordinate systems.

    Converts both the positions and the associated vectors (forces,
    velocities, etc.) from one coordinate system to another.

    Parameters
    ----------
    pos : array_like, shape (..., 3)
        Positions in the *source* coordinate system.
    vec : array_like, shape (..., 3)
        Vector field values in the *source* coordinate system.
    from_sys, to_sys : {'cart', 'sph', 'cyl'}
        Source and target coordinate systems.

    Returns
    -------
    pos_new : np.ndarray, shape (..., 3)
        Positions in the target coordinate system.
    vec_new : np.ndarray, shape (..., 3)
        Vectors rotated into the target coordinate system.

    Examples
    --------
    >>> pos_sph, F_sph = convert_vectors(pos_cart, F_cart, 'cart', 'sph')
    >>> pos_cart, v_cart = convert_vectors(pos_cyl, v_cyl, 'cyl', 'cart')
    """
    from_sys = from_sys.lower()
    to_sys = to_sys.lower()

    if from_sys not in _VALID_SYSTEMS:
        raise ValueError(f"from_sys must be one of {_VALID_SYSTEMS}, got '{from_sys}'")
    if to_sys not in _VALID_SYSTEMS:
        raise ValueError(f"to_sys must be one of {_VALID_SYSTEMS}, got '{to_sys}'")

    pos = _as_3vec(pos)
    vec = _as_3vec(vec)

    if from_sys == to_sys:
        return np.array(pos), np.array(vec)

    # Convert positions
    pos_new = convert_coords(pos, from_sys, to_sys)

    # Get angles in the appropriate system for the rotation matrix.
    # Strategy: always build the rotation from Cartesian, then compose
    # the forward or inverse transforms as needed.
    #
    # For cart->sph or sph->cart: use spherical angles (theta, phi)
    # For cart->cyl or cyl->cart: use cylindrical phi
    # For sph<->cyl: chain through cart

    if {from_sys, to_sys} == {"cart", "sph"}:
        # Need spherical angles — get them from whichever side has them
        if from_sys == "cart":
            sph = pos_new  # we just converted cart -> sph
        else:
            sph = pos      # input is already sph

        theta, phi = sph[..., 1], sph[..., 2]
        R = _rotation_cart_to_sph(theta, phi)  # (..., 3, 3)

        if from_sys == "cart":
            # R @ vec_cart = vec_sph
            vec_new = np.einsum("...ij,...j->...i", R, vec)
        else:
            # R^T @ vec_sph = vec_cart
            vec_new = np.einsum("...ji,...j->...i", R, vec)

    elif {from_sys, to_sys} == {"cart", "cyl"}:
        if from_sys == "cart":
            cyl = pos_new
        else:
            cyl = pos

        phi = cyl[..., 1]
        R = _rotation_cart_to_cyl(phi)  # (..., 3, 3)

        if from_sys == "cart":
            vec_new = np.einsum("...ij,...j->...i", R, vec)
        else:
            vec_new = np.einsum("...ji,...j->...i", R, vec)

    else:
        # sph <-> cyl: chain through Cartesian
        if from_sys == "sph":
            pos_cart, vec_cart = convert_vectors(pos, vec, "sph", "cart")
            _, vec_new = convert_vectors(pos_cart, vec_cart, "cart", "cyl")
        else:
            pos_cart, vec_cart = convert_vectors(pos, vec, "cyl", "cart")
            _, vec_new = convert_vectors(pos_cart, vec_cart, "cart", "sph")

    return pos_new, vec_new


# =====================================================================
# Public API: convert_to_vel_los
# =====================================================================
def convert_to_vel_los(
    xv: np.ndarray,
    reference_xv: np.ndarray | list | None = None,
) -> np.ndarray | float:
    """
    Compute line-of-sight (radial) velocity from Galactocentric phase-space
    coordinates.

    Projects the velocity vector onto the unit radial direction:
    ``v_los = v . r_hat``.  Optionally subtracts a reference point first
    (e.g. progenitor phase-space vector).

    Parameters
    ----------
    xv : array_like, shape (..., 6)
        Phase-space coordinates [x, y, z, vx, vy, vz].
        Positions in kpc, velocities in km/s.
        Accepts (6,), (N, 6), (M, N, 6), or any higher batch dims.
    reference_xv : array_like or None, optional
        Reference point(s) to subtract before computing v_los.
        Must be broadcastable to the shape of *xv*.
        None (default) means no subtraction.

    Returns
    -------
    v_los : float or np.ndarray
        Line-of-sight velocity in km/s.  Scalar when input is (6,),
        otherwise shape (...).

    Examples
    --------
    >>> xv = np.array([8.0, 0.0, 0.0, 0.0, 220.0, 0.0])
    >>> convert_to_vel_los(xv)
    0.0

    >>> xv = np.random.randn(100, 6)
    >>> v_los = convert_to_vel_los(xv)           # shape (100,)

    >>> xv = np.random.randn(5, 1000, 6)
    >>> ref = np.random.randn(5, 1, 6)           # broadcasts over N
    >>> v_los = convert_to_vel_los(xv, ref)      # shape (5, 1000)
    """
    xv = np.asarray(xv, dtype=float)
    if xv.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {xv.shape[-1]}")

    if reference_xv is not None:
        reference_xv = np.asarray(reference_xv, dtype=float)
        if reference_xv.size > 0:
            xv = xv - reference_xv

    pos = xv[..., :3]
    vel = xv[..., 3:6]

    r_mag = np.linalg.norm(pos, axis=-1, keepdims=True)
    if not np.all(r_mag > 0):
        raise ValueError("Position vectors cannot have zero magnitude")

    v_los = np.sum(vel * (pos / r_mag), axis=-1)

    if xv.ndim == 1:
        return float(v_los)
    return v_los