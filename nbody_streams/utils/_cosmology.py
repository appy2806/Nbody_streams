"""
nbody_streams.utils._cosmology
===============================

Flat LCDM background and the comoving/peculiar frame transforms.

Units are Gyr, kpc, km/s throughout.

The comoving/peculiar equation of motion is an exact change of variables of the
physical one.  With ``r = a x``, ``v_pec = a x'`` and ``H = a'/a``,

.. code-block:: text

    v_pec' = -grad(Phi) - H v_pec - a'' x - u_dot
      ==>  r'' = (H' + H^2 - a''/a) r - grad(Phi) - u_dot,

and ``H' = a''/a - H^2`` makes the bracket vanish identically, leaving

.. code-block:: text

    r'' = -grad(Phi)(r, t) - u_dot(t).

Phi is absent from the cancellation, so this holds for an evolving potential as
much as a static one: ``a(t)`` and ``H(t)`` are needed only to convert
coordinates in and out, and ``a''(t)`` not at all.  The centre term ``u_dot``
does *not* cancel -- dropping it moves 5 Gyr endpoints by tens of kpc.  See
:func:`~nbody_streams.agama_helper.center_acceleration` for that term.

This module is NumPy only, apart from ``scipy.optimize.curve_fit`` inside
:meth:`FlatLCDM.fit` and the pandas-backed table reader that
:meth:`FlatLCDM.from_snapshot_times` defers to.

Examples
--------
>>> from nbody_streams import utils
>>> cosmo = utils.FlatLCDM.from_snapshot_times(sim_dir)
>>> r0, v0 = utils.comoving_to_physical(x0, v_pec0, 13.78, cosmo)
>>> x, v_pec = utils.physical_to_comoving(xv[..., :3], xv[..., 3:], t, cosmo)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

__all__ = [
    "KPC_PER_GYR_PER_KMS",
    "FlatLCDM",
    "comoving_to_physical",
    "physical_to_comoving",
]

#: (kpc/Gyr) per (km/s), i.e. 1 Gyr = 1.0227 kpc/(km/s).
KPC_PER_GYR_PER_KMS = 1.0227121650537077


# ---------------------------------------------------------------------------
# Flat LCDM background
# ---------------------------------------------------------------------------

class FlatLCDM:
    r"""
    Flat LCDM background in closed form.

    .. code-block:: text

        a(t) = (Om/Ol)^(1/3) sinh(1.5 H0 sqrt(Ol) t)^(2/3)
        H(a) = H0 sqrt(Om/a^3 + Ol)
        t(a) = arcsinh(sqrt(Ol/Om) a^(3/2)) / (1.5 H0 sqrt(Ol))
        z(a) = 1/a - 1

    Parameters
    ----------
    hubble : float, optional
        Dimensionless h, with ``H0 = 100 h km/s/Mpc``, by default 0.702 (FIRE m12i).
    omega_matter : float, optional
        Present-day matter density, by default 0.272; ``omega_lambda = 1 - omega_matter``.

    Attributes
    ----------
    H0 : float
        Hubble constant in (km/s)/kpc, i.e. ``0.1 * hubble``.
    omega_lambda : float
        ``1 - omega_matter``.

    See Also
    --------
    comoving_to_physical, physical_to_comoving
    nbody_streams.agama_helper.center_acceleration

    Notes
    -----
    Any flat LCDM, not just FIRE's: against ``astropy.cosmology.FlatLambdaCDM``
    with ``Tcmb0=0``, for h in 0.5-0.8 and Om in 0.15-0.5, ``a`` agrees to 8e-12,
    ``H`` to 7e-16 and ``t(z)`` to 1.1e-10 Gyr.  Assumes flat
    (``Ol = 1 - Om``), ``w = -1`` and no radiation.

    The missing radiation is not an approximation *here*, because gizmo's own
    background omits it too: the m12i ``snapshot_times.txt`` a(t) matches this
    closed form to 5.2e-7, against 1.5e-3 for a radiation-inclusive model.  For
    a code that does include radiation the age would be 6.0 Myr high at z = 0,
    2.3 Myr at z = 10 and 4 percent at z = 100, and refitting (h, Om) is a poor
    remedy -- it recovers only a factor of 3 in a(t) while making H(a) worse.
    Radiation keeps H(a) and a''(a) elementary,

    .. code-block:: text

        H(a)    = H0 sqrt(Or/a^4 + Om/a^3 + Ol)
        a''(a)  = -(H0^2 a / 2) [Om/a^3 + 2 Or/a^4 - 2 Ol],

    but not t(a): the age integral becomes
    ``(1/H0) int a da / sqrt(Ol a^4 + Om a + Or)``, an elliptic integral.  The
    substitution ``x = a^(3/2)`` that produces the arcsinh above only collapses
    the quartic when ``Or = 0``.  Quadrature plus monotone inversion is the way
    in if it is ever needed.

    Agrees with the FIRE m12i ``CosmologyClass`` to 3e-9 in ``a`` and 8e-9 in
    ``H``.  The closed form matters for ``a''``: differentiating a spline
    through tabulated a(t) twice gives errors of order the value itself.

    Examples
    --------
    >>> cosmo = FlatLCDM()                       # FIRE m12i defaults
    >>> float(cosmo.scale_factor(13.8))
    1.0002...
    >>> float(cosmo.time(a=1.0))
    13.79...
    """

    def __init__(self, hubble: float = 0.702, omega_matter: float = 0.272):
        self.hubble = float(hubble)
        self.omega_matter = float(omega_matter)
        self.omega_lambda = 1.0 - self.omega_matter
        self.H0 = 0.1 * self.hubble          # (km/s)/kpc
        self._static = False

    # -- constructors -------------------------------------------------------

    @classmethod
    def static(cls) -> "FlatLCDM":
        """
        Degenerate background with ``a = 1``, ``H = 0``, for a non-cosmological
        potential.

        The comoving/peculiar frame then coincides with the physical one and
        both of :func:`comoving_to_physical` / :func:`physical_to_comoving`
        become the identity.

        Returns
        -------
        FlatLCDM
        """
        obj = cls(hubble=0.0, omega_matter=1.0)
        obj._static = True
        return obj

    @classmethod
    def fit(cls, t, a, t_min: float = 0.5) -> "FlatLCDM":
        """
        Least-squares fit of ``(hubble, omega_matter)`` to a tabulated a(t).

        Parameters
        ----------
        t : array_like, shape (n,)
            Times [Gyr].
        a : array_like, shape (n,)
            Scale factors at those times.
        t_min : float, optional
            Skip rows below this time in Gyr, where radiation matters, by
            default 0.5.

        Returns
        -------
        FlatLCDM

        Raises
        ------
        ValueError
            If *t* and *a* have different lengths, or fewer than two rows
            survive the *t_min* cut.
        """
        from scipy.optimize import curve_fit

        t = np.asarray(t, dtype=float).ravel()
        a = np.asarray(a, dtype=float).ravel()
        if t.size != a.size:
            raise ValueError(f"t and a must be the same length; got {t.size} and {a.size}.")

        keep = np.isfinite(t) & np.isfinite(a) & (t > float(t_min))
        if keep.sum() < 2:
            raise ValueError(
                f"only {int(keep.sum())} finite sample(s) with t > {t_min} Gyr; "
                "nothing to fit. Lower t_min or check the table."
            )

        params, _ = curve_fit(
            lambda tt, h, om: cls(h, om).scale_factor(tt),
            t[keep], a[keep], p0=[0.7, 0.27], maxfev=20000,
        )
        return cls(*params)

    @classmethod
    def from_snapshot_times(
        cls,
        sim_dir: Union[str, Path],
        t_min: float = 0.5,
    ) -> "FlatLCDM":
        """
        Fit ``(hubble, omega_matter)`` to the a(t) table in ``snapshot_times.txt``.

        Columns are located by
        :func:`~nbody_streams.agama_helper.read_snapshot_times`, which is
        header-driven with a statistical fallback -- so this does not depend on
        the column order of any one FIRE run.

        Parameters
        ----------
        sim_dir : str or Path
            Directory containing ``snapshot_times.txt``.
        t_min : float, optional
            Skip rows below this time in Gyr, where radiation matters, by
            default 0.5.

        Returns
        -------
        FlatLCDM
            Recovers h = 0.702, Om = 0.272 for m12i, reproducing the table to 5e-7.

        Raises
        ------
        ImportError
            If ``pandas`` is not installed (needed by the table reader).
        FileNotFoundError
            If ``snapshot_times.txt`` is not found in *sim_dir*.
        """
        # Deferred: keeps utils importable without touching the agama_helper
        # subpackage, and pandas is a lazy dependency of the reader.
        from ..agama_helper import read_snapshot_times

        df = read_snapshot_times(sim_dir)
        return cls.fit(df["time[Gyr]"].values, df["scale-factor"].values, t_min=t_min)

    # -- background functions ----------------------------------------------

    def _a_of(self, t=None, a=None, z=None):
        """Scale factor from exactly one of ``t`` [Gyr], ``a`` or ``z``."""
        if sum(x is not None for x in (t, a, z)) != 1:
            raise ValueError("give exactly one of t, a, z")
        if a is not None:
            return np.asarray(a, dtype=float)
        if z is not None:
            z = np.asarray(z, dtype=float)
            return np.ones_like(z) if self._static else 1.0 / (1.0 + z)
        return self.scale_factor(t)

    def scale_factor(self, t=None, z=None):
        """
        Scale factor ``a`` at time *t* [Gyr] or redshift *z*.

        ``a(t) = (Om/Ol)^(1/3) sinh(1.5 H0 sqrt(Ol) t)^(2/3)``, or the identity
        ``a = 1/(1+z)``.

        Parameters
        ----------
        t : array_like, optional
            Time [Gyr].  Exactly one of *t*, *z*.
        z : array_like, optional
            Redshift.

        Returns
        -------
        ndarray
            Scale factor, same shape as the input.
        """
        if (t is None) == (z is None):
            raise ValueError("give exactly one of t, z")
        if z is not None:
            z = np.asarray(z, dtype=float)
            return np.ones_like(z) if self._static else 1.0 / (1.0 + z)
        if self._static:
            return np.ones_like(np.asarray(t, dtype=float))
        arg = (1.5 * self.H0 * KPC_PER_GYR_PER_KMS * np.sqrt(self.omega_lambda)
               * np.asarray(t, dtype=float))
        return (self.omega_matter / self.omega_lambda) ** (1 / 3) * np.sinh(arg) ** (2 / 3)

    def time(self, a=None, z=None):
        """
        Age of the universe [Gyr] at scale factor *a* or redshift *z*.

        The exact inverse of :meth:`scale_factor`: from
        ``a = (Om/Ol)^(1/3) sinh(1.5 H0 sqrt(Ol) t)^(2/3)``,

        .. code-block:: text

            t = arcsinh(sqrt(Ol/Om) a^(3/2)) / (1.5 H0 sqrt(Ol)),

        so no root-finding or spline is involved.

        Parameters
        ----------
        a : array_like, optional
            Scale factor.  Exactly one of *a*, *z*.
        z : array_like, optional
            Redshift; ``a = 1/(1+z)``.

        Returns
        -------
        ndarray
            Time [Gyr].
        """
        if (a is None) == (z is None):
            raise ValueError("give exactly one of a, z")
        a = 1.0 / (1.0 + np.asarray(z, dtype=float)) if a is None \
            else np.asarray(a, dtype=float)
        if self._static:
            return np.zeros_like(a)
        return (np.arcsinh(np.sqrt(self.omega_lambda / self.omega_matter) * a ** 1.5)
                / (1.5 * self.H0 * KPC_PER_GYR_PER_KMS * np.sqrt(self.omega_lambda)))

    def redshift(self, t=None, a=None):
        """
        ``z = 1/a - 1``, from time *t* [Gyr] or from *a* directly.

        Parameters
        ----------
        t : array_like, optional
            Time [Gyr].  Exactly one of *t*, *a*.
        a : array_like, optional
            Scale factor.

        Returns
        -------
        ndarray
            Redshift.
        """
        return 1.0 / self._a_of(t, a) - 1.0

    def hubble_parameter(self, t=None, a=None, z=None):
        """
        ``H = H0 sqrt(Om/a^3 + Ol)`` [(km/s)/kpc].

        Parameters
        ----------
        t : array_like, optional
            Time [Gyr].
        a : array_like, optional
            Scale factor.
        z : array_like, optional
            Redshift.  Exactly one of *t*, *a*, *z*.

        Returns
        -------
        ndarray
            Hubble parameter [(km/s)/kpc].
        """
        a = self._a_of(t, a, z)
        if self._static:
            return np.zeros_like(np.asarray(a, dtype=float))
        return self.H0 * np.sqrt(self.omega_matter / a ** 3 + self.omega_lambda)

    def a_double_dot(self, t=None, a=None, z=None):
        """
        ``a'' = -a H0^2 (Om/a^3 - 2 Ol) / 2`` in ((km/s)/kpc)^2.

        Cancels out of the physical equation of motion (see the module
        docstring); kept for cross-checks against the comoving formulation.

        Parameters
        ----------
        t : array_like, optional
            Time [Gyr].
        a : array_like, optional
            Scale factor.
        z : array_like, optional
            Redshift.  Exactly one of *t*, *a*, *z*.

        Returns
        -------
        ndarray
            ``a''`` in ((km/s)/kpc)^2.
        """
        a = self._a_of(t, a, z)
        if self._static:
            return np.zeros_like(np.asarray(a, dtype=float))
        return -0.5 * a * self.H0 ** 2 * (self.omega_matter / a ** 3
                                          - 2 * self.omega_lambda)

    def __repr__(self) -> str:
        if self._static:
            return "FlatLCDM.static()"
        return f"FlatLCDM(hubble={self.hubble:.6g}, omega_matter={self.omega_matter:.6g})"


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def _aH(cosmo: FlatLCDM, t, ndim: int):
    """a(t) and H(t), shaped to broadcast against a trailing axis of 3."""
    a = cosmo.scale_factor(t)
    H = cosmo.hubble_parameter(a=a)
    if ndim > 1:
        return np.asarray(a, float)[..., None], np.asarray(H, float)[..., None]
    return a, H


def comoving_to_physical(x, v_pec, t, cosmo: FlatLCDM):
    """
    Comoving position and peculiar velocity to physical.

    ``r = a x``, ``v = H r + v_pec``.

    Parameters
    ----------
    x : array_like, shape (3,) or (..., 3)
        Comoving position [kpc].
    v_pec : array_like, shape (3,) or (..., 3)
        Peculiar velocity [km/s].
    t : float or array_like
        Time [Gyr], broadcastable against ``x[..., 0]``.
    cosmo : FlatLCDM
        Background cosmology.

    Returns
    -------
    r : ndarray
        Physical position [kpc].
    v : ndarray
        Physical velocity [km/s].

    See Also
    --------
    physical_to_comoving : the exact inverse.
    """
    x = np.asarray(x, dtype=float)
    a, H = _aH(cosmo, t, x.ndim)
    r = x * a
    return r, np.asarray(v_pec, dtype=float) + H * r


def physical_to_comoving(r, v, t, cosmo: FlatLCDM):
    """
    Physical position and velocity to comoving position and peculiar velocity.

    ``x = r/a``, ``v_pec = v - H r``.

    Parameters
    ----------
    r : array_like, shape (3,) or (..., 3)
        Physical position [kpc].
    v : array_like, shape (3,) or (..., 3)
        Physical velocity [km/s].
    t : float or array_like
        Time [Gyr], broadcastable against ``r[..., 0]``.
    cosmo : FlatLCDM
        Background cosmology.

    Returns
    -------
    x : ndarray
        Comoving position [kpc].
    v_pec : ndarray
        Peculiar velocity [km/s].

    See Also
    --------
    comoving_to_physical : the exact inverse.
    """
    r = np.asarray(r, dtype=float)
    a, H = _aH(cosmo, t, r.ndim)
    return r / a, np.asarray(v, dtype=float) - H * r
