"""
agama_helper._coefs
~~~~~~~~~~~~~~~~~~~
Structured representations of Agama expansion coefficient tables.

Two expansion types are supported:

* **Multipole** (spherical harmonic BFE): :class:`MultipoleCoefs`
* **CylSpline** (azimuthal harmonic + 2-D spline BFE): :class:`CylSplineCoefs`

Both dataclasses expose:

- ``.zeroed(keep=...)``       — return a modified copy with unselected terms zeroed
- ``.to_coef_string()``       — round-trip back to the Agama text format

Parsing entrypoints:

- :func:`read_mult_coefs`     — file path or raw coef string → :class:`MultipoleCoefs`
- :func:`read_cylspl_coefs`   — file path or raw coef string → :class:`CylSplineCoefs`

Both parsers accept either a filesystem path *or* the raw text content of the
file, so they work identically whether the string was read from disk or loaded
from an HDF5 archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import h5py
import numpy as np

from ._io import _extract_int_from_group, _resolve_coef_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_lmax_pairs(lmax: int, mmax: int | None = None) -> list[tuple[int, int]]:
    """
    Generate (l, m) pairs for a spherical harmonic expansion.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum order (l ≥ 0).
    mmax : int, optional
        If given, the azimuthal order is capped at ``min(l, mmax)`` for each l.

    Returns
    -------
    list of (int, int)
        Sorted by l then m (non-negative m only).

    Examples
    --------
    >>> generate_lmax_pairs(2)
    [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

    >>> generate_lmax_pairs(2, mmax=1)
    [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
    """
    assert lmax >= 0, "lmax must be >= 0"
    if mmax is not None:
        assert mmax >= 0, "mmax must be >= 0 when specified"
    return [
        (l, m)
        for l in range(lmax + 1)
        for m in range(min(l, mmax) + 1 if mmax is not None else l + 1)
    ]


def _add_negative_m(lm_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Expand (l, m) pairs to include their negative-m counterparts.

    Each (l, m) with m > 0 gains a companion (l, -m).  The result is
    de-duplicated and sorted by l then m.

    Parameters
    ----------
    lm_pairs : list of (int, int)

    Returns
    -------
    list of (int, int)
        Expanded, sorted list.
    """
    expanded: set[tuple[int, int]] = set()
    for l, m in lm_pairs:
        expanded.add((l, m))
        if m != 0:
            expanded.add((l, -m))
    return sorted(expanded, key=lambda pair: (pair[0], pair[1]))


def _source_to_lines(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> list[str]:
    """
    Return lines from any coef source: file path, HDF5 path, or raw string.

    Delegates source resolution to :func:`~agama_helper._io._resolve_coef_string`.
    """
    return _resolve_coef_string(source, group_name, dataset_name).splitlines()


#: Section markers an Agama CylSpline export may carry, keyed by the lower-cased
#: marker text.  A CylSpline written by ``Potential.export()`` always carries all
#: three; older or hand-made files may carry ``#Phi`` alone.
_CYLSPL_SECTIONS: dict[str, str] = {
    "#phi": "phi",
    "#dphi/dr": "dphi_dR",
    "#dphi/dz": "dphi_dz",
}


def _split_cylspl_sections(lines: list[str]) -> dict[str, tuple[int, int]]:
    """
    Split a CylSpline line stream into ``#Phi`` / ``#dPhi/dR`` / ``#dPhi/dz`` blocks.

    Returns
    -------
    dict
        Maps ``"phi"``/``"dphi_dR"``/``"dphi_dz"`` to a ``(start, stop)`` pair of
        line indices bounding that section's body (marker line excluded).

    Notes
    -----
    ``#dPhi/dR`` and ``#dPhi/dz`` differ only in their final character, so the
    lower-cased comparison in :data:`_CYLSPL_SECTIONS` stays unambiguous.  If no
    marker is present at all, the whole stream is reported as the ``phi``
    section — this keeps hand-made single-section files readable.
    """
    marks: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        key = _CYLSPL_SECTIONS.get(line.strip().lower())
        if key is not None:
            marks.append((key, i))

    if not marks:
        return {"phi": (0, len(lines))}

    bounds: dict[str, tuple[int, int]] = {}
    for j, (key, idx) in enumerate(marks):
        stop = marks[j + 1][1] if j + 1 < len(marks) else len(lines)
        bounds[key] = (idx + 1, stop)
    return bounds


def _detect_expansion_type(coef_string: str) -> str:
    """Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown."""
    for line in coef_string.splitlines()[:15]:
        s = line.strip()
        if s.startswith("type=") or s.startswith("type ="):
            return s.split("=", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Optional time axis — shared machinery
# ---------------------------------------------------------------------------

def _as_times(times) -> np.ndarray:
    """Coerce *times* to a 1-D float64 array, rejecting anything else."""
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"times must be a 1-D sequence; got an array of shape {arr.shape}."
        )
    return arr


def _as_existing_path(source) -> Path | None:
    """Return *source* as a :class:`Path` when it names a filesystem entry."""
    if isinstance(source, Path):
        return source
    s = str(source)
    if "\n" in s:
        return None
    try:
        p = Path(s)
        return p if p.exists() else None
    except (OSError, ValueError):
        return None


class _CoefTimeAxisMixin:
    """
    Time-axis behaviour shared by :class:`MultipoleCoefs` and :class:`CylSplineCoefs`.

    Time is always the **last** array axis and is always optional.  The ``times``
    field is authoritative: ``has_time_axis`` is ``times is not None``, and
    :meth:`validate` is what checks that the coefficient arrays agree with it.

    Subclasses supply ``snapshot``, ``validate``, ``to_coef_string`` and
    ``_COEF_EXT``.
    """

    #: Filename extension used by :meth:`to_coef_files`.
    _COEF_EXT: str = ".coef"

    # --- time-axis introspection ------------------------------------------

    @property
    def has_time_axis(self) -> bool:
        """``True`` when this object carries a trailing time axis."""
        return self.times is not None

    @property
    def n_times(self) -> int | None:
        """Number of time samples, or ``None`` without a time axis."""
        return None if self.times is None else int(len(self.times))

    def _time_index(self, i) -> int:
        """Normalise a time index, honouring negative indexing."""
        if not self.has_time_axis:
            raise ValueError(
                f"{type(self).__name__} has no time axis (times is None), so time "
                f"index {i!r} is meaningless. Attach one with with_times() or "
                "build a series with stack_coefs()/read_coefs(group_name='all')."
            )
        nt = self.n_times
        idx = int(i)
        if idx < 0:
            idx += nt
        if not 0 <= idx < nt:
            raise IndexError(f"time index {i} is out of range for n_times={nt}.")
        return idx

    def __getitem__(self, i):
        """Alias of :meth:`snapshot` — ``coefs[3]`` is the snapshot at t index 3."""
        return self.snapshot(i)

    # --- writing ----------------------------------------------------------

    def to_coef_strings(self) -> list[str]:
        """
        Serialise every time sample.

        Returns
        -------
        list of str
            One Agama coef string per time.  A one-element list when this object
            has no time axis.
        """
        self.validate()
        if not self.has_time_axis:
            return [self.to_coef_string()]
        return [self.to_coef_string(t=i) for i in range(self.n_times)]

    def to_coef_files(
        self,
        out_dir: Union[str, Path],
        name_fmt: str = "snap_{i:04d}{ext}",
    ) -> list[str]:
        """
        Write one plain-text Agama coefficient file per time sample.

        Parameters
        ----------
        out_dir : str or Path
            Destination directory; created if absent.
        name_fmt : str, optional
            ``str.format`` template receiving ``i`` (the time index) and ``ext``
            (``".coef_mult"`` or ``".coef_cylsp"``).

        Returns
        -------
        list of str
            Absolute paths of the written files, in time order.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, coef_str in enumerate(self.to_coef_strings()):
            p = out / name_fmt.format(i=i, ext=self._COEF_EXT)
            p.write_text(coef_str, encoding="utf-8")
            paths.append(str(p.resolve()))
        return paths

    def to_h5(
        self,
        path: Union[str, Path],
        group_fmt: str = "snap_{i:04d}",
        dataset_name: str = "coefs",
        overwrite: bool = True,
        write_times: bool = True,
    ) -> str:
        """
        Write every time sample into one HDF5 archive.

        The layout matches :func:`~agama_helper._io.write_snapshot_coefs_to_h5`,
        so the result round-trips through ``read_coefs(..., group_name="all")``
        and loads directly with
        :func:`~agama_helper._load.load_agama_evolving_potential`.

        Parameters
        ----------
        path : str or Path
            Destination archive.  Opened in append mode when it already exists.
        group_fmt : str, optional
            ``str.format`` template receiving the time index ``i``.
        dataset_name : str, optional
            Dataset name within each group, by default ``"coefs"``.
        overwrite : bool, optional
            Replace an existing dataset of the same name, by default ``True``.
        write_times : bool, optional
            Also write the root-level ``"times"`` dataset, by default ``True``.
            Ignored without a time axis.

        Returns
        -------
        str
            The archive path.
        """
        from ._io import write_coef_to_h5

        path = Path(path)
        for i, coef_str in enumerate(self.to_coef_strings()):
            write_coef_to_h5(
                path,
                coef_str,
                group_name=group_fmt.format(i=i),
                dataset_name=dataset_name,
                overwrite=overwrite,
            )
        if write_times and self.has_time_axis:
            with h5py.File(path, "a") as f:
                if "times" in f:
                    del f["times"]
                f.create_dataset("times", data=np.asarray(self.times, dtype=float))
        return str(path)

    def to_evolving_ini(
        self,
        ini_path: Union[str, Path],
        out_dir: Union[str, Path, None] = None,
        interp_linear: bool = True,
    ) -> str:
        """
        Write the coefficient files plus an Agama ``Evolving`` ``.ini`` config.

        Parameters
        ----------
        ini_path : str or Path
            Destination ``.ini`` file.
        out_dir : str or Path, optional
            Directory for the per-snapshot coef files; defaults to *ini_path*'s
            parent.  File names are prefixed with the ``.ini`` stem.
        interp_linear : bool, optional
            Linear (``True``, default) vs. cubic-spline interpolation in time.

        Returns
        -------
        str
            Absolute path of the written ``.ini``.
        """
        self.validate()
        if not self.has_time_axis:
            raise ValueError(
                f"{type(self).__name__}.to_evolving_ini() needs a time axis: an "
                "Evolving config is a list of (time, file) pairs and this object "
                "has times=None."
            )
        from ._load import create_evolving_ini

        ini_path = Path(ini_path)
        out = Path(out_dir) if out_dir is not None else ini_path.parent
        paths = self.to_coef_files(out, name_fmt=ini_path.stem + "_{i:04d}{ext}")
        return create_evolving_ini(
            list(np.asarray(self.times, dtype=float)),
            paths,
            ini_path,
            interp_linear=interp_linear,
        )

    # --- materialisation --------------------------------------------------

    def materialize_potential(self, *, center=None, interp_linear: bool = True, gpu: bool = False):
        """
        Turn these coefficients into a live potential object.

        Without a time axis this delegates to
        :func:`~agama_helper._load.load_agama_potential`; with one it delegates to
        :func:`~agama_helper._load.load_agama_evolving_potential` using
        ``times=self.times``.  Either way the *live* arrays are serialised, so
        any manual surgery done on ``phi`` is picked up automatically.

        Parameters
        ----------
        center : array-like or str or Path, optional
            Galactic centre — same forms as
            :func:`~agama_helper._load.load_agama_potential`.
        interp_linear : bool, optional
            Time interpolation mode.  Ignored without a time axis.
        gpu : bool, optional
            Return a GPU potential instead of an ``agama.Potential``.

        Returns
        -------
        agama.Potential or a GPU potential object
        """
        self.validate()
        from . import _load

        if not self.has_time_axis:
            return _load.load_agama_potential(self, center=center, gpu=gpu)
        return _load.load_agama_evolving_potential(
            self,
            times=np.asarray(self.times, dtype=float),
            center=center,
            interp_linear=interp_linear,
            gpu=gpu,
        )


# ---------------------------------------------------------------------------
# MultipoleCoefs
# ---------------------------------------------------------------------------

@dataclass
class MultipoleCoefs(_CoefTimeAxisMixin):
    """
    Structured representation of a Multipole (spherical harmonic BFE) potential.

    Holds either a single snapshot or a whole time series.  Time, when present,
    is the **last** array axis.

    Attributes
    ----------
    R_grid : ndarray, shape (nR,)
        Radial grid points [kpc].  Identical at every time, by construction.
    lm_labels : list of (l, m)
        Ordered (l, m) pairs corresponding to columns in *phi* and *dphi_dr*.
        Identical at every time, by construction.
    phi : ndarray, shape (nR, n_lm) or (nR, n_lm, nt)
        Potential coefficients Φ_{l,m}(r), with the trailing axis present only
        when *times* is set.
    dphi_dr : ndarray or None
        Radial derivative coefficients ∂Φ/∂r, same shape as *phi*.  ``None`` if
        absent in the source file.
    metadata : dict
        Header parameters parsed from the coefficient file: ``lmax``,
        ``gridSizeR``, ``symmetry``, ``type``, etc.
    times : ndarray or None, shape (nt,)
        Sample times.  ``None`` (the default) means no time axis; the field is
        last so existing positional construction keeps working.
    """

    R_grid: np.ndarray
    lm_labels: list[tuple[int, int]]
    phi: np.ndarray
    dphi_dr: np.ndarray | None
    metadata: dict = field(default_factory=dict)
    times: np.ndarray | None = None

    _COEF_EXT = ".coef_mult"

    # --- convenience properties -------------------------------------------

    @property
    def lmax(self) -> int:
        """Maximum l order present in *lm_labels*."""
        return max(l for l, _ in self.lm_labels) if self.lm_labels else 0

    @property
    def l_values(self) -> list[int]:
        """Sorted unique l values."""
        return sorted({l for l, _ in self.lm_labels})

    @property
    def m_values(self) -> list[int]:
        """Sorted unique m values (includes negatives)."""
        return sorted({m for _, m in self.lm_labels})

    # --- analysis ---------------------------------------------------------

    def radial_power(self, l: int, use_quadrature: bool = True) -> np.ndarray:
        """
        Radial power spectrum for harmonic order *l*.

        Parameters
        ----------
        l : int
            Harmonic order.
        use_quadrature : bool, optional
            If ``True`` (default), power = Σ_m Φ_{l,m}²(r).
            If ``False``, power = Σ_m |Φ_{l,m}(r)|.

        Returns
        -------
        ndarray, shape (nR,) or (nR, nt)
            Power at each radial grid point, co-indexed with *R_grid*.  The
            trailing time axis is present iff this object carries one — the
            meaning of the quantity is unchanged.
        """
        cols = [i for i, (li, _) in enumerate(self.lm_labels) if li == l]
        if not cols:
            shape = (len(self.R_grid),)
            if self.has_time_axis:
                shape += (self.n_times,)
            return np.zeros(shape)
        block = self.phi[:, cols]
        return (block ** 2).sum(axis=1) if use_quadrature else np.abs(block).sum(axis=1)

    def total_power(self, l: int, use_quadrature: bool = True) -> Union[float, np.ndarray]:
        """
        Total power for harmonic order *l* summed over all radial bins.

        Parameters
        ----------
        l : int
            Harmonic order.
        use_quadrature : bool, optional
            Forwarded to :meth:`radial_power`.

        Returns
        -------
        float or ndarray, shape (nt,)
            A scalar without a time axis, one value per time with one.
        """
        radial = self.radial_power(l, use_quadrature)
        if not self.has_time_axis:
            return float(radial.sum())
        return radial.sum(axis=0)

    # --- modification -----------------------------------------------------

    def zeroed(self, keep_lm: list) -> "MultipoleCoefs":
        """
        Return a copy with all (l, m) terms **not** in *keep_lm* zeroed out.

        Negative-m counterparts of any (l, m) with m > 0 are added
        automatically via :func:`_add_negative_m`.

        Parameters
        ----------
        keep_lm : list of int or (int, int)
            Terms to retain.  Each element may be:

            * An ``int`` *l* — keep **all** (l, m) pairs present in the
              expansion for that angular order.  Convenient shorthand:
              ``keep_lm=[0, 2]`` keeps all monopole and quadrupole terms.
            * A ``(l, m)`` tuple — keep that specific harmonic pair.

            Mixing both forms is supported.  Raises :exc:`TypeError` for any
            element that is neither an int nor a 2-tuple of ints.

        Returns
        -------
        MultipoleCoefs
            New instance; *R_grid*, *lm_labels*, *metadata* are shared
            references; *phi* and *dphi_dr* are new arrays.
        """
        import warnings

        normalised: list[tuple[int, int]] = []
        for item in keep_lm:
            if isinstance(item, (int, np.integer)):
                l = int(item)
                found = [(li, m) for li, m in self.lm_labels if li == l]
                if not found:
                    warnings.warn(
                        f"l={l} is not present in this expansion; ignoring.",
                        stacklevel=2,
                    )
                normalised.extend(found)
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and all(isinstance(x, (int, np.integer)) for x in item)
            ):
                normalised.append((int(item[0]), int(item[1])))
            else:
                raise TypeError(
                    f"keep_lm elements must be an int l (all m for that l) "
                    f"or a (l, m) tuple of ints; got {type(item).__name__!r}: {item!r}"
                )

        keep_set = set(_add_negative_m(normalised))
        mask = np.array([lm in keep_set for lm in self.lm_labels])
        # Broadcast the (l, m) mask along axis 1 for a 2-D *or* 3-D phi.
        bcast = mask.reshape((1, mask.size) + (1,) * (np.ndim(self.phi) - 2))
        new_phi = np.where(bcast, self.phi, 0.0)
        new_dphi = (
            np.where(bcast, self.dphi_dr, 0.0)
            if self.dphi_dr is not None
            else None
        )
        return MultipoleCoefs(
            R_grid=self.R_grid,
            lm_labels=self.lm_labels,
            phi=new_phi,
            dphi_dr=new_dphi,
            metadata=self.metadata,
            times=self.times,
        )

    def copy(self) -> "MultipoleCoefs":
        """Return a deep copy — no array, list or dict is shared with *self*."""
        return MultipoleCoefs(
            R_grid=np.array(self.R_grid, copy=True),
            lm_labels=list(self.lm_labels),
            phi=np.array(self.phi, copy=True),
            dphi_dr=None if self.dphi_dr is None else np.array(self.dphi_dr, copy=True),
            metadata=dict(self.metadata),
            times=None if self.times is None else np.array(self.times, copy=True),
        )

    def column(self, l: int, m: int) -> int:
        """
        Index of the ``(l, m)`` column in *phi*, *dphi_dr* and *lm_labels*.

        Parameters
        ----------
        l, m : int
            Harmonic pair.  Negative *m* is a distinct column.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the pair is not present in this expansion.
        """
        try:
            return self.lm_labels.index((int(l), int(m)))
        except ValueError:
            raise KeyError(
                f"(l={l}, m={m}) is not present in this expansion. "
                f"lmax={self.lmax}, l values present: {self.l_values}."
            ) from None

    def snapshot(self, i: int) -> "MultipoleCoefs":
        """
        Time-less view at time index *i*.

        The returned object shares memory with *self* (``phi[..., i]`` is a
        NumPy view); call :meth:`copy` on it for an independent object.
        """
        ti = self._time_index(i)
        return MultipoleCoefs(
            R_grid=self.R_grid,
            lm_labels=self.lm_labels,
            phi=self.phi[..., ti],
            dphi_dr=None if self.dphi_dr is None else self.dphi_dr[..., ti],
            metadata=self.metadata,
        )

    def with_times(
        self,
        times,
        *,
        phi: np.ndarray | None = None,
        dphi_dr: np.ndarray | None = None,
    ) -> "MultipoleCoefs":
        """
        Attach or relabel the time axis.

        This is the hand-off point for user-computed resampling: compute your own
        interpolation of ``phi`` onto whatever new time grid you want — 100 steps
        to 1000 to 10 — and attach the result here.  Nothing is interpolated,
        smoothed or resampled by this method.

        Parameters
        ----------
        times : array-like, shape (nt,)
            New time samples.
        phi : ndarray, shape (nR, n_lm, nt), optional
            New coefficient block.  ``nt`` is taken from its trailing axis and
            must match ``len(times)``; the leading two axes must match the
            existing grid exactly.  When omitted, only the time labels change and
            ``len(times)`` must equal the current ``n_times``.
        dphi_dr : ndarray, optional
            New derivative block, same shape as *phi*.  Required whenever *phi*
            is supplied and *self* already carries a ``dphi_dr``.

        Returns
        -------
        MultipoleCoefs
        """
        t = _as_times(times)
        nR, n_lm = len(self.R_grid), len(self.lm_labels)

        if phi is None:
            if dphi_dr is not None:
                raise ValueError(
                    "dphi_dr can only be supplied together with phi; on its own "
                    "there is no time axis to attach it to."
                )
            if not self.has_time_axis:
                raise ValueError(
                    "with_times(times) without phi only relabels an existing time "
                    "axis, but this object has times=None. Supply phi=... to "
                    "attach a time axis."
                )
            if len(t) != self.n_times:
                raise ValueError(
                    f"with_times(times) without phi only relabels the time axis, so "
                    f"len(times)={len(t)} must equal the current n_times="
                    f"{self.n_times}. Supply phi=... to change the number of times."
                )
            new_phi, new_dphi = self.phi, self.dphi_dr
        else:
            new_phi = np.asarray(phi, dtype=float)
            if new_phi.ndim != 3 or new_phi.shape[:2] != (nR, n_lm):
                raise ValueError(
                    f"phi has shape {new_phi.shape}, expected "
                    f"({nR}, {n_lm}, nt) — the leading axes must match the "
                    "existing R_grid and lm_labels exactly; grids are never "
                    "interpolated."
                )
            if len(t) != new_phi.shape[-1]:
                raise ValueError(
                    f"len(times)={len(t)} does not match phi.shape[-1]="
                    f"{new_phi.shape[-1]}."
                )
            if dphi_dr is None:
                if self.dphi_dr is not None:
                    raise ValueError(
                        "phi was supplied but dphi_dr was not, and this object "
                        f"already carries a dphi_dr of shape "
                        f"{np.shape(self.dphi_dr)} which cannot be reconciled with "
                        f"the new phi of shape {new_phi.shape}. Pass a matching "
                        "dphi_dr, or drop the old one first (obj.dphi_dr = None) "
                        "— note that Agama cannot load a Multipole file without a "
                        "#dPhi/dr section."
                    )
                new_dphi = None
            else:
                new_dphi = np.asarray(dphi_dr, dtype=float)
                if new_dphi.shape != new_phi.shape:
                    raise ValueError(
                        f"dphi_dr has shape {new_dphi.shape}, expected "
                        f"{new_phi.shape} (same as phi)."
                    )

        out = MultipoleCoefs(
            R_grid=self.R_grid,
            lm_labels=self.lm_labels,
            phi=new_phi,
            dphi_dr=new_dphi,
            metadata=self.metadata,
            times=t,
        )
        out.validate()
        return out

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """
        Assert internal shape consistency, naming the offending field on failure.

        Direct field assignment is legal — this is what catches the damage.  It
        runs automatically at the top of every materialise and write entry point.

        Raises
        ------
        ValueError
            On any shape inconsistency between *R_grid*, *lm_labels*, *phi*,
            *dphi_dr* and *times*.
        """
        R = np.asarray(self.R_grid)
        if R.ndim != 1:
            raise ValueError(
                f"MultipoleCoefs.R_grid must be 1-D; got shape {R.shape}."
            )
        nR, n_lm = R.shape[0], len(self.lm_labels)

        phi = np.asarray(self.phi)
        if self.times is None:
            expected = (nR, n_lm)
            hint = f"(len(R_grid)={nR}, len(lm_labels)={n_lm}; times is None)"
        else:
            t = np.asarray(self.times)
            if t.ndim != 1:
                raise ValueError(
                    f"MultipoleCoefs.times must be 1-D; got shape {t.shape}."
                )
            expected = (nR, n_lm, t.shape[0])
            hint = (
                f"(len(R_grid)={nR}, len(lm_labels)={n_lm}, len(times)={t.shape[0]})"
            )
        if phi.shape != expected:
            raise ValueError(
                f"MultipoleCoefs.phi has shape {phi.shape}, expected {expected} "
                f"{hint}."
            )

        if self.dphi_dr is not None:
            d = np.asarray(self.dphi_dr)
            if d.shape != phi.shape:
                raise ValueError(
                    f"MultipoleCoefs.dphi_dr has shape {d.shape}, expected "
                    f"{phi.shape} (identical to phi)."
                )

    # --- serialisation ----------------------------------------------------

    def to_coef_string(self, t: int | None = None) -> str:
        """
        Serialise one snapshot back to the Agama Multipole text format.

        Parameters
        ----------
        t : int, optional
            Time index.  **Required** when this object carries a time axis, and
            rejected when it does not.  Use :meth:`to_coef_strings` for all
            times at once.

        Returns
        -------
        str
            Full text suitable for writing to a ``.coef_mult`` file or
            passing to :func:`~agama_helper._io._write_tmp_coef`.

        Raises
        ------
        ValueError
            If *t* is inconsistent with the presence of a time axis, or if
            *dphi_dr* is ``None`` — Agama fails with ``RuntimeError: Error
            loading Multipole potential`` on a ``#Phi``-only Multipole file, so
            writing one is refused rather than deferred to load time.
        """
        self.validate()
        if self.has_time_axis:
            if t is None:
                raise ValueError(
                    f"MultipoleCoefs carries a time axis (n_times={self.n_times}); "
                    "to_coef_string(t=<index>) needs an explicit time index. Use "
                    "to_coef_strings() to serialise every time."
                )
            ti = self._time_index(t)
            phi = self.phi[..., ti]
            dphi_dr = None if self.dphi_dr is None else self.dphi_dr[..., ti]
        else:
            if t is not None:
                raise ValueError(
                    f"MultipoleCoefs has no time axis (times is None), so t={t!r} "
                    "is meaningless; call to_coef_string() with no arguments."
                )
            phi, dphi_dr = self.phi, self.dphi_dr

        if dphi_dr is None:
            raise ValueError(
                "MultipoleCoefs.dphi_dr is None. Agama cannot load a Multipole "
                "coefficient file that carries only a #Phi section, so refusing "
                "to write one. Read from a '_DR' coefficient file, or attach "
                "dphi_dr before serialising."
            )

        meta = self.metadata
        lines: list[str] = [
            "[Potential]",
            f"type={meta.get('type', 'Multipole')}",
            f"gridSizeR={meta.get('gridSizeR', len(self.R_grid))}",
            f"lmax={meta.get('lmax', self.lmax)}",
            f"symmetry={meta.get('symmetry', 'None')}",
            "Coefficients",
        ]
        col_header = "#radius\t" + "\t".join(
            f"l={l},m={m}" for l, m in self.lm_labels
        )
        # #Phi section
        lines.append("#Phi")
        lines.append(col_header)
        for ri, r in enumerate(self.R_grid):
            row = [f"{r:.13g}"] + [f"{v:.13g}" for v in phi[ri]]
            lines.append("\t".join(row))
        # #dPhi/dr section
        lines.append("")
        lines.append("#dPhi/dr")
        lines.append(col_header)
        for ri, r in enumerate(self.R_grid):
            row = [f"{r:.13g}"] + [f"{v:.13g}" for v in dphi_dr[ri]]
            lines.append("\t".join(row))
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CylSplineCoefs
# ---------------------------------------------------------------------------

@dataclass
class CylSplineCoefs(_CoefTimeAxisMixin):
    """
    Structured representation of a CylSpline (azimuthal harmonic BFE) potential.

    The CylSpline expansion stores Φ(R, z) as a set of 2-D spline tables,
    one per azimuthal order m.  Holds either a single snapshot or a whole time
    series; time, when present, is the **last** axis of every table.

    Attributes
    ----------
    m_values : list of int
        All azimuthal orders present (may include negatives), sorted.  Identical
        at every time, by construction.
    R_grid : ndarray, shape (nR,)
        Radial (cylindrical R) grid [kpc], positive values.
    z_grid : ndarray, shape (nz,)
        Vertical grid [kpc], symmetric about z = 0.
    phi : dict[int, ndarray]
        ``phi[m]`` has shape ``(nR, nz)``, or ``(nR, nz, nt)`` when *times* is
        set, and holds the Φ_m(R, z) table.
    metadata : dict
        Header parameters: ``mmax``, ``gridSizeR``, ``gridSizez``,
        ``symmetry``, ``type``.
    dphi_dR : dict[int, ndarray] or None
        ``∂Φ_m/∂R`` tables from the file's ``#dPhi/dR`` section, same shapes as
        *phi*.  ``None`` when the source carried no such section.
    dphi_dz : dict[int, ndarray] or None
        ``∂Φ_m/∂z`` tables from the file's ``#dPhi/dz`` section.  ``None`` when
        the source carried no such section.
    times : ndarray or None, shape (nt,)
        Sample times.  ``None`` (the default) means no time axis; the field is
        last so existing positional construction keeps working.
    """

    m_values: list[int]
    R_grid: np.ndarray
    z_grid: np.ndarray
    phi: dict[int, np.ndarray]
    metadata: dict = field(default_factory=dict)
    dphi_dR: dict[int, np.ndarray] | None = None
    dphi_dz: dict[int, np.ndarray] | None = None
    times: np.ndarray | None = None

    _COEF_EXT = ".coef_cylsp"

    # --- modification -----------------------------------------------------

    def zeroed(
        self,
        keep_m: list[int],
        include_negative: bool = True,
    ) -> "CylSplineCoefs":
        """
        Return a copy with all m terms **not** in *keep_m* zeroed out.

        Parameters
        ----------
        keep_m : list of int
            Azimuthal orders to retain.
        include_negative : bool, optional
            If ``True`` (default), automatically include the negative-m
            counterpart for each positive m in *keep_m*.

        Returns
        -------
        CylSplineCoefs
            New instance with unselected m tables replaced by zero arrays.
        """
        keep_set: set[int] = set(keep_m)
        if include_negative:
            keep_set |= {-m for m in keep_m if m != 0}

        def _mask(tables: dict[int, np.ndarray] | None):
            if tables is None:
                return None
            return {
                m: (table.copy() if m in keep_set else np.zeros_like(table))
                for m, table in tables.items()
            }

        return CylSplineCoefs(
            m_values=self.m_values,
            R_grid=self.R_grid,
            z_grid=self.z_grid,
            phi=_mask(self.phi),
            metadata=self.metadata,
            dphi_dR=_mask(self.dphi_dR),
            dphi_dz=_mask(self.dphi_dz),
            times=self.times,
        )

    def copy(self) -> "CylSplineCoefs":
        """Return a deep copy — no array, list or dict is shared with *self*."""

        def _dup(tables: dict[int, np.ndarray] | None):
            if tables is None:
                return None
            return {m: np.array(t, copy=True) for m, t in tables.items()}

        return CylSplineCoefs(
            m_values=list(self.m_values),
            R_grid=np.array(self.R_grid, copy=True),
            z_grid=np.array(self.z_grid, copy=True),
            phi=_dup(self.phi),
            metadata=dict(self.metadata),
            dphi_dR=_dup(self.dphi_dR),
            dphi_dz=_dup(self.dphi_dz),
            times=None if self.times is None else np.array(self.times, copy=True),
        )

    def snapshot(self, i: int) -> "CylSplineCoefs":
        """
        Time-less view at time index *i*.

        Every table is a NumPy view into *self*; call :meth:`copy` on the result
        for an independent object.
        """
        ti = self._time_index(i)

        def _slice(tables: dict[int, np.ndarray] | None):
            if tables is None:
                return None
            return {m: t[..., ti] for m, t in tables.items()}

        return CylSplineCoefs(
            m_values=self.m_values,
            R_grid=self.R_grid,
            z_grid=self.z_grid,
            phi=_slice(self.phi),
            metadata=self.metadata,
            dphi_dR=_slice(self.dphi_dR),
            dphi_dz=_slice(self.dphi_dz),
        )

    def with_times(
        self,
        times,
        *,
        phi: dict[int, np.ndarray] | None = None,
        dphi_dR: dict[int, np.ndarray] | None = None,
        dphi_dz: dict[int, np.ndarray] | None = None,
    ) -> "CylSplineCoefs":
        """
        Attach or relabel the time axis.

        Nothing is interpolated, smoothed or resampled here: compute your own
        time-resampled tables and attach them.

        Parameters
        ----------
        times : array-like, shape (nt,)
            New time samples.
        phi : dict[int, ndarray], optional
            New tables, keyed by **exactly** the existing ``m_values``, each of
            shape ``(nR, nz, nt)``.  When omitted, only the time labels change
            and ``len(times)`` must equal the current ``n_times``.
        dphi_dR, dphi_dz : dict[int, ndarray], optional
            New derivative tables, same keys and shapes as *phi*.  Required
            whenever *phi* is supplied and *self* already carries them.

        Returns
        -------
        CylSplineCoefs
        """
        t = _as_times(times)
        nR, nz = len(self.R_grid), len(self.z_grid)
        expected_keys = sorted(self.m_values)

        def _check_tables(name: str, tables: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            if not isinstance(tables, dict):
                raise TypeError(
                    f"{name} must be a dict keyed by m; got {type(tables).__name__}."
                )
            if sorted(tables) != expected_keys:
                raise ValueError(
                    f"{name} is keyed by m={sorted(tables)}, which differs from "
                    f"the existing m_values={expected_keys}. Labels are never "
                    "reconciled."
                )
            out: dict[int, np.ndarray] = {}
            for m in expected_keys:
                arr = np.asarray(tables[m], dtype=float)
                if arr.shape != (nR, nz, len(t)):
                    raise ValueError(
                        f"{name}[{m}] has shape {arr.shape}, expected "
                        f"({nR}, {nz}, {len(t)}) — the leading axes must match "
                        "the existing R_grid and z_grid exactly, and the "
                        "trailing axis must match len(times)."
                    )
                out[m] = arr
            return out

        if phi is None:
            if dphi_dR is not None or dphi_dz is not None:
                raise ValueError(
                    "dphi_dR / dphi_dz can only be supplied together with phi; on "
                    "their own there is no time axis to attach them to."
                )
            if not self.has_time_axis:
                raise ValueError(
                    "with_times(times) without phi only relabels an existing time "
                    "axis, but this object has times=None. Supply phi=... to "
                    "attach a time axis."
                )
            if len(t) != self.n_times:
                raise ValueError(
                    f"with_times(times) without phi only relabels the time axis, so "
                    f"len(times)={len(t)} must equal the current n_times="
                    f"{self.n_times}. Supply phi=... to change the number of times."
                )
            new_phi, new_dR, new_dz = self.phi, self.dphi_dR, self.dphi_dz
        else:
            new_phi = _check_tables("phi", phi)
            new_dR, new_dz = None, None
            for name, supplied, existing in (
                ("dphi_dR", dphi_dR, self.dphi_dR),
                ("dphi_dz", dphi_dz, self.dphi_dz),
            ):
                if supplied is None:
                    if existing is not None:
                        raise ValueError(
                            f"phi was supplied but {name} was not, and this object "
                            f"already carries a {name} that cannot be reconciled "
                            f"with the new time axis (nt={len(t)}). Pass a matching "
                            f"{name}, or drop the old one first "
                            f"(obj.{name} = None)."
                        )
                    continue
                checked = _check_tables(name, supplied)
                if name == "dphi_dR":
                    new_dR = checked
                else:
                    new_dz = checked

        out = CylSplineCoefs(
            m_values=self.m_values,
            R_grid=self.R_grid,
            z_grid=self.z_grid,
            phi=new_phi,
            metadata=self.metadata,
            dphi_dR=new_dR,
            dphi_dz=new_dz,
            times=t,
        )
        out.validate()
        return out

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """
        Assert internal shape consistency, naming the offending field on failure.

        Direct field assignment is legal — this is what catches the damage.  It
        runs automatically at the top of every materialise and write entry point.

        Raises
        ------
        ValueError
            On any shape or key inconsistency between *m_values*, *R_grid*,
            *z_grid*, *phi*, *dphi_dR*, *dphi_dz* and *times*.
        """
        R = np.asarray(self.R_grid)
        z = np.asarray(self.z_grid)
        if R.ndim != 1:
            raise ValueError(
                f"CylSplineCoefs.R_grid must be 1-D; got shape {R.shape}."
            )
        if z.ndim != 1:
            raise ValueError(
                f"CylSplineCoefs.z_grid must be 1-D; got shape {z.shape}."
            )
        nR, nz = R.shape[0], z.shape[0]

        if self.times is None:
            expected = (nR, nz)
            hint = f"(len(R_grid)={nR}, len(z_grid)={nz}; times is None)"
        else:
            t = np.asarray(self.times)
            if t.ndim != 1:
                raise ValueError(
                    f"CylSplineCoefs.times must be 1-D; got shape {t.shape}."
                )
            expected = (nR, nz, t.shape[0])
            hint = f"(len(R_grid)={nR}, len(z_grid)={nz}, len(times)={t.shape[0]})"

        expected_keys = sorted(self.m_values)
        if self.phi is None:
            raise ValueError("CylSplineCoefs.phi must not be None.")
        for name, tables in (
            ("phi", self.phi),
            ("dphi_dR", self.dphi_dR),
            ("dphi_dz", self.dphi_dz),
        ):
            if tables is None:
                continue
            if sorted(tables) != expected_keys:
                raise ValueError(
                    f"CylSplineCoefs.{name} is keyed by m={sorted(tables)}, which "
                    f"differs from m_values={expected_keys}."
                )
            for m in expected_keys:
                arr = np.asarray(tables[m])
                if arr.shape != expected:
                    raise ValueError(
                        f"CylSplineCoefs.{name}[{m}] has shape {arr.shape}, "
                        f"expected {expected} {hint}."
                    )

    # --- serialisation ----------------------------------------------------

    def to_coef_string(self, t: int | None = None) -> str:
        """
        Serialise one snapshot back to the Agama CylSpline text format.

        Parameters
        ----------
        t : int, optional
            Time index.  **Required** when this object carries a time axis, and
            rejected when it does not.  Use :meth:`to_coef_strings` for all
            times at once.

        Returns
        -------
        str
            Full text suitable for writing to a ``.coef_cylsp`` file or
            passing to :func:`~agama_helper._io._write_tmp_coef`.

        Notes
        -----
        The ``#dPhi/dR`` and ``#dPhi/dz`` sections are emitted only when the
        corresponding attributes are populated.  A ``#Phi``-only file is a legal
        CylSpline export — Agama reconstructs the derivatives from the spline —
        but writing all three sections keeps the round-trip lossless.
        """
        self.validate()
        if self.has_time_axis:
            if t is None:
                raise ValueError(
                    f"CylSplineCoefs carries a time axis (n_times={self.n_times}); "
                    "to_coef_string(t=<index>) needs an explicit time index. Use "
                    "to_coef_strings() to serialise every time."
                )
            ti = self._time_index(t)

            def _at(tables):
                return None if tables is None else {m: v[..., ti] for m, v in tables.items()}
        else:
            if t is not None:
                raise ValueError(
                    f"CylSplineCoefs has no time axis (times is None), so t={t!r} "
                    "is meaningless; call to_coef_string() with no arguments."
                )

            def _at(tables):
                return tables

        phi_t, dR_t, dz_t = _at(self.phi), _at(self.dphi_dR), _at(self.dphi_dz)

        meta = self.metadata
        lines: list[str] = [
            "[Potential]",
            f"type={meta.get('type', 'CylSpline')}",
            f"gridSizeR={meta.get('gridSizeR', len(self.R_grid))}",
            f"gridSizez={meta.get('gridSizez', len(self.z_grid))}",
            f"mmax={meta.get('mmax', max(abs(m) for m in self.m_values) if self.m_values else 0)}",
            f"symmetry={meta.get('symmetry', 'None')}",
            "Coefficients",
        ]
        z_header = "\t".join(f"{z:.14g}" for z in self.z_grid)

        def _emit(marker: str, tables: dict[int, np.ndarray]) -> None:
            lines.append(marker)
            for m in sorted(self.m_values):
                lines.append(f"{m}\t#m")
                lines.append(f"#R(row)\\z(col)\t{z_header}")
                table = tables[m]
                for ri, r in enumerate(self.R_grid):
                    row_vals = " ".join(f"{v:.14g}" for v in table[ri])
                    lines.append(f"{r:.14g} {row_vals}")

        _emit("#Phi", phi_t)
        for marker, tables in (("#dPhi/dR", dR_t), ("#dPhi/dz", dz_t)):
            if tables is not None:
                lines.append("")
                _emit(marker, tables)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stacking time-less snapshots into a series
# ---------------------------------------------------------------------------

def _check_stackable(items: list) -> None:
    """Common pre-conditions for stacking: time-less, valid, matching headers."""
    for i, it in enumerate(items):
        if it.has_time_axis:
            raise ValueError(
                f"item {i} already carries a time axis (n_times={it.n_times}); "
                "stack_coefs() stacks time-less snapshots. Use with_times() to "
                "relabel an existing series."
            )
        it.validate()

    ref = items[0].metadata
    for i, it in enumerate(items[1:], start=1):
        if it.metadata != ref:
            bad = sorted(
                k for k in set(ref) | set(it.metadata)
                if ref.get(k) != it.metadata.get(k)
            )
            detail = ", ".join(
                f"{k}: {ref.get(k)!r} vs {it.metadata.get(k)!r}" for k in bad
            )
            raise ValueError(
                f"item {i} has header metadata differing from item 0 on "
                f"{bad} ({detail}). Snapshots of one series must share their "
                "expansion header."
            )


def _describe_grid_mismatch(name: str, a, b) -> str:
    """Explain how grid *a* differs from reference *b* — shape, or first value."""
    sa, sb = np.shape(a), np.shape(b)
    if sa != sb:
        return f"shapes {sa} vs {sb}"
    diff = np.flatnonzero(np.asarray(a) != np.asarray(b))
    if diff.size == 0:
        return f"shapes {sa} vs {sb}"
    i = int(diff[0])
    return (
        f"same shape {sa} but differing values, first at index {i}: "
        f"{np.asarray(a)[i]!r} vs {np.asarray(b)[i]!r} ({diff.size} of {sa[0]} differ)"
    )


def _check_presence(items: list, attr: str) -> bool:
    """Return whether *attr* is populated, requiring all-or-nothing across items."""
    have = [getattr(it, attr) is not None for it in items]
    if any(have) and not all(have):
        missing = [i for i, h in enumerate(have) if not h]
        raise ValueError(
            f"{attr} is present on some snapshots and absent on others "
            f"(missing at indices {missing}). Stack a homogeneous set, or drop "
            f"{attr} everywhere first."
        )
    return all(have)


def _stack_mult(items: list["MultipoleCoefs"], times: np.ndarray) -> "MultipoleCoefs":
    _check_stackable(items)
    ref = items[0]
    for i, it in enumerate(items[1:], start=1):
        if list(it.lm_labels) != list(ref.lm_labels):
            raise ValueError(
                f"item {i} has lm_labels differing from item 0 "
                f"(n_lm={len(it.lm_labels)} vs {len(ref.lm_labels)}). Labels are "
                "never reconciled across a time series."
            )
        if not np.array_equal(it.R_grid, ref.R_grid):
            raise ValueError(
                f"item {i} has an R_grid differing from item 0 "
                f"({_describe_grid_mismatch('R_grid', it.R_grid, ref.R_grid)}). "
                "Grids are never interpolated across a time series."
            )
    has_dphi = _check_presence(items, "dphi_dr")

    return MultipoleCoefs(
        R_grid=np.array(ref.R_grid, copy=True),
        lm_labels=list(ref.lm_labels),
        phi=np.stack([np.asarray(it.phi, dtype=float) for it in items], axis=-1),
        dphi_dr=(
            np.stack([np.asarray(it.dphi_dr, dtype=float) for it in items], axis=-1)
            if has_dphi
            else None
        ),
        metadata=dict(ref.metadata),
        times=times,
    )


def _stack_cylspl(items: list["CylSplineCoefs"], times: np.ndarray) -> "CylSplineCoefs":
    _check_stackable(items)
    ref = items[0]
    for i, it in enumerate(items[1:], start=1):
        if list(it.m_values) != list(ref.m_values):
            raise ValueError(
                f"item {i} has m_values={list(it.m_values)}, differing from item "
                f"0's {list(ref.m_values)}. Labels are never reconciled across a "
                "time series."
            )
        for name, a, b in (("R_grid", it.R_grid, ref.R_grid),
                           ("z_grid", it.z_grid, ref.z_grid)):
            if not np.array_equal(a, b):
                raise ValueError(
                    f"item {i} has a {name} differing from item 0 "
                    f"({_describe_grid_mismatch(name, a, b)}). Grids are never "
                    "interpolated across a time series."
                )
    has_dR = _check_presence(items, "dphi_dR")
    has_dz = _check_presence(items, "dphi_dz")

    def _stack_tables(attr: str) -> dict[int, np.ndarray]:
        return {
            m: np.stack(
                [np.asarray(getattr(it, attr)[m], dtype=float) for it in items],
                axis=-1,
            )
            for m in ref.m_values
        }

    return CylSplineCoefs(
        m_values=list(ref.m_values),
        R_grid=np.array(ref.R_grid, copy=True),
        z_grid=np.array(ref.z_grid, copy=True),
        phi=_stack_tables("phi"),
        metadata=dict(ref.metadata),
        dphi_dR=_stack_tables("dphi_dR") if has_dR else None,
        dphi_dz=_stack_tables("dphi_dz") if has_dz else None,
        times=times,
    )


def stack_coefs(
    items: Sequence[Union["MultipoleCoefs", "CylSplineCoefs"]],
    times,
) -> Union["MultipoleCoefs", "CylSplineCoefs"]:
    """
    Stack time-less coefficient snapshots into one object with a time axis.

    Grids and labels are never reconciled: ``R_grid``, ``z_grid``,
    ``lm_labels``, ``m_values`` and the header ``metadata`` must be identical
    across all *items*, and any difference is a hard error.

    Parameters
    ----------
    items : sequence of MultipoleCoefs or CylSplineCoefs
        Time-less snapshots, in time order.  All must be the same expansion type.
    times : array-like, shape (len(items),)
        Sample time for each item.

    Returns
    -------
    MultipoleCoefs or CylSplineCoefs
        A single object whose arrays carry a trailing axis of length
        ``len(items)``.

    Raises
    ------
    TypeError
        If *items* mixes expansion types, or holds something that is not a
        coefficient object.
    ValueError
        If *times* does not match *items* in length, if any item already has a
        time axis, or if grids, labels or metadata differ between items.

    Examples
    --------
    >>> snaps = [read_coefs(p) for p in sorted(paths)]
    >>> series = stack_coefs(snaps, times=np.linspace(6.0, 14.0, len(snaps)))
    >>> series.phi.shape
    (25, 81, 11)
    """
    items = list(items)
    if not items:
        raise ValueError("stack_coefs() needs at least one coefficient object.")

    t = _as_times(times)
    if len(t) != len(items):
        raise ValueError(
            f"times (len={len(t)}) must have exactly one entry per item "
            f"(len={len(items)})."
        )

    kinds = {type(it).__name__ for it in items}
    if len(kinds) != 1:
        raise TypeError(
            f"Cannot stack a mixture of expansion types: {sorted(kinds)}. "
            "Multipole and CylSpline expansions never stack."
        )

    first = items[0]
    if isinstance(first, MultipoleCoefs):
        return _stack_mult(items, t)
    if isinstance(first, CylSplineCoefs):
        return _stack_cylspl(items, t)
    raise TypeError(
        f"stack_coefs() expects MultipoleCoefs or CylSplineCoefs objects; got "
        f"{type(first).__name__}."
    )


# ---------------------------------------------------------------------------
# Source resolution for single-or-many reads
# ---------------------------------------------------------------------------

def _resolve_snapshot_strings(
    source,
    group_name,
    dataset_name: str,
    times,
) -> tuple[list[str], np.ndarray | None]:
    """
    Expand any accepted *source* into raw coef strings plus resolved times.

    Returns
    -------
    coef_strings : list of str
        One entry per snapshot, in time order.
    times : ndarray or None
        ``None`` signals a single-snapshot read, which carries no time axis.

    Notes
    -----
    Times resolve in the order: explicit *times* argument, then the ``.h5`` root
    ``times`` dataset, then the ``.ini`` timestamps.  When a time axis is
    requested and none of those yield times, this raises — an index-based axis
    is never invented.
    """
    # 1. An explicit sequence of sources — always a time series.
    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("An empty sequence of coefficient sources was given.")
        strings = [_resolve_coef_string(s, group_name, dataset_name) for s in source]
        if times is None:
            raise ValueError(
                "A sequence of coefficient sources always produces a time axis, "
                "but no times were supplied and a bare sequence carries none. "
                "Pass times=... explicitly."
            )
        t = _as_times(times)
        if len(t) != len(strings):
            raise ValueError(
                f"times (len={len(t)}) does not match the number of sources "
                f"({len(strings)})."
            )
        return strings, t

    path = _as_existing_path(source)

    # 2. An Agama Evolving .ini — group_name is ignored, time axis always present.
    if path is not None and path.suffix.lower() == ".ini":
        from ._load import _parse_evolving_ini

        ini_times, coef_paths, _ = _parse_evolving_ini(path)
        if not coef_paths:
            raise ValueError(
                f"No Timestamps entries found in {path}; nothing to read."
            )
        strings = [Path(p).read_text(encoding="utf-8") for p in coef_paths]
        t = _as_times(times) if times is not None else _as_times(ini_times)
        if len(t) != len(strings):
            raise ValueError(
                f"times (len={len(t)}) does not match the {len(strings)} "
                f"coefficient files listed in {path}."
            )
        return strings, t

    # 3. Multiple HDF5 groups — "all", or an explicit ordered sequence.
    wants_many_groups = group_name == "all" or isinstance(group_name, (list, tuple))
    if wants_many_groups:
        if path is None or path.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError(
                f"group_name={group_name!r} selects multiple HDF5 groups, but the "
                f"source is not an existing .h5/.hdf5 archive (got {source!r})."
            )
        with h5py.File(path, "r") as f:
            # The root "times" dataset is written co-indexed with the archive's
            # numerically sorted group order, so keep that order around to map
            # an explicitly requested subset or reordering back onto it.
            canonical = sorted(
                (k for k in f.keys() if k != "times"), key=_extract_int_from_group
            )
            if group_name == "all":
                groups = canonical
                if not groups:
                    raise ValueError(f"{path} contains no snapshot groups.")
            else:
                groups = list(group_name)
                missing = [g for g in groups if g not in f]
                if missing:
                    raise KeyError(f"groups {missing} are not present in {path}.")
            stored_times = np.asarray(f["times"]) if "times" in f else None
            strings = []
            for g in groups:
                raw = f[g][dataset_name][()]
                strings.append(
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                )

        if times is not None:
            t = _as_times(times)
        elif stored_times is not None:
            stored = _as_times(stored_times)
            if groups is canonical:
                t = stored
            else:
                # An explicit group_name may subset or reorder the archive.  Take
                # each group's stored time from its canonical position rather than
                # assuming the caller's order matches the file's.
                if len(stored) != len(canonical):
                    raise ValueError(
                        f"{path} has {len(canonical)} snapshot groups but its root "
                        f"'times' dataset has {len(stored)} entries, so a stored "
                        "time cannot be matched to each requested group. Pass "
                        "times=... explicitly."
                    )
                pos = {g: i for i, g in enumerate(canonical)}
                t = stored[[pos[g] for g in groups]]
        else:
            raise ValueError(
                f"A time axis was requested (group_name={group_name!r}) but no "
                f"times were supplied and {path} has no root 'times' dataset. "
                "Pass times=..., or write the archive with "
                "write_snapshot_coefs_to_h5(times=...) / to_h5(write_times=True). "
                "An index-based time axis is never invented."
            )
        if len(t) != len(strings):
            raise ValueError(
                f"times (len={len(t)}) does not match the number of groups read "
                f"({len(strings)})."
            )
        return strings, t

    # 4. A single snapshot — unchanged behaviour, no time axis.
    if times is not None:
        raise ValueError(
            f"times was supplied but group_name={group_name!r} selects a single "
            "snapshot, which has no time axis. Use group_name='all', a sequence "
            "of group names, a sequence of sources, or an Evolving .ini to build "
            "a time series."
        )
    return [_resolve_coef_string(source, group_name, dataset_name)], None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _read_mult_snapshot(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> MultipoleCoefs:
    """
    Parse **one** Agama Multipole snapshot into a :class:`MultipoleCoefs`.

    Internal single-snapshot parser behind :func:`read_mult_coefs`; it never
    produces a time axis.  See :func:`read_mult_coefs` for the public surface.

    Raises
    ------
    ValueError
        If the ``#Phi`` (or ``#rho``) section cannot be located in the data.
    """
    lines = _source_to_lines(source, group_name, dataset_name)

    # Parse header metadata (key=value lines before "Coefficients")
    meta: dict = {}
    for line in lines:
        s = line.strip()
        if s == "Coefficients":
            break
        if "=" in s and not s.startswith("[") and not s.startswith("#"):
            k, _, v = s.partition("=")
            meta[k.strip()] = v.strip()

    gridSizeR = int(meta.get("gridSizeR", 0))

    # Locate #Phi and #dPhi/dr section markers
    phi_header_idx: int | None = None
    dphi_header_idx: int | None = None
    for i, line in enumerate(lines):
        s = line.strip()
        if phi_header_idx is None and (s.startswith("#Phi") or s.startswith("#rho")):
            phi_header_idx = i
        elif s.startswith("#dPhi/dr"):
            dphi_header_idx = i

    if phi_header_idx is None:
        raise ValueError("Could not locate #Phi or #rho section in coefficient data.")

    def _parse_section(header_idx: int) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
        """Parse one coefficient section starting at the section marker line."""
        col_line = lines[header_idx + 1].strip()
        tokens = col_line.split("\t")
        # tokens[0] is '#radius'; rest are 'l=X,m=Y'
        lm_labels = []
        for tok in tokens[1:]:
            l_part, m_part = tok.split(",")
            lm_labels.append((int(l_part.split("=")[1]), int(m_part.split("=")[1])))
        data_start = header_idx + 2
        R_list, phi_list = [], []
        for line in lines[data_start: data_start + gridSizeR]:
            vals = line.strip().split("\t")
            R_list.append(float(vals[0]))
            phi_list.append([float(v) for v in vals[1:]])
        return np.array(R_list), lm_labels, np.array(phi_list)

    R_grid, lm_labels, phi = _parse_section(phi_header_idx)
    dphi_dr = None
    if dphi_header_idx is not None:
        _, _, dphi_dr = _parse_section(dphi_header_idx)

    return MultipoleCoefs(
        R_grid=R_grid,
        lm_labels=lm_labels,
        phi=phi,
        dphi_dr=dphi_dr,
        metadata=meta,
    )


def _read_cylspl_snapshot(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> CylSplineCoefs:
    """
    Parse **one** Agama CylSpline snapshot into a :class:`CylSplineCoefs`.

    Internal single-snapshot parser behind :func:`read_cylspl_coefs`; it never
    produces a time axis.  See :func:`read_cylspl_coefs` for the public surface.

    Notes
    -----
    An Agama CylSpline export written by ``Potential.export()`` carries three
    sections — ``#Phi``, ``#dPhi/dR`` and ``#dPhi/dz`` — each repeating the full
    set of ``m`` blocks.  The sections are split *before* the ``#m`` blocks are
    scanned, so ``phi`` always comes from ``#Phi``; *dphi_dR* / *dphi_dz* are
    populated from their own sections and are ``None`` when absent.

    Raises
    ------
    ValueError
        If ``gridSizeR``, ``gridSizez``, or ``mmax`` cannot be found in the
        header, if no m-blocks are detected, or if a derivative section covers a
        different set of ``m`` values than ``#Phi``.
    """
    lines = _source_to_lines(source, group_name, dataset_name)

    # Parse header metadata
    meta: dict = {}
    for line in lines:
        s = line.strip()
        if s == "Coefficients":
            break
        if "=" in s and not s.startswith("[") and not s.startswith("#"):
            k, _, v = s.partition("=")
            meta[k.strip()] = v.strip()

    gridSizeR = int(meta.get("gridSizeR", 0))
    gridSizez = int(meta.get("gridSizez", meta.get("gridSizeZ", 0)))
    if gridSizeR == 0 or gridSizez == 0:
        raise ValueError(
            "Could not determine gridSizeR/gridSizez from file header. "
            f"Parsed metadata: {meta}"
        )

    # Split into #Phi / #dPhi/dR / #dPhi/dz sections *before* scanning m-blocks:
    # each section repeats every m, so a section-blind scan would silently keep
    # only the last section's tables.
    sections = _split_cylspl_sections(lines)
    if "phi" not in sections:
        raise ValueError(
            "Could not locate a #Phi section in CylSpline coefficient data."
        )

    def _parse_section(lo: int, hi: int) -> tuple[list[int], dict[int, np.ndarray], np.ndarray, np.ndarray]:
        """Parse every ``\\t#m`` block within ``lines[lo:hi]``."""
        m_start: dict[int, int] = {}
        for i in range(lo, hi):
            if "\t#m" in lines[i]:
                m_start[int(lines[i].split("\t")[0].strip())] = i
        if not m_start:
            raise ValueError(
                "No azimuthal m-blocks found in CylSpline coefficient data."
            )

        ordered_m = sorted(m_start)

        # z-grid comes from the first m-block header of this section
        z_tokens = lines[m_start[ordered_m[0]] + 1].strip().split("\t")[1:]
        z_vals = np.array([float(z) for z in z_tokens])

        tables: dict[int, np.ndarray] = {}
        R_vals: np.ndarray | None = None
        for m in ordered_m:
            start = m_start[m]
            rows, R_list = [], []
            for row_line in lines[start + 2: start + 2 + gridSizeR]:
                vals = row_line.strip().split()
                R_list.append(float(vals[0]))
                rows.append([float(v) for v in vals[1: 1 + gridSizez]])
            tables[m] = np.array(rows)
            if R_vals is None:
                R_vals = np.array(R_list)

        return ordered_m, tables, R_vals if R_vals is not None else np.array([]), z_vals

    m_values, phi_dict, R_grid, z_grid = _parse_section(*sections["phi"])

    deriv: dict[str, dict[int, np.ndarray] | None] = {"dphi_dR": None, "dphi_dz": None}
    for key in ("dphi_dR", "dphi_dz"):
        if key not in sections:
            continue
        d_m, d_tables, _, _ = _parse_section(*sections[key])
        if d_m != m_values:
            raise ValueError(
                f"The {key} section covers m={d_m}, which differs from the #Phi "
                f"section's m={m_values}. Refusing to guess a correspondence."
            )
        deriv[key] = d_tables

    return CylSplineCoefs(
        m_values=m_values,
        R_grid=R_grid,
        z_grid=z_grid,
        phi=phi_dict,
        metadata=meta,
        dphi_dR=deriv["dphi_dR"],
        dphi_dz=deriv["dphi_dz"],
    )


def _read_auto_snapshot(coef_str: str) -> Union[MultipoleCoefs, CylSplineCoefs]:
    """Dispatch one coef string to the parser named by its ``type=`` header."""
    exp_type = _detect_expansion_type(coef_str)
    if exp_type == "Multipole":
        return _read_mult_snapshot(coef_str)
    if exp_type == "CylSpline":
        return _read_cylspl_snapshot(coef_str)
    raise ValueError(
        f"Could not determine expansion type from header (got '{exp_type}'). "
        "Expected 'Multipole' or 'CylSpline'."
    )


def read_mult_coefs(
    source,
    group_name="snap_000",
    dataset_name: str = "coefs",
    times=None,
) -> MultipoleCoefs:
    """
    Read one or many Agama Multipole snapshots into a :class:`MultipoleCoefs`.

    A single-group read returns exactly what it always has, with ``times`` set to
    ``None``.  Any multi-snapshot form returns one object whose ``phi`` and
    ``dphi_dr`` carry a trailing time axis.

    Parameters
    ----------
    source, group_name, dataset_name, times
        See :func:`read_coefs` — the signature is identical; only the expansion
        type is fixed here instead of auto-detected.

    Returns
    -------
    MultipoleCoefs
    """
    strings, t = _resolve_snapshot_strings(source, group_name, dataset_name, times)
    if t is None:
        return _read_mult_snapshot(strings[0])
    return stack_coefs([_read_mult_snapshot(s) for s in strings], t)


def read_cylspl_coefs(
    source,
    group_name="snap_000",
    dataset_name: str = "coefs",
    times=None,
) -> CylSplineCoefs:
    """
    Read one or many Agama CylSpline snapshots into a :class:`CylSplineCoefs`.

    A single-group read returns exactly what it always has, with ``times`` set to
    ``None``.  Any multi-snapshot form returns one object whose ``phi`` (and the
    derivative tables, when present) carry a trailing time axis.

    Parameters
    ----------
    source, group_name, dataset_name, times
        See :func:`read_coefs` — the signature is identical; only the expansion
        type is fixed here instead of auto-detected.

    Returns
    -------
    CylSplineCoefs
    """
    strings, t = _resolve_snapshot_strings(source, group_name, dataset_name, times)
    if t is None:
        return _read_cylspl_snapshot(strings[0])
    return stack_coefs([_read_cylspl_snapshot(s) for s in strings], t)


def read_coefs(
    source,
    group_name="snap_000",
    dataset_name: str = "coefs",
    times=None,
) -> Union[MultipoleCoefs, CylSplineCoefs]:
    """
    Read an Agama expansion into a structured dataclass — one snapshot or many.

    The expansion type (Multipole or CylSpline) is detected automatically from
    the file header, so there is no need to know it in advance.  There is no
    separate series reader and no separate series class: a single-group read
    behaves exactly as it always has, and every multi-snapshot form returns the
    same dataclass with a trailing time axis attached.

    Parameters
    ----------
    source : str, Path, or sequence
        Any of:

        * Path to a plain-text coefficient file.
        * Path to an ``.h5`` / ``.hdf5`` archive — see *group_name*.
        * Raw text content of a coefficient file.
        * Path to an Agama Evolving ``.ini`` config — every listed snapshot is
          read, *group_name* is ignored, and a time axis is always present.
        * A sequence of any of the above — read in the given order, with a time
          axis always present.

    group_name : str or sequence of str, optional
        Selects HDF5 groups:

        * ``str`` (default ``"snap_000"``) — one group; **no time axis**.
        * ``"all"`` — every group in the archive, numerically sorted
          (``"snap_0042"`` sorts as 42); time axis present.
        * sequence of ``str`` — exactly those groups, in the given order; time
          axis present.

    dataset_name : str, optional
        Dataset within each HDF5 group, by default ``"coefs"``.
    times : array-like, optional
        Sample times for the time axis.  Resolution order is: this argument,
        then the ``.h5`` root ``"times"`` dataset, then the ``.ini`` timestamps.
        When a time axis is requested and none of those yield times, a
        :exc:`ValueError` is raised — an index-based axis is never invented.
        Supplying *times* for a single-snapshot read is an error.

        The root ``"times"`` dataset is co-indexed with the archive's
        numerically sorted group order, so an explicit *group_name* that
        subsets or reorders the archive still gets each group's own time.

    Returns
    -------
    MultipoleCoefs or CylSplineCoefs
        With ``times is None`` for a single-snapshot read, and a populated
        ``times`` otherwise.

    Raises
    ------
    ValueError
        If the expansion type cannot be determined, if times cannot be resolved
        for a requested time axis, or if the snapshots disagree on grids, labels
        or header metadata.
    TypeError
        If the snapshots mix Multipole and CylSpline expansions.

    Examples
    --------
    >>> mc = read_coefs("potential/090.dark.none_8.coef_mult")   # single, as before
    >>> mc.times is None
    True
    >>> series = read_coefs("MW_mult.h5", group_name="all")      # whole archive
    >>> series.phi.shape, series.n_times
    ((25, 81, 11), 11)
    >>> some = read_coefs("MW_mult.h5", group_name=["snap_090", "snap_095"],
    ...                   times=[6.0, 6.5])
    >>> ev = read_coefs("potential/MW_mult.ini")                 # times from the .ini
    >>> files = read_coefs(sorted(glob("potential/*.coef_mult")), times=t_gyr)
    """
    strings, t = _resolve_snapshot_strings(source, group_name, dataset_name, times)
    parsed = [_read_auto_snapshot(s) for s in strings]
    if t is None:
        return parsed[0]
    return stack_coefs(parsed, t)


