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
from typing import Union

import numpy as np

from ._io import _resolve_coef_string


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
# MultipoleCoefs
# ---------------------------------------------------------------------------

@dataclass
class MultipoleCoefs:
    """
    Structured representation of a Multipole (spherical harmonic BFE) potential.

    Attributes
    ----------
    R_grid : ndarray, shape (nR,)
        Radial grid points [kpc].
    lm_labels : list of (l, m)
        Ordered (l, m) pairs corresponding to columns in *phi* and *dphi_dr*.
    phi : ndarray, shape (nR, n_lm)
        Potential coefficients Φ_{l,m}(r).
    dphi_dr : ndarray or None, shape (nR, n_lm)
        Radial derivative coefficients ∂Φ/∂r.  ``None`` if absent in the
        source file.
    metadata : dict
        Header parameters parsed from the coefficient file: ``lmax``,
        ``gridSizeR``, ``symmetry``, ``type``, etc.
    """

    R_grid: np.ndarray
    lm_labels: list[tuple[int, int]]
    phi: np.ndarray
    dphi_dr: np.ndarray | None
    metadata: dict = field(default_factory=dict)

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
        ndarray, shape (nR,)
            Power at each radial grid point, co-indexed with *R_grid*.
        """
        cols = [i for i, (li, _) in enumerate(self.lm_labels) if li == l]
        if not cols:
            return np.zeros(len(self.R_grid))
        block = self.phi[:, cols]
        return (block ** 2).sum(axis=1) if use_quadrature else np.abs(block).sum(axis=1)

    def total_power(self, l: int, use_quadrature: bool = True) -> float:
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
        float
        """
        return float(self.radial_power(l, use_quadrature).sum())

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
        new_phi = np.where(mask[np.newaxis, :], self.phi, 0.0)
        new_dphi = (
            np.where(mask[np.newaxis, :], self.dphi_dr, 0.0)
            if self.dphi_dr is not None
            else None
        )
        return MultipoleCoefs(
            R_grid=self.R_grid,
            lm_labels=self.lm_labels,
            phi=new_phi,
            dphi_dr=new_dphi,
            metadata=self.metadata,
        )

    # --- serialisation ----------------------------------------------------

    def to_coef_string(self) -> str:
        """
        Serialise back to the Agama Multipole text format.

        Returns
        -------
        str
            Full text suitable for writing to a ``.coef_mult`` file or
            passing to :func:`~agama_helper._io._write_tmp_coef`.
        """
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
            row = [f"{r:.13g}"] + [f"{v:.13g}" for v in self.phi[ri]]
            lines.append("\t".join(row))
        # #dPhi/dr section (if present)
        if self.dphi_dr is not None:
            lines.append("")
            lines.append("#dPhi/dr")
            lines.append(col_header)
            for ri, r in enumerate(self.R_grid):
                row = [f"{r:.13g}"] + [f"{v:.13g}" for v in self.dphi_dr[ri]]
                lines.append("\t".join(row))
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CylSplineCoefs
# ---------------------------------------------------------------------------

@dataclass
class CylSplineCoefs:
    """
    Structured representation of a CylSpline (azimuthal harmonic BFE) potential.

    The CylSpline expansion stores Φ(R, z) as a set of 2-D spline tables,
    one per azimuthal order m.

    Attributes
    ----------
    m_values : list of int
        All azimuthal orders present (may include negatives), sorted.
    R_grid : ndarray, shape (nR,)
        Radial (cylindrical R) grid [kpc], positive values.
    z_grid : ndarray, shape (nz,)
        Vertical grid [kpc], symmetric about z = 0.
    phi : dict[int, ndarray]
        ``phi[m]`` has shape ``(nR, nz)`` and holds the Φ_m(R, z) table.
    metadata : dict
        Header parameters: ``mmax``, ``gridSizeR``, ``gridSizez``,
        ``symmetry``, ``type``.
    dphi_dR : dict[int, ndarray] or None
        ``∂Φ_m/∂R`` tables from the file's ``#dPhi/dR`` section, same shapes as
        *phi*.  ``None`` when the source carried no such section.
    dphi_dz : dict[int, ndarray] or None
        ``∂Φ_m/∂z`` tables from the file's ``#dPhi/dz`` section.  ``None`` when
        the source carried no such section.
    """

    m_values: list[int]
    R_grid: np.ndarray
    z_grid: np.ndarray
    phi: dict[int, np.ndarray]
    metadata: dict = field(default_factory=dict)
    dphi_dR: dict[int, np.ndarray] | None = None
    dphi_dz: dict[int, np.ndarray] | None = None

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
        )

    # --- serialisation ----------------------------------------------------

    def to_coef_string(self) -> str:
        """
        Serialise back to the Agama CylSpline text format.

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

        _emit("#Phi", self.phi)
        for marker, tables in (("#dPhi/dR", self.dphi_dR), ("#dPhi/dz", self.dphi_dz)):
            if tables is not None:
                lines.append("")
                _emit(marker, tables)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def read_mult_coefs(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> MultipoleCoefs:
    """
    Read an Agama Multipole expansion into a :class:`MultipoleCoefs` dataclass.

    Parameters
    ----------
    source : str or Path
        Any of:

        * Path to a plain-text ``.coef_mult`` file.
        * Path to an ``.h5`` / ``.hdf5`` archive — reads from
          ``group_name/dataset_name``.
        * Raw text content of a ``.coef_mult`` file (e.g. from
          :func:`~agama_helper._io.read_coef_string` or
          :meth:`MultipoleCoefs.to_coef_string`).

    group_name : str, optional
        HDF5 group to read when *source* is an HDF5 file, by default
        ``"snap_000"``.
    dataset_name : str, optional
        HDF5 dataset within the group, by default ``"coefs"``.

    Returns
    -------
    MultipoleCoefs

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


def read_cylspl_coefs(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> CylSplineCoefs:
    """
    Read an Agama CylSpline expansion into a :class:`CylSplineCoefs` dataclass.

    Parameters
    ----------
    source : str or Path
        Any of:

        * Path to a plain-text ``.coef_cylsp`` file.
        * Path to an ``.h5`` / ``.hdf5`` archive — reads from
          ``group_name/dataset_name``.
        * Raw text content of a ``.coef_cylsp`` file.

    group_name : str, optional
        HDF5 group to read when *source* is an HDF5 file, by default
        ``"snap_000"``.
    dataset_name : str, optional
        HDF5 dataset within the group, by default ``"coefs"``.

    Returns
    -------
    CylSplineCoefs

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


def read_coefs(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> Union[MultipoleCoefs, CylSplineCoefs]:
    """
    Read an Agama expansion coefficient file into a structured dataclass.

    The expansion type (Multipole or CylSpline) is detected automatically
    from the file header — no need to know it in advance.

    Parameters
    ----------
    source : str or Path
        Any of:

        * Path to a plain-text ``.coef_mult`` or ``.coef_cylsp`` file.
        * Path to an ``.h5`` / ``.hdf5`` archive — reads from
          ``group_name/dataset_name``.
        * Raw text content of either file type.

    group_name : str, optional
        HDF5 group when *source* is an HDF5 file, by default ``"snap_000"``.
    dataset_name : str, optional
        HDF5 dataset within the group, by default ``"coefs"``.

    Returns
    -------
    MultipoleCoefs or CylSplineCoefs
        The appropriate dataclass, depending on the expansion type found in
        *source*.

    Raises
    ------
    ValueError
        If the expansion type cannot be determined from the header.

    Examples
    --------
    >>> mc = read_coefs("potential/090.dark.none_8.coef_mult")
    >>> cc = read_coefs("potential/090.bar.none_8.coef_cylsp")
    >>> mc = read_coefs("MW_mult.h5", group_name="snap_090")
    >>> cc = read_coefs("MW_cylsp.h5", group_name="snap_090")
    """
    coef_str = _resolve_coef_string(source, group_name, dataset_name)
    exp_type = _detect_expansion_type(coef_str)
    if exp_type == "Multipole":
        return read_mult_coefs(coef_str)
    if exp_type == "CylSpline":
        return read_cylspl_coefs(coef_str)
    raise ValueError(
        f"Could not determine expansion type from header (got '{exp_type}'). "
        "Expected 'Multipole' or 'CylSpline'."
    )
