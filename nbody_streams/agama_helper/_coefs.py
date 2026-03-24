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
    """

    m_values: list[int]
    R_grid: np.ndarray
    z_grid: np.ndarray
    phi: dict[int, np.ndarray]
    metadata: dict = field(default_factory=dict)

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
        new_phi = {
            m: (table.copy() if m in keep_set else np.zeros_like(table))
            for m, table in self.phi.items()
        }
        return CylSplineCoefs(
            m_values=self.m_values,
            R_grid=self.R_grid,
            z_grid=self.z_grid,
            phi=new_phi,
            metadata=self.metadata,
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
            "#Phi",
        ]
        z_header = "\t".join(f"{z:.14g}" for z in self.z_grid)
        for m in sorted(self.m_values):
            lines.append(f"{m}\t#m")
            lines.append(f"#R(row)\\z(col)\t{z_header}")
            table = self.phi[m]
            for ri, r in enumerate(self.R_grid):
                row_vals = " ".join(f"{v:.14g}" for v in table[ri])
                lines.append(f"{r:.14g} {row_vals}")
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

    Raises
    ------
    ValueError
        If ``gridSizeR``, ``gridSizez``, or ``mmax`` cannot be found in the
        header, or if no m-blocks are detected.
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

    # Locate m-block start lines (lines containing "\t#m")
    m_values: list[int] = []
    m_start: dict[int, int] = {}
    for i, line in enumerate(lines):
        if "\t#m" in line:
            m_val = int(line.split("\t")[0].strip())
            m_values.append(m_val)
            m_start[m_val] = i

    if not m_values:
        raise ValueError("No azimuthal m-blocks found in CylSpline coefficient data.")

    # Parse z-grid from first m-block header
    first_m = m_values[0]
    z_header_line = lines[m_start[first_m] + 1]
    z_tokens = z_header_line.strip().split("\t")[1:]   # skip "#R(row)\z(col)" label
    z_grid = np.array([float(z) for z in z_tokens])

    # Parse R-grid and phi tables for every m
    R_grid: np.ndarray | None = None
    phi_dict: dict[int, np.ndarray] = {}
    for m in m_values:
        start = m_start[m]
        rows = []
        R_vals = []
        for row_line in lines[start + 2: start + 2 + gridSizeR]:
            vals = row_line.strip().split()
            R_vals.append(float(vals[0]))
            rows.append([float(v) for v in vals[1: 1 + gridSizez]])
        phi_dict[m] = np.array(rows)
        if R_grid is None:
            R_grid = np.array(R_vals)

    return CylSplineCoefs(
        m_values=sorted(m_values),
        R_grid=R_grid if R_grid is not None else np.array([]),
        z_grid=z_grid,
        phi=phi_dict,
        metadata=meta,
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
