"""
agama_helper._fire
~~~~~~~~~~~~~~~~~~
FIRE-simulation-specific convenience wrappers (Arora et al. 2022).

These functions encode FIRE path conventions (``potential/10kpc/``,
``snapshot_times.txt``) and are not needed for generic Agama workflows.
All heavy-lifting is delegated to the generic modules
(:mod:`agama_helper._io`, :mod:`agama_helper._load`, :mod:`agama_helper._coefs`)
and, for the background cosmology, to
:mod:`nbody_streams.utils._cosmology`.

Sections
--------
1. Snapshot time table
2. FIRE evolving-potential ``.ini`` helper
3. FIRE potential loader
4. Cubic-spline resampling of a coefficient time series
5. Comoving host frame: rotation, centre splines, centre acceleration
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..utils._cosmology import KPC_PER_GYR_PER_KMS
from ._coefs import MultipoleCoefs, _add_negative_m, read_cylspl_coefs, read_mult_coefs
from ._io import _cleanup_tmp_file, _write_tmp_coef
from ._load import create_evolving_ini


# ---------------------------------------------------------------------------
# Snapshot time table
# ---------------------------------------------------------------------------

def read_snapshot_times(
    sim_dir: Union[Path, str],
    sep: str = r'\s+',
):
    r"""
    Read ``snapshot_times.txt`` from a FIRE simulation directory.

    Returns a DataFrame with canonical columns ``snap``, ``scale-factor``,
    ``redshift``, ``time[Gyr]``, ``time_width[Myr]``.  Column detection is
    header-driven with a statistical fallback when the comment header is
    absent or non-standard.

    Parameters
    ----------
    sim_dir : str or Path
        Directory containing ``snapshot_times.txt``.
    sep : str, optional
        Separator passed to ``pd.read_csv``.  Default ``r'\s+'`` (whitespace
        regex).  Regex separators require ``engine='python'`` (applied
        automatically).

    Returns
    -------
    pandas.DataFrame
        Canonical columns plus any extras found in the file.  Missing
        canonical columns are present but filled with ``NaN``.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.
    FileNotFoundError
        If ``snapshot_times.txt`` is not found in *sim_dir*.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for read_snapshot_times. "
            "Install it with: pip install pandas"
        ) from exc

    snapshot_file = Path(sim_dir) / "snapshot_times.txt"
    if not snapshot_file.exists():
        raise FileNotFoundError(f"snapshot_times.txt not found in {sim_dir}")

    def _normalize(tok: str) -> str:
        s = tok.strip().lower()
        s = re.sub(r"[\[\]\(\)\,]", "", s)
        s = re.sub(r"[^0-9a-z]+", "_", s)
        return re.sub(r"_+", "_", s).strip("_")

    token_to_canonical = {
        "i": "snap", "snap": "snap", "index": "snap",
        "scale_factor": "scale-factor", "scale-factor": "scale-factor",
        "a": "scale-factor", "scalefactor": "scale-factor",
        "redshift": "redshift", "z": "redshift",
        "time_gyr": "time[Gyr]", "timegyr": "time[Gyr]",
        "time": "time[Gyr]", "t": "time[Gyr]",
        "lookback_time_gyr": "lookback-time[Gyr]",
        "lookback": "lookback-time[Gyr]",
        "lookback_time": "lookback-time[Gyr]",
        "time_width_myr": "time_width[Myr]",
        "timewidth": "time_width[Myr]",
        "time_width": "time_width[Myr]",
        "time-width": "time_width[Myr]",
    }

    # Collect comment lines; the last one with ≥2 alphabetic words is the header
    comment_lines: list[str] = []
    with open(snapshot_file, "r") as fh:
        for raw in fh:
            if raw.lstrip().startswith("#"):
                comment_lines.append(raw.rstrip("\n"))

    header_tokens: list[str] | None = None
    for line in reversed(comment_lines):
        words = re.split(r"\s+", line.lstrip("#").strip())
        if sum(bool(re.search(r"[A-Za-z]", w)) for w in words) >= 2:
            header_tokens = words
            break

    sep_to_use = sep if sep is not None else r'\s+'
    df = pd.read_csv(
        snapshot_file, sep=sep_to_use, comment="#",
        header=None, engine="python",
    )

    canonical_cols = ["snap", "scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]

    if df.shape[0] == 0:
        return pd.DataFrame(columns=canonical_cols)

    ncols = df.shape[1]
    mapped_names: list[str] | None = None

    if header_tokens:
        normed = [_normalize(t) for t in header_tokens]
        mapped = [token_to_canonical.get(n) for n in normed]
        alpha_idx = [i for i, t in enumerate(header_tokens) if re.search(r"[A-Za-z]", t)]
        plausible = [mapped[i] if mapped[i] is not None else header_tokens[i] for i in alpha_idx]
        if len(plausible) == ncols:
            mapped_names = plausible
        elif len(plausible) >= ncols:
            mapped_names = plausible[-ncols:]
        else:
            mapped_names = plausible + [f"col{j}" for j in range(len(plausible), ncols)]

    if mapped_names is None:
        # Statistical column-assignment fallback
        def _stats(col: int) -> dict:
            c = df[col].dropna().values.astype(float)
            return dict(
                min=c.min() if c.size else np.nan,
                max=c.max() if c.size else np.nan,
                is_int=bool(np.allclose(c, np.round(c), atol=1e-8)) if c.size else False,
                monotone=bool(np.all(np.diff(c) >= 0)) if c.size > 1 else True,
            )

        stats = {c: _stats(c) for c in df.columns}
        scorers = {
            "snap":            lambda c: 2 * stats[c]["is_int"] + stats[c]["monotone"],
            "scale-factor":    lambda c: 3 * int(0 <= stats[c]["min"] and stats[c]["max"] <= 1.2),
            "redshift":        lambda c: 2 * int(stats[c]["min"] >= 0 and stats[c]["max"] > 1)
                                         + int(stats[c]["max"] > 10),
            "time[Gyr]":       lambda c: 2 * int(-1 <= stats[c]["min"] and stats[c]["max"] <= 20),
            "time_width[Myr]": lambda c: int(stats[c]["max"] < 1e5),
        }
        assigned: dict[str, int] = {}
        available = set(df.columns)
        for canon, scorer in scorers.items():
            best = max(available, key=scorer, default=None)
            if best is not None and scorer(best) > 0:
                assigned[canon] = best
                available.remove(best)
        col_to_canon = {v: k for k, v in assigned.items()}
        mapped_names = [col_to_canon.get(c, f"col{c}") for c in df.columns]

    if len(mapped_names) != ncols:
        mapped_names = [f"col{i}" for i in range(ncols)]
    df.columns = mapped_names

    for col in canonical_cols:
        if col not in df.columns:
            df[col] = np.nan

    try:
        df["snap"] = pd.to_numeric(df["snap"], errors="coerce").astype("Int64")
    except Exception:
        df["snap"] = pd.to_numeric(df["snap"], errors="coerce")

    for c in ["scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    extra = [c for c in df.columns if c not in canonical_cols]
    return df[canonical_cols + extra].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FIRE evolving potential helper
# ---------------------------------------------------------------------------

def create_fire_evolving_ini(
    sim_dir: Union[Path, str],
    model_pattern: str,
    output_filename: str,
    snap_range: Optional[tuple[int, int]] = None,
    verbose: bool = True,
) -> str:
    """
    Write an Agama ``Evolving`` ``.ini`` file from a FIRE simulation directory.

    Reads snapshot times from ``snapshot_times.txt`` via
    :func:`read_snapshot_times`, filters by *snap_range*, and writes the
    config file into ``<sim_dir>/potential/10kpc/<output_filename>``.

    Parameters
    ----------
    sim_dir : str or Path
        FIRE simulation root containing ``snapshot_times.txt`` and
        ``potential/10kpc/``.
    model_pattern : str
        Pattern for individual snapshot filenames, e.g.
        ``"*.dark.none_4.coef_mul_spl"`` (``*`` is replaced by the integer
        snapshot number).
    output_filename : str
        Name of the ``.ini`` file to write inside ``potential/10kpc/``.
    snap_range : (int, int), optional
        Inclusive ``(start, end)`` snapshot range to include.  If ``None``,
        all available snapshots are used.
    verbose : bool, optional
        Print progress, by default ``True``.

    Returns
    -------
    str
        Absolute path of the written ``.ini`` file.

    Raises
    ------
    FileNotFoundError
        If ``snapshot_times.txt`` is absent or any expected coefficient file
        is missing.
    """
    sim_dir = Path(sim_dir)
    pot_dir = sim_dir / "potential" / "10kpc"
    pot_dir.mkdir(parents=True, exist_ok=True)

    df = read_snapshot_times(sim_dir)
    if snap_range is not None:
        df = df[(df["snap"] >= snap_range[0]) & (df["snap"] <= snap_range[1])]
    df = df.dropna(subset=["time[Gyr]", "snap"])

    times = df["time[Gyr]"].tolist()
    coef_paths: list[str] = []
    missing: list[str] = []
    for snap_num in df["snap"]:
        filename = str(int(snap_num)) + model_pattern.replace("*", "")
        path = pot_dir / filename
        coef_paths.append(str(path))
        if not path.exists():
            missing.append(str(path))

    if missing:
        sample = "\n".join(missing[:10]) + ("\n  ..." if len(missing) > 10 else "")
        raise FileNotFoundError(f"Missing {len(missing)} coefficient file(s):\n{sample}")

    output_path = pot_dir / output_filename
    result = create_evolving_ini(times, coef_paths, output_path)
    if verbose:
        print(f"Written: {result}  ({len(times)} snapshots)")
    return result


# ---------------------------------------------------------------------------
# FIRE potential loader  (Arora et al. 2022)
# ---------------------------------------------------------------------------

def load_fire_pot(
    sim_dir: Union[Path, str],
    nsnap: int,
    sym: str = "n",
    lmax: int = 4,
    kind: str = "whole",
    keep_lm_mult: Optional[list[tuple[int, int]]] = None,
    keep_m_cylspl: Optional[list[int]] = None,
    include_negative_m: bool = True,
    file_ext: str = "DR",
    out_acc: bool = False,
    halo: Optional[str] = None,
    verbose: bool = True,
    return_coefs: bool = False,
    save_modified: bool = False,
    save_dir: Optional[str] = None,
):
    """
    Load a FIRE potential snapshot as an Agama potential object.

    Reads pre-computed Multipole and CylSpline coefficient files following the
    FIRE ``potential/10kpc/`` layout (Arora et al. 2022).  Optionally zeroes
    selected harmonic terms before loading — working entirely in memory via
    the :class:`~agama_helper._coefs.MultipoleCoefs` /
    :class:`~agama_helper._coefs.CylSplineCoefs` dataclasses.

    Parameters
    ----------
    sim_dir : str or Path
        FIRE simulation root directory.
    nsnap : int
        Snapshot number.
    sym : str, optional
        Symmetry flag: ``'n'`` none (default), ``'a'`` axisymmetric,
        ``'s'`` spherical, ``'t'`` triaxial.
    lmax : int, optional
        Multipole lmax / CylSpline mmax order used in the filename, by default 4.
    kind : {'whole', 'dark', 'bar'}, optional
        Component to load, by default ``'whole'``.
    keep_lm_mult : list of (int, int), optional
        If provided, only these (l, m) pairs are retained in the Multipole
        expansion; all others are zeroed.
    keep_m_cylspl : list of int, optional
        If provided, only these azimuthal orders m are retained in the
        CylSpline expansion.
    include_negative_m : bool, optional
        Automatically include negative-m counterparts when filtering,
        by default ``True``.
    file_ext : str, optional
        Filename suffix after the expansion type token, by default ``"DR"``.
    out_acc : bool, optional
        Read from the ``out_acc/`` sub-directory, by default ``False``.
    halo : str, optional
        Halo label in filename (e.g. ``"MW"``, ``"LMC"``), by default
        ``None`` (omitted).
    verbose : bool, optional
        Print info messages, by default ``True``.
    return_coefs : bool, optional
        If ``True``, return the :class:`~agama_helper._coefs.MultipoleCoefs`
        dataclass instead of an ``agama.Potential``, by default ``False``.
    save_modified : bool, optional
        If ``True`` and coef modifications are requested, write the modified
        coefficient strings to *save_dir* (or the original directory if
        ``save_dir=None``), by default ``False``.
    save_dir : str, optional
        Directory for modified files when ``save_modified=True``.

    Returns
    -------
    agama.Potential, MultipoleCoefs, CylSplineCoefs, or tuple
        Normally an ``agama.Potential``.  When ``return_coefs=True``:
        a :class:`~agama_helper._coefs.MultipoleCoefs` for ``kind='dark'``,
        a :class:`~agama_helper._coefs.CylSplineCoefs` for ``kind='bar'``,
        or a ``(MultipoleCoefs, CylSplineCoefs)`` tuple for
        ``kind='whole'``.
    """
    try:
        import agama
    except ImportError as exc:
        raise ImportError("agama is required for load_fire_pot.") from exc

    sym_map = {"a": "axi", "s": "sph", "t": "triax", "n": "none"}
    if sym not in sym_map:
        raise ValueError(f"Unknown sym '{sym}'. Allowed: {list(sym_map)}")
    sym_label = sym_map[sym]

    sim_dir = Path(sim_dir)
    sub = "out_acc/" if out_acc else ""
    base = sim_dir / "potential" / "10kpc" / sub

    def _build_path(component: str, ext_suffix: str) -> Path:
        name = f"{nsnap}.{component}.{sym_label}_{lmax}"
        if halo:
            name += f".{halo}"
        name += ext_suffix
        if file_ext:
            name += f"_{file_ext}"
        return base / name

    dark_path = _build_path("dark", ".coef_mul")
    bar_path = _build_path("bar", ".coef_cylsp")
    if verbose:
        print(f"Multipole : {dark_path}")
        print(f"CylSpline : {bar_path}")

    # -- Multipole string (with optional selective zeroing) --
    def _prepare_mult() -> str:
        coef_str = dark_path.read_text(encoding="utf-8")
        if keep_lm_mult is not None:
            keep = _add_negative_m(keep_lm_mult) if include_negative_m else keep_lm_mult
            if verbose:
                print(f"Multipole keep (l,m): {keep}")
            coef_str = read_mult_coefs(coef_str).zeroed(keep).to_coef_string()
            if save_modified:
                out = Path(save_dir) / (dark_path.name + ".modified") if save_dir else dark_path.with_suffix(".modified")
                out.write_text(coef_str, encoding="utf-8")
                if verbose:
                    print(f"  Saved modified multipole → {out}")
        return coef_str

    # -- CylSpline string (with optional selective zeroing) --
    def _prepare_cylspl() -> str:
        coef_str = bar_path.read_text(encoding="utf-8")
        if keep_m_cylspl is not None:
            coefs = read_cylspl_coefs(coef_str).zeroed(keep_m_cylspl, include_negative=include_negative_m)
            if verbose:
                print(f"CylSpline keep m: {coefs.m_values}")
            coef_str = coefs.to_coef_string()
            if save_modified:
                out = Path(save_dir) / (bar_path.name + ".modified") if save_dir else bar_path.with_suffix(".modified")
                out.write_text(coef_str, encoding="utf-8")
                if verbose:
                    print(f"  Saved modified CylSpline → {out}")
        return coef_str

    # Early return: coef dataclass(es), respecting kind
    if return_coefs:
        if kind == "dark":
            return read_mult_coefs(_prepare_mult())
        if kind == "bar":
            return read_cylspl_coefs(_prepare_cylspl())
        # kind == "whole": return both
        return read_mult_coefs(_prepare_mult()), read_cylspl_coefs(_prepare_cylspl())

    # Materialise and build Agama potential objects
    tmps: list[str] = []
    dark_pot = bar_pot = None
    try:
        if kind in ("whole", "dark"):
            tmps.append(_write_tmp_coef(_prepare_mult()))
            dark_pot = agama.Potential(file=tmps[-1])
        if kind in ("whole", "bar"):
            tmps.append(_write_tmp_coef(_prepare_cylspl()))
            bar_pot = agama.Potential(file=tmps[-1])
    finally:
        for t in tmps:
            _cleanup_tmp_file(t)

    if kind == "dark":
        return dark_pot
    if kind == "bar":
        return bar_pot
    return agama.Potential(dark_pot, bar_pot)


# ---------------------------------------------------------------------------
# Cubic-spline resampling of a coefficient time series
# ---------------------------------------------------------------------------

def refine_times(times, factor: int = 10) -> np.ndarray:
    """
    Subdivide every interval of *times* by *factor*, keeping the original nodes.

    FIRE snapshot cadence is uneven — on ``m12i`` the spacing varies by ~12x
    across the run — so a plain ``np.linspace`` over the full range would *not*
    land on the original sample times.  Refining interval-by-interval keeps every
    original node, at indices ``0, factor, 2*factor, ...``, which makes node
    preservation directly checkable after a resample.

    Parameters
    ----------
    times : array-like, shape (nt,)
        Original sample times, strictly increasing.
    factor : int, optional
        Sub-intervals per original interval, by default 10.

    Returns
    -------
    ndarray, shape (factor * (nt - 1) + 1,)

    Examples
    --------
    >>> refine_times([0.0, 1.0, 3.0], factor=2)
    array([0. , 0.5, 1. , 2. , 3. ])
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            f"times must be a 1-D array of at least 2 samples; got shape {times.shape}."
        )
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing to be refined.")
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1; got {factor}.")

    pieces = [
        np.linspace(times[i], times[i + 1], factor + 1)[:-1]
        for i in range(len(times) - 1)
    ]
    return np.concatenate(pieces + [times[-1:]])


def _spline_block(block, times: np.ndarray, times_new: np.ndarray) -> np.ndarray:
    """Cubic-spline every coefficient series along the trailing time axis."""
    import agama

    lead, nt = np.shape(block)[:-1], np.shape(block)[-1]
    flat = np.asarray(block, dtype=float).reshape(-1, nt)
    out = np.empty((flat.shape[0], times_new.size), dtype=float)
    for k in range(flat.shape[0]):
        out[k] = agama.Spline(times, flat[k])(times_new)
    return out.reshape(lead + (times_new.size,))


def spline_resample_coefs(coefs, times_new):
    """
    Resample a coefficient time series onto a new time grid with cubic splines.

    Builds one ``agama.Spline`` per coefficient series along the time axis and
    evaluates it on *times_new*.  Grids and labels are untouched — only the
    trailing time axis changes — and the result is an ordinary coefficient object,
    so it materialises and writes like any other.

    Works for both expansion types.  Derivative blocks (``dphi_dr`` for Multipole,
    ``dphi_dR`` / ``dphi_dz`` for CylSpline) are resampled alongside ``phi`` when
    present; Agama cannot load a Multipole file without a ``#dPhi/dr`` section, so
    dropping it is never the right default.

    Parameters
    ----------
    coefs : MultipoleCoefs or CylSplineCoefs
        Must already carry a time axis.
    times_new : array-like, shape (nt_new,)
        New sample times.  Any length; see :func:`refine_times` for a grid that
        contains the original nodes.

    Returns
    -------
    MultipoleCoefs or CylSplineCoefs
        New object with ``n_times == len(times_new)``.

    Raises
    ------
    ImportError
        If agama is unavailable.
    ValueError
        If *coefs* has no time axis.

    Notes
    -----
    ``agama.Spline`` is a **natural** cubic spline.  ``scipy.interpolate.CubicSpline``
    agrees with it to machine precision *only* with ``bc_type="natural"``; its
    default ``"not-a-knot"`` differs by up to ~6e-3 relative near the endpoints on
    real FIRE coefficient series.  This helper therefore requires agama rather than
    silently falling back to a spline with different boundary conditions.

    Because the spline passes through every input node, resampling onto a grid that
    contains the original times reproduces those times' coefficients exactly.

    Examples
    --------
    >>> ser = read_coefs("mult_halo.h5", group_name="all")
    >>> fine = spline_resample_coefs(ser, refine_times(ser.times, 10))
    >>> fine.n_times, np.allclose(fine.phi[..., ::10], ser.phi)
    (3001, True)
    >>> pot = fine.materialize_potential()
    """
    from ._coefs import CylSplineCoefs, MultipoleCoefs

    try:
        import agama  # noqa: F401
    except ImportError:
        raise ImportError(
            "spline_resample_coefs requires agama for agama.Spline. "
            "scipy.interpolate.CubicSpline is equivalent only with "
            "bc_type='natural' — its default 'not-a-knot' gives a different "
            "spline near the endpoints — so no fallback is applied here."
        ) from None

    if not isinstance(coefs, (MultipoleCoefs, CylSplineCoefs)):
        raise TypeError(
            f"spline_resample_coefs expects MultipoleCoefs or CylSplineCoefs; got "
            f"{type(coefs).__name__}."
        )

    if not coefs.has_time_axis:
        raise ValueError(
            f"{type(coefs).__name__} has no time axis (times is None); there is "
            "nothing to resample. Build a series with read_coefs(group_name='all') "
            "or stack_coefs() first."
        )

    times = np.asarray(coefs.times, dtype=float)
    times_new = np.asarray(times_new, dtype=float)
    if times_new.ndim != 1:
        raise ValueError(
            f"times_new must be 1-D; got an array of shape {times_new.shape}."
        )

    if isinstance(coefs, MultipoleCoefs):
        return coefs.with_times(
            times_new,
            phi=_spline_block(coefs.phi, times, times_new),
            dphi_dr=(
                None if coefs.dphi_dr is None
                else _spline_block(coefs.dphi_dr, times, times_new)
            ),
        )

    if isinstance(coefs, CylSplineCoefs):
        extra = {}
        for name in ("dphi_dR", "dphi_dz"):
            tables = getattr(coefs, name)
            if tables is not None:
                extra[name] = {
                    m: _spline_block(v, times, times_new) for m, v in tables.items()
                }
        return coefs.with_times(
            times_new,
            phi={m: _spline_block(v, times, times_new) for m, v in coefs.phi.items()},
            **extra,
        )


# ---------------------------------------------------------------------------
# Comoving host frame: rotation, centre splines, centre acceleration
# ---------------------------------------------------------------------------
#
# The physical equation of motion in a comoving simulation is
#
#     r'' = -grad(Phi)(r, t) - u_dot(t),
#
# an exact change of variables away from the comoving/peculiar one (see
# :mod:`nbody_streams.utils._cosmology`).  The expansion terms cancel
# identically; the centre term ``u_dot`` does not.  These helpers build that
# term from the FIRE centre-of-mass splines and hand it to Agama as a
# ``UniformAcceleration`` component, so it composes with the evolving host
# potential like any other.

def read_rotation(
    sim_dir: Union[Path, str],
    nsnap: int = 600,
    spl: bool = True,
    subdir: str = "potential/10kpc",
) -> np.ndarray:
    """
    Read the (3, 3) rotation into the present-day principal-axis frame.

    Parameters
    ----------
    sim_dir : str or Path
        FIRE simulation root directory.
    nsnap : int, optional
        Snapshot defining the frame, by default 600.
    spl : bool, optional
        Use ``<nsnap>_coords_spl.txt`` rather than ``<nsnap>_coords.txt``,
        by default ``True``.
    subdir : str, optional
        Location of the coords files under *sim_dir*, by default
        ``"potential/10kpc"``.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.  Apply as ``xyz @ R.T``.

    Raises
    ------
    FileNotFoundError
        If the coords file does not exist.
    ValueError
        If fewer than three uncommented rows are present, or the last three do
        not form a (3, 3) block.

    Notes
    -----
    The rotation is taken as the **last three uncommented rows**, not a fixed
    line offset: m12m and m12f label the block with a
    ``# rotation to principal-axis frame`` comment that m12i and m12b omit, so
    any fixed ``skip_header`` is wrong for half the suite.
    """
    suffix = "_coords_spl.txt" if spl else "_coords.txt"
    path = Path(sim_dir) / subdir / f"{int(nsnap)}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"coords file not found: {path}")

    rows = [
        line for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < 3:
        raise ValueError(
            f"{path} has only {len(rows)} uncommented row(s); the rotation "
            "block is the last three."
        )

    rot = np.array([[float(v) for v in line.split()] for line in rows[-3:]], dtype=float)
    if rot.shape != (3, 3):
        raise ValueError(
            f"last three rows of {path} give shape {rot.shape}, not (3, 3)."
        )
    return rot


def read_center_splines(file_pattern: Union[Path, str]) -> list:
    """
    Load the three comoving centre-of-mass splines.

    Parameters
    ----------
    file_pattern : str or Path
        Path with one ``{}`` for the component, e.g.
        ``".../m12i_reg_spl_{}.pickle"``.  A ``.txt`` suffix reads the knot and
        coefficient columns written by :func:`write_center_splines` instead,
        and rebuilds an identical ``BSpline``; anything else is unpickled.

    Returns
    -------
    list of scipy.interpolate.BSpline
        Three splines giving the comoving centre [kpc] against time [Gyr], in
        x, y, z order.

    Raises
    ------
    FileNotFoundError
        If any of the three component files is missing.
    ValueError
        If a ``.txt`` file has no parsable ``degree N`` header.

    Notes
    -----
    The non-``.txt`` branch calls :func:`pickle.load`, which executes arbitrary
    code from the file.  Only load pickles you produced yourself; prefer the
    ``.txt`` form written by :func:`write_center_splines`, which additionally
    survives scipy and numpy upgrades.
    """
    from scipy import interpolate

    out = []
    for c in "xyz":
        path = Path(str(file_pattern).format(c))
        if not path.exists():
            raise FileNotFoundError(f"centre-spline file not found: {path}")
        if path.suffix == ".txt":
            header = path.read_text().splitlines()[0]
            match = re.search(r"degree\s+(\d+)", header)
            if match is None:
                raise ValueError(
                    f"{path}: expected a 'degree N' header written by "
                    f"write_center_splines; got {header!r}."
                )
            degree = int(match.group(1))
            knots, coeff = np.loadtxt(path, unpack=True)
            out.append(interpolate.BSpline(
                knots, coeff[:len(knots) - degree - 1], degree, extrapolate=True))
        else:
            with open(path, "rb") as fh:
                out.append(pickle.load(fh))
    return out


def write_center_splines(file_pattern: Union[Path, str], center_splines) -> list[str]:
    """
    Write the three centre splines as ``(knots, coefficients)`` text.

    Reload with :func:`read_center_splines`, which reconstructs the same
    ``scipy.interpolate.BSpline`` bit for bit.  Unlike a pickle this survives
    scipy and numpy upgrades, and carries no code-execution risk on read.

    Parameters
    ----------
    file_pattern : str or Path
        Path with one ``{}`` for the component, e.g.
        ``".../m12i_reg_spl_{}.txt"``.  Must end in ``.txt`` to be readable
        back by :func:`read_center_splines`.
    center_splines : sequence of scipy.interpolate.BSpline
        Three BSplines, in x, y, z order.

    Returns
    -------
    list of str
        The three paths written.

    Raises
    ------
    ValueError
        If *center_splines* does not hold exactly three splines, or a spline
        has more coefficients than knots.
    """
    splines = list(center_splines)
    if len(splines) != 3:
        raise ValueError(f"expected 3 centre splines (x, y, z); got {len(splines)}.")

    written: list[str] = []
    for c, sp in zip("xyz", splines):
        if len(sp.c) > len(sp.t):
            raise ValueError(
                f"spline '{c}' has {len(sp.c)} coefficients but only {len(sp.t)} "
                "knots; it cannot be stored as two aligned columns."
            )
        col = np.zeros(len(sp.t))
        col[:len(sp.c)] = sp.c
        path = Path(str(file_pattern).format(c))
        np.savetxt(
            path, np.column_stack([sp.t, col]), fmt="%.17g",
            header=f"degree {sp.k}, {len(sp.c)} coefficients\nknot  coefficient",
        )
        written.append(str(path))
    return written


def center_acceleration_table(
    center_splines,
    rotation: np.ndarray,
    cosmo,
    t_range: Optional[tuple[float, float]] = None,
    n_samples: int = 1201,
) -> np.ndarray:
    """
    Tabulate ``-u_dot``, the galactic-centre correction, on a uniform time grid.

    This is the ``(n_samples, 4)`` array behind :func:`center_acceleration`;
    see there for the physics and for the parameters.

    Parameters
    ----------
    center_splines : str, Path, or sequence
        A ``{}`` path pattern, or three splines from
        :func:`read_center_splines`.
    rotation : ndarray, shape (3, 3)
        From :func:`read_rotation`.
    cosmo : nbody_streams.utils.FlatLCDM
        Background cosmology.
    t_range : (float, float), optional
        Range [Gyr] to tabulate; defaults to the splines' knot range.
    n_samples : int, optional
        Number of rows, by default 1201.

    Returns
    -------
    ndarray, shape (n_samples, 4)
        Columns ``t`` [Gyr] and ``-u_dot`` x, y, z [(km/s)^2/kpc], in the
        integration frame.

    Raises
    ------
    ValueError
        If *rotation* is not (3, 3), *t_range* is empty, or *n_samples* < 2.

    Notes
    -----
    *t_range* is taken as given and is **not** clipped to the splines' knot
    range, so it can reach the last snapshot: the FIRE centre splines end at
    snapshot 598, 4.4 Myr short of snapshot 600, and initial conditions at the
    present day would otherwise sit outside the table.  The splines extrapolate
    with their final polynomial piece over that gap, which is what the
    pipeline's own ``599/600_coords_spl.txt`` already contain.
    """
    if isinstance(center_splines, (str, os.PathLike)):
        center_splines = read_center_splines(center_splines)

    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3); got {rotation.shape}.")

    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2; got {n_samples}.")

    if t_range is None:
        s = center_splines[0]
        lo, hi = (float(s.t[s.k]), float(s.t[-s.k - 1])) if hasattr(s, "k") \
            else (float(s.x[0]), float(s.x[-1]))
    else:
        lo, hi = float(min(t_range)), float(max(t_range))
        if hi <= lo:
            raise ValueError(f"t_range {tuple(t_range)} is empty")

    t = np.linspace(lo, hi, n_samples)
    a = cosmo.scale_factor(t)
    H = cosmo.hubble_parameter(a=a)
    d1 = np.column_stack([sp.derivative(1)(t) for sp in center_splines])   # kpc/Gyr
    d2 = np.column_stack([sp.derivative(2)(t) for sp in center_splines])   # kpc/Gyr^2

    K = KPC_PER_GYR_PER_KMS
    u_dot = (a[:, None] * d2 / K ** 2 + (a * H)[:, None] * d1 / K) @ rotation.T
    return np.column_stack([t, -u_dot])


def write_center_acceleration(
    path: Union[Path, str],
    center_splines,
    rotation: np.ndarray,
    cosmo,
    t_range: Optional[tuple[float, float]] = None,
    n_samples: int = 1201,
    ini: bool = True,
) -> str:
    """
    Write the centre-acceleration table, to reload without the splines.

    Reload with ``agama.Potential(type="UniformAcceleration", file=path)``,
    with ``agama.Potential(<path>.ini)``, or on the GPU with
    ``PotentialGPU(type="UniformAcceleration", file=path)``; all give forces
    identical to :func:`center_acceleration`, since the table is parsed by the
    same reader either way.  The ``.ini`` can be inlined as a component of a
    master file alongside the host.

    Parameters
    ----------
    path : str or Path
        Destination for the 4-column table.
    center_splines : str, Path, or sequence
        A ``{}`` path pattern, or three splines from
        :func:`read_center_splines`.
    rotation : ndarray, shape (3, 3)
        From :func:`read_rotation`.
    cosmo : nbody_streams.utils.FlatLCDM
        Background cosmology.
    t_range : (float, float), optional
        Range [Gyr] to tabulate; defaults to the splines' knot range.
    n_samples : int, optional
        Table length, by default 1201.
    ini : bool, optional
        Also write a one-section ``.ini`` beside it, by default ``True``.

    Returns
    -------
    str
        The path written.
    """
    table = center_acceleration_table(center_splines, rotation, cosmo, t_range, n_samples)
    path = Path(path)
    np.savetxt(
        path, table, fmt="%.17g",
        header="t [Gyr]   -u_dot x y z [(km/s)^2/kpc], integration frame",
    )
    if ini:
        path.with_suffix(".ini").write_text(
            "[Potential]\ntype = UniformAcceleration\n"
            f"file = {path.resolve()}\n"
        )
    return str(path)


def center_acceleration(
    center_splines,
    rotation: np.ndarray,
    cosmo,
    t_range: Optional[tuple[float, float]] = None,
    n_samples: int = 1201,
    gpu: bool = False,
):
    r"""
    Galactic-centre correction as a ``UniformAcceleration`` potential.

    The centre's peculiar velocity is ``u = a x_com'``, so

    .. code-block:: text

        u_dot = a x_com'' + H a x_com',

    a *physical* acceleration despite ``x_com`` being comoving.  With
    ``x_com'`` in kpc/Gyr, ``x_com''`` in kpc/Gyr^2 and
    ``1 Gyr = K kpc/(km/s)``,

    .. code-block:: text

        u_dot [(km/s)^2/kpc] = a x_com''/K^2 + a H x_com'/K,

    rotated into the integration frame.  The potential carries ``-u_dot``, a
    fictitious force subtracted from the host force.

    Parameters
    ----------
    center_splines : str, Path, or sequence
        A ``{}`` path pattern, or three splines from
        :func:`read_center_splines`.
    rotation : ndarray, shape (3, 3)
        From :func:`read_rotation`.
    cosmo : nbody_streams.utils.FlatLCDM
        Background cosmology.
    t_range : (float, float), optional
        Range [Gyr] to tabulate; defaults to the splines' knot range.  Agama
        extrapolates linearly beyond the table.
    n_samples : int, optional
        Table length, by default 1201.  Agama re-splines it, so this is not the
        snapshot count: a 5 Gyr orbit shifts by 50 pc at 301 samples, 4 pc at
        601, and 1201 sits at the integrator's noise floor.
    gpu : bool, optional
        Return a :class:`~agama_helper.PotentialGPU` instead of an
        ``agama.Potential``, by default ``False``.  Both interpolate the table
        with Agama's regularized natural cubic spline, so the two agree to
        machine precision.

    Returns
    -------
    agama.Potential or PotentialGPU
        Compose with the host, e.g. ``agama.Potential(host, acc)`` on the CPU
        or ``host_gpu + acc_gpu`` on the GPU.

    See Also
    --------
    write_center_acceleration : same table, written to disk.
    nbody_streams.utils.FlatLCDM

    Notes
    -----
    Units are the package-wide (Msol, kpc, km/s), set by
    ``agama_helper``'s import-time ``agama.setUnits`` call.

    The kpc/Gyr^2 conversion **divides** by ``K^2``; multiplying is wrong by
    9.4 percent.

    Time units are the usual trap.  Agama's integration variable is
    kpc/(km/s) = 0.977792 Gyr, and that one variable drives both the dynamics
    and an ``Evolving`` potential's clock.  Three things must share a
    convention: the ``.ini`` timestamps, the time column of this table, and
    ``timestart``/``time``.  Stamping the ``.ini`` in Gyr keeps the snapshot
    sequence exact but makes elapsed time 2.27 percent short; dividing all
    three by 0.977792 makes both exact.  Mixing the two desynchronises the
    centre correction from the host and is not benign.  The centre splines are
    not among the three -- they stay fit in Gyr and are evaluated in Gyr here.

    The m12i centre splines are quintic smoothing fits with crowded end knots
    and their second derivative spikes there: ``|u_dot|`` reaches
    1396 (km/s)^2/kpc at the first knot against 40-160 through the interior,
    while the position residuals stay a flat 0.2-0.4 kpc.  Pass *t_range* to
    trim the end intervals if it matters.

    Examples
    --------
    >>> from nbody_streams import agama_helper as ah
    >>> from nbody_streams.utils import FlatLCDM
    >>> cosmo = FlatLCDM.from_snapshot_times(sim_dir)
    >>> rot = ah.read_rotation(sim_dir, nsnap=600)
    >>> acc = ah.center_acceleration(f"{sim_dir}/m12i_reg_spl_{{}}.txt", rot, cosmo)
    >>> host = ah.load_agama_evolving_potential("m12i_mult.h5", times)
    >>> pot = agama.Potential(host, acc)
    """
    table = center_acceleration_table(center_splines, rotation, cosmo, t_range, n_samples)

    if gpu:
        from ._potential import PotentialGPU
        return PotentialGPU(type="UniformAcceleration", file=table)

    try:
        import agama
    except ImportError as exc:
        raise ImportError("agama is required for center_acceleration.") from exc
    return agama.Potential(type="UniformAcceleration", file=table)
