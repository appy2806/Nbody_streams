"""
agama_helper._fire
~~~~~~~~~~~~~~~~~~
FIRE-simulation-specific convenience wrappers (Arora et al. 2022).

These functions encode FIRE path conventions (``potential/10kpc/``,
``snapshot_times.txt``) and are not needed for generic Agama workflows.
All heavy-lifting is delegated to the generic modules
(:mod:`agama_helper._io`, :mod:`agama_helper._load`, :mod:`agama_helper._coefs`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

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
