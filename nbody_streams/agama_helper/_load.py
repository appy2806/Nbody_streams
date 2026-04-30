"""
agama_helper._load
~~~~~~~~~~~~~~~~~~
Load Agama potential objects from coefficient files, HDF5 archives, or
in-memory coefficient strings.

All temporary files are created in RAM-backed storage (``/dev/shm`` on
Linux/WSL2) and cleaned up immediately after the potential is constructed,
even on failure.

Public surface
--------------
load_agama_potential           -- single snapshot, any source type
load_agama_evolving_potential  -- time-evolving, from HDF5 or Agama .ini file
create_evolving_ini            -- write an Agama Evolving .ini file
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import h5py

import numpy as np

from ._coefs import CylSplineCoefs, MultipoleCoefs
from ._io import (
    _cleanup_tmp_file,
    _extract_int_from_group,
    _resolve_center,
    _resolve_coef_string,
    _write_tmp_coef,
)


def _parse_evolving_ini(ini_path: Path) -> tuple[list[float], list[str], bool]:
    """
    Parse an Agama ``Evolving`` potential ``.ini`` config file.

    Returns
    -------
    times : list of float
    coef_paths : list of str
        Absolute paths resolved relative to the ``.ini`` file's directory.
    interp_linear : bool
    """
    text = ini_path.read_text(encoding="utf-8")
    in_timestamps = False
    times: list[float] = []
    coef_paths: list[str] = []
    interp_linear = True
    ini_dir = ini_path.parent

    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("["):
            continue
        if s.lower() == "timestamps":
            in_timestamps = True
            continue
        if in_timestamps:
            parts = s.split(None, 1)
            if len(parts) == 2:
                times.append(float(parts[0]))
                p = Path(parts[1].strip())
                coef_paths.append(str(p if p.is_absolute() else ini_dir / p))
        elif "=" in s:
            k, _, v = s.partition("=")
            if k.strip().lower() == "interplinear":
                interp_linear = v.strip().lower() not in ("false", "0", "no")

    return times, coef_paths, interp_linear


def _require_agama():
    try:
        import agama
        return agama
    except ImportError:
        raise ImportError(
            "agama is required for loading potentials. "
            "See https://github.com/GalacticDynamics-Oxford/Agama"
        )


# ---------------------------------------------------------------------------
# Single snapshot
# ---------------------------------------------------------------------------

def load_agama_potential(
    source: Union[str, Path, "MultipoleCoefs", "CylSplineCoefs"],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
    center=None,
    keep_lm_mult: Optional[list] = None,
    keep_m_cylspl: Optional[list[int]] = None,
    include_negative_m: bool = True,
    gpu: bool = False,
):
    """
    Load a single-snapshot Agama potential from any coefficient source.

    Accepts four source types transparently:

    * **Plain-text file** — path to a ``.coef_mult`` or ``.coef_cylsp`` file
      produced by Agama's ``Potential.export()``.
    * **HDF5 archive** — path to a ``.h5``/``.hdf5`` file written by
      :func:`~agama_helper._io.write_coef_to_h5`; the dataset is read from
      ``group_name/dataset_name``.
    * **Raw coefficient string** — the text content of a coef file, e.g. the
      output of :meth:`~agama_helper._coefs.MultipoleCoefs.to_coef_string`.
    * **Coef dataclass** — a :class:`~agama_helper._coefs.MultipoleCoefs` or
      :class:`~agama_helper._coefs.CylSplineCoefs` instance (e.g. after
      calling ``.zeroed(...)``); serialised to string automatically.

    Optionally filters harmonic terms in memory before constructing the
    ``agama.Potential``, then cleans up all temporary files.

    Parameters
    ----------
    source : str, Path, MultipoleCoefs, or CylSplineCoefs
        Any of the four source types described above.
    group_name : str, optional
        HDF5 group when *source* is an HDF5 file, by default ``"snap_000"``.
    dataset_name : str, optional
        HDF5 dataset within the group, by default ``"coefs"``.
    center : array-like or str or Path, optional
        Galactic centre passed to ``agama.Potential``.  Accepted forms:

        * Length-3 sequence ``[x, y, z]`` — static centre.
        * (N, 4) array ``[time, x, y, z]`` — time-varying centre; written to
          a temporary file automatically.
        * (N, 7) array ``[time, x, y, z, vx, vy, vz]`` — time-varying centre
          + velocity.
        * File path (str or Path) — passed through to Agama as-is.
    keep_lm_mult : list of int or (int, int), optional
        **Multipole only.** Zero all (l, m) terms *not* matching this list.
        Each element may be an int *l* (keep all m for that order) or a
        ``(l, m)`` tuple.  Raises :exc:`TypeError` if source is CylSpline.
    keep_m_cylspl : list of int, optional
        **CylSpline only.** Zero all azimuthal orders *not* in this list.
        Raises :exc:`TypeError` if source is Multipole.
    include_negative_m : bool, optional
        Auto-include negative-m counterparts when filtering, by default
        ``True``.
    gpu : bool, optional
        When ``True``, return a :class:`~agama_helper._potential.PotentialGPU`
        object instead of an ``agama.Potential``.  Requires CuPy and nvcc.
        All filtering (``keep_lm_mult`` / ``keep_m_cylspl``) is applied before
        building the GPU potential.  ``center`` is forwarded to
        ``ShiftedPotentialGPU``; supply it as a length-3 array or an (N,4) /
        (N,7) trajectory array.

    Returns
    -------
    agama.Potential or GPU potential object
    """
    agama = _require_agama()

    # Resolve source to a coef string
    if isinstance(source, (MultipoleCoefs, CylSplineCoefs)):
        coef_str = source.to_coef_string()
    else:
        coef_str = _resolve_coef_string(source, group_name, dataset_name)

    # Reject Evolving configs early — before any filtering checks
    from ._coefs import _detect_expansion_type
    exp_type = _detect_expansion_type(coef_str)
    if exp_type == "Evolving":
        raise TypeError(
            "The source contains an Evolving (time-varying) potential config. "
            "Use load_agama_evolving_potential() instead of load_agama_potential()."
        )

    # Apply selective harmonic filtering when requested
    if keep_lm_mult is not None or keep_m_cylspl is not None:
        from ._coefs import read_cylspl_coefs, read_mult_coefs

        if keep_lm_mult is not None:
            if exp_type != "Multipole":
                raise TypeError(
                    f"keep_lm_mult was specified, but the expansion type is "
                    f"'{exp_type}' (not 'Multipole'). "
                    "Use keep_m_cylspl for CylSpline expansions."
                )
            coef_str = read_mult_coefs(coef_str).zeroed(keep_lm_mult).to_coef_string()
        elif keep_m_cylspl is not None:
            if exp_type != "CylSpline":
                raise TypeError(
                    f"keep_m_cylspl was specified, but the expansion type is "
                    f"'{exp_type}' (not 'CylSpline'). "
                    "Use keep_lm_mult for Multipole expansions."
                )
            coef_str = (
                read_cylspl_coefs(coef_str)
                .zeroed(keep_m_cylspl, include_negative=include_negative_m)
                .to_coef_string()
            )

    # --- GPU path ---
    if gpu:
        from ._potential import PotentialGPU, ShiftedPotentialGPU
        tmp = _write_tmp_coef(coef_str)
        try:
            pot = PotentialGPU(file=tmp)
        finally:
            _cleanup_tmp_file(tmp)
        if center is not None:
            pot = ShiftedPotentialGPU(pot, center=np.asarray(center, dtype=np.float64))
        return pot

    # Resolve center (handles static, time-varying array, and file path)
    center_arg, center_tmp = _resolve_center(center) if center is not None else (None, None)

    tmp = _write_tmp_coef(coef_str)
    try:
        pot = (
            agama.Potential(file=tmp)
            if center_arg is None
            else agama.Potential(file=tmp, center=center_arg)
        )
    finally:
        _cleanup_tmp_file(tmp)
        if center_tmp is not None:
            _cleanup_tmp_file(center_tmp)
    return pot


# ---------------------------------------------------------------------------
# Time-evolving potential
# ---------------------------------------------------------------------------

def load_agama_evolving_potential(
    source: Union[str, Path],
    times: Optional[Sequence[float]] = None,
    *,
    group_names: Optional[Sequence[str]] = None,
    dataset_name: str = "coefs",
    center=None,
    interp_linear: bool = True,
    keep_lm_mult: Optional[list] = None,
    keep_m_cylspl: Optional[list[int]] = None,
    include_negative_m: bool = True,
    gpu: bool = False,
):
    """
    Build a time-evolving Agama potential from an HDF5 archive or a native
    Agama ``Evolving`` ``.ini`` config file.

    Accepts two source types:

    * **HDF5 archive** (``.h5``/``.hdf5``) — groups contain coefficient
      strings written by :func:`~agama_helper._io.write_snapshot_coefs_to_h5`.
      Times can be embedded in the archive (``"times"`` dataset) or supplied
      via *times*.
    * **Agama Evolving ``.ini`` file** — Agama's native config format::

          [Potential]
          type = Evolving
          Timestamps
          <time1>  <path/to/snap1.coef_mult>
          <time2>  <path/to/snap2.coef_mult>
          ...

      Paths in the ``.ini`` are resolved relative to the file's directory.
      Useful when Agama already generated or you already have an ``.ini``.

    In both cases, filtering (``keep_lm_mult`` / ``keep_m_cylspl``) is applied
    to every snapshot in memory before construction, and all temporary files
    are removed afterwards even if construction fails.

    Parameters
    ----------
    source : str or Path
        Path to an HDF5 archive or an Agama Evolving ``.ini`` file.
    times : sequence of float, optional
        Simulation times [Gyr or internal units].  For HDF5 sources, if
        omitted the embedded ``"times"`` dataset is used.  For ``.ini``
        sources, if omitted the timestamps in the file are used.  An explicit
        value always overrides stored/parsed times.
    group_names : sequence of str, optional
        **HDF5 only.** Explicit group ordering.  If ``None`` (default), all
        groups are used, sorted numerically (``"snap_042"`` → 42).
    dataset_name : str, optional
        **HDF5 only.** Dataset name within each group, by default ``"coefs"``.
    center : array-like or str or Path, optional
        Galactic centre — same forms as :func:`load_agama_potential`.
    interp_linear : bool, optional
        Linear (``True``, default) vs. cubic-spline interpolation.  For
        ``.ini`` sources, the value parsed from the file is used unless this
        argument overrides it.
    keep_lm_mult : list of int or (int, int), optional
        **Multipole only.** Filter applied to every snapshot.
        Raises :exc:`TypeError` if the coef files are CylSpline.
    keep_m_cylspl : list of int, optional
        **CylSpline only.** Filter applied to every snapshot.
        Raises :exc:`TypeError` if the coef files are Multipole.
    include_negative_m : bool, optional
        Auto-include negative-m counterparts when filtering, by default
        ``True``.
    gpu : bool, optional
        When ``True``, return an
        :class:`~agama_helper._potential.EvolvingPotentialGPU` object instead
        of an ``agama.Potential``.  Each snapshot is loaded as a GPU potential
        and interpolated linearly (``interp_linear=True``) or with nearest-
        neighbour (``interp_linear=False``).  ``center`` is forwarded to
        ``ShiftedPotentialGPU``.

    Returns
    -------
    agama.Potential or EvolvingPotentialGPU
        Time-evolving potential object.

    Raises
    ------
    ValueError
        If times cannot be resolved, or if the number of snapshots does not
        match the number of times.
    """
    agama = _require_agama()

    source = Path(source)
    is_h5 = source.suffix.lower() in (".h5", ".hdf5")

    # ------------------------------------------------------------------
    # Resolve coef sources and times depending on source type
    # ------------------------------------------------------------------
    if is_h5:
        with h5py.File(source, "r") as f:
            all_groups = [k for k in f.keys() if k != "times"]
            stored_times = np.asarray(f["times"]) if "times" in f else None

        if group_names is None:
            group_names = sorted(all_groups, key=_extract_int_from_group)
        else:
            group_names = list(group_names)

        if times is not None:
            resolved_times = list(times)
        elif stored_times is not None:
            resolved_times = stored_times.tolist()
        else:
            raise ValueError(
                "times was not provided and no 'times' dataset was found in "
                f"{source}. Pass times explicitly or embed them when writing "
                "with write_snapshot_coefs_to_h5(times=...)."
            )

        if len(group_names) != len(resolved_times):
            raise ValueError(
                f"len(group_names)={len(group_names)} does not match "
                f"len(times)={len(resolved_times)}."
            )

        def _iter_coef_strings():
            for grp in group_names:
                yield _resolve_coef_string(source, grp, dataset_name)

        ini_interp_linear = interp_linear  # no override from file for H5

    else:
        # Native Agama .ini file
        ini_times, coef_paths, ini_interp_linear = _parse_evolving_ini(source)

        resolved_times = list(times) if times is not None else ini_times
        if not resolved_times:
            raise ValueError(
                f"No timestamps found in {source}. "
                "Check that the file has a 'Timestamps' section."
            )
        if len(coef_paths) != len(resolved_times):
            raise ValueError(
                f"len(coef_paths)={len(coef_paths)} does not match "
                f"len(times)={len(resolved_times)} parsed from {source}."
            )

        def _iter_coef_strings():
            for p in coef_paths:
                yield Path(p).read_text(encoding="utf-8")

    # Use ini_interp_linear only as a fallback when caller didn't touch the default
    # (we always honour the caller's explicit interp_linear kwarg)

    # ------------------------------------------------------------------
    # Build optional per-snapshot filter
    # ------------------------------------------------------------------
    _filter_fn = None
    if keep_lm_mult is not None or keep_m_cylspl is not None:
        from ._coefs import _detect_expansion_type, read_cylspl_coefs, read_mult_coefs

        # Detect type from the first snapshot
        _probe = next(_iter_coef_strings())
        _exp_type = _detect_expansion_type(_probe)
        if keep_lm_mult is not None:
            if _exp_type != "Multipole":
                raise TypeError(
                    f"keep_lm_mult was specified, but the expansion type is "
                    f"'{_exp_type}' (not 'Multipole'). "
                    "Use keep_m_cylspl for CylSpline expansions."
                )
            def _filter_fn(cs):  # noqa: E306
                return read_mult_coefs(cs).zeroed(keep_lm_mult).to_coef_string()
        else:
            if _exp_type != "CylSpline":
                raise TypeError(
                    f"keep_m_cylspl was specified, but the expansion type is "
                    f"'{_exp_type}' (not 'CylSpline'). "
                    "Use keep_lm_mult for Multipole expansions."
                )
            def _filter_fn(cs):  # noqa: E306
                return (
                    read_cylspl_coefs(cs)
                    .zeroed(keep_m_cylspl, include_negative=include_negative_m)
                    .to_coef_string()
                )

    # ------------------------------------------------------------------
    # GPU path: build EvolvingPotentialGPU from per-snapshot GPU pots
    #
    # Performance notes vs Agama CPU:
    #   - Per snapshot, _build_multipole_data / _build_cylspline_data run
    #     NumPy/SciPy quintic spline construction (band-solve per (l,m) pair).
    #     Agama does the equivalent in C++, which is ~100x faster per snapshot.
    #   - We avoid the temp-file round-trip (the coef string is already in RAM)
    #     and use a ThreadPoolExecutor to parallelise the CPU-side spline work
    #     across snapshots.  NumPy/SciPy release the GIL, so threads achieve
    #     genuine parallelism.  GPU uploads stay on the main thread to avoid
    #     CUDA context complications.
    # ------------------------------------------------------------------
    if gpu:
        import os as _os
        from concurrent.futures import ThreadPoolExecutor
        from ._potential import (
            EvolvingPotentialGPU, ShiftedPotentialGPU,
            MultipolePotentialGPU, CylSplinePotentialGPU,
            _build_multipole_data, _build_cylspline_data,
        )
        from ._coefs import MultipoleCoefs, CylSplineCoefs, read_coefs as _rc

        # Materialise all coef strings up-front (generator can only be iterated once).
        all_cs: list[str] = []
        for cs in _iter_coef_strings():
            if _filter_fn is not None:
                cs = _filter_fn(cs)
            all_cs.append(cs)

        def _cpu_build(cs: str):
            """Parse coef string and run spline construction — CPU only, no GPU."""
            mc = _rc(cs)
            if isinstance(mc, MultipoleCoefs):
                return ('mul', _build_multipole_data(mc))
            else:
                return ('cyl', _build_cylspline_data(mc))

        # Parallel CPU spline construction across snapshots.
        n_workers = min(len(all_cs), _os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            built = list(pool.map(_cpu_build, all_cs))

        # GPU upload — sequential in main thread (safe, no context issues).
        gpu_pots: list = []
        for kind, data in built:
            if kind == 'mul':
                gpu_pots.append(MultipolePotentialGPU._from_data(data))
            else:
                gpu_pots.append(CylSplinePotentialGPU._from_data(data))

        use_interp = ini_interp_linear if not is_h5 else interp_linear
        pot = EvolvingPotentialGPU(gpu_pots, resolved_times, interpolate=bool(use_interp))
        if center is not None:
            pot = ShiftedPotentialGPU(pot, center=np.asarray(center, dtype=np.float64))
        return pot

    # ------------------------------------------------------------------
    # Materialise coef strings → temp files, build Evolving config
    # ------------------------------------------------------------------
    center_arg, center_tmp = _resolve_center(center) if center is not None else (None, None)

    tmp_paths: list[str] = []
    config_path: Optional[str] = None
    try:
        for cs in _iter_coef_strings():
            if _filter_fn is not None:
                cs = _filter_fn(cs)
            tmp_paths.append(_write_tmp_coef(cs))

        use_interp = ini_interp_linear if not is_h5 else interp_linear
        config = (
            "[Potential]\n"
            "type = Evolving\n"
            f"interpLinear = {bool(use_interp)}\n"
            "Timestamps\n"
            + "\n".join(f"{t} {p}" for t, p in zip(resolved_times, tmp_paths))
        )
        config_path = _write_tmp_coef(config)
        pot = (
            agama.Potential(file=config_path)
            if center_arg is None
            else agama.Potential(file=config_path, center=center_arg)
        )
    finally:
        for p in tmp_paths:
            _cleanup_tmp_file(p)
        if config_path:
            _cleanup_tmp_file(config_path)
        if center_tmp is not None:
            _cleanup_tmp_file(center_tmp)

    return pot


# ---------------------------------------------------------------------------
# Generic Evolving .ini writer
# ---------------------------------------------------------------------------

def create_evolving_ini(
    times: Sequence[float],
    coef_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    interp_linear: bool = True,
) -> str:
    """
    Write an Agama ``Evolving`` potential ``.ini`` config from explicit file paths.

    For FIRE simulations use :func:`~agama_helper._fire.create_fire_evolving_ini`,
    which reads snapshot times from ``snapshot_times.txt`` automatically.

    Parameters
    ----------
    times : sequence of float
        Simulation times, one per coefficient file.
    coef_paths : sequence of str or Path
        Absolute paths to coefficient files, one per snapshot.
    output_path : str or Path
        Destination ``.ini`` file.  Parent directories are created as needed.
    interp_linear : bool, optional
        ``True`` (default) for linear interpolation.

    Returns
    -------
    str
        Absolute path of the written ``.ini`` file.
    """
    if len(times) != len(coef_paths):
        raise ValueError(
            f"times (len={len(times)}) and coef_paths (len={len(coef_paths)}) "
            "must have the same length."
        )
    lines = [
        "[Potential]",
        "type = Evolving",
        f"interpLinear = {bool(interp_linear)}",
        "Timestamps",
    ] + [f"{t} {p}" for t, p in zip(times, coef_paths)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)
