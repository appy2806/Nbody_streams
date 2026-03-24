"""
agama_helper._io
~~~~~~~~~~~~~~~~
HDF5 archive I/O, fast temporary-file helpers, and the shared source-resolution
helper used by all coefficient readers.

Public surface
--------------
write_coef_to_h5              -- store one coef string in an HDF5 group
write_snapshot_coefs_to_h5   -- batch-write many snapshots
read_coef_string              -- read raw coef text from a file or HDF5 group

Everything else (underscore-prefixed) is for internal use only.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Temporary-file helpers  (internal)
# ---------------------------------------------------------------------------

def _get_fast_tmp_dir() -> str:
    """Return fastest writable temp dir: /dev/shm (RAM-backed on Linux) or system tmp."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return str(shm)
    return tempfile.gettempdir()


def _write_tmp_coef(coef_string: str) -> str:
    """Write *coef_string* to a fast temp file; caller owns cleanup."""
    path = Path(_get_fast_tmp_dir()) / f"agama_{uuid.uuid4().hex}.coef"
    path.write_text(coef_string, encoding="utf-8")
    return str(path)


def _cleanup_tmp_file(path: Union[str, Path]) -> None:
    """Remove a temp coef file, ignoring missing-file errors."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _extract_int_from_group(name: str) -> int:
    """Extract first integer from a group name for numeric sorting (e.g. 'snap_042' → 42)."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else abs(hash(name)) % (10 ** 9)


def _resolve_center(center) -> Tuple[Any, Optional[str]]:
    """
    Prepare a ``center`` value for ``agama.Potential``.

    Agama accepts three forms:

    * **Static** — a length-3 sequence ``[x, y, z]``.
    * **Time-varying position** — an (N, 4) array with columns
      ``[time, x, y, z]``; written to a temp file.
    * **Time-varying position + velocity** — an (N, 7) array with columns
      ``[time, x, y, z, vx, vy, vz]``; written to a temp file.
    * **File path** — a ``str`` or :class:`~pathlib.Path` already pointing to
      a file in one of the above tabular formats; passed through as-is.

    Returns
    -------
    center_arg : any
        Value to pass as ``center=`` to ``agama.Potential``.
    tmp_path : str or None
        Path of a temporary file created here; ``None`` if no file was
        created.  **Caller is responsible for cleanup.**
    """
    if isinstance(center, (str, Path)):
        return str(center), None

    arr = np.asarray(center, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(
                f"A 1-D center must have exactly 3 elements [x, y, z]; "
                f"got {arr.shape[0]}."
            )
        return arr.tolist(), None
    elif arr.ndim == 2:
        if arr.shape[1] not in (4, 7):
            raise ValueError(
                f"A 2-D center array must have 4 columns (time, x, y, z) or "
                f"7 columns (time, x, y, z, vx, vy, vz); got shape {arr.shape}."
            )
        content = "\n".join(" ".join(f"{v:.14g}" for v in row) for row in arr)
        tmp = _write_tmp_coef(content)
        return tmp, tmp
    else:
        raise ValueError(
            f"center must be a 1-D sequence (length 3), a 2-D array "
            f"(N×4 or N×7), or a file path; got array of shape {arr.shape}."
        )


# ---------------------------------------------------------------------------
# Source resolution  (internal, shared with _coefs.py)
# ---------------------------------------------------------------------------

def _resolve_coef_string(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> str:
    """
    Return the raw Agama coefficient text from any source type.

    Resolution order:

    1. If *source* is a :class:`~pathlib.Path` (or a string without newlines
       that names an existing file):

       * ``.h5`` / ``.hdf5`` extension → read from the HDF5 group
         ``group_name/dataset_name``.
       * Any other extension → read as a plain text file.

    2. Otherwise treat *source* as the raw coefficient text itself (e.g. a
       string already loaded from HDF5 or produced by ``to_coef_string()``).

    Parameters
    ----------
    source : str or Path
        File path, HDF5 path, or raw coef text.
    group_name : str, optional
        HDF5 group to read when *source* is an ``.h5`` file.
    dataset_name : str, optional
        Dataset within the group, by default ``"coefs"``.

    Returns
    -------
    str
        Raw UTF-8 coefficient text.
    """
    # Fast path: Path objects always go through the filesystem
    if isinstance(source, Path):
        if source.suffix.lower() in (".h5", ".hdf5"):
            with h5py.File(source, "r") as f:
                raw = f[group_name][dataset_name][()]
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return source.read_text(encoding="utf-8")

    # String: if no newlines, check whether it names a file
    s = str(source)
    if "\n" not in s:
        p = Path(s)
        try:
            if p.exists():
                if p.suffix.lower() in (".h5", ".hdf5"):
                    with h5py.File(p, "r") as f:
                        raw = f[group_name][dataset_name][()]
                    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                return p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass

    # Fallback: treat as raw coef string content
    return s


# ---------------------------------------------------------------------------
# HDF5 write  (public)
# ---------------------------------------------------------------------------

def write_coef_to_h5(
    h5_path: Union[str, Path],
    coef_string: str,
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
    overwrite: bool = False,
    metadata: Optional[dict] = None,
) -> None:
    """
    Store an Agama coefficient text string in an HDF5 archive.

    The archive is created if absent; existing archives are opened in append
    mode so other groups are preserved.

    Parameters
    ----------
    h5_path : str or Path
        Destination HDF5 file.
    coef_string : str
        Full text content of an Agama ``.coef_*`` file.
    group_name : str, optional
        HDF5 group to write into, by default ``"snap_000"``.
    dataset_name : str, optional
        Scalar string dataset name, by default ``"coefs"``.
    overwrite : bool, optional
        Replace an existing dataset if ``True``; raise :exc:`RuntimeError`
        otherwise (default).
    metadata : dict, optional
        Key/value pairs attached as HDF5 group attributes, e.g.
        ``{"lmax": 8, "snap": 0}``.

    Raises
    ------
    RuntimeError
        If ``overwrite=False`` and the target dataset already exists.
    """
    h5_path = Path(h5_path)
    dt = h5py.string_dtype(encoding="utf-8")
    mode = "a" if h5_path.exists() else "w"
    with h5py.File(h5_path, mode) as f:
        grp = f.require_group(group_name)
        if dataset_name in grp:
            if overwrite:
                del grp[dataset_name]
            else:
                raise RuntimeError(
                    f"{group_name}/{dataset_name} already exists; "
                    "pass overwrite=True to replace."
                )
        grp.create_dataset(dataset_name, data=coef_string, dtype=dt)
        if metadata:
            for k, v in metadata.items():
                grp.attrs[k] = v


def write_snapshot_coefs_to_h5(
    snapshot_ids: Sequence[int],
    coef_file_patterns: Sequence[str],
    h5_output_paths: Sequence[Union[str, Path]],
    group_fmt: str = "snap_{snap:03d}",
    dataset_name: str = "coefs",
    overwrite: bool = True,
    encoding: str = "utf-8",
    times: Optional[Sequence[float]] = None,
) -> None:
    """
    Batch-write Agama coefficient files for multiple snapshots into HDF5 archives.

    For each ``snap_id`` and each ``(pattern, h5_path)`` pair, formats
    ``pattern.format(snap=snap_id)`` to find the source file, reads it, and
    stores the text under ``group_fmt.format(snap=snap_id)``.

    Optionally embeds the simulation times directly in each HDF5 archive so
    that :func:`~agama_helper._load.load_agama_evolving_potential` can be
    called without supplying times explicitly.  The dataset ``"times"`` is
    written at the root of each HDF5 file (one shared array per archive).

    Parameters
    ----------
    snapshot_ids : sequence of int
        Ordered snapshot indices, e.g. ``range(90, 101)``.
    coef_file_patterns : sequence of str
        ``str.format``-style patterns, e.g.
        ``"potential/{snap:03d}.MW.none_8.coef_mult"``.
    h5_output_paths : sequence of str or Path
        One HDF5 output path per entry in *coef_file_patterns*.
    group_fmt : str, optional
        Template for HDF5 group names, by default ``"snap_{snap:03d}"``.
    dataset_name : str, optional
        Dataset name within each group, by default ``"coefs"``.
    overwrite : bool, optional
        Forwarded to :func:`write_coef_to_h5`, default ``True``.
    times : sequence of float, optional
        Simulation times aligned with *snapshot_ids*.  When provided, stored
        as a root-level ``"times"`` dataset in each output HDF5 file so that
        :func:`~agama_helper._load.load_agama_evolving_potential` can be
        called without passing *times* explicitly.
    encoding : str, optional
        Encoding for reading source files, by default ``"utf-8"``.

    Raises
    ------
    ValueError
        If *coef_file_patterns* and *h5_output_paths* have different lengths,
        or if *times* is provided but its length differs from *snapshot_ids*.
    FileNotFoundError
        If a formatted coefficient file path does not exist.

    Examples
    --------
    >>> write_snapshot_coefs_to_h5(
    ...     snapshot_ids=range(90, 101),
    ...     coef_file_patterns=[
    ...         "potential/{snap:03d}.MW.none_8.coef_mult",
    ...         "potential/{snap:03d}.MW.none_8.coef_cylsp",
    ...     ],
    ...     h5_output_paths=["data/MW_mult.h5", "data/MW_cylsp.h5"],
    ...     times=snapshot_times_gyr,   # optional: embed for load_agama_evolving_potential
    ... )
    """
    snap_list = list(snapshot_ids)
    if len(coef_file_patterns) != len(h5_output_paths):
        raise ValueError(
            f"coef_file_patterns (len={len(coef_file_patterns)}) and "
            f"h5_output_paths (len={len(h5_output_paths)}) must have the same length."
        )
    if times is not None and len(times) != len(snap_list):
        raise ValueError(
            f"times (len={len(times)}) must match snapshot_ids (len={len(snap_list)})."
        )
    for snap_id in snap_list:
        group_name = group_fmt.format(snap=snap_id)
        for pattern, out_path in zip(coef_file_patterns, h5_output_paths):
            src = Path(pattern.format(snap=snap_id))
            if not src.exists():
                raise FileNotFoundError(
                    f"Coefficient file not found: {src}  (snap={snap_id})"
                )
            write_coef_to_h5(
                out_path,
                src.read_text(encoding=encoding),
                group_name=group_name,
                dataset_name=dataset_name,
                overwrite=overwrite,
            )

    # Embed times in every output archive so load_agama_evolving_potential
    # can be called without explicit times.
    if times is not None:
        times_arr = np.asarray(times, dtype=float)
        for out_path in h5_output_paths:
            with h5py.File(out_path, "a") as f:
                if "times" in f:
                    del f["times"]
                f.create_dataset("times", data=times_arr)


# ---------------------------------------------------------------------------
# Raw string reader  (public)
# ---------------------------------------------------------------------------

def read_coef_string(
    source: Union[str, Path],
    group_name: str = "snap_000",
    dataset_name: str = "coefs",
) -> str:
    """
    Read the raw Agama coefficient text from a file or HDF5 archive.

    Parameters
    ----------
    source : str or Path
        Path to a plain-text ``.coef_*`` file **or** a ``.h5``/``.hdf5``
        archive.  When reading from HDF5, *group_name* selects the snapshot
        group.
    group_name : str, optional
        HDF5 group to read, by default ``"snap_000"``.  Ignored when *source*
        is a plain-text file.
    dataset_name : str, optional
        Dataset within the HDF5 group, by default ``"coefs"``.

    Returns
    -------
    str
        Raw UTF-8 coefficient text, identical to the content of the original
        ``.coef_*`` file.
    """
    return _resolve_coef_string(source, group_name, dataset_name)
