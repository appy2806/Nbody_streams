"""
nbody_streams.nbody_io

I/O utilities for N-body snapshots and restart data.

Primary API:
- class ParticleReader: read HDF5-based snapshot sets
- _save_snapshot(path, snapshot_dict): hidden helper to write snapshots
- _load_restart(path): hidden helper to read simple npz restarts
"""
from __future__ import annotations
import os, h5py, glob, math, warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

from .species import Species

def _make_times_ns(raw):
    """
    Convert a raw np.loadtxt array into SimpleNamespace(snap=int_array, time=float_array).
    Returns None if conversion fails or input is None.
    """
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return SimpleNamespace(snap=arr[:, 0].astype(int), time=arr[:, 1].astype(float))


def _is_uniform(arr: np.ndarray, rtol: float = 1e-10) -> tuple[bool, float]:
    """
    Check whether all elements of *arr* are equal to within relative tolerance.

    Returns
    -------
    (is_uniform, representative_value)
    """
    v0 = float(arr.flat[0])
    return bool(np.allclose(arr, v0, rtol=rtol, atol=0.0)), v0


# ---------------------------------------------------------------------------
# Worker functions — must be at module scope to be picklable.
# ---------------------------------------------------------------------------

# Worker function must live at module scope so it can be pickled by ProcessPoolExecutor.
def _worker_write_shared(args):
    """
    Worker executed in a separate process. Reads one snapshot from disk and writes
    it into the provided shared-memory buffers at the destination index.

    Args (tuple):
        snap_index (int)
        dest_idx (int)
        file_path (str)
        shm_dark_name (str or None)
        shape_dark (tuple)  # (num_snapshots, N_dark, 6) -- ALWAYS passed
        dtype_name (str)
        shm_star_name (str or None)
        shape_star (tuple or None)  # (num_snapshots, N_star, 6) or None
    Returns:
        snap_index (int)
    """
    (
        snap_index,
        dest_idx,
        file_path,
        shm_dark_name,
        shape_dark,
        dtype_name,
        shm_star_name,
        shape_star,
    ) = args

    import numpy as _np
    from multiprocessing import shared_memory as _shared_memory
    import h5py as _h5py
    
    dtype = _np.dtype(dtype_name)
    dset = f"snap.{snap_index:03d}"

    # Read snapshot (each worker opens its own file handle)
    with _h5py.File(file_path, "r") as f:
        raw = f["snapshots"][dset][:]

    # Number of dark / star particles (from shapes passed)
    # shape_dark is always provided by the parent, even if shm_dark is None.
    num_dark = int(shape_dark[1])
    num_star = int(shape_star[1]) if shape_star is not None else None

    # Write into dark shared memory if provided
    if shm_dark_name is not None:
        shm_d = _shared_memory.SharedMemory(name=shm_dark_name)
        dark_view = _np.ndarray(shape_dark, dtype=dtype, buffer=shm_d.buf)
        # raw[0:num_dark] -> (N_dark,6)
        dark_view[dest_idx, :, :] = raw[0:num_dark]
        shm_d.close()

    # Write into star shared memory if provided
    if shm_star_name is not None and shape_star is not None:
        shm_s = _shared_memory.SharedMemory(name=shm_star_name)
        star_view = _np.ndarray(shape_star, dtype=dtype, buffer=shm_s.buf)
        # raw indexing: stars start after dark block
        start = num_dark
        end = start + num_star
        star_view[dest_idx, :, :] = raw[start:end]
        shm_s.close()

    return snap_index


def _worker_write_shared_multi(args):
    """
    Worker for N-species parallel extraction.

    Args (tuple):
        snap_index (int)
        dest_idx (int)
        file_path (str)
        species_shm_info (list of (shm_name_or_None, shape, start_idx, end_idx))
            One entry per species (in order).  ``shm_name`` is None when the
            caller does not want this species extracted.
        dtype_name (str)

    Returns:
        snap_index (int)
    """
    snap_index, dest_idx, file_path, species_shm_info, dtype_name = args

    import numpy as _np
    from multiprocessing import shared_memory as _shared_memory
    import h5py as _h5py

    dtype = _np.dtype(dtype_name)
    dset = f"snap.{snap_index:03d}"

    with _h5py.File(file_path, "r") as f:
        raw = f["snapshots"][dset][:]

    for shm_name, shape, start_idx, end_idx in species_shm_info:
        if shm_name is None or shape is None:
            continue
        shm = _shared_memory.SharedMemory(name=shm_name)
        view = _np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        view[dest_idx, :, :] = raw[start_idx:end_idx]
        shm.close()

    return snap_index


class ParticleReader:
    """
    A class to read N-body simulation data from one or more HDF5 files.

    This reader assumes a consistent data structure across files, with particle
    data stored in a '/snapshots' group where each dataset is named 'snap.###'.

    Units assumed:
      - Length: kpc
      - Mass: Msun
      - Velocity: km/s
      - Time: physical time in Gyr (if a snapshot.times file is provided)

    Usage
    ----------
    Simreader = ParticleReader(sim_dir, verbose=True)
    part = Simreader.read_snapshot(100) # # load in snap=100, also accepts float as time.
    orbits = Simreader.extract_orbits('star', 4)    

    Parameters
    ----------
    sim_pattern : str
        A path or glob pattern for the HDF5 files (e.g., 'path/to/sim.hdf5'
        or 'path/to/sim.*.hdf5').
    times_file_path : str or Path, optional
        Path to the snapshot times file. If not provided, the reader will look
        for a file named 'snapshot.times' in the parent directory of the
        first matched simulation file (i.e., `Path(first_file).parent / 'snapshot.times'`).
        If that file does not exist it will be set to None.
    verbose : bool, optional
        If True, print status messages. Default is False.
    """

    def __init__(self, sim_pattern: str, times_file_path: str = None, verbose: bool = False):
        self._verbose = bool(verbose)
        if self._verbose:
            print(f"🔍 Initializing ParticleReader for pattern: {sim_pattern}")

        self.file_list = sorted(glob.glob(sim_pattern))
        if not self.file_list:
            raise FileNotFoundError(f"No HDF5 files found matching pattern: {sim_pattern}")
        if self._verbose:
            print(f"   Found {len(self.file_list)} simulation file(s).")

        # read properties and map snapshots
        self._read_properties()
        self._map_snapshots_to_files()

        # Try to load explicit times file if user passed one
        self.Times = None
        if times_file_path is not None:
            try:
                self.Times = _make_times_ns(np.loadtxt(str(times_file_path), comments="#"))
                if self._verbose:
                    print(f"✅  Successfully loaded time mapping from: {times_file_path}")
            except Exception:
                if self._verbose:
                    print(f"⚠️   Could not load provided times file: {times_file_path}. Ignoring.")
                self.Times = None
        else:
            # try default location next to first matched file
            default_path = Path(self.file_list[0]).parent / "snapshot.times"
            if default_path.exists():
                try:
                    self.Times = _make_times_ns(np.loadtxt(str(default_path), comments="#"))
                    if self._verbose:
                        print(f"✅  Loaded snapshot.times from: {default_path}")
                except Exception:
                    if self._verbose:
                        print(f"⚠️   snapshot.times exists but could not be read: {default_path}")
                    self.Times = None

        # If no snapshot.times found, create a fail-safe file (do not overwrite existing file)
        if self.Times is None:
            parent_dir = Path(self.file_list[0]).parent
            times_path = parent_dir / "snapshot.times"
            if not times_path.exists() and len(self.Snapshots) > 0:
                snaps = np.array(self.Snapshots, dtype=int)
                if getattr(self, "_timestep", 0.0) and float(self._timestep) > 0.0:
                    times = (snaps - snaps.min()).astype(float) * float(self._timestep)
                else:
                    times = np.arange(len(snaps), dtype=float)
                arr = np.column_stack([snaps, times])
                try:
                    np.savetxt(str(times_path), arr, fmt="%d %.10e",
                            header="snap_index time", comments="# ")
                    if self._verbose:
                        print(f"ℹ️  Created default snapshot.times at: {times_path}")
                    self.Times = _make_times_ns(arr)
                except Exception as e:
                    if self._verbose:
                        print(f"⚠️  Could not write default snapshot.times ({e}). Times disabled.")
                    self.Times = None
            else:
                # if file exists but earlier load failed, try one more time
                if times_path.exists():
                    try:
                        self.Times = _make_times_ns(np.loadtxt(str(times_path), comments="#"))
                        if self._verbose:
                            print(f"✅  Loaded snapshot.times from: {times_path}")
                    except Exception:
                        self.Times = None
                else:
                    self.Times = None

    def _read_properties(self):
        """
        Read simulation properties from the first HDF5 file.

        Supports two HDF5 schemas:

        **New multi-species schema** (written by this version):
          ``/properties`` group carries ``n_species`` and ``species_names``
          attributes, followed by per-species sub-groups that may store mass /
          softening as a scalar dataset **or** as a ``m_array`` / ``eps_array``
          dataset (smart storage written by :func:`_save_snapshot`).

        **Legacy two-species schema** (files written by earlier versions):
          ``/properties/dark`` and ``/properties/star`` sub-groups only; no
          ``n_species`` attribute.

        In both cases the following backward-compatible instance attributes are
        set after returning from this method::

            self.species_list  : list[Species]    (primary, ordered)
            self.num_dark      : int
            self.mass_dark     : float
            self._eps_dark     : float
            self.num_star      : int
            self.mass_star     : float
            self._eps_star     : float
            self._timestep     : float
        """
        default_mass = 1.0
        default_eps = 0.0
        default_timestep = 0.0

        with h5py.File(self.file_list[0], "r") as f:
            props = f.get("properties", None)

            # ------------------------------------------------------------------
            # Read timestep (shared by both schemas)
            # ------------------------------------------------------------------
            if props is not None and "time_step" in props:
                self._timestep = float(props["time_step"][()])
            elif props is not None and "/properties/time_step" in f:
                try:
                    self._timestep = float(f["/properties/time_step"][()])
                except Exception:
                    self._timestep = default_timestep
            else:
                self._timestep = default_timestep

            # ------------------------------------------------------------------
            # Detect schema
            # ------------------------------------------------------------------
            if props is None:
                # No properties at all — legacy fallback with neutral defaults
                self.species_list = [
                    Species(name="dark", N=0, mass=default_mass,
                            softening=default_eps),
                ]
                self._timestep = default_timestep

            elif "n_species" in props.attrs:
                # ---- NEW multi-species schema --------------------------------
                raw_names = props.attrs["species_names"]
                names = [
                    n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
                    for n in raw_names
                ]
                species_list: list[Species] = []
                for name in names:
                    grp = props.get(name, None)
                    if grp is None:
                        continue
                    N_sp = int(grp["N"][()]) if "N" in grp else 0
                    # Smart mass: scalar 'm' or per-particle 'm_array'
                    if "m_array" in grp:
                        mass_val: float | np.ndarray = grp["m_array"][:]
                    elif "m" in grp:
                        mass_val = float(grp["m"][()])
                    else:
                        mass_val = default_mass
                    # Smart softening: scalar 'eps' or per-particle 'eps_array'
                    if "eps_array" in grp:
                        eps_val: float | np.ndarray = grp["eps_array"][:]
                    elif "eps" in grp:
                        eps_val = float(grp["eps"][()])
                    else:
                        eps_val = default_eps
                    species_list.append(
                        Species(name=name, N=N_sp, mass=mass_val,
                                softening=eps_val)
                    )
                self.species_list = species_list

            else:
                # ---- Legacy two-species schema --------------------------------
                species_list = []

                dark = props.get("dark", None)
                if dark is not None:
                    n_d = int(dark["N"][()]) if "N" in dark else 0
                    m_d = float(dark["m"][()]) if "m" in dark else default_mass
                    e_d = float(dark["eps"][()]) if "eps" in dark else default_eps
                    if n_d > 0:
                        species_list.append(
                            Species(name="dark", N=n_d, mass=m_d, softening=e_d)
                        )

                star = props.get("star", None)
                if star is not None:
                    n_s = int(star["N"][()]) if "N" in star else 0
                    m_s = float(star["m"][()]) if "m" in star else default_mass
                    e_s = float(star["eps"][()]) if "eps" in star else default_eps
                    if n_s > 0:
                        species_list.append(
                            Species(name="star", N=n_s, mass=m_s, softening=e_s)
                        )

                if not species_list:
                    # properties group exists but has no dark/star entries
                    species_list.append(
                        Species(name="dark", N=0, mass=default_mass,
                                softening=default_eps)
                    )
                self.species_list = species_list

        # ----------------------------------------------------------------------
        # Set legacy backward-compat attributes from species_list
        # ----------------------------------------------------------------------
        dark_sp = next((s for s in self.species_list if s.name == "dark"), None)
        star_sp = next((s for s in self.species_list if s.name == "star"), None)

        self.num_dark = dark_sp.N if dark_sp else 0
        self.mass_dark = (
            float(dark_sp.mass)
            if dark_sp and np.isscalar(dark_sp.mass)
            else (float(dark_sp.mass.flat[0]) if dark_sp else default_mass)
        )
        self._eps_dark = (
            float(dark_sp.softening)
            if dark_sp and np.isscalar(dark_sp.softening)
            else (float(dark_sp.softening.flat[0]) if dark_sp else default_eps)
        )
        self.num_star = star_sp.N if star_sp else 0
        self.mass_star = (
            float(star_sp.mass)
            if star_sp and np.isscalar(star_sp.mass)
            else (float(star_sp.mass.flat[0]) if star_sp else default_mass)
        )
        self._eps_star = (
            float(star_sp.softening)
            if star_sp and np.isscalar(star_sp.softening)
            else (float(star_sp.softening.flat[0]) if star_sp else self._eps_dark)
        )

        if self._verbose:
            print("Simulation properties:")
            for s in self.species_list:
                m_str = (f"{s.mass:.2e}" if np.isscalar(s.mass)
                         else f"array[{s.N}]")
                h_str = (f"{s.softening:.3f}" if np.isscalar(s.softening)
                         else f"array[{s.N}]")
                print(f"  [{s.name}] N={s.N:,}, mass={m_str}, eps={h_str}")
            print(f"  time_step: {self._timestep}")

    def _map_snapshots_to_files(self):
        """
        Scan HDF5 files and map snapshot index -> file path for fast lookups.

        Sets:
          - self._snap_to_file_map : dict mapping snap_index -> file_path
          - self.Snapshots : np.ndarray sorted snapshot indices (dtype=int)
        """
        self._snap_to_file_map = {}
        if self._verbose:
            print("   Scanning files to map snapshot locations...")
        for file_path in self.file_list:
            with h5py.File(file_path, "r") as f:
                if "snapshots" not in f:
                    continue
                for snap_name in f["snapshots"].keys():
                    # expected snap_name like 'snap.012' or 'snap.12'
                    try:
                        snap_index = int(snap_name.split(".")[-1])
                    except ValueError:
                        # skip non-standard snapshot names
                        continue
                    self._snap_to_file_map[snap_index] = file_path

        # Create a sorted numpy array of snapshot indices for easy access
        if len(self._snap_to_file_map) > 0:
            self.Snapshots = np.array(sorted(self._snap_to_file_map.keys()), dtype=int)
        else:
            self.Snapshots = np.array([], dtype=int)

        if self._verbose:
            print(f"   ...found {len(self._snap_to_file_map)} total snapshots.")

    def read_snapshot(self, identifier: int | float):
        """
        Read a single snapshot by index or physical time.

        Parameters
        ----------
        identifier : int or float
            If int: snapshot index (e.g., 151).
            If float: physical time in Gyr (e.g., 7.5) — requires snapshot.times file.

        Returns
        -------
        types.SimpleNamespace
            - part.dark / part.star: dicts containing 'posvel' (N x 6) and 'mass' arrays.
            - part.snap: int, snapshot index.
            - part.time: float or None, physical time in Gyr if available.
        """
        if isinstance(identifier, (float, np.floating)):
            if self.Times is None:
                raise ValueError("❌ Time-based lookup requires a snapshot.times file, which was not loaded.")
            closest_time_idx = np.argmin(np.abs(self.Times.time - identifier))
            snap_index = int(self.Times.snap[closest_time_idx])
        elif isinstance(identifier, (int, np.integer)):
            snap_index = int(identifier)
        else:
            raise TypeError("❌ Identifier must be an integer (snapshot index) or a float (time).")

        if snap_index not in self._snap_to_file_map:
            raise ValueError(f"❌ Snapshot index {snap_index} not found in any of the simulation files.")

        file_to_open = self._snap_to_file_map[snap_index]
        dataset_name = f"snap.{snap_index:03d}"

        with h5py.File(file_to_open, "r") as f:
            raw_data = f["snapshots"][dataset_name][:]

        # Build per-species slices using species_list (handles N species)
        part_species: dict[str, dict] = {}
        idx = 0
        for s in self.species_list:
            posvel = raw_data[idx : idx + s.N]
            mass_arr = (
                np.full(s.N, float(s.mass))
                if np.isscalar(s.mass)
                else np.asarray(s.mass)
            )
            part_species[s.name] = {"posvel": posvel, "mass": mass_arr}
            idx += s.N

        # Assemble SimpleNamespace with backward-compatible dark/star attributes
        part = SimpleNamespace()
        part.species = part_species
        part.dark = part_species.get("dark", {"posvel": np.empty((0, 6)),
                                              "mass": np.empty(0)})
        part.star = part_species.get("star", {"posvel": np.empty((0, 6)),
                                              "mass": np.empty(0)})

        part.snap = snap_index
        if self.Times is not None:
            mask = self.Times.snap == snap_index
            part.time = float(self.Times.time[mask][0]) if mask.any() else None
        else:
            part.time = None

        if self._verbose:
            print(f"Read snapshot {snap_index} from {file_to_open} "
                  f"(time={part.time})")

        return part

    def extract_orbits(
        self,
        particle_type: str = "star",
        min_parallel_workers: int = 4,
        snap_indices=None,
    ):
        """
        Extract orbits for selected particle types across snapshots,
        using parallel worker processes that write directly into shared memory.

        All requested snapshot data is loaded into RAM at once.  Use
        *snap_indices* to limit memory usage when you have many snapshots or
        large particle counts; for very large N (> ~500k) combined with many
        snapshots, prefer iterating over ``read_snapshot`` instead.

        Parameters
        ----------
        particle_type : str, True, or False
            Which particle types to load:

            * ``'all'`` or ``True`` — all species in the file.
            * ``False`` — return ``None`` immediately.
            * ``'dark'``, ``'star'`` — backward-compatible single-species load.
            * Any other species name present in the file (e.g. ``'bh'``).

        min_parallel_workers : int, optional
            Maximum number of parallel worker processes.  Actual workers =
            ``min(min_parallel_workers, cpu_count, num_snapshots)``.

        snap_indices : array-like of int or None, optional
            Snapshot indices to extract (e.g. ``range(0, 1000, 10)`` for every
            10th snapshot).  Must be valid keys in the snapshot map.
            ``None`` (default) extracts all available snapshots.

        Returns
        -------
        types.SimpleNamespace
            * ``orbits.species`` : dict mapping species name ->
              ndarray of shape ``(num_snapshots, N_k, 6)``.
            * ``orbits.dark`` / ``orbits.star`` : backward-compatible aliases
              to the corresponding entry in ``orbits.species`` (if present).
            * ``orbits.Times`` : 1-D ndarray of length ``num_snapshots`` with
              physical times, or ``None``.
            * ``orbits.Snaps`` : 1-D int ndarray of snapshot indices.
        """
        if not particle_type:
            return None

        # Resolve which species names to extract
        all_names = [s.name for s in self.species_list]
        if particle_type in ("all", True):
            types_to_process = list(all_names)
        elif isinstance(particle_type, str):
            if particle_type not in all_names:
                raise ValueError(
                    f"particle_type '{particle_type}' not found in species list "
                    f"{all_names}."
                )
            types_to_process = [particle_type]
        else:
            raise TypeError(
                f"particle_type must be str, True, or False; got {type(particle_type)!r}"
            )

        available = sorted(self._snap_to_file_map.keys())
        if snap_indices is None:
            all_snap_indices = available
        else:
            requested = sorted(int(i) for i in snap_indices)
            missing = [i for i in requested if i not in self._snap_to_file_map]
            if missing:
                raise ValueError(
                    f"snap_indices contains indices not found in snapshot map: {missing}"
                )
            all_snap_indices = requested

        num_snapshots = len(all_snap_indices)

        # Memory estimate and warning
        total_bytes = 0
        for s in self.species_list:
            if s.name in types_to_process:
                total_bytes += num_snapshots * s.N * 6 * 8  # float64
        total_gb = total_bytes / 1e9
        if total_gb > 4.0:
            warnings.warn(
                f"extract_orbits will allocate ~{total_gb:.1f} GB of RAM "
                f"({num_snapshots} snapshots, species: {types_to_process}). "
                "Use snap_indices to load a subset, or iterate over read_snapshot() "
                "instead.",
                ResourceWarning,
                stacklevel=2,
            )

        orbits = SimpleNamespace()

        if num_snapshots == 0:
            if self._verbose:
                print("No snapshots found; returning empty orbits.")
            orbits.species = {}
            for s in self.species_list:
                if s.name in types_to_process:
                    orbits.species[s.name] = np.zeros((0, s.N, 6), dtype=np.float64)
            orbits.dark = orbits.species.get("dark", np.zeros((0, 0, 6)))
            orbits.star = orbits.species.get("star", np.zeros((0, 0, 6)))
            orbits.Times = None
            orbits.Snaps = np.array([], dtype=int)
            return orbits

        dtype = np.float64
        dtype_name = np.dtype(dtype).name

        # Build start/end indices for each species in the flat snapshot array
        species_ranges: dict[str, tuple[int, int]] = {}
        idx = 0
        for s in self.species_list:
            species_ranges[s.name] = (idx, idx + s.N)
            idx += s.N

        # Per-species shared memory
        shm_map: dict[str, shared_memory.SharedMemory] = {}
        view_map: dict[str, np.ndarray] = {}
        species_shapes: dict[str, tuple] = {}

        def _nbytes(shape, dt):
            return int(np.prod(shape)) * int(np.dtype(dt).itemsize)

        try:
            for s in self.species_list:
                if s.name not in types_to_process:
                    continue
                shape = (num_snapshots, s.N, 6)
                species_shapes[s.name] = shape
                shm = shared_memory.SharedMemory(create=True, size=_nbytes(shape, dtype))
                shm_map[s.name] = shm
                view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                view[:] = 0.0
                view_map[s.name] = view

            # Build args for each snapshot
            args_list = []
            for dest_idx, snap in enumerate(all_snap_indices):
                file_path = self._snap_to_file_map[snap]
                # One entry per species: (shm_name_or_None, shape_or_None, start, end)
                species_shm_info = []
                for s in self.species_list:
                    if s.name in shm_map:
                        shm_name = shm_map[s.name].name
                        shape = species_shapes[s.name]
                        start, end = species_ranges[s.name]
                        species_shm_info.append((shm_name, shape, start, end))
                    else:
                        species_shm_info.append((None, None, None, None))
                args = (
                    int(snap),
                    int(dest_idx),
                    str(file_path),
                    species_shm_info,
                    dtype_name,
                )
                args_list.append(args)

            cpu_avail = os.cpu_count() or 1
            max_workers = min(min_parallel_workers, cpu_avail, num_snapshots)

            if self._verbose:
                print(f"Spawning up to {max_workers} workers for "
                      f"{num_snapshots} snapshots...")

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_worker_write_shared_multi, a): a[0]
                    for a in args_list
                }
                for fut in as_completed(futures):
                    _ = fut.result()

            # Copy out of shared memory into regular arrays
            orbits.species = {}
            for name, view in view_map.items():
                orbits.species[name] = np.array(view, copy=True)

        finally:
            for shm in shm_map.values():
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

        # Backward-compatible aliases
        orbits.dark = orbits.species.get("dark", np.empty((0, 0, 6), dtype=np.float64))
        orbits.star = orbits.species.get("star", np.empty((0, 0, 6), dtype=np.float64))

        # Time information
        orbits.Snaps = np.array(all_snap_indices, dtype=int)
        if self.Times is not None:
            try:
                snap_idxs = self.Times.snap
                time_vals = self.Times.time
                if np.array_equal(snap_idxs, orbits.Snaps):
                    orbits.Times = time_vals
                else:
                    time_map = dict(zip(snap_idxs, time_vals))
                    orbits.Times = np.array(
                        [time_map.get(s, np.nan) for s in all_snap_indices],
                        dtype=float,
                    )
            except Exception:
                orbits.Times = None
                if self._verbose:
                    print("Warning: could not align Times with snapshot indices.")
        else:
            orbits.Times = None
            if self._verbose:
                print("Times not available — orbits.Snaps attached.")

        if self._verbose:
            print("Orbit extraction complete.")

        return orbits
    
def _save_snapshot(
    phase_space: np.ndarray,
    snap_index: int,
    time: float,
    output_dir: Path,
    *,
    # --- New multi-species path (preferred) ---
    species: list[Species] | None = None,
    # --- Legacy kwargs (still honoured when species=None) ---
    num_dark: int | None = None,
    num_star: int | None = None,
    mass_dark: float | None = None,
    mass_star: float | None = None,
    time_step: float | None = None,
    eps_dark: float | None = None,
    eps_star: float | None = None,
    single_file: bool | None = None,
    num_files_to_write: int | None = None,
    total_expected_snapshots: int | None = None,
) -> None:
    """
    Write a snapshot compatible with :class:`ParticleReader`.

    Two modes
    ---------
    **Multi-species** (``species`` kwarg provided):
      Writes a ``/properties`` group with ``n_species`` / ``species_names``
      attributes and per-species sub-groups.  Per-species mass and softening
      are stored as a **scalar dataset** when all values are equal (saves
      space), or as a compressed ``m_array`` / ``eps_array`` dataset when
      values vary.

    **Legacy two-species** (``species=None``, default):
      Writes ``/properties/dark`` and ``/properties/star`` sub-groups exactly
      as in previous versions.  All existing callers continue to work.

    File-distribution modes (both paths)
    -------------------------------------
    * ``single_file=True`` (or default) → ``<output_dir>/snapshot.h5``
    * ``num_files_to_write=N`` → ``snapshot.000.h5 … snapshot.(N-1).h5``

    Datasets are created under group ``'snapshots'`` as ``snap.###``.
    Existing datasets are **never** overwritten.
    Properties are written only on first creation of the file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if time_step is None:
        time_step = 0.0

    # ---- Determine target filename ----------------------------------------
    if single_file is None:
        single_file = (num_files_to_write is None)

    if single_file:
        fname = output_dir / "snapshot.h5"
    else:
        num_files = int(num_files_to_write) if (
            num_files_to_write is not None and num_files_to_write > 0
        ) else 1
        if num_files == 1:
            fname = output_dir / "snapshot.h5"
        else:
            if total_expected_snapshots is not None and total_expected_snapshots > 0:
                per_file = math.ceil(total_expected_snapshots / num_files)
                file_idx = min(int(snap_index) // per_file, num_files - 1)
            else:
                file_idx = int(snap_index) % num_files
            fname = output_dir / f"snapshot.{file_idx:03d}.h5"

    # ---- Write HDF5 ----------------------------------------------------------
    with h5py.File(fname, "a") as f:
        snaps = f.require_group("snapshots")
        dset_name = f"snap.{snap_index:03d}"
        if dset_name in snaps:
            return  # never overwrite
        snaps.create_dataset(dset_name, data=phase_space, compression="gzip")
        snaps.attrs[f"snap_time.{snap_index:03d}"] = float(time)

        props = f.require_group("properties")

        if species is not None:
            # ------------------------------------------------------------------
            # NEW multi-species path
            # ------------------------------------------------------------------
            if "n_species" not in props.attrs:
                props.attrs["n_species"] = len(species)
                props.attrs["species_names"] = np.array(
                    [s.name.encode("utf-8") for s in species]
                )

            for s in species:
                if s.name in props:
                    continue  # already written in a previous snapshot of this file
                grp = props.create_group(s.name)
                grp.create_dataset("N", data=int(s.N))

                # Smart mass storage
                m_arr = (
                    np.full(s.N, float(s.mass), dtype=np.float64)
                    if np.isscalar(s.mass)
                    else np.asarray(s.mass, dtype=np.float64)
                )
                m_uniform, m_val = _is_uniform(m_arr)
                if m_uniform:
                    grp.create_dataset("m", data=float(m_val))
                else:
                    grp.create_dataset("m_array", data=m_arr, compression="gzip")

                # Smart softening storage
                h_arr = (
                    np.full(s.N, float(s.softening), dtype=np.float64)
                    if np.isscalar(s.softening)
                    else np.asarray(s.softening, dtype=np.float64)
                )
                h_uniform, h_val = _is_uniform(h_arr)
                if h_uniform:
                    grp.create_dataset("eps", data=float(h_val))
                else:
                    grp.create_dataset("eps_array", data=h_arr, compression="gzip")

            if "time_step" not in props:
                props.create_dataset("time_step", data=float(time_step))

        else:
            # ------------------------------------------------------------------
            # LEGACY two-species path (backward compatible)
            # ------------------------------------------------------------------
            N = phase_space.shape[0]
            if num_dark is None and num_star is None:
                num_dark = N
                num_star = 0
            elif num_star is None:
                num_star = N - int(num_dark)
            num_dark = int(num_dark)
            num_star = int(num_star)

            if mass_dark is None:
                mass_dark = 1.0
            if mass_star is None:
                mass_star = 1.0
            if eps_dark is None:
                eps_dark = 0.0
            if eps_star is None:
                eps_star = 0.0

            if "dark" not in props:
                dark = props.create_group("dark")
                dark.create_dataset("N", data=num_dark)
                dark.create_dataset("m", data=float(mass_dark))
                dark.create_dataset("eps", data=float(eps_dark))
            if "star" not in props:
                star = props.create_group("star")
                star.create_dataset("N", data=num_star)
                star.create_dataset("m", data=float(mass_star))
                star.create_dataset("eps", data=float(eps_star))
            if "time_step" not in props:
                props.create_dataset("time_step", data=float(time_step))

def _save_restart(
    phase_space: np.ndarray,
    time: float,
    step: int,
    output_dir: Path,
    snapshot_counter: int,
    *,
    mass_arr: np.ndarray | None = None,
    softening_arr: np.ndarray | None = None,
    species_names: list[str] | None = None,
    species_N: list[int] | None = None,
) -> None:
    """
    Save a restart file for crash recovery.

    Parameters
    ----------
    phase_space : ndarray, shape (N, 6)
    time : float
    step : int
    output_dir : Path
    snapshot_counter : int
    mass_arr : ndarray, shape (N,), optional
        Per-particle masses (multi-species path).
    softening_arr : ndarray, shape (N,), optional
        Per-particle softening lengths (multi-species path).
    species_names : list[str], optional
        Ordered species names (multi-species path).
    species_N : list[int], optional
        Per-species particle counts (multi-species path).

    Notes
    -----
    Old restart files that lack the species keys are loaded gracefully by
    :func:`_load_restart` (missing fields are returned as ``None``).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    restart_file = Path(output_dir) / "restart.npz"

    save_kwargs: dict = dict(
        phase_space=phase_space,
        time=np.float64(time),
        step=np.int64(step),
        snapshot_counter=np.int64(snapshot_counter),
    )
    if mass_arr is not None:
        save_kwargs["mass_arr"] = np.asarray(mass_arr, dtype=np.float64)
    if softening_arr is not None:
        save_kwargs["softening_arr"] = np.asarray(softening_arr, dtype=np.float64)
    if species_names is not None:
        save_kwargs["species_names"] = np.array(
            [n.encode("utf-8") for n in species_names]
        )
    if species_N is not None:
        save_kwargs["species_N"] = np.array(species_N, dtype=np.int64)

    np.savez_compressed(restart_file, **save_kwargs)


def _load_restart(
    output_dir: Path,
) -> tuple[
    np.ndarray, float, int, int,
    np.ndarray | None, np.ndarray | None,
    list[str] | None, list[int] | None,
] | None:
    """
    Load a restart file if one exists.

    Returns
    -------
    tuple of length 8, or ``None`` if no restart file found::

        (phase_space, time, step, snapshot_counter,
         mass_arr, softening_arr, species_names, species_N)

    The last four elements are ``None`` when the file was written by an
    older version of the package (backward-compatible).
    """
    restart_file = Path(output_dir) / "restart.npz"
    if not restart_file.exists():
        return None

    data = np.load(restart_file, allow_pickle=False)
    phase_space = data["phase_space"]
    time = float(data["time"])
    step = int(data["step"])
    snapshot_counter = int(data["snapshot_counter"]) if "snapshot_counter" in data.files else 0

    mass_arr = data["mass_arr"] if "mass_arr" in data.files else None
    softening_arr = data["softening_arr"] if "softening_arr" in data.files else None

    species_names: list[str] | None = None
    if "species_names" in data.files:
        raw = data["species_names"]
        species_names = [
            n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
            for n in raw
        ]

    species_N: list[int] | None = None
    if "species_N" in data.files:
        species_N = [int(n) for n in data["species_N"]]

    return (phase_space, time, step, snapshot_counter,
            mass_arr, softening_arr, species_names, species_N)

def _update_snapshot_times(output_dir: Path, snap_index: int, time: float) -> None:
    """
    Append or update snapshot.times in output_dir.
    Ensures unique snap_index entries and does not clobber existing file.
    Format: two-columns 'snap_index time' as integers and floats.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    times_path = out / "snapshot.times"

    # load existing if present
    if times_path.exists():
        try:
            arr = np.loadtxt(str(times_path), comments="#")
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
        except Exception:
            arr = np.empty((0, 2))
    else:
        arr = np.empty((0, 2))

    # make sure arr is 2D
    if arr.size == 0:
        arr2 = np.array([[int(snap_index), float(time)]], dtype=float)
    else:
        # update or append
        # ensure integer snap indices in first column
        snap_ids = arr[:, 0].astype(int)
        mask = snap_ids == int(snap_index)
        if mask.any():
            arr[mask, 1] = float(time)
            arr2 = arr
        else:
            arr2 = np.vstack([arr, [int(snap_index), float(time)]])

    # sort rows by snap index
    arr2 = arr2[np.argsort(arr2[:, 0])]
    np.savetxt(str(times_path), arr2, fmt="%d %.10e", header="snap_index time", comments="# ")