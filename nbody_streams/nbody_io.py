"""
nbody_streams.nbody_io

I/O utilities for N-body snapshots and restart data.

Primary API:
- class ParticleReader: read HDF5-based snapshot sets
- _save_snapshot(path, snapshot_dict): hidden helper to write snapshots
- _load_restart(path): hidden helper to read simple npz restarts
"""
from __future__ import annotations
import os, h5py, glob, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

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
                self.Times = np.loadtxt(str(times_file_path), comments="#")
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
                    self.Times = np.loadtxt(str(default_path), comments="#")
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
                    self.Times = arr
                except Exception as e:
                    if self._verbose:
                        print(f"⚠️  Could not write default snapshot.times ({e}). Times disabled.")
                    self.Times = None
            else:
                # if file exists but earlier load failed, try one more time
                if times_path.exists():
                    try:
                        self.Times = np.loadtxt(str(times_path), comments="#")
                        if self._verbose:
                            print(f"✅  Loaded snapshot.times from: {times_path}")
                    except Exception:
                        self.Times = None
                else:
                    self.Times = None

    def _read_properties(self):
        """
        Read simulation properties from the first HDF5 file.
        Robust to missing 'properties', 'dark', or 'star' groups.
        """
        default_mass = 1.0
        default_eps = 0.0
        default_N = 0
        default_timestep = 0.0

        with h5py.File(self.file_list[0], "r") as f:
            props = f.get("properties", None)
            if props is None:
                # No properties group at all -> set sensible defaults
                self.num_dark = default_N
                self.mass_dark = default_mass
                self._eps_dark = default_eps

                self.num_star = 0
                self.mass_star = default_mass
                self._eps_star = default_eps

                self._timestep = default_timestep
            else:
                # dark
                dark = props.get("dark", None)
                if dark is None:
                    self.num_dark = default_N
                    self.mass_dark = default_mass
                    self._eps_dark = default_eps
                else:
                    self.num_dark = int(dark["N"][()]) if "N" in dark else default_N
                    self.mass_dark = float(dark["m"][()]) if "m" in dark else default_mass
                    self._eps_dark = float(dark["eps"][()]) if "eps" in dark else default_eps

                # star (may be missing)
                star = props.get("star", None)
                if star is None:
                    self.num_star = 0
                    self.mass_star = default_mass
                    self._eps_star = self._eps_dark
                else:
                    self.num_star = int(star["N"][()]) if "N" in star else 0
                    self.mass_star = float(star["m"][()]) if "m" in star else default_mass
                    self._eps_star = float(star["eps"][()]) if "eps" in star else self._eps_dark

                # time_step (may be at properties root)
                if "time_step" in props:
                    self._timestep = float(props["time_step"][()])
                else:
                    # fallback: check absolute path (some writers used /properties/time_step)
                    if "/properties/time_step" in f:
                        try:
                            self._timestep = float(f["/properties/time_step"][()])
                        except Exception:
                            self._timestep = default_timestep
                    else:
                        self._timestep = default_timestep

        if self._verbose:
            print("✅   Simulation properties:")
            print(f"      - N_dark: {self.num_dark}, N_star: {self.num_star}")
            print(f"      - M_dark: {self.mass_dark:.2e}, M_star: {self.mass_star:.2e}")
            print(f"      - time_step: {self._timestep}")

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
        if isinstance(identifier, float):
            if self.Times is None:
                raise ValueError("❌ Time-based lookup requires a snapshot.times file, which was not loaded.")
            time_col = self.Times[:, 1]
            closest_time_idx = np.argmin(np.abs(time_col - identifier))
            snap_index = int(self.Times[closest_time_idx, 0])
        elif isinstance(identifier, int):
            snap_index = identifier
        else:
            raise TypeError("❌ Identifier must be an integer (snapshot index) or a float (time).")

        if snap_index not in self._snap_to_file_map:
            raise ValueError(f"❌ Snapshot index {snap_index} not found in any of the simulation files.")

        file_to_open = self._snap_to_file_map[snap_index]
        dataset_name = f"snap.{snap_index:03d}"

        with h5py.File(file_to_open, "r") as f:
            raw_data = f["snapshots"][dataset_name][:]

        part = SimpleNamespace(dark={}, star={})

        dm_end_index = self.num_dark
        part.dark["posvel"] = raw_data[0:dm_end_index]
        part.star["posvel"] = raw_data[dm_end_index : dm_end_index + self.num_star]

        part.dark["mass"] = np.full(self.num_dark, self.mass_dark)
        part.star["mass"] = np.full(self.num_star, self.mass_star)

        part.snap = snap_index
        if self.Times is not None:
            time_row = self.Times[self.Times[:, 0] == snap_index]
            part.time = float(time_row[0, 1]) if len(time_row) > 0 else None
        else:
            part.time = None

        if self._verbose:
            print(f"✅   Read snapshot {snap_index} from {file_to_open} (time={part.time})")

        return part

    def extract_orbits(self, particle_type: str = "star", min_parallel_workers: int = 4):
        """
        Extract orbits for selected particle types across all available snapshots.
        using parallel worker processes that write directly into shared memory.

        WARNING: This loads *all* snapshot phase-space data for the chosen particle
        types into memory. It can be very memory intensive for many snapshots and
        large N.

        Parameters
        ----------
        particle_type : {'dark', 'star', 'all', True} or False
            Which particle types to load. Use 'dark', 'star', or 'all' (or True).
            If False, the function returns None.
        min_parallel_workers : Number of parallel workers. Default = 4. 
            Set to min(min_parallel_workers, CPUsavail, num_snapshots)
            
        Returns
        -------
        types.SimpleNamespace
            - orbits.dark / orbits.star : ndarray with shape (num_snapshots, N_particles, 6)
              (snapshot-major ordering)
            - orbits.Times : 1D ndarray of length num_snapshots with physical times (Gyr),
              or None if no times were loaded.
            - orbits.Snaps : 1D ndarray of snapshot indices (present only if Times not available).
        """

        if not particle_type:
            return None

        types_to_process = []
        if particle_type in ["dark", "all", True]:
            types_to_process.append("dark")
        if particle_type in ["star", "all", True]:
            types_to_process.append("star")

        if not types_to_process:
            raise ValueError(f"Invalid particle_type: {particle_type}. Use 'dark', 'star', or 'all'.")

        all_snap_indices = sorted(self._snap_to_file_map.keys())
        num_snapshots = len(all_snap_indices)

        orbits = SimpleNamespace()

        if num_snapshots == 0:
            if self._verbose:
                print("⚠️ No snapshots found; returning empty orbits.")
            if "dark" in types_to_process:
                orbits.dark = np.zeros((0, self.num_dark, 6), dtype=np.float64)
            if "star" in types_to_process:
                orbits.star = np.zeros((0, self.num_star, 6), dtype=np.float64)
            orbits.Times = None
            orbits.Snaps = np.array([], dtype=int)
            return orbits

        # Shared memory shapes: (num_snapshots, N_particles, 6)
        shape_dark = (num_snapshots, self.num_dark, 6)
        shape_star = (num_snapshots, self.num_star, 6) if "star" in types_to_process else None

        dtype = np.float64
        dtype_name = np.dtype(dtype).name

        shm_dark = None
        shm_star = None
        shared_dark = None
        shared_star = None

        def _nbytes_for_shape(s, dt):
            return int(np.prod(s)) * int(np.dtype(dt).itemsize)

        try:
            # Create shared memory buffers (only when the corresponding type is requested)
            if "dark" in types_to_process:
                bytes_dark = _nbytes_for_shape(shape_dark, dtype)
                shm_dark = shared_memory.SharedMemory(create=True, size=bytes_dark)
                shared_dark = np.ndarray(shape_dark, dtype=dtype, buffer=shm_dark.buf)
                shared_dark[:] = 0.0

            if "star" in types_to_process:
                bytes_star = _nbytes_for_shape(shape_star, dtype)
                shm_star = shared_memory.SharedMemory(create=True, size=bytes_star)
                shared_star = np.ndarray(shape_star, dtype=dtype, buffer=shm_star.buf)
                shared_star[:] = 0.0

            # Build argument list for each snapshot (pass shape_dark always so worker knows num_dark)
            args_list = []
            for dest_idx, snap in enumerate(all_snap_indices):
                file_path = self._snap_to_file_map[snap]
                args = (
                    int(snap),
                    int(dest_idx),
                    str(file_path),
                    shm_dark.name if shm_dark is not None else None,
                    shape_dark,           # ALWAYS pass shape_dark
                    dtype_name,
                    shm_star.name if shm_star is not None else None,
                    shape_star,           # pass shape_star or None
                )
                args_list.append(args)

            cpu_avail = os.cpu_count() or 1
            max_workers = min(min_parallel_workers, cpu_avail, num_snapshots)

            if self._verbose:
                print(f"   Spawning up to {max_workers} workers to read {num_snapshots} snapshots...")

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_worker_write_shared, a): a[0] for a in args_list}
                for fut in as_completed(futures):
                    # ensure exceptions are raised here for visibility
                    _ = fut.result()

            # Copy shared memory views into normal numpy arrays for return
            if "dark" in types_to_process:
                orbits.dark = np.array(shared_dark, copy=True)
            if "star" in types_to_process:
                orbits.star = np.array(shared_star, copy=True)

        finally:
            # Cleanup shared memory
            if shm_dark is not None:
                try:
                    shm_dark.close()
                    shm_dark.unlink()
                except FileNotFoundError:
                    pass
            if shm_star is not None:
                try:
                    shm_star.close()
                    shm_star.unlink()
                except FileNotFoundError:
                    pass

        # Attach time information (1D array aligned with snapshots) or Snaps
        if self.Times is not None:
            try:
                times_arr = np.asarray(self.Times)
                if times_arr.ndim == 1:
                    if times_arr.shape[0] == num_snapshots:
                        orbits.Times = times_arr.astype(float)
                    else:
                        orbits.Times = None
                        if self._verbose:
                            print("   Warning: Times is 1D but length doesn't match number of snapshots; omitting orbits.Times.")
                else:
                    snap_idxs = times_arr[:, 0].astype(int)
                    time_vals = times_arr[:, 1].astype(float)
                    if np.array_equal(snap_idxs, np.array(all_snap_indices, dtype=int)):
                        orbits.Times = time_vals
                    else:
                        time_map = dict(zip(snap_idxs, time_vals))
                        orbits.Times = np.array([time_map.get(snap, np.nan) for snap in all_snap_indices], dtype=float)
            except Exception:
                orbits.Times = None
                if self._verbose:
                    print("   Warning: could not parse Times array into orbits.Times.")
        else:
            orbits.Snaps = np.array(all_snap_indices, dtype=int)
            orbits.Times = None
            if self._verbose:
                print("   Times not available — attaching orbits.Snaps (snapshot indices).")

        if self._verbose:
            print("✅  ...extraction complete.")

        return orbits
    
def _save_snapshot(
    phase_space: np.ndarray,
    snap_index: int,
    time: float,
    output_dir: Path,
    *,
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
    Write snapshot(s) compatible with ParticleReader.

    Modes:
      - single_file=True -> all snapshots go into <output_dir>/snapshots.h5 (default if single_file=None and num_files_to_write is None)
      - num_files_to_write=N -> create N files named snapshot_part{0..N-1}.h5 and distribute snapshots.
         If total_expected_snapshots is provided distribution is contiguous chunks; else round-robin.

    Datasets created under group 'snapshots' as 'snap.###'.
    If dataset exists already it will NOT be overwritten.
    Properties are written to each file the first time the file is created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if time_step is None:
        time_step = 0.0

    # Determine file target
    if single_file is None:
        single_file = (num_files_to_write is None)

    if single_file:
        fname = output_dir / "snapshot.h5"
    else:
        # multi-file mode
        num_files = int(num_files_to_write) if (num_files_to_write is not None and num_files_to_write > 0) else 1
        if num_files == 1:
            fname = output_dir / "snapshot.h5"
        else:
            if total_expected_snapshots is not None and total_expected_snapshots > 0:
                per_file = math.ceil(total_expected_snapshots / num_files)
                file_idx = int(snap_index) // per_file
                file_idx = min(file_idx, num_files - 1)
            else:
                # fallback to round-robin if total expected not given
                file_idx = int(snap_index) % num_files
            fname = output_dir / f"snapshot.{file_idx:03d}.h5"

    # Open file in append mode and create group/dataset if missing
    with h5py.File(fname, "a") as f:
        snaps = f.require_group("snapshots")
        dset_name = f"snap.{snap_index:03d}"
        if dset_name in snaps:
            # do not overwrite existing dataset
            return
        snaps.create_dataset(dset_name, data=phase_space, compression="gzip")

        # write properties only if not present
        props = f.require_group("properties")
        if "dark" not in props:
            dark = props.create_group("dark")
            dark.create_dataset("N", data=num_dark)
            dark.create_dataset("m", data=mass_dark)
            dark.create_dataset("eps", data=eps_dark)
        if "star" not in props:
            star = props.create_group("star")
            star.create_dataset("N", data=num_star)
            star.create_dataset("m", data=mass_star)
            star.create_dataset("eps", data=eps_star)
        if "time_step" not in props:
            props.create_dataset("time_step", data=time_step)

        # optional attribute with physical time for convenience
        snaps.attrs[f"snap_time.{snap_index:03d}"] = float(time)

def _save_restart(
    phase_space: np.ndarray,
    time: float,
    step: int,
    output_dir: Path,
    snapshot_counter: int,
) -> None:
    """
    Save restart file for crash recovery.
    
    Parameters
    ----------
    phase_space : np.ndarray, shape (N, 6)
        Phase space coordinates.
    time : float
        Current time.
    step : int
        Current step number.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    restart_file = output_dir / "restart.npz"
    np.savez_compressed(
        restart_file, 
        phase_space=phase_space, 
        time=time, 
        step=step, 
        snapshot_counter=int(snapshot_counter)
        )

def _load_restart(output_dir: Path) -> tuple[np.ndarray, float, int, int] | None:
    """
    Load restart file if it exists.
    
    Parameters
    ----------
    output_dir : Path
        Output directory.
    
    Returns
    -------
    tuple | None
        (phase_space, time, step) if found, else None.
    """
    restart_file = output_dir / "restart.npz"
    if restart_file.exists():
        data = np.load(restart_file)
        phase_space = data["phase_space"]
        time = float(data["time"])
        step = int(data["step"])        
        snapshot_counter = int(data["snapshot_counter"]) if "snapshot_counter" in data.files else 0
        return phase_space, time, step, snapshot_counter
    return None

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