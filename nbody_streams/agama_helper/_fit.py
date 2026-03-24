"""
agama_helper._fit
~~~~~~~~~~~~~~~~~
Fit Agama Multipole and CylSpline potential expansions from particle snapshots.

The primary entry point :func:`fit_potential` replicates the multi-species
expansion workflow described in Arora et al. (2022) for cosmological zoom
simulations:

* Dark matter (+ hot gas, if present) → spherical harmonic **Multipole** BFE
* Stars (+ cold gas, if present)      → azimuthal harmonic **CylSpline** BFE

Particles should already be in the desired halo-centric frame before calling
this function.  Alternatively, supply ``center`` and ``rotation`` and the
transform ``np.dot(pos - center, R.T)`` is applied internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

try:
    import agama as _agama
    AGAMA_AVAILABLE = True
except ImportError:
    AGAMA_AVAILABLE = False


def _require_agama() -> None:
    if not AGAMA_AVAILABLE:
        raise ImportError(
            "agama is required for fitting potentials. "
            "See https://github.com/GalacticDynamics-Oxford/Agama"
        )


# ---------------------------------------------------------------------------
# Snapshot dictionary
# ---------------------------------------------------------------------------

def create_snapshot_dict(
    pos_dark: np.ndarray,
    mass_dark: np.ndarray,
    pos_star: np.ndarray | None = None,
    mass_star: np.ndarray | None = None,
    pos_gas: np.ndarray | None = None,
    mass_gas: np.ndarray | None = None,
    temperature_gas: np.ndarray | None = None,
) -> dict:
    """
    Pack particle arrays into a FIRE-like snapshot dictionary.

    Creates the minimal species dictionary expected by :func:`fit_potential`.
    Positions are stored as provided; no centering or rotation is applied here.

    Parameters
    ----------
    pos_dark : ndarray, shape (N_dark, 3)
        Dark matter positions [kpc].
    mass_dark : ndarray, shape (N_dark,)
        Dark matter masses [M☉].
    pos_star : ndarray, shape (N_star, 3), optional
        Stellar positions [kpc].
    mass_star : ndarray, shape (N_star,), optional
        Stellar masses [M☉].
    pos_gas : ndarray, shape (N_gas, 3), optional
        Gas positions [kpc].
    mass_gas : ndarray, shape (N_gas,), optional
        Gas masses [M☉].
    temperature_gas : ndarray, shape (N_gas,), optional
        Gas temperatures [K].  When provided, :func:`fit_potential` splits gas
        into cold (→ CylSpline disk) and hot (→ Multipole halo) components
        using ``cold_temp_log10_thresh``.

    Returns
    -------
    dict
        ``{"dark": {"host.distance": pos, "mass": mass}, "star": {...}, "gas": {...}}``
        Any species omitted has an empty sub-dict.
    """
    pos_dark = np.asarray(pos_dark, dtype=float)
    mass_dark = np.asarray(mass_dark, dtype=float)
    if pos_dark.ndim != 2 or pos_dark.shape[1] != 3:
        raise ValueError("pos_dark must be shape (N, 3).")
    if mass_dark.shape[0] != pos_dark.shape[0]:
        raise ValueError("mass_dark length must match pos_dark rows.")

    snap: dict = {
        "dark": {"host.distance": pos_dark, "mass": mass_dark},
        "star": {},
        "gas": {},
    }

    if pos_star is not None or mass_star is not None:
        if pos_star is None or mass_star is None:
            raise ValueError("Both pos_star and mass_star must be provided together.")
        pos_star = np.asarray(pos_star, dtype=float)
        mass_star = np.asarray(mass_star, dtype=float)
        if pos_star.ndim != 2 or pos_star.shape[1] != 3:
            raise ValueError("pos_star must be shape (N, 3).")
        if mass_star.shape[0] != pos_star.shape[0]:
            raise ValueError("mass_star length must match pos_star rows.")
        snap["star"]["host.distance"] = pos_star
        snap["star"]["mass"] = mass_star

    if pos_gas is not None or mass_gas is not None:
        if pos_gas is None or mass_gas is None:
            raise ValueError("Both pos_gas and mass_gas must be provided together.")
        pos_gas = np.asarray(pos_gas, dtype=float)
        mass_gas = np.asarray(mass_gas, dtype=float)
        if pos_gas.ndim != 2 or pos_gas.shape[1] != 3:
            raise ValueError("pos_gas must be shape (N, 3).")
        if mass_gas.shape[0] != pos_gas.shape[0]:
            raise ValueError("mass_gas length must match pos_gas rows.")
        snap["gas"]["host.distance"] = pos_gas
        snap["gas"]["mass"] = mass_gas
        if temperature_gas is not None:
            temperature_gas = np.asarray(temperature_gas, dtype=float)
            if temperature_gas.shape[0] != pos_gas.shape[0]:
                raise ValueError("temperature_gas length must match pos_gas rows.")
            snap["gas"]["temperature"] = temperature_gas

    return snap


# ---------------------------------------------------------------------------
# Potential fitting
# ---------------------------------------------------------------------------

def fit_potential(
    part: Mapping[str, Mapping[str, np.ndarray]],
    nsnap: int,
    *,
    sym: str | Sequence[str] = "n",
    pole_l: int | Sequence[int] = 4,
    rmax_sel: float,
    rmax_exp: float = 500.0,
    file_ext: str = "spline",
    save_dir: Union[str, Path] = "./",
    halo: Optional[str] = None,
    spec_ind: Optional[Mapping[str, Iterable[int]]] = None,
    kind: str = "whole",
    center: Optional[Union[np.ndarray, Sequence[float]]] = None,
    rotation: Optional[np.ndarray] = None,
    verbose: bool = True,
    subsample_factor: float = 1.0,
    cold_temp_log10_thresh: float = 4.5,
) -> dict[str, list[str]]:
    """
    Fit Agama Multipole and CylSpline potentials from a multi-species snapshot.

    Implements the expansion workflow of Arora et al. (2022) for cosmological
    zoom simulations:

    * Dark matter + hot gas → **Multipole** (spherical harmonic BFE)
    * Stars + cold gas      → **CylSpline** (azimuthal harmonic BFE)

    Particles are assumed to be in the desired frame before calling.
    Supply ``center`` and/or ``rotation`` and the transform
    ``pos_frame = np.dot(pos - center, rotation.T)`` is applied to each
    species internally, so the fitting is done in the rotated frame.

    Parameters
    ----------
    part : dict
        Snapshot dictionary as returned by :func:`create_snapshot_dict`, or
        any compatible dict with ``part[species]["host.distance"]`` (positions,
        shape (N, 3)) and ``part[species]["mass"]`` (masses, shape (N,)).
    nsnap : int
        Snapshot index used in output filenames (zero-padded to 3 digits).
    sym : str or sequence of str, optional
        Symmetry flag(s): ``'n'`` none (default), ``'a'`` axisymmetric,
        ``'s'`` spherical, ``'t'`` triaxial.  A list triggers one fit per
        symmetry.
    pole_l : int or sequence of int, optional
        Multipole lmax / CylSpline mmax order(s), by default 4.  A list
        triggers one fit per order.
    rmax_sel : float
        Selection aperture [kpc]: only particles within this radius contribute
        to the fit.  Required; must be > 0.
    rmax_exp : float, optional
        Maximum radius of the expansion radial grid [kpc], by default 500.
    file_ext : str, optional
        Suffix appended after the expansion-type token in output filenames,
        by default ``"spline"``.
    save_dir : str or Path, optional
        Directory to write coefficient files.  Created if absent.
        By default ``"./"``.
    halo : str, optional
        Optional halo label inserted in output filenames, e.g. ``"MW"`` or
        ``"LMC"``.  If ``None`` (default) the label is omitted.
    spec_ind : dict, optional
        ``{species: index_array}`` to restrict to a subset of particles within
        a species.  If ``None``, all particles of each species are used.
    kind : {'whole', 'dark', 'bar'}, optional
        Which expansion to fit: ``'whole'`` (default) fits both; ``'dark'``
        fits Multipole only; ``'bar'`` fits CylSpline only.
    center : array-like of length 3, optional
        Halo centre in the original coordinate frame [kpc].  Positions are
        shifted by ``pos - center`` before fitting.  ``None`` (default) means
        positions are already centred.
    rotation : ndarray, shape (3, 3), optional
        Rotation matrix aligning the halo into the desired frame (e.g.
        disk in the xy-plane).  Applied as ``np.dot(pos - center, rotation.T)``
        after centring.  ``None`` (default) means no rotation is applied.
    verbose : bool, optional
        Print progress messages, by default ``True``.
    subsample_factor : float, optional
        Multiply all particle masses by this factor to re-weight a subsampled
        snapshot, by default 1.0 (no reweighting).
    cold_temp_log10_thresh : float, optional
        log₁₀(T/K) threshold separating cold (→ CylSpline) from hot
        (→ Multipole) gas, by default 4.5.

    Returns
    -------
    dict
        ``{"multipole": [path, ...], "cylspline": [path, ...]}`` listing files
        written for each expansion type.
    """
    _require_agama()
    import agama

    if rmax_sel <= 0:
        raise ValueError("rmax_sel must be > 0.")

    allowed_syms = {"n": "none", "a": "axi", "s": "sph", "t": "triax"}
    syms = [sym] if isinstance(sym, str) else list(sym)
    for s in syms:
        if s not in allowed_syms:
            raise ValueError(f"Unknown symmetry '{s}'. Allowed: {list(allowed_syms)}")

    pole_ls = [pole_l] if isinstance(pole_l, int) else list(pole_l)
    if any(not isinstance(l, int) or l < 0 for l in pole_ls):
        raise ValueError("pole_l entries must be non-negative integers.")

    if kind not in ("whole", "dark", "bar"):
        raise ValueError("kind must be one of {'whole', 'dark', 'bar'}.")

    # Validate and prepare center / rotation
    _center: np.ndarray | None = None
    _rotation: np.ndarray | None = None
    if center is not None:
        _center = np.asarray(center, dtype=float).ravel()
        if _center.shape != (3,):
            raise ValueError("center must be a 3-element array.")
    if rotation is not None:
        _rotation = np.asarray(rotation, dtype=float)
        if _rotation.shape != (3, 3):
            raise ValueError("rotation must be a (3, 3) array.")

    def _transform(pos: np.ndarray) -> np.ndarray:
        """Shift to halo centre then rotate into target frame."""
        if _center is not None:
            pos = pos - _center
        if _rotation is not None:
            pos = np.dot(pos, _rotation.T)
        return pos

    # Detect species with required arrays
    species_keys = [
        k for k, v in part.items()
        if isinstance(v, dict) and "mass" in v and "host.distance" in v
    ]
    if not species_keys:
        raise ValueError(
            "No valid species found in `part`. Each species dict must have "
            "'host.distance' and 'mass' arrays."
        )
    if verbose:
        ignored = set(part.keys()) - set(species_keys)
        if ignored:
            print(f"Ignoring non-particle keys in snapshot: {ignored}")

    # Build per-species index maps
    spec_ind_map: dict[str, np.ndarray] = {}
    for sp in species_keys:
        n = int(np.asarray(part[sp]["mass"]).shape[0])
        if spec_ind and sp in spec_ind:
            spec_ind_map[sp] = np.asarray(list(spec_ind[sp]), dtype=int)
        else:
            spec_ind_map[sp] = np.arange(n, dtype=int)

    # Transform positions, compute distances, load masses
    pos_t: dict[str, np.ndarray] = {}
    dist: dict[str, np.ndarray] = {}
    masses: dict[str, np.ndarray] = {}

    for sp in ("dark", "star", "gas"):
        if sp in part and "host.distance" in part[sp]:
            raw_pos = np.asarray(part[sp]["host.distance"])
            raw_mass = np.asarray(part[sp]["mass"])
            idx = spec_ind_map.get(sp, np.arange(raw_pos.shape[0], dtype=int))
            if np.any(idx >= raw_pos.shape[0]) or np.any(idx < 0):
                raise IndexError(f"spec_ind for '{sp}' contains out-of-range indices.")
            p = _transform(raw_pos[idx])
            pos_t[sp] = p
            dist[sp] = np.linalg.norm(p, axis=1)
            masses[sp] = raw_mass[idx] * float(subsample_factor)
        else:
            pos_t[sp] = np.empty((0, 3), dtype=float)
            dist[sp] = np.empty((0,), dtype=float)
            masses[sp] = np.empty((0,), dtype=float)

    sel = {sp: dist[sp] < rmax_sel for sp in ("dark", "star", "gas")}

    # Gas cold/hot split
    tsel_cold = np.zeros(dist["gas"].shape[0], dtype=bool)
    if "gas" in part and "temperature" in part.get("gas", {}):
        temp_all = np.asarray(part["gas"]["temperature"])
        temp_idx = spec_ind_map.get("gas", np.arange(temp_all.shape[0], dtype=int))
        if np.any(temp_all[temp_idx] <= 0.0):
            raise ValueError("All gas temperatures must be > 0 to compute log10.")
        tsel_cold = np.log10(temp_all[temp_idx]) < cold_temp_log10_thresh

    # Build CylSpline (bar) arrays: stars + cold gas
    pos_bar_list: list[np.ndarray] = []
    m_bar_list: list[np.ndarray] = []

    if masses["star"].size > 0:
        ps = pos_t["star"][sel["star"]]
        ms = masses["star"][sel["star"]]
        if ps.size > 0:
            pos_bar_list.append(ps)
            m_bar_list.append(ms)
            if verbose:
                print(f"  {ps.shape[0]} star particles selected (r < {rmax_sel} kpc).")

    pos_gas_hot = np.empty((0, 3), dtype=float)
    m_gas_hot = np.empty((0,), dtype=float)
    if masses["gas"].size > 0:
        pg = pos_t["gas"][sel["gas"]]
        mg = masses["gas"][sel["gas"]]
        if pg.size > 0:
            tc = tsel_cold[sel["gas"]]
            if tc.any():
                pos_bar_list.append(pg[tc])
                m_bar_list.append(mg[tc])
                if verbose:
                    print(f"  {tc.sum()} cold-gas particles → CylSpline.")
            pos_gas_hot = pg[~tc]
            m_gas_hot = mg[~tc]

    # Build Multipole (dark + hot gas) arrays
    pd_sel = pos_t["dark"][sel["dark"]]
    md_sel = masses["dark"][sel["dark"]]
    if pos_gas_hot.size:
        pos_mul = np.vstack([pd_sel, pos_gas_hot]) if pd_sel.size else pos_gas_hot
        m_mul = np.hstack([md_sel, m_gas_hot]) if md_sel.size else m_gas_hot
    else:
        pos_mul = pd_sel
        m_mul = md_sel

    pos_bar = np.vstack(pos_bar_list) if pos_bar_list else np.empty((0, 3), dtype=float)
    m_bar = np.hstack(m_bar_list) if m_bar_list else np.empty((0,), dtype=float)

    if pos_mul.size == 0 and kind in ("whole", "dark"):
        raise ValueError("No particles selected for Multipole (dark + hot gas).")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, list[str]] = {"multipole": [], "cylspline": []}
    nsnap_str = f"{int(nsnap):03d}"

    for s in syms:
        sym_label = allowed_syms[s]
        for l in pole_ls:
            if verbose:
                print(f"Fitting sym='{s}' ({sym_label}), order={l}, rmax_exp={rmax_exp} kpc")

            if kind in ("whole", "dark") and pos_mul.size > 0:
                p_dark = agama.Potential(
                    type="Multipole",
                    particles=(pos_mul, m_mul),
                    lmax=l, symmetry=s,
                    rmin=0.1, rmax=rmax_exp,
                )
                fname = f"{nsnap_str}.dark.{sym_label}_{l}"
                if halo:
                    fname += f".{halo}"
                fname += f".coef_mul_{file_ext}"
                path = str(save_dir_path / fname)
                p_dark.export(path)
                output_files["multipole"].append(path)
                if verbose:
                    print(f"  Saved multipole → {path}")

            if kind in ("whole", "bar"):
                if pos_bar.size == 0:
                    if verbose:
                        print("  No disk/bar particles; skipping CylSpline.")
                else:
                    p_bar = agama.Potential(
                        type="CylSpline",
                        particles=(pos_bar, m_bar),
                        mmax=l, symmetry=s,
                        rmin=0.1, rmax=rmax_exp,
                    )
                    fname = f"{nsnap_str}.bar.{sym_label}_{l}"
                    if halo:
                        fname += f".{halo}"
                    fname += f".coef_cylsp_{file_ext}"
                    path = str(save_dir_path / fname)
                    p_bar.export(path)
                    output_files["cylspline"].append(path)
                    if verbose:
                        print(f"  Saved CylSpline → {path}")

    if verbose:
        print("Done fitting potential models.")
    return output_files


# ---------------------------------------------------------------------------
# Benchmark helpers (used by __main__ only)
# ---------------------------------------------------------------------------

def _sample_spherical(n: int, scale_radius: float = 50.0) -> np.ndarray:
    """Sample *n* positions from a spherically declining density profile."""
    u = np.random.random(n)
    r = -scale_radius * np.log(1.0 - u * 0.95)
    cos_theta = 2.0 * np.random.random(n) - 1.0
    phi = 2.0 * np.pi * np.random.random(n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    return np.column_stack([r * sin_theta * np.cos(phi),
                            r * sin_theta * np.sin(phi),
                            r * cos_theta])


def _sample_disk(n: int, r_scale: float = 3.0, z_sigma: float = 0.2) -> np.ndarray:
    """Sample *n* positions from a thin exponential disk."""
    u = np.random.random(n)
    R = -r_scale * np.log(1.0 - u)
    phi = 2.0 * np.pi * np.random.random(n)
    z = np.random.normal(scale=z_sigma, size=n)
    return np.column_stack([R * np.cos(phi), R * np.sin(phi), z])


if __name__ == "__main__":
    import time
    np.random.seed(13)
    N_dark, N_star, N_gas = 20_000, 5_000, 3_000
    print("Sampling synthetic snapshot...")
    snap = create_snapshot_dict(
        pos_dark=_sample_spherical(N_dark, 30.0),
        mass_dark=np.ones(N_dark) * (1e6 / N_dark),
        pos_star=_sample_disk(N_star, 3.0, 0.15),
        mass_star=np.ones(N_star) * (5e5 / N_star),
        pos_gas=_sample_disk(N_gas, 4.0, 0.3),
        mass_gas=np.ones(N_gas) * (1e5 / N_gas),
    )
    t0 = time.perf_counter()
    out = fit_potential(
        snap, nsnap=0,
        sym=["n", "a"], pole_l=[2, 4],
        rmax_sel=600.0, rmax_exp=100.0,
        save_dir="./demo_output", file_ext="spline",
    )
    print(f"Elapsed: {time.perf_counter()-t0:.3f} s")
    for k, v in out.items():
        print(f"  {k}: {len(v)} files")
