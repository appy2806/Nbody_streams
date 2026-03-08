"""Plotting functions for stream and N-body visualization.

Public API (flat via ``nbody_streams.viz.*``):
- ``plot_density``          -- projected surface / volume density image (SPH / histogram)
- ``plot_mollweide``        -- Healpix Mollweide sky projection (requires healpy)
- ``plot_stream_sky``       -- 2x3 observed-coordinate diagnostic panels (requires agama)
- ``plot_stream_evolution`` -- 3-panel N-body evolution plot
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import matplotlib.axes
import matplotlib.colors
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =====================================================================
# Private helpers
# =====================================================================
def _gauss_filter_surf_dens(
    mass_particle_bin: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2-D mass histogram and return surface density.

    Parameters
    ----------
    mass_particle_bin : ndarray, shape (Nx, Ny)
        2-D histogram of total particle masses.
    xedges, yedges : ndarray
        Bin edges from ``np.histogram2d``.
    **kwargs
        Forwarded to ``scipy.ndimage.gaussian_filter`` (e.g. ``sigma``).

    Returns
    -------
    ndarray, shape (Nx, Ny)
        Smoothed surface mass density.
    """
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(mass_particle_bin, **kwargs)
    surface_area = np.diff(xedges)[0] * np.diff(yedges)[0]
    return smoothed / surface_area


def _generate_ticks(vmin: float, vmax: float) -> np.ndarray:
    """Return 3-4 'nice' ticks within [vmin, vmax]."""
    span = vmax - vmin
    eps = 1e-12

    for n in (3, 4):
        step_raw = span / (n - 1)
        if step_raw >= 1.0:
            step = float(round(step_raw))
            if step <= 0:
                step = 1.0
        else:
            step = round(step_raw * 2.0) / 2.0
            if step <= 0:
                step = step_raw

        start = vmin
        ticks = start + np.arange(n) * step

        if ticks[-1] > vmax + eps:
            k = int(np.ceil((ticks[-1] - vmax) / step))
            start -= k * step
            ticks = start + np.arange(n) * step

        inside = ticks[(ticks >= vmin - eps) & (ticks <= vmax + eps)]
        if inside.size >= 3:
            return np.unique(np.round(inside, 12))

    return np.round(np.linspace(vmin, vmax, 3), 12)


def _aggregate_data_chunk(chunk_data):
    """Helper for parallel pixel aggregation (pandas path)."""
    import pandas as pd

    pixel_indices, weights = chunk_data
    df_chunk = pd.DataFrame({"pixel": pixel_indices, "weight": weights})
    return df_chunk.groupby("pixel").sum()


def _extract_particles_at_step(part_xv: np.ndarray, time_step: int):
    """
    Return particle positions at *time_step*, handling both
    (N, T, 6)  and  (N, 3)/(N, 6)  layouts.
    """
    if part_xv.ndim == 3:
        return part_xv[:, time_step, :3]
    # Already a single snapshot
    return part_xv[:, :3]


# =====================================================================
# plot_density -- projected surface / volume density image
# =====================================================================
def plot_density(
    pos: np.ndarray | None = None,
    mass: np.ndarray | None = None,
    snap=None,
    spec: str = "dark",
    gridsize: float = 200.0,
    resolution: int = 512,
    xval: str = "x",
    yval: str = "z",
    density_kind: str = "surface",
    method: str = "sph",
    slice_width: float = 0.0,
    slice_axis: str | None = None,
    return_dens: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    colorbar_ax: matplotlib.axes.Axes | bool | None = None,
    scale_size: float = 0,
    cmap: matplotlib.colors.Colormap | str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    smooth_sigma: float = 1.0,
    **kwargs: Any,
) -> None | tuple[matplotlib.image.AxesImage, np.ndarray]:
    """
    Generate a projected density image (imshow) from particle data.

    Accepts particle arrays directly, or a ``ParticleReader`` snapshot from
    which positions and masses are extracted by species.

    Parameters
    ----------
    pos : ndarray, shape (N, 3), optional
        Particle positions in kpc.  Required when *snap* is *None*.
    mass : ndarray, shape (N,), optional
        Particle masses in M_sun.  Defaults to uniform unit mass when *None*.
    snap : ParticleReader snapshot, optional
        Object returned by ``ParticleReader.read_snapshot()``.  When provided,
        ``pos`` and ``mass`` are extracted from ``snap[spec]``.
    spec : str
        Particle species key -- ``'dark'``, ``'star'``, ``'gas'``, etc.
        Used for default colormap / colorbar label selection and for reading
        from *snap*.
    gridsize : float
        Total size of the rendered region in kpc.  The grid spans
        ``[-gridsize/2, gridsize/2]`` along both projected axes.
        Default ``200.0``.
    resolution : int
        Grid resolution (pixels per axis).  Default ``512``.
    xval, yval : str
        Projection axes: ``'x'``, ``'y'``, or ``'z'``.
    density_kind : {'surface', 'volume'}
        ``'surface'`` returns M/kpc^2; ``'volume'`` divides further by
        2 * *slice_width* to give M/kpc^3.
    method : {'sph', 'histogram', 'gauss_smooth'}
        Density estimation method.

        * ``'sph'`` *(default)* -- physics-motivated SPH kernel splatting via
          :func:`~nbody_streams.viz.sph_kernels.render_surface_density`.
          Smoothing lengths are computed automatically (GPU -> CPU fallback).
        * ``'histogram'`` -- raw 2-D mass histogram divided by pixel area.
        * ``'gauss_smooth'`` -- histogram followed by a Gaussian filter
          (sigma = *smooth_sigma* pixels).
    slice_width : float
        If > 0, retain only particles within +/- *slice_width* kpc of the
        origin along *slice_axis*.
    slice_axis : str, optional
        Axis along which the slice is applied.  Inferred as the remaining
        axis when *None*.
    return_dens : bool
        If *True*, return ``(im_obj, density_array)`` where
        ``density_array`` is the raw (pre-log10) density map produced by
        *method* -- SPH grid, Gaussian-smoothed map, or raw histogram,
        depending on *method*.
    ax : Axes, optional
        Existing axes.  A new 3x3 in figure is created when *None*.
    colorbar_ax : Axes or True or None
        Axes for the colorbar, or *True* for an auto-inset horizontal bar.
    scale_size : float
        If > 0, draw a physical scale bar of this length in kpc.
    cmap : colormap or str, optional
        Defaults to cmasher palettes when available, else ``'cubehelix'``.
    vmin, vmax : float, optional
        Colour limits in log10 density units.  When equal, matplotlib
        auto-scales.
    smooth_sigma : float
        Gaussian smoothing width in pixels.  Only used when
        ``method='gauss_smooth'``.  Default ``1.0``.
    **kwargs
        Advanced SPH options extracted by name (remaining kwargs are
        forwarded to the colorbar label ``ax.text`` call):

        * ``arch`` (str, default ``'auto'``) -- SPH compute backend:
          ``'auto'``, ``'gpu'``, or ``'cpu'``.
        * ``k_neighbors`` (int, default ``32``) -- neighbours used when
          computing SPH smoothing lengths.
        * ``chunk_size`` (int, default ``10_000_000``) -- GPU tile size
          for SPH splatting.
    """
    # --- Extract advanced SPH options from kwargs ---
    arch        = kwargs.pop("arch",         "auto")
    k_neighbors = kwargs.pop("k_neighbors",  32)
    chunk_size  = kwargs.pop("chunk_size",   10_000_000)

    # --- Validate method ---
    valid_methods = ("sph", "histogram", "gauss_smooth")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods!r}; got {method!r}")

    # --- Validate density_kind ---
    density_kind = density_kind.lower()
    if density_kind not in ("volume", "surface"):
        warnings.warn(f"density_kind '{density_kind}' invalid, forcing 'surface'.")
        density_kind = "surface"
    if density_kind == "volume" and slice_width == 0:
        warnings.warn("slice_width=0 invalid for volume density, forcing 1.")
        slice_width = 1.0

    spec = spec.lower()
    xval = xval.lower()
    yval = yval.lower()
    if slice_axis is not None:
        slice_axis = slice_axis.lower()

    from matplotlib import offsetbox
    from matplotlib.lines import Line2D

    # --- Extract pos / mass ---
    if snap is not None:
        try:
            spec_data = snap[spec] if hasattr(snap, "__getitem__") else getattr(snap, spec)
        except (KeyError, AttributeError):
            raise ValueError(
                f"snap has no species '{spec}'. "
                "Pass pos/mass directly or check the species name."
            )
        pos = np.asarray(spec_data["posvel"])[:, :3]
        mass = np.asarray(spec_data["mass"])

    if pos is None:
        raise ValueError(
            "Either pos=(N,3) or a ParticleReader snap object is required."
        )

    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3).")

    if mass is None:
        mass = np.ones(pos.shape[0], dtype=np.float32)
    elif np.isscalar(mass):
        mass = np.full(pos.shape[0], float(mass), dtype=np.float32)
    else:
        mass = np.asarray(mass, dtype=np.float32).ravel()
        if mass.shape[0] != pos.shape[0]:
            raise ValueError("mass length must match pos length.")

    # --- Axes mapping ---
    axes_kind = {"x": 0, "y": 1, "z": 2}
    if xval not in axes_kind or yval not in axes_kind:
        raise ValueError("xval and yval must be one of 'x', 'y', 'z'.")
    if xval == yval:
        raise ValueError("xval and yval must be different axes.")
    if slice_axis is None:
        leftover = set(axes_kind) - {xval, yval}
        slice_axis = leftover.pop() if len(leftover) == 1 else "y"
    if slice_axis not in axes_kind:
        raise ValueError("slice_axis must be one of 'x', 'y', 'z'.")

    xi = axes_kind[xval]
    yi = axes_kind[yval]
    zi = axes_kind[slice_axis]

    # --- Slice ---
    if slice_width > 0:
        mask = np.abs(pos[:, zi]) <= slice_width
        if mask.sum() == 0:
            warnings.warn("Slice contains zero particles; result will be empty.")
        pos = pos[mask]
        mass = mass[mask]

    x_2d = pos[:, xi]
    y_2d = pos[:, yi]

    # --- Compute density ---
    if method == "sph":
        from .sph_kernels import render_surface_density
        calc_density, _ = render_surface_density(
            x_2d, y_2d, mass,
            resolution=resolution, gridsize=gridsize,
            k_neighbors=k_neighbors, arch=arch, chunk_size=chunk_size,
        )
    else:
        half = gridsize / 2.0
        bins = np.linspace(-half, half, resolution + 1)
        mass_bin, xedges, yedges = np.histogram2d(x_2d, y_2d, weights=mass, bins=bins)
        if method == "gauss_smooth":
            calc_density = _gauss_filter_surf_dens(
                mass_bin, xedges, yedges, sigma=smooth_sigma
            )
        else:  # "histogram"
            calc_density = mass_bin / (np.diff(xedges)[0] * np.diff(yedges)[0])

    if slice_width > 0 and density_kind == "volume":
        calc_density = calc_density / (2.0 * slice_width)

    if return_dens:
        dens_to_return = calc_density.copy()

    # --- Log scale (zeros -> 1 so log10 = 0, rendered as background colour) ---
    calc_density[calc_density == 0] = 1
    calc_density = np.log10(calc_density)

    # --- Colormap ---
    if cmap is None:
        try:
            import cmasher as cmr
            colormaps = {"star": cmr.ghostlight, "gas": cmr.gothic, "dark": cmr.eclipse}
            cmap = colormaps.get(spec, cmr.eclipse)
        except Exception:
            cmap = "cubehelix"

    vmins = {"star": 3.0, "gas": 4.0, "dark": 6.5}
    vmaxs = {"star": 10.0, "gas": 9.5, "dark": 9.5}
    vmin_use = vmins.get(spec, 5.0) if vmin is None else vmin
    vmax_use = vmaxs.get(spec, 9.5) if vmax is None else vmax
    if vmin_use == vmax_use:
        vmin_use, vmax_use = None, None

    # --- Plot ---
    if ax is None:
        fig = plt.figure(figsize=(3, 3), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("k")

    im_obj = ax.imshow(
        calc_density.T,
        interpolation="bilinear",
        cmap=cmap,
        origin="lower",
        vmin=vmin_use,
        vmax=vmax_use,
        aspect=1,
        extent=[-gridsize/2, gridsize/2, -gridsize/2, gridsize/2],
    )
    vmin_use, vmax_use = im_obj.get_clim()

    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.tick_params(bottom=False, left=False)

    # --- Scale bar ---
    if scale_size > 0:
        class _AnchoredHScaleBar(offsetbox.AnchoredOffsetbox):
            def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                         pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                         frameon=True, **kw):
                if not ax:
                    ax = plt.gca()
                trans = ax.get_xaxis_transform()
                size_bar = offsetbox.AuxTransformBox(trans)
                size_bar.add_artist(Line2D([0, size], [0, 0], **kw))
                size_bar.add_artist(Line2D([0, 0], [-extent/2, extent/2], **kw))
                size_bar.add_artist(Line2D([size, size], [-extent/2, extent/2], **kw))
                txt = offsetbox.TextArea(
                    label, textprops=dict(color="white", fontsize=8)
                )
                self.vpac = offsetbox.VPacker(
                    children=[size_bar, txt], align="center", pad=ppad, sep=sep
                )
                super().__init__(
                    loc, pad=pad, borderpad=borderpad,
                    child=self.vpac, prop=prop, frameon=frameon,
                )

        ob = _AnchoredHScaleBar(
            size=scale_size,
            label=f'{{\\bf {int(scale_size)} kpc}}',
            loc=3, frameon=False, pad=0.2, sep=0.3, borderpad=0.7,
            color="white", linewidth=2.0, ax=ax,
        )
        ax.add_artist(ob)

    # --- Colorbar ---
    if colorbar_ax:
        if isinstance(colorbar_ax, matplotlib.axes.Axes):
            pass
        elif colorbar_ax is True:
            colorbar_ax = ax.inset_axes((0.35, 0.12, 0.6, 0.025))
        else:
            raise ValueError("colorbar_ax must be an Axes object or True.")

        cbar = plt.colorbar(
            im_obj, cax=colorbar_ax, orientation="horizontal",
            aspect=25, extend="both",
        )
        cbar.ax.tick_params(
            size=3, width=0.5, labelsize=10, color="white", labelcolor="white",
        )
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.12)

        ticks = _generate_ticks(vmin_use, vmax_use)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([rf'$\mathbf{{10^{{{int(t)}}}}}$' for t in ticks])

        if density_kind == "volume":
            label = rf'$\mathbf{{{{\rho_{{{spec}}}[M_{{\odot}}/kpc^3]}}}}$'
        else:
            label = rf'$\mathbf{{{{\Sigma_{{{spec}}}[M_{{\odot}}/kpc^2]}}}}$'
        ax.text(
            0.7, 0.2, label, ha="center", va="center", transform=ax.transAxes,
            color="w", fontsize=12,
            bbox=dict(facecolor="none", edgecolor="none"), **kwargs,
        )

    if return_dens:
        return im_obj, dens_to_return
    return None


# =====================================================================
# plot_mollweide -- Healpix Mollweide sky projection
# =====================================================================
def plot_mollweide(
    pos: np.ndarray,
    weights: np.ndarray | None = None,
    initial_nside: int = 60,
    normalize: bool = False,
    log_scale: bool = True,
    filter_radius: tuple[float, float] = (0, 0),
    return_map: bool = False,
    smooth_fwhm_deg: float | None = None,
    verbose: bool = False,
    add_traj: np.ndarray | None = None,
    add_end_pt: bool = False,
    add_traj_dist: bool = False,
    density_threshold: float = 1e5,
    cmap: str = "bone",
    **kwargs,
) -> np.ndarray | None:
    """
    Mollweide projection of a 3-D particle field using Healpix.

    Supports automatic nside upscaling, Gaussian smoothing, optional
    trajectory overlays, and large-dataset parallelism via vaex or pandas.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Cartesian positions.
    weights : ndarray, optional
        Per-particle weights (e.g. masses).  Default: uniform.
    initial_nside : int
        Starting Healpix resolution.
    normalize : bool
        Compute fractional variation relative to median density.
    log_scale : bool
        Convert density to log10 before plotting.
    filter_radius : (float, float)
        ``(radius, tol)`` or ``(rmin, rmax)`` radial filter.
    return_map : bool
        Return the smoothed sky map array.
    smooth_fwhm_deg : float, optional
        Smoothing FWHM in degrees.  Auto-set from nside if *None*.
    verbose : bool
        Print diagnostic messages.
    add_traj : ndarray, optional
        (M, 3) trajectory to overplot on the map.
    add_end_pt : bool
        Mark the last trajectory point with a star.
    density_threshold : float
        Particle count that triggers nside upscaling.
    cmap : str
        Colormap name.
    **kwargs
        Forwarded to ``hp.newvisufunc.projview``.

    Returns
    -------
    ndarray or None
        Smoothed sky map if *return_map* is True.

    Notes
    -----
    Requires ``healpy``.  For large datasets (>1M particles), ``vaex``
    is recommended for fast pixel aggregation; otherwise ``pandas``
    with multiprocessing is used as a fallback.
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpy is required for plot_mollweide.  "
            "Install via: pip install nbody_streams[healpy]"
        )

    # vaex > pandas > numpy for pixel aggregation
    try:
        import vaex
        has_vaex = True
    except ImportError:
        has_vaex = False

    pos = np.asarray(pos)

    # --- Radial filter ---
    if filter_radius[0] > 0 and filter_radius[1] > 0 and filter_radius[0] >= filter_radius[1]:
        distances = np.linalg.norm(pos, axis=1)
        mask = np.isclose(distances, filter_radius[0], atol=filter_radius[1])
        pos = pos[mask]
        if weights is not None:
            weights = weights[mask]
    elif filter_radius[0] >= 0 and filter_radius[1] > filter_radius[0]:
        distances = np.linalg.norm(pos, axis=1)
        mask = (distances >= filter_radius[0]) & (distances <= filter_radius[1])
        pos = pos[mask]
        if weights is not None:
            weights = weights[mask]

    # --- Dynamic nside ---
    nside = initial_nside
    if pos.shape[0] > density_threshold:
        if verbose:
            print("Dynamically adjusting nside based on particle count")
        nside = min(512, int(initial_nside * (pos.shape[0] / density_threshold) ** 0.5))
        if verbose:
            print(f"  nside: {nside}")

    # --- Spherical coords -> pixel indices ---
    pos_spherical = hp.rotator.vec2dir(pos.T, lonlat=False).T
    pixel_indices = hp.ang2pix(nside, pos_spherical[:, 0], pos_spherical[:, 1])
    num_pixels = hp.nside2npix(nside)

    if weights is None:
        weights = np.ones(pos.shape[0])

    # --- Pixel aggregation ---
    if has_vaex:
        df = vaex.from_arrays(pixel=pixel_indices, weight=weights)
        pixel_data = df.groupby("pixel", agg={"weight": vaex.agg.sum("weight")})
        pixel_ids = pixel_data["pixel"].values
        total_mass = pixel_data["weight"].values
    elif pos.shape[0] > 1_000_000:
        import pandas as pd
        from multiprocessing import cpu_count, Pool

        num_chunks = max(4, min(cpu_count() - 1, 8))
        chunk_size = len(pixel_indices) // num_chunks
        chunks = [
            (pixel_indices[i : i + chunk_size], weights[i : i + chunk_size])
            for i in range(0, len(pixel_indices), chunk_size)
        ]
        with Pool(num_chunks) as pool:
            chunk_results = pool.map(_aggregate_data_chunk, chunks)

        df_combined = pd.concat(chunk_results).groupby("pixel").sum()
        pixel_ids = df_combined.index.values.astype(np.int64)
        total_mass = df_combined["weight"].values
    else:
        unique_pixels, inverse = np.unique(pixel_indices, return_inverse=True)
        total_mass = np.bincount(inverse, weights=weights)
        pixel_ids = unique_pixels.astype(np.int64)

    # --- Build sky map ---
    area_deg2 = hp.nside2pixarea(nside, degrees=True)
    sky_map = np.zeros(num_pixels)
    sky_map[pixel_ids] = total_mass / area_deg2
    if log_scale:
        sky_map[pixel_ids] = np.log10(total_mass / area_deg2)

    if normalize:
        med = np.median(sky_map[sky_map > 0])
        sky_map = sky_map / med - 1

    # --- Smoothing ---
    if smooth_fwhm_deg is None:
        if verbose:
            print("Computing dynamic FWHM smoothing")
        smooth_fwhm_rad = 3 * np.sqrt(hp.nside2pixarea(nside))
    else:
        smooth_fwhm_rad = np.radians(smooth_fwhm_deg)

    map_smoothed = hp.smoothing(sky_map, fwhm=smooth_fwhm_rad)

    # --- Plot ---
    hp.newvisufunc.projview(
        map_smoothed,
        coord=["G"],
        projection_type="mollweide",
        extend="both",
        flip="astro",
        cmap=cmap,
        **kwargs,
    )

    if add_traj is not None and len(add_traj) > 0:
        add_traj = np.asarray(add_traj)
        theta_traj, phi_traj = hp.rotator.vec2dir(add_traj.T, lonlat=False)
        hp.newvisufunc.newprojplot(theta_traj, phi_traj, c="lime", ls="--")
        if add_end_pt:
            hp.newvisufunc.newprojplot(
                theta_traj[-1], phi_traj[-1], marker="*", c="lime", linewidth=0.25,
            )

    return map_smoothed if return_map else None


# =====================================================================
# plot_stream_sky -- 2x3 observed-coordinate panels (requires agama)
# =====================================================================
def plot_stream_sky(
    xv_stream: np.ndarray,
    color: str = "ro",
    ax: np.ndarray | None = None,
    xv_prog: np.ndarray | list | None = None,
    alpha_lim: tuple[float | None, float | None] = (None, None),
    delta_lim: tuple[float | None, float | None] = (None, None),
    phi1_lim: tuple[float | None, float | None] = (None, None),
    phi2_lim: tuple[float | None, float | None] = (None, None),
    ms: float = 0.5,
    mew: float = 0,
) -> tuple[Figure | None, np.ndarray]:
    """
    2x3 diagnostic panel: (RA, Dec), (RA, v_los), (phi1, phi2),
    (X, Y), (X, Z), (Y, Z).

    Parameters
    ----------
    xv_stream : ndarray, shape (N, 6)
        Stream particles in galactocentric coordinates.
    color : str
        Matplotlib format string.
    ax : ndarray of Axes (2, 3), optional
        Existing axes to plot into.
    xv_prog : ndarray (6,), optional
        Progenitor phase-space vector for observed-frame computation.
    alpha_lim, delta_lim : tuple
        RA / Dec axis limits.
    phi1_lim, phi2_lim : tuple
        Stream coordinate axis limits.
    ms, mew : float
        Marker size / edge width.

    Returns
    -------
    fig : Figure or None
    ax : ndarray of Axes (2, 3)

    Notes
    -----
    Requires ``agama`` (used by ``coords.get_observed_stream_coords``).
    """
    from ..coords.streams import get_observed_stream_coords

    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Set aspect ratio (skip RA-vlos panel)
    for i, a in enumerate(ax.flatten()):
        if i != 1:
            a.set_aspect("equal")

    ax[0, 0].set_xlabel(r'$\alpha$ [deg]')
    ax[0, 0].set_ylabel(r'$\delta$ [deg]')
    ax[0, 1].set_xlabel(r'$\alpha$ [deg]')
    ax[0, 1].set_ylabel(r'$v_{\rm los}$ [km/s]')
    ax[0, 2].set_xlabel(r'$\phi_1$ [deg]')
    ax[0, 2].set_ylabel(r'$\phi_2$ [deg]')
    ax[1, 0].set_xlabel('X [kpc]')
    ax[1, 0].set_ylabel('Y [kpc]')
    ax[1, 1].set_xlabel('X [kpc]')
    ax[1, 1].set_ylabel('Z [kpc]')
    ax[1, 2].set_xlabel('Y [kpc]')
    ax[1, 2].set_ylabel('Z [kpc]')

    ra, dec, vlos, phi1, phi2 = get_observed_stream_coords(xv_stream, xv_prog)

    ax[0, 0].plot(ra, dec, color, ms=ms, mew=mew)
    ax[0, 1].plot(ra, vlos, color, ms=ms, mew=mew)
    ax[0, 2].plot(phi1, phi2, color, ms=ms, mew=mew)

    ax[0, 0].set_xlim(alpha_lim)
    ax[0, 0].set_ylim(delta_lim)
    ax[0, 1].set_xlim(alpha_lim)
    ax[0, 2].set_xlim(phi1_lim)
    ax[0, 2].set_ylim(phi2_lim)

    ax[1, 0].plot(xv_stream[:, 0], xv_stream[:, 1], color, ms=ms, mew=mew)
    ax[1, 1].plot(xv_stream[:, 0], xv_stream[:, 2], color, ms=ms, mew=mew)
    ax[1, 2].plot(xv_stream[:, 1], xv_stream[:, 2], color, ms=ms, mew=mew)
    plt.tight_layout()

    if return_fig:
        return fig, ax
    return None, ax


# =====================================================================
# plot_stream_evolution -- 3-panel N-body evolution
# =====================================================================
def plot_stream_evolution(
    prog_xv: np.ndarray | dict,
    times: np.ndarray | None = None,
    part_xv: np.ndarray | None = None,
    bound_mass: np.ndarray | None = None,
    time_step: int = -1,
    x_axis: int = 0,
    y_axis: int = 2,
    LMC_traj: np.ndarray | None = None,
    three_d_plot: bool = False,
    interactive: bool = False,
    dpi: int = 200,
    figsize: tuple[float, float] = (12, 3),
) -> tuple[Figure, list]:
    """
    Three-panel evolution plot: galactocentric distance, bound fraction
    (or 3-D trajectory), and projected particle positions.

    Parameters
    ----------
    prog_xv : ndarray (T, 6) **or** dict
        Progenitor trajectory.  If a *dict*, it is treated as a legacy
        ``Nbody_out`` dictionary with keys ``'prog_xv'``, ``'times'``,
        ``'part_xv'``, and optionally ``'bound_mass'``.
    times : ndarray (T,), optional
        Time array (required when *prog_xv* is an array).
    part_xv : ndarray, optional
        Particle positions.  Shape ``(N, T, 6)`` for multi-snapshot or
        ``(N, 3)`` / ``(N, 6)`` for a single snapshot.
    bound_mass : ndarray (T,), optional
        Bound mass history for the middle panel.
    time_step : int
        Snapshot index for the right panel (default: last).
    x_axis, y_axis : int
        Coordinate indices for the right panel (0=X, 1=Y, 2=Z).
    LMC_traj : ndarray (M, 4), optional
        ``[t, x, y, z]`` LMC trajectory to overplot.
    three_d_plot : bool
        Use a 3-D middle panel instead of bound-fraction.
    interactive : bool
        Enable IPython widget backend (requires ``three_d_plot=True``).

    Returns
    -------
    fig : Figure
    ax : list of 3 Axes
    """
    from scipy.signal import find_peaks

    # --- Unpack dict if provided ---
    if isinstance(prog_xv, dict):
        Nbody_out = prog_xv
        prog_xv = np.asarray(Nbody_out["prog_xv"])
        times = np.asarray(Nbody_out["times"])
        part_xv = np.asarray(Nbody_out["part_xv"]) if "part_xv" in Nbody_out else None
        bound_mass = np.asarray(Nbody_out["bound_mass"]) if "bound_mass" in Nbody_out else None
        if LMC_traj is None and "LMC_traj" in Nbody_out:
            LMC_traj = np.asarray(Nbody_out["LMC_traj"])

    prog_xv = np.asarray(prog_xv)
    times = np.asarray(times)

    if interactive and not three_d_plot:
        raise ValueError("interactive=True requires three_d_plot=True")
    if interactive:
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic("matplotlib", "widget")
        except (ImportError, AttributeError):
            raise RuntimeError("Interactive mode requires IPython")

    # --- Figure ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = [None, None, None]
    ax[0] = fig.add_subplot(131)
    ax[2] = fig.add_subplot(133)
    ax[1] = fig.add_subplot(132, projection="3d" if three_d_plot else None)
    plt.subplots_adjust(wspace=0.25)

    axes_label = {0: r'{\bf X [kpc]}', 1: r'{\bf Y [kpc]}', 2: r'{\bf Z [kpc]}'}
    ax[0].set_xlabel(r'{\bf T [Gyr]}')
    ax[0].set_ylabel(r'{\bf $\mathbf{d_{cen}}$ [kpc]}')
    ax[2].set_xlabel(axes_label[x_axis])
    ax[2].set_ylabel(axes_label[y_axis])

    if three_d_plot:
        ax[1].set_xlabel(axes_label[0])
        ax[1].set_ylabel(axes_label[1])
        ax[1].set_zlabel(axes_label[2])
    else:
        ax[1].set_xlabel(r'{\bf T [Gyr]}')
        ax[1].set_ylabel(r'{\bf Bound Frac}')

    # --- Left panel: distance vs time ---
    distances = np.linalg.norm(prog_xv[:, :3], axis=1)
    t_slice = times[:time_step] if time_step != -1 else times
    d_slice = distances[:time_step] if time_step != -1 else distances
    ax[0].plot(t_slice, d_slice, c="r", lw=0.5)

    # --- Middle panel ---
    if not three_d_plot:
        if bound_mass is not None:
            bound_frac = bound_mass / bound_mass[0]
            ax[1].plot(times, bound_frac, c="r", lw=0.5)
        else:
            ax[1].plot(times, np.zeros_like(times), c="r", lw=0.5)
    else:
        px, py, pz = prog_xv[:, 0], prog_xv[:, 1], prog_xv[:, 2]
        if time_step == -1:
            ax[1].plot(px, py, pz, c="r", lw=0.5)
        else:
            ax[1].plot(px[:time_step+1], py[:time_step+1], pz[:time_step+1], c="r", lw=0.5)

        if part_xv is not None:
            pts = _extract_particles_at_step(part_xv, time_step)
            ax[1].scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.15, alpha=0.15, c="m")
            ax[1].xaxis.pane.fill = False
            ax[1].yaxis.pane.fill = False
            ax[1].zaxis.pane.fill = False
            ax[1].grid(False)

            # PCA-based viewing angle
            data = pts - pts.mean(axis=0)
            if len(data) > 1 and not np.allclose(data, 0):
                try:
                    cov = np.cov(data, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    pc3 = eigvecs[:, eigvals.argsort()[0]]
                    azim = np.degrees(np.arctan2(pc3[1], pc3[0]))
                    elev = 90 - np.degrees(np.arccos(np.clip(pc3[2] / np.linalg.norm(pc3), -1, 1)))
                    ax[1].view_init(elev=max(0, min(90, elev)), azim=azim % 360)
                except Exception:
                    ax[1].view_init(elev=30, azim=45)
            else:
                ax[1].view_init(elev=30, azim=45)

    # --- Pericenter markers ---
    peaks, _ = find_peaks(-distances)
    for peak in peaks:
        ax[0].axvline(times[peak], color="gray", lw=0.5, alpha=0.5, ls="--")
        if not three_d_plot:
            ax[1].axvline(times[peak], color="gray", lw=0.5, alpha=0.5, ls="--")

    # --- Right panel: particle scatter ---
    ax[2].scatter(
        prog_xv[0, x_axis], prog_xv[0, y_axis],
        facecolor="none", edgecolor="r", s=50, marker="*", linewidth=0.3, zorder=3,
    )
    ax[2].scatter(
        prog_xv[-1, x_axis], prog_xv[-1, y_axis],
        facecolor="none", edgecolor="k", s=50, marker="*", linewidth=0.3, zorder=3,
    )

    if part_xv is not None:
        pts = _extract_particles_at_step(part_xv, time_step)
        ax[2].scatter(pts[:, x_axis], pts[:, y_axis], s=0.15, alpha=0.15, c="m")

    if time_step == -1:
        ax[2].plot(prog_xv[:, x_axis], prog_xv[:, y_axis], lw=0.5, ls="--", c="gray")
    else:
        ax[2].plot(
            prog_xv[:time_step+1, x_axis], prog_xv[:time_step+1, y_axis],
            lw=0.5, ls="--", c="gray",
        )

    if three_d_plot:
        ax[1].relim()
        ax[1].autoscale_view()
        ax[1].set_autoscale_on(False)
    ax[2].relim()
    ax[2].autoscale_view()
    ax[2].autoscale(False)
    ax[0].autoscale(False)

    # --- LMC trajectory ---
    if LMC_traj is not None and len(LMC_traj) > 0:
        LMC_traj = np.asarray(LMC_traj)
        ax[0].plot(
            LMC_traj[:, 0], np.linalg.norm(LMC_traj[:, 1:], axis=1),
            c="gray", lw=2, alpha=0.4, ls="--",
        )
        ax[2].plot(LMC_traj[:, x_axis+1], LMC_traj[:, y_axis+1], c="gray", lw=2, alpha=0.4, ls="--")
        ax[2].scatter(
            LMC_traj[-1, x_axis+1], LMC_traj[-1, y_axis+1],
            facecolor="none", edgecolor="gray", s=50, marker="*", linewidth=1, zorder=3,
        )
        if three_d_plot:
            ax[1].plot(LMC_traj[:, 1], LMC_traj[:, 2], LMC_traj[:, 3], c="gray", lw=2, alpha=0.4, ls="--")
            ax[1].scatter(
                LMC_traj[-1, 1], LMC_traj[-1, 2], LMC_traj[-1, 3],
                facecolor="none", edgecolor="gray", s=50, marker="*", linewidth=1, zorder=3,
            )

    if x_axis == 2 or y_axis == 2:
        ax[2].plot([-10, 10], [0, 0], c="g", alpha=0.5)

    if interactive and three_d_plot:
        ax[1].mouse_init()
        def on_move(event):
            if event.inaxes == ax[1]:
                ax[1].view_init(elev=ax[1].elev, azim=ax[1].azim)
        fig.canvas.mpl_connect("motion_notify_event", on_move)

    return fig, ax
