# Graph Report - Nbody_streams  (2026-11-15)

## Corpus Check
- 110 files · ~219,762 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2908 nodes · 5495 edges · 300 communities (186 shown, 114 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 433 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `51eb9c2a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- _make_ic
- MultipolePotentialGPU
- sph_kernels.py
- _load.py
- _nfw_potential
- test_utils.py
- __init__.py
- `nbody_streams.viz` — Visualization
- convert_coords
- create_particle_spray_stream
- CylSplinePotentialGPU
- ShiftedPotentialGPU
- run_simulation
- generate_stream_coords
- _AnalyticBase
- tree_gravity_gpu
- CylSplineCoefs dataclass
- __init__.py
- test_phase2_analytic.py
- test_advanced.py
- CellData
- test_newtons_third_law.py
- _build.py
- .radial_power
- `nbody_streams.utils` — Analysis Utilities
- fit_dehnen_profile
- `nbody_streams.coords` — Coordinate Transforms
- load_agama_evolving_potential
- create_evolving_ini
- Treecode.h
- Treecode
- kpc km/s Msun unit system
- create_snapshot_dict
- .dark
- Species
- _potential.py
- nbody_io.py
- `nbody_streams.agama_helper`
- Changelog
- test_multi_species.py
- Box
- _force.py
- buildTree.cu
- NFWPotentialGPU
- _random_xv
- API
- r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I
- Particle4
- CompositePotentialGPU
- treewalk_warp
- treeGPU_interface.cu
- Ellipsoidal radius and particle selection (numba-accelerated).      Computes the
- Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti
- Generate uniformly random points on the surface of a sphere.      Parameters
- Generate num_pts points on a sphere using the Fibonacci spiral.      Points are
- Implementation internals
- Find the density peak by locating the minimum gravitational potential.      Uses
- computeMultipoles.cu
- TreeGPU
- test_chandrasekhar.py
- vec<3,double>
- vec<2,float>
- _find_cudart
- .alloc
- make_df_force_extra
- .test_circular_orbit_decays
- computeMultipoles.cu
- TreeGPU
- transforms.py
- _jeans_sigma_r
- _spheroid.py
- main.py
- vec<4,float>
- plot_density
- test_recentre.py
- compute_nbody_potential_gpu
- Writers and materialization
- run_nbody_gpu
- Load a pre-defined spherical spiral grid scaled to a given radius.      The grid
- GPU potential evaluation (PotentialGPU)
- FIRE helpers
- Dynamical friction
- cuda_primitives.h
- Quick-start workflow
- _random_xv
- _io.py
- TestBoundCenterPhi
- UniformAccelerationGPU
- AGAMA_GPU
- Quadrupole
- vector3
- Loading Agama potentials
- test_phase1_multipole.py
- Physics and formulae
- plots.py
- Quick start
- convert_to_vel_los
- compute_nbody_forces_gpu
- convert_vectors
- render_surface_density
- _cylspl_potential_kernel.cu
- AGAMA_GPU — Development Notes
- test_phase3_cylspline.py
- cuda_mem
- plot_stream_evolution
- ._state
- get_smoothing_lengths
- EvolvingPotentialGPU
- Public API
- vec<3,float>
- _multipole_potential_kernel.cu
- run_nbody_gpu_tree
- TreeGPU
- GroupData
- empirical_density_profile
- Potential
- Chandrasekhar dynamical friction
- _validate_species
- HDF5 I/O
- run_nbody_gpu
- TestConservationMultiSpecies
- Empirical radial profiles
- CoM galactocentric radius vs time chart
- _is_uniform
- nbody_streams — Documentation
- CLAUDE.md
- ._from_data
- _chandrasekhar module
- FIRE simulation helpers
- force_extra hook
- libtreeGPU.so shared library
- load_agama_evolving_potential
- load_agama_potential
- matplotlib dependency
- graphify knowledge graph rules
- NBODY_UNITS constant
- fit_double_spheroid_profile
- make_df_force_extra
- cuda_alive
- MultipoleCoefs dataclass
- `nbody_streams.fields`
- `nbody_streams.nbody_io` — HDF5 Snapshot I/O
- nbody_streams Changelog
- read_coefs unified API
- render_surface_density SPH
- _StepWatchdog
- Unreleased feat-dynamicFric
- version 1.0.0 initial release
- version 1.1.0 float4 vectorization
- version 1.2.0 fast_sims
- version 1.3.0 multi-species
- version 2.0.0 GPU tree code
- version 2.1.0 SPH renderer
- version 2.2.0 agama_helper
- write_snapshot_coefs_to_h5
- create_fire_evolving_ini
- fit_potential
- load_agama_potential
- load_fire_pot
- read_coefs
- read_snapshot_times
- write_coef_to_h5
- write_snapshot_coefs_to_h5
- _bound_center_phi
- core-stalling suppression
- make_df_force_extra
- _shrinking_sphere_com
- sigma(r) jeans method
- sigma(r) local_circular method
- sigma(r) quasispherical method
- Kahan compensated summation
- nbody_streams Documentation Index
- ParticleReader class
- Quick start guide
- G_DEFAULT constant
- empirical_circular_velocity_profile
- empirical_velocity_dispersion_profile
- fit_plummer_profile
- Morton Z-curve sort
- render_cpu SPH kernel
- render_gpu SPH kernel
- add_perturber subhalo
- create_ic_particle_spray_chen2025
- create_particle_spray_stream
- dynamical friction progenitor orbit
- create_ic_particle_spray_fardal2015
- run_restricted_nbody
- custom stripping times
- truncated NFW perturber profile
- _GPUPotBase
- Abstract base --- subclasses implement _phi, _grad, _hess, _rho.
- Fallback saver: Stores data in a simple .npy format.          Parameters     ---
- Compute total accelerations on GPU.          Combines self-gravity (GPU) with op
- Agama external potential integration
- CPU direct N-body backend
- CPU tree FMM backend falcON
- GPU direct N-body backend
- GPU Barnes-Hut tree backend
- KDK symplectic leapfrog integrator
- make_plummer_sphere
- nbody_streams package
- ParticleReader
- PerformanceWarning
- Species
- nbody_streams.agama_helper subpackage
- nbody_streams.coords subpackage
- nbody_streams.cuda_kernels subpackage
- nbody_streams.fast_sims subpackage
- nbody_streams.fields subpackage
- nbody_streams.io subpackage
- nbody_streams.run subpackage
- nbody_streams.sim subpackage
- nbody_streams.species subpackage
- nbody_streams.tree_gpu subpackage
- nbody_streams.utils subpackage
- nbody_streams.viz subpackage
- h5py dependency
- numba dependency
- numpy dependency
- scipy dependency
- buildTree phase
- CellData data structure
- computeForces phase
- computeMultipoles phase
- ctypes interface _force.py
- libtreeGPU.so
- makeGroups phase
- Particle4 data structure
- Quadrupole tensor data structure
- run_nbody_gpu_tree
- _StepWatchdog
- tree_gravity_gpu
- TreeGPU handle
- compute_nbody_forces_cpu
- Reading API
- [2.3.0] - 2026-05-12
- Upgrade Notes
- [1.3.0] - 2026-02-24
- [2.1.0] - 2026-03-07
- [2.2.0] - 2026-03-24
- vec<3,double>
- vec<4,double>
- vec<3,float>
- [2.0.0] - 2026-02-28

## God Nodes (most connected - your core abstractions)
1. `Treecode` - 75 edges
2. `PotentialGPU()` - 74 edges
3. `MultipolePotentialGPU` - 64 edges
4. `Species` - 58 edges
5. `tree_gravity_gpu()` - 49 edges
6. `read_coefs()` - 47 edges
7. `CylSplinePotentialGPU` - 45 edges
8. `MultipoleCoefs` - 44 edges
9. `SpheroidPotentialGPU` - 44 edges
10. `CylSplineCoefs` - 42 edges

## Surprising Connections (you probably didn't know these)
- `Backend dispatch table` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png
- `run_nbody_gpu` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png
- `run_nbody_cpu` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png
- `run_nbody_gpu_tree` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png
- `Chandrasekhar dynamical friction` --references--> `Satellite A final surface density (M_sat=5e10 Msun, post-inspiral)`  [INFERRED]
  docs/dynamical_friction.md → output/df_tutorial/satA_final_density.png

## Import Cycles
- 1-file cycle: `nbody_streams/__init__.py -> nbody_streams/__init__.py`

## Communities (300 total, 114 thin omitted)

### Community 0 - "_make_ic"
Cohesion: 0.06
Nodes (44): _com(), _ke(), _make_ic(), _pe(), ndarray, Path, tests/test_physics.py ===================== Physics-level validation tests for n, KE = (1/2) sum_i m_i |v_i|^2. (+36 more)

### Community 1 - "MultipolePotentialGPU"
Cohesion: 0.07
Nodes (29): _AnalyticBase, AnalyticPotentialGPU(), _prep_xyz(), ndarray, Factory function matching Agama's constructor syntax::          pot = AnalyticPo, Abstract base for analytic GPU potentials.      Inherits ``__add__`` / ``__radd_, Arithmetic precision of the kernels, ``'float64'`` or ``'float32'``., Agama-compatible eval --- returns any combination of potential, acceleration, (+21 more)

### Community 2 - "sph_kernels.py"
Cohesion: 0.13
Nodes (21): _argsort_morton_2d(), _cubic_spline_2d_gpu(), _cubic_spline_2d_scalar(), _get_smoothing_lengths_cpu(), _get_smoothing_lengths_gpu(), ndarray, sph_kernels.py ============== SPH smoothing-length computation and 2-D surface-d, GPU smoothing-length computation via CuPy KDTree (internal). (+13 more)

### Community 3 - "_load.py"
Cohesion: 0.12
Nodes (18): get_gpu_info(), Get information about available GPU(s).          Returns     -------     info :, nbody_streams - lightweight direct N-body utilities., ndarray, nbody_streams.sim Unified high-level simulation entry point for multi-species N-, Run a direct N-body simulation with one or more particle species.      This is t, run_simulation(), _emit_performance_warnings() (+10 more)

### Community 4 - "_nfw_potential"
Cohesion: 0.15
Nodes (14): chandrasekhar_friction(), Chandrasekhar dynamical-friction acceleration at the satellite CoM.      Compute, _nfw_potential(), Constant sigma function for analytic tests., DF force must oppose the velocity., No friction when satellite is at rest., DF acceleration should scale linearly with satellite mass., Validate against a direct evaluation of BT2008 eq. 8.13. (+6 more)

### Community 5 - "test_utils.py"
Cohesion: 0.11
Nodes (21): empirical_circular_velocity_profile(), Generate uniformly random points on the surface of a sphere.      Parameters, r"""Compute the circular velocity profile :math:`v_{\rm circ}(r) = \sqrt{G\,M(<r, uniform_spherical_grid(), plummer_enclosed_mass(), plummer_v_circ(), tests/test_utils.py ===================  Tests for ``nbody_streams.utils`` using, Circular velocity should match analytic Plummer. (+13 more)

### Community 6 - "__init__.py"
Cohesion: 0.07
Nodes (50): ArrayLike, float32, ImportError, KERNEL_TYPES, _cupy.py ~~~~~~~~ Optional-CuPy shim for ``agama_helper``.  CuPy is an *optional, nbody_streams.cuda_kernels CUDA kernels for N-body force and potential computati, _compute_forces_cpu(), compute_nbody_forces_cpu() (+42 more)

### Community 7 - "`nbody_streams.viz` — Visualization"
Cohesion: 0.22
Nodes (9): `get_smoothing_lengths`, Morton Z-curve sort note, `nbody_streams.viz` — Visualization, `plot_density`, `plot_mollweide`, `plot_stream_evolution`, `plot_stream_sky`, `render_cpu` and `render_gpu` (+1 more)

### Community 8 - "convert_coords"
Cohesion: 0.17
Nodes (9): convert_coords(), Convert positions between coordinate systems.      Parameters     ----------, _random_positions(), Generate random Cartesian positions away from the origin., Arbitrary leading batch dimensions should work., With mollweide=True, phi should be in (-pi, pi]., Point on +x axis: rho=1, theta=pi/2, phi=0., Point on +x axis: R=1, phi=0, z=0. (+1 more)

### Community 9 - "create_particle_spray_stream"
Cohesion: 0.05
Nodes (44): _compute_vel_disp_from_Potential(), _create_perturber_potential(), _dynamical_friction_acceleration(), _find_prog_pot_Nparticles(), _get_prog_GalaxyModel(), _integrate_orbit_with_dynamical_friction(), Any, ndarray (+36 more)

### Community 10 - "CylSplinePotentialGPU"
Cohesion: 0.08
Nodes (27): _clamped_left_cubic_deriv_batch(), CylSplinePotentialGPU, _determine_asympt_cylspline(), _natural_cubic_deriv_batch(), _prep_xyz(), ndarray, Vectorized form of ``_sph_harm_agama``: processes all points simultaneously., Fit outer PowerLaw multipole coefficients matching Agama's determineAsympt(). (+19 more)

### Community 11 - "ShiftedPotentialGPU"
Cohesion: 0.10
Nodes (13): Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Shifted modifier: evaluates ``inner`` at ``xyz - center(t)``.      Parameters, Scaled modifier: ``Phi_s(x,t) = a(t)*s(t) * Phi(x*s(t))`` where ``s(t) = 1/scale, Return (s, a) = (1/scale(t), ampl(t)) with linear extrapolation., Rotate symmetric tensors stored as (..., 6) [xx, yy, zz, xy, yz, zx]: R D R^T., Common machinery for ``Phi(x, t) = Phi_inner(R(t)^T x, t)``.      Nested rotatio, Total rotation R(t) (3, 3): ``Phi(x, t) = Phi_base(R(t)^T x, t)``. (+5 more)

### Community 12 - "run_simulation"
Cohesion: 0.06
Nodes (31): Backend dispatch table, Backend-specific kwargs, Contents, Definition, Dynamical friction kwargs, Example, Example: GPU tree backend, Example: multi-species dark matter + stars (+23 more)

### Community 13 - "generate_stream_coords"
Cohesion: 0.31
Nodes (5): generate_stream_coords(), Convert galactocentric phase space into stream-aligned coordinates     (phi1, ph, Circular orbit particles spread along a stream-like arc., None and [] should give identical results., TestGenerateStreamCoords

### Community 14 - "_AnalyticBase"
Cohesion: 0.17
Nodes (15): CompletedProcess, _check(), test_no_cupy.py ~~~~~~~~~~~~~~~ Regression tests for the optional-CuPy contract, Coefficient dataclasses, text and HDF5 round-trips are pure NumPy., Every GPU entry point fails loudly with an actionable message., With the real environment, ``CUPY_AVAILABLE`` tracks the import., Execute *body* in a fresh interpreter where ``import cupy`` fails., ``import nbody_streams.agama_helper`` must not require CuPy. (+7 more)

### Community 15 - "tree_gravity_gpu"
Cohesion: 0.11
Nodes (29): _apply_modifiers(), euler_matrix(), Agama's ``orientation=[alpha, beta, gamma]`` rotation matrix.      ``R = Rz(alph, Static rotation: Agama's ``orientation=`` modifier.      ``Phi(x) = Phi_inner(R^, Rotation about the z-axis: Agama's ``rotation=`` modifier.      ``Phi(x, t) = Ph, Rotation angle [rad] at time *t*., Wrap *pot* with Scaled/Rotating/Tilted/Shifted modifiers if requested.      Orde, _rot_z() (+21 more)

### Community 17 - "__init__.py"
Cohesion: 0.21
Nodes (12): create_snapshot_dict(), fit_potential(), ndarray, Path, agama_helper._fit ~~~~~~~~~~~~~~~~~ Fit Agama Multipole and CylSpline potential, Fit Agama Multipole and CylSpline potentials from a multi-species snapshot., Sample *n* positions from a spherically declining density profile., Sample *n* positions from a thin exponential disk. (+4 more)

### Community 18 - "test_phase2_analytic.py"
Cohesion: 0.08
Nodes (31): Triaxial double-power-law potential with time-dependent shape.      Same density, Axis ratios ``(p, q)`` at time *t*., Density normalisation at time *t* [Msol/kpc^3]., Node table for the shape at *t*, and the amplitude 4 pi G rho0 p q         (Hess, SpheroidPotentialGPU, test_spheroid_shape.py ~~~~~~~~~~~~~~~~~~~~~~ Tests for ``SpheroidPotentialGPU``, Finite-mass profile: ``mass=`` sets rho0 exactly as Agama does., rho0 * p * q is constant, so M(<m) does not change with the shape. (+23 more)

### Community 19 - "test_advanced.py"
Cohesion: 0.28
Nodes (8): Checks which high-performance features are actually active., Runs a tiny GPU simulation and verifies snapshot output., Quick helper to generate test particles (Plummer-like spread)., Runs a tiny CPU simulation and verifies snapshot output., setup_dummy_data(), test_cpu_integration(), test_environment_summary(), test_gpu_integration()

### Community 21 - "CellData"
Cohesion: 0.12
Nodes (28): dim3, buildOctant(), buildOctantBatch(), buildOctreeHost(), collect_leaves(), compute_level_begIdx(), computeBoundingBox(), computeGridAndBlockSize() (+20 more)

### Community 22 - "test_newtons_third_law.py"
Cohesion: 0.14
Nodes (26): Subdivide every interval of *times* by *factor*, keeping the original nodes., Resample a coefficient time series onto a new time grid with cubic splines., refine_times(), spline_resample_coefs(), Tests for cubic-spline resampling of a coefficient time series.  ``spline_resamp, A #Phi-only CylSpline (like the FIRE archives) must not gain fake derivatives., Each series here is base * SCALES[k], so the spline of a column tracks it., Pins why this helper requires agama instead of falling back to scipy.      ``aga (+18 more)

### Community 23 - "_build.py"
Cohesion: 0.67
Nodes (3): main(), nbody_streams.tree_gpu._build Console-script entry point: nbody-build-tree  Comp, _tree_gpu_dir()

### Community 24 - ".radial_power"
Cohesion: 0.03
Nodes (43): Stack time-less coefficient snapshots into one object with a time axis.      Gri, Read an Agama expansion into a structured dataclass — one snapshot or many., read_coefs(), stack_coefs(), cylsp3_snaps(), cylsp_snaps(), mult_series(), mult_snaps() (+35 more)

### Community 26 - "`nbody_streams.utils` — Analysis Utilities"
Cohesion: 0.07
Nodes (30): Centre finding, `comoving_to_physical` / `physical_to_comoving`, `compute_iterative_boundness` (deprecated), Cosmology and the comoving frame, Density-profile fitting, `empirical_circular_velocity_profile`, `empirical_density_profile`, Empirical radial profiles (+22 more)

### Community 28 - "`nbody_streams.coords` — Coordinate Transforms"
Cohesion: 0.25
Nodes (8): `convert_coords`, `convert_to_vel_los`, `convert_vectors`, Coordinate conventions, `generate_stream_coords`, `get_observed_stream_coords`, `nbody_streams.coords` — Coordinate Transforms, `to_stream_coords`

### Community 31 - "Treecode.h"
Cohesion: 0.11
Nodes (21): double4, Treecode<real_t>::computeForces(), addMonopole(), addQuadrupole(), __device__, double4, float3, float4 (+13 more)

### Community 32 - "Treecode"
Cohesion: 0.03
Nodes (60): double4, float4, Treecode, build_max_work, BUILD_PENDING_WORK_BYTES, buildTree, cell_max, computeForces (+52 more)

### Community 35 - ".dark"
Cohesion: 0.11
Nodes (13): Convenience constructor for dark-matter particles., _plummer_like(), run_simulation emits PerformanceWarning for large CPU direct N., Very small Plummer-like ICs for integration tests., Without species kwarg, old dark/star HDF5 schema is written., Verify that debug_energy=True works without errors on all dispatcher paths     t, debug_energy=False (default) must not print energy lines., run_simulation must raise TypeError on unknown kwargs for CPU path. (+5 more)

### Community 36 - "Species"
Cohesion: 0.23
Nodes (11): _mock_stream(), Tests for nbody_streams.coords — coordinate and vector transforms., A progenitor plus a thin stream spread along its orbit, deliberately tilted., The progenitor must land exactly at (phi1, phi2) = (0, 0).      It *defines* the, R must stay a proper rotation: R^T R = I and det R = +1., The roll must earn its cost: it should flatten the stream, not just pin it., Batched input: every stream's progenitor pinned, not just the first., test_optimizer_reduces_phi2_objective() (+3 more)

### Community 37 - "_potential.py"
Cohesion: 0.15
Nodes (18): empirical_velocity_anisotropy_profile(), fit_dehnen_profile(), fit_plummer_profile(), r"""Compute the velocity anisotropy parameter :math:`\beta(r)`.      .. math::, r"""Fit a triaxial Dehnen profile by mapping to ellipsoidal radius.      Compute, r"""Fit a spherical Plummer profile to particle data.      Bins particles into l, ndarray, nbody_streams.utils._validation ================================  Shared input-v (+10 more)

### Community 38 - "nbody_io.py"
Cohesion: 0.29
Nodes (17): check_nan_inf(), direct_sum_max_eps(), generate_plummer(), hdr(), Return (result, elapsed_ms) with a surrounding device sync., Small-N O(N^2) direct-sum with tree_gpu max softening convention:       eps2_ij, rel_err_stats(), subhdr() (+9 more)

### Community 40 - "`nbody_streams.agama_helper`"
Cohesion: 0.14
Nodes (14): Center parameter, Contents, `create_snapshot_dict(pos_dark, mass_dark, pos_star=None, mass_star=None, pos_gas=None, mass_gas=None, temperature_gas=None) -> dict`, `fit_potential(snap, nsnap, *, sym="n", lmax=4, rmax_sel=300.0, rmax_exp=None, file_ext="", save_dir="potential/", halo=None, kind="both", center=None, rotation=None, verbose=True, subsample_factor=1, cold_temp_log10_thresh=4.5) -> dict[str, list[str]]`, Full example: MW potential pipeline, Gotchas, HDF5 I/O, `nbody_streams.agama_helper` (+6 more)

### Community 41 - "Changelog"
Cohesion: 0.20
Nodes (9): [1.0.0] - 2025-XX-XX, [1.2.0] - 2026-02-14, Added, Added, Changelog, Features, Future Roadmap, Planned (+1 more)

### Community 42 - "test_multi_species.py"
Cohesion: 0.07
Nodes (46): center_acceleration(), center_acceleration_table(), ndarray, Cubic-spline every coefficient series along the trailing time axis., Read the (3, 3) rotation into the present-day principal-axis frame.      Paramet, Load the three comoving centre-of-mass splines.      Parameters     ----------, Write the three centre splines as ``(knots, coefficients)`` text.      Reload wi, Tabulate ``-u_dot``, the galactic-centre correction, on a uniform time grid. (+38 more)

### Community 43 - "Box"
Cohesion: 0.07
Nodes (23): _as_times(), _check_presence(), _check_stackable(), _describe_grid_mismatch(), ndarray, Time-less view at time index *i*.          The returned object shares memory wit, Attach or relabel the time axis.          Nothing is interpolated, smoothed or r, Assert internal shape consistency, naming the offending field on failure. (+15 more)

### Community 44 - "_force.py"
Cohesion: 0.13
Nodes (17): CDLL, _coerce_eps(), _find_cudart(), _gpu_ptr(), ndarray, _force.py  --  ctypes interface to the GPU tree-code gravity shared library.  Su, Return a contiguous float32 CuPy array of length n for eps.      Accepts:, Load libcudart; try versioned names as fallback. (+9 more)

### Community 45 - "buildTree.cu"
Cohesion: 0.08
Nodes (20): MultipoleCoefs, Parse **one** Agama Multipole snapshot into a :class:`MultipoleCoefs`.      Inte, Dispatch one coef string to the parser named by its ``type=`` header., Read one or many Agama Multipole snapshots into a :class:`MultipoleCoefs`., Structured representation of a Multipole (spherical harmonic BFE) potential., Maximum l order present in *lm_labels*., Sorted unique l values., Sorted unique m values (includes negatives). (+12 more)

### Community 46 - "NFWPotentialGPU"
Cohesion: 0.05
Nodes (49): Raise a descriptive :class:`ImportError` if CuPy is unavailable.      Parameters, require_cupy(), _build_dehnen_gpu(), _build_king_gpu(), _build_single(), _build_spheroid_direct(), _build_spheroid_gpu(), _coerce() (+41 more)

### Community 47 - "_random_xv"
Cohesion: 0.31
Nodes (7): nbody_streams.coords - coordinate transformations.  Cartesian/spherical/cylindri, get_observed_stream_coords(), ndarray, Stream coordinate generation and observable conversions.  Generate stream-aligne, Project galactocentric positions or phase-space vectors into a     pre-computed, Convert galactocentric phase-space coordinates to observed sky     coordinates (, to_stream_coords()

### Community 48 - "API"
Cohesion: 0.07
Nodes (28): AGAMA potential helper (`agama_helper`), Analysis utilities (`utils`), API, Caveats and known limitations, Coordinates (`coords`), Direct vs tree accuracy, Dynamical friction with external potentials, Fast stream generation (`fast_sims`) (+20 more)

### Community 49 - "r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I"
Cohesion: 0.50
Nodes (3): double2, vec<2,double>, __device__

### Community 50 - "Particle4"
Cohesion: 0.07
Nodes (38): Box, centre, __device__, hsize, __forceinline__ Box<T> ChildBox(), __forceinline__ int Particle4<double>::get_idx(), __forceinline__ int Particle4<double>::get_oct(), __forceinline__ int Particle4<double>::set_idx() (+30 more)

### Community 51 - "CompositePotentialGPU"
Cohesion: 0.13
Nodes (22): generate_plummer(), Net force = sum(m_i * a_i) should be ~0 for an isolated system., Net torque = sum(r x (m*a)) should be small., Compare 0.5*sum(m*phi_tree) against 0.5*sum(m*phi_direct)., For a spherically symmetric distribution, forces should point inward., Error should not degrade significantly as N grows (same distribution)., Plummer model ICs matching the C++ binary (double precision + R<100 rejection)., Calling tree_gravity_gpu twice with identical input should give identical output (+14 more)

### Community 52 - "treewalk_warp"
Cohesion: 0.16
Nodes (21): approxAcc(), __device__, float2, float3, float4, __forceinline__, __global__, int2 (+13 more)

### Community 53 - "treeGPU_interface.cu"
Cohesion: 0.15
Nodes (25): float4, __global__, real_t, k_extract_unsorted(), k_load_pos_mass(), k_load_pos_mass_eps(), Res, w (+17 more)

### Community 54 - "Ellipsoidal radius and particle selection (numba-accelerated).      Computes the"
Cohesion: 0.06
Nodes (23): DehnenSphericalPotentialGPU, HernquistPotentialGPU, IsochronePotentialGPU, PlummerPotentialGPU, _precision_dtype(), Plummer sphere: Phi(r) = -GM / sqrt(r^2 + b^2)      Constructor: PlummerPotentia, Hernquist: Phi(r) = -GM / (r + a)      Constructor: HernquistPotentialGPU(mass=1, Dehnen spherical: Phi(r) = -(GM/a) * (1 - (r/(r+a))^(2-gamma)) / (2-gamma)     S (+15 more)

### Community 55 - "Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti"
Cohesion: 0.08
Nodes (27): _is_uniform(), _load_restart(), _make_times_ns(), ParticleReader, ndarray, Path, nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr, Append or update snapshot.times in output_dir.     Ensures unique snap_index ent (+19 more)

### Community 56 - "Generate uniformly random points on the surface of a sphere.      Parameters"
Cohesion: 0.08
Nodes (27): _as_existing_path(), _check_unique_names(), _CoefTimeAxisMixin, _is_source_sequence(), Path, Return lines from any coef source: file path, HDF5 path, or raw string.      Del, Split a CylSpline line stream into ``#Phi`` / ``#dPhi/dR`` / ``#dPhi/dz`` blocks, Expand any accepted *source* into raw coef strings plus resolved times.      Ret (+19 more)

### Community 57 - "Generate num_pts points on a sphere using the Fibonacci spiral.      Points are"
Cohesion: 0.07
Nodes (39): _aH(), comoving_to_physical(), FlatLCDM, physical_to_comoving(), Path, nbody_streams.utils._cosmology ===============================  Flat LCDM backgr, Degenerate background with ``a = 1``, ``H = 0``, for a non-cosmological, Least-squares fit of ``(hubble, omega_matter)`` to a tabulated a(t).          Pa (+31 more)

### Community 58 - "Implementation internals"
Cohesion: 0.08
Nodes (25): Availability flag, Build system details, Building the shared library, ctypes interface (`_force.py`), `cuda_alive`, Data structures, `G_DEFAULT`, GPU memory layout (+17 more)

### Community 59 - "Find the density peak by locating the minimum gravitational potential.      Uses"
Cohesion: 0.06
Nodes (41): DiskAnsatzPotentialGPU, LogHaloPotentialGPU, _make_kernels(), MiyamotoNagaiPotentialGPU, _analytic_potentials.py ~~~~~~~~~~~~~~~~~~~~~~~ Phase 2: GPU-accelerated analyti, Build the (phi, force, deriv, rho) kernels of one model.  *params*     unpacks `, Miyamoto-Nagai disk: Phi = -GM / sqrt(R^2 + (sqrt(z^2+b^2)+a)^2)     where R^2 =, LogHalo: Phi = (v0^2/2) * ln(rc^2 + x^2 + y^2/p^2 + z^2/q^2)      Constructor ma (+33 more)

### Community 60 - "computeMultipoles.cu"
Cohesion: 0.16
Nodes (12): level, Treecode<real_t>::buildTree(), __global__, __out, leafMultipoles(), nodeMultipoles(), cellGroups(), __global__ (+4 more)

### Community 61 - "TreeGPU"
Cohesion: 0.08
Nodes (32): CylSplineCoefs, Return a copy with all m terms **not** in *keep_m* zeroed out.          Paramete, Return a deep copy — no array, list or dict is shared with *self*., Read one or many Agama CylSpline snapshots into a :class:`CylSplineCoefs`., Structured representation of a CylSpline (azimuthal harmonic BFE) potential., read_cylspl_coefs(), Regression tests for section-aware CylSpline coefficient parsing.  An Agama ``Cy, ``to_coef_string`` output for a Phi-only file must not have changed. (+24 more)

### Community 62 - "test_chandrasekhar.py"
Cohesion: 0.24
Nodes (7): Iterative shrinking-sphere centre-of-mass estimator.      At each iteration the, _shrinking_sphere_com(), CoM should converge close to the true cloud centre., All particles at same position → CoM = that position exactly., With one dominant dense clump + sparse background, CoM should         land in th, r_sphere must be positive and smaller than the full cloud extent., TestShrinkingSphereCom

### Community 63 - "vec<3,double>"
Cohesion: 0.35
Nodes (10): _pts(), test_phase3_cylspline.py ~~~~~~~~~~~~~~~~~~~~~~~~ Tests CylSplinePotentialGPU ag, Random Cartesian points inside a sphere, avoiding z-axis., _rel_deriv(), _rel_force(), _rel_phi(), test_accuracy(), test_mmax_sweep() (+2 more)

### Community 64 - "vec<2,float>"
Cohesion: 0.13
Nodes (8): _build_particle_arrays(), Definition of a particle species in an N-body simulation.      Parameters     --, Convenience constructor for stellar particles., Build combined per-particle mass and softening arrays from a species list., Species, TestBuildParticleArrays, TestParticleReaderMultiSpecies, TestSpeciesDataclass

### Community 65 - "_find_cudart"
Cohesion: 0.33
Nodes (8): error_report(), generate_plummer(), Run with a small N and check that forces point inward., Plummer model ICs matching the C++ binary (double precision + R<100 rejection)., Print relative-error statistics between ref and test arrays., test_accuracy(), test_performance(), test_sanity()

### Community 66 - ".alloc"
Cohesion: 0.18
Nodes (11): Coefficient time axis, `column(l, m) -> int`  (Multipole only), `copy() -> MultipoleCoefs \| CylSplineCoefs`, Introspection properties, Methods (both classes unless noted), Module-level: `stack_coefs(items, times) -> MultipoleCoefs | CylSplineCoefs`, `radial_power` / `total_power` with a time axis, `snapshot(i) -> MultipoleCoefs \| CylSplineCoefs`  (alias: `obj[i]`) (+3 more)

### Community 67 - "make_df_force_extra"
Cohesion: 0.14
Nodes (11): _is_gpu_potential(), make_df_force_extra(), Build a ``force_extra`` closure that applies Chandrasekhar dynamical     frictio, True when *pot* is from the PotentialGPU family (nbody_streams.agama_helper)., DF is a rigid-body force — all particles within the core get the same         ac, Net DF acceleration should point opposite to bulk CoM velocity., Between correction steps, CoM should be predicted, not recalculated., Without Agama the factory should raise ImportError. (+3 more)

### Community 68 - ".test_circular_orbit_decays"
Cohesion: 0.25
Nodes (8): empirical_velocity_dispersion_profile(), make_uneven_grid(), Create a 1-D grid with unequally spaced nodes.      The grid starts at 0, the se, r"""Compute the velocity dispersion :math:`\sigma_v(r)`.      Parameters     ---, Velocity dispersion should be finite and positive at all radii., Grid starts at 0, second node at xmin, last at xmax., test_empirical_velocity_dispersion(), test_make_uneven_grid()

### Community 69 - "computeMultipoles.cu"
Cohesion: 0.33
Nodes (6): nbody_streams.utils - analysis and diagnostic utilities., compute_iterative_boundness(), iterative_unbinding(), Iterative unbinding to determine bound particles.      Computes the gravitationa, A virialised Plummer sphere should be >90% bound., test_iterative_unbinding()

### Community 71 - "TreeGPU"
Cohesion: 0.33
Nodes (6): double_power_law_density(), fit_double_spheroid_profile(), r"""Construct a Zhao (1996) double-power-law density profile normalised to total, r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I, Fitted profile should recover the Plummer mass to ~30%., test_fit_double_spheroid_profile()

### Community 72 - "transforms.py"
Cohesion: 0.17
Nodes (22): _as_3vec(), _cart_to_cyl(), _cart_to_sph(), _cyl_to_cart(), _cyl_to_sph(), _propagate_nans(), ndarray, Coordinate system and vector field conversions.  Cartesian <-> spherical <-> cyl (+14 more)

### Community 73 - "_jeans_sigma_r"
Cohesion: 0.09
Nodes (25): compute_sigma_r(), _jeans_sigma_r(), ndarray, nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica, Local circular-speed approximation to the 1-D velocity dispersion.      Uses the, Compute a radial velocity-dispersion profile from an Agama potential.      Param, Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when     possibl, Jeans-equation estimate of the isotropic 1-D velocity dispersion.      Uses the (+17 more)

### Community 74 - "_spheroid.py"
Cohesion: 0.19
Nodes (21): _as_agama_scale(), _pts(), test_time_modifiers.py ~~~~~~~~~~~~~~~~~~~~~~ Numerical tests for the time-depen, Guard: the fixture is coarse enough that not-a-knot is measurably wrong.      If, Our (T,2) [t, scale] -> Agama's (T,3) [t, ampl=1, scale]., A coarsely sampled LMC-like infall --- the regime where the spline choice bites., The interpolated center itself, vs agama.Spline(reg=True) per component., _rel() (+13 more)

### Community 75 - "main.py"
Cohesion: 0.15
Nodes (20): _calculate_particle_distances_sq(), _calculate_Rsphall_and_extract(), _compute_weighted_structure_tensor(), _density_peak_center(), fibonacci_sphere_grid(), find_center(), find_center_position(), fit_iterative_ellipsoid() (+12 more)

### Community 76 - "vec<4,float>"
Cohesion: 0.22
Nodes (9): Coefficient dataclasses, `CylSplineCoefs`, `MultipoleCoefs`, `radial_power(l, use_quadrature=True) -> ndarray`, `to_coef_string() -> str`, `to_coef_string() -> str`, `total_power(l, use_quadrature=True) -> float`, `zeroed(keep_lm) -> MultipoleCoefs` (+1 more)

### Community 77 - "plot_density"
Cohesion: 0.15
Nodes (8): Axes, AxesImage, Colormap, plot_density(), Any, Generate a projected density image (imshow) from particle data.      Accepts par, imshow extent should be [-gridsize/2, gridsize/2] on both axes., TestPlotDensity

### Community 78 - "test_recentre.py"
Cohesion: 0.16
Nodes (19): estimate_centre(), _gpu_centre(), _gpu_sub_cast(), _gpu_sub_cast_soa(), Float64 -> float32 position conversion relative to a nearby centre.  float32 kee, Single-launch subsample centre for a C-contiguous (N, 3) float64 CuPy array., Mass-weighted mean of a strided subsample of ``pos`` (float64, shape (3,))., Return float32 positions relative to a centre, and that centre.      Parameters (+11 more)

### Community 79 - "compute_nbody_potential_gpu"
Cohesion: 0.50
Nodes (4): empirical_velocity_rms_profile(), r"""Compute the root-mean-square velocity profile :math:`v_{\rm rms}(r)`.      P, RMS velocity should be finite and positive., test_empirical_velocity_rms()

### Community 80 - "Writers and materialization"
Cohesion: 0.29
Nodes (7): `materialize_potential(*, center=None, interp_linear=True, gpu=False)`, `to_coef_files(out_dir, name_fmt="snap_{i:04d}{ext}") -> list[str]`, `to_coef_string(t=None) -> str`, `to_coef_strings() -> list[str]`, `to_evolving_ini(ini_path, out_dir=None, interp_linear=True) -> str`, `to_h5(path, group_fmt="snap_{i:04d}", dataset_name="coefs", overwrite=True, write_times=True) -> str`, Writers and materialization

### Community 81 - "run_nbody_gpu"
Cohesion: 0.18
Nodes (11): Constants, Contents, Example: direct method with specific kernel, Example: tree method, make_plummer_sphere, Method selection guidance, `nbody_streams.run`, Parameters (+3 more)

### Community 83 - "GPU potential evaluation (PotentialGPU)"
Cohesion: 0.18
Nodes (11): Accuracy (vs Agama CPU), API (matches `agama.Potential`), CuPy is optional, Factory: `PotentialGPU(...)`, GPU potential evaluation (PotentialGPU), Requirements, Rotation modifiers: `orientation=` and `rotation=`, Single precision: `precision=` (+3 more)

### Community 84 - "FIRE helpers"
Cohesion: 0.33
Nodes (6): `create_fire_evolving_ini(sim_dir, model_pattern, output_filename, snap_range=None, verbose=True) -> str`, FIRE helpers, `load_fire_pot(sim_dir, nsnap, sym="n", lmax=4, kind="whole", keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, file_ext="DR", out_acc=False, halo=None, verbose=True, return_coefs=False, save_modified=False, save_dir=None)`, `read_snapshot_times(sim_dir, sep=r'\s+') -> pandas.DataFrame`, `refine_times(times, factor=10) -> ndarray`, `spline_resample_coefs(coefs, times_new) -> MultipoleCoefs | CylSplineCoefs`

### Community 85 - "Dynamical friction"
Cohesion: 0.09
Nodes (23): Caveats, Centre-of-mass detection, Chandrasekhar formula (BT2008 eq. 8.13), Contents, Core-stalling suppression, Coulomb logarithm, df_* kwargs reference, Dynamical friction (+15 more)

### Community 86 - "cuda_primitives.h"
Cohesion: 0.19
Nodes (21): buildOctantBody(), __device__, float2, minmax_block(), addBoxSize(), atomicAdd_double(), __forceinline__(), __device__ (+13 more)

### Community 87 - "Quick-start workflow"
Cohesion: 0.40
Nodes (5): 1 — Fit and save coefficient files, 2 — Pack text files into HDF5, 3 — Inspect and modify coefficients, 4 — Load Agama potential, Quick-start workflow

### Community 88 - "_random_xv"
Cohesion: 0.16
Nodes (11): Write a snapshot compatible with :class:`ParticleReader`.      Two modes     ---, _save_snapshot(), NDArray, Validate *species* list against *phase_space* shape.      Checks     ------, _validate_species(), ndarray, _random_xv(), Old file with only dark particles (no star group). (+3 more)

### Community 89 - "_io.py"
Cohesion: 0.07
Nodes (55): _add_negative_m(), _detect_expansion_type(), generate_lmax_pairs(), agama_helper._coefs ~~~~~~~~~~~~~~~~~~~ Structured representations of Agama expa, Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown., Generate (l, m) pairs for a spherical harmonic expansion.      Parameters     --, Expand (l, m) pairs to include their negative-m counterparts.      Each (l, m) w, create_fire_evolving_ini() (+47 more)

### Community 90 - "TestBoundCenterPhi"
Cohesion: 0.16
Nodes (10): _bound_center_phi(), Phi-energy iterative bound-particle centre.      Finds the centre of the gravita, Return a cluster at r0 on the x-axis with a bulk y-velocity., Simple pairwise Plummer phi (exact for small N test)., Returned CoM should be close to the cluster centre., For a self-gravitating cluster with low velocity dispersion, bound         parti, make_df_force_extra closure should accept and use phi kwarg., Unbound particles (phi + 0.5 v_rel^2 >= 0) should get zero DF. (+2 more)

### Community 91 - "UniformAccelerationGPU"
Cohesion: 0.07
Nodes (30): NFWPotentialGPU, Navarro-Frenk-White potential optimized with CuPy ElementwiseKernels.      Const, PotentialGPU(), GPU potential factory : mirrors the ``agama.Potential`` API.      Usage     ----, _nfw_exact(), test_analytic_precision.py ~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests for the ``precision=, |force| and the radial Hessian d2Phi/dr2 of the NFW potential., Types Agama builds as a Multipole run its mixed-precision kernel. (+22 more)

### Community 92 - "AGAMA_GPU"
Cohesion: 0.10
Nodes (20): Accuracy, AGAMA_GPU, Analytic — `_analytic_potentials.py`, API, Coefficient objects — one snapshot or a whole time series, Composite types, CylSpl BFE - `CylSplinePotentialGPU`, File layout (+12 more)

### Community 93 - "Quadrupole"
Cohesion: 0.20
Nodes (7): real_t, Quadrupole, __device__, q0, q1, real2_t, real4_t

### Community 94 - "vector3"
Cohesion: 0.24
Nodes (8): maxeach(), mineach(), operator *(), vector3, x, y, z, REAL

### Community 95 - "Loading Agama potentials"
Cohesion: 0.50
Nodes (4): `create_evolving_ini(times, coef_paths, output_path, interp_linear=True) -> str`, `load_agama_evolving_potential(source, times=None, *, group_names=None, dataset_name="coefs", center=None, interp_linear=True, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, gpu=False)`, `load_agama_potential(source, group_name="snap_000", dataset_name="coefs", center=None, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, gpu=False)`, Loading Agama potentials

### Community 96 - "test_phase1_multipole.py"
Cohesion: 0.29
Nodes (14): _make_from_agama(), _make_from_coef_file(), _pts(), test_phase1_multipole.py ~~~~~~~~~~~~~~~~~~~~~~~~ Tests MultipolePotentialGPU ag, Load coef file, cap at lmax, return (agama_pot, gpu_pot)., Export agama_pot as Multipole BFE, load into GPU., _rel_force(), _rel_phi() (+6 more)

### Community 97 - "Physics and formulae"
Cohesion: 0.09
Nodes (22): Run GPU-accelerated N-body simulation with leapfrog (KDK) integration., run_nbody_gpu(), cuda_alive(), Return True if the CUDA context has no pending errors.      Uses cudaGetLastErro, nbody_streams.tree_gpu ====================== GPU Barnes-Hut tree code for N-bod, ndarray, nbody_streams.tree_gpu.run_gpu_tree ==================================== KDK lea, Background-thread watchdog for GPU leapfrog loops.      Per-step overhead is ~1 (+14 more)

### Community 99 - "plots.py"
Cohesion: 0.13
Nodes (17): Figure, nbody_streams.viz - visualization and plotting.  Projected density maps, Mollwei, _aggregate_data_chunk(), _extract_particles_at_step(), _gauss_filter_surf_dens(), _generate_ticks(), plot_mollweide(), plot_stream_sky() (+9 more)

### Community 100 - "Quick start"
Cohesion: 0.15
Nodes (13): Choosing a backend, Core package, CPU fallback (no GPU required), GPU support (CuPy), GPU tree code (one-time build), Hello world: Plummer sphere self-gravity, Installation, Next steps (+5 more)

### Community 101 - "convert_to_vel_los"
Cohesion: 0.27
Nodes (5): convert_to_vel_los(), Compute line-of-sight (radial) velocity from Galactocentric phase-space     coor, Circular orbit at solar position: v_los should be 0., Purely radial velocity along +x: v_los = vx., TestConvertToVelLos

### Community 102 - "compute_nbody_forces_gpu"
Cohesion: 0.22
Nodes (16): Compute gravitational accelerations and potential using the GPU tree code     (B, tree_gravity_gpu(), Time the tree_gravity_gpu() Python helper function., time_helper_api(), _hernquist(), test_guards.py -- Regression tests for GPU-tree input guards and conventions., Cluster (M=1e5 Msun, a=1 pc) at 40 kpc moving at 200 km/s evolves like at rest., More than nleaf particles at one point: the depth cap turns them into a     sing (+8 more)

### Community 103 - "convert_vectors"
Cohesion: 0.24
Nodes (6): convert_vectors(), Rotate vector fields between coordinate systems.      Converts both the position, sph->cyl->sph chains through cart; verify it still round-trips., Rotation must preserve vector magnitudes., A purely radial velocity on +x axis should map to v_rho only., TestConvertVectors

### Community 104 - "render_surface_density"
Cohesion: 0.26
Nodes (5): High-level SPH surface-density renderer with GPU/CPU dispatch.      Computes smo, render_surface_density(), Total mass in grid should approximately match input mass., Morton sort should produce a valid density grid.          Grids are NOT expected, TestRenderSurfaceDensity

### Community 105 - "_cylspl_potential_kernel.cu"
Cohesion: 0.36
Nodes (7): binSearch_device(), __device__, cylspl_eval_device(), eval_outer_asympt(), evalCubic1D(), evalCubic4_in_z(), sphHarm_device()

### Community 106 - "AGAMA_GPU — Development Notes"
Cohesion: 0.18
Nodes (10): AGAMA_GPU — Development Notes, Architecture decisions, File layout, INI loading (multi-section, case-insensitive), Kernel design, Non-uniform grid resampling, Numerical precision, Outer extrapolation (+2 more)

### Community 107 - "test_phase3_cylspline.py"
Cohesion: 0.12
Nodes (33): Build the acceleration spline from Agama's ``file=`` argument.      Accepts what, _read_accel_table(), _acc_table(), _acc_table_jumpy(), _pts(), test_uniform_acceleration.py ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests for the time-dep, Guard: SciPy's default spline is *not* what Agama uses.      If this ever stops, The whole point: the force must not depend on position. (+25 more)

### Community 108 - "cuda_mem"
Cohesion: 0.14
Nodes (7): cuda_mem, n, ptr, T, host_mem, n, ptr

### Community 109 - "plot_stream_evolution"
Cohesion: 0.31
Nodes (3): plot_stream_evolution(), Three-panel evolution plot: galactocentric distance, bound fraction     (or 3-D, TestPlotStreamEvolution

### Community 110 - "._state"
Cohesion: 0.50
Nodes (4): _newtons_third_law(), Test that net force is ~0 for isolated system.          Returns:         bool: T, Test all precision modes., test_all_precisions()

### Community 111 - "get_smoothing_lengths"
Cohesion: 0.15
Nodes (10): get_smoothing_lengths(), Compute SPH smoothing lengths as the distance to the *k_neighbors*-th     neares, _close_figures(), orbit_data(), Tests for nbody_streams.viz -- visualization functions.  All tests use the Agg b, Close all matplotlib figures after each test., Particles in a dense cluster should get smaller h than isolated ones., 3-D pos should be accepted (D is not restricted to 2). (+2 more)

### Community 112 - "EvolvingPotentialGPU"
Cohesion: 0.17
Nodes (5): _error(), _MissingCuPy, _MissingKernel, Stand-in for a CuPy kernel constructed while CuPy is unavailable., Stub standing in for the ``cupy`` module when it is not installed.

### Community 113 - "Public API"
Cohesion: 0.25
Nodes (8): `create_ic_particle_spray_chen2025`, `create_ic_particle_spray_fardal2015`, `create_particle_spray_stream`, Dynamical friction, `nbody_streams.fast_sims` — Fast Stream Generation, Perturber potential, Public API, `run_restricted_nbody`

### Community 114 - "vec<3,float>"
Cohesion: 0.22
Nodes (9): `center_acceleration(center_splines, rotation, cosmo, t_range=None, n_samples=1201, gpu=False)`, `center_acceleration_table(...) -> ndarray (n_samples, 4)`, Comoving host frame, End-to-end, Gotchas: units and end knots, `read_center_splines(file_pattern) -> list[BSpline]`, `read_rotation(sim_dir, nsnap=600, spl=True, subdir="potential/10kpc") -> ndarray`, `write_center_acceleration(path, center_splines, rotation, cosmo, t_range=None, n_samples=1201, ini=True) -> str` (+1 more)

### Community 115 - "_multipole_potential_kernel.cu"
Cohesion: 0.26
Nodes (12): F, MULT_EVAL_PARAMS, __device__, multipole_density_kernel(), multipole_eval_device(), multipole_force_kernel(), multipole_force_kernel_f32(), multipole_hess_kernel() (+4 more)

### Community 116 - "run_nbody_gpu_tree"
Cohesion: 0.18
Nodes (14): Load from an Agama .coef_mul_DR file or HDF5 archive., _dirs(), pots(), test_multipole_precision.py ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests for ``MultipolePot, dF over 10 pc at 40 kpc: ~1e-7 f_ns R/d, below the pure-float32 4e-4., _rel(), test_force_on_axis(), test_hessian_and_density_stay_float64() (+6 more)

### Community 117 - "TreeGPU"
Cohesion: 0.24
Nodes (4): _max_n(), Pre-allocated GPU tree handle for time-stepping loops.      Parameters     -----, Barnes-Hut opening angle used by the next force evaluation., TreeGPU

### Community 118 - "GroupData"
Cohesion: 0.17
Nodes (4): CompositePotentialGPU, _GPUPotBase, Sum of GPU potential components : mirrors agama.Potential composite.      Each c, Mixin giving every GPU potential class ``+`` composition and ``sum()`` support.

### Community 119 - "empirical_density_profile"
Cohesion: 0.33
Nodes (6): empirical_density_profile(), r"""Compute the mass-density radial profile :math:`\rho(r)`.      Parameters, plummer_density(), rho(r) = (3M / 4*pi*a^3) (1 + r^2/a^2)^{-5/2}, Density profile should match analytic Plummer within ~20% at intermediate radii., test_empirical_density_profile()

### Community 120 - "Potential"
Cohesion: 0.33
Nodes (4): Potential, When quasispherical DF fails, _common should use Jeans, not MW table., The fallback must NOT use the old hardcoded MW sigma array., TestCommonJeansFallback

### Community 121 - "Chandrasekhar dynamical friction"
Cohesion: 0.50
Nodes (5): Chandrasekhar dynamical friction, Orbital decay mass dependence comparison (Sat A vs Sat B, t_DF proportional to 1/M_sat), Satellite A final surface density (M_sat=5e10 Msun, post-inspiral), Satellite A orbital decay with/without DF (M_sat=5e10 Msun), Satellite B orbit stability plot (M_sat=1e7 Msun, DF negligible)

### Community 122 - "_validate_species"
Cohesion: 0.22
Nodes (9): Added, Added, Changed, Changed, Fixed, Fixed, Fixed, Fixed (+1 more)

### Community 124 - "run_nbody_gpu"
Cohesion: 0.20
Nodes (10): Example: basic self-gravity, Example: resume from crash, Example: with external potential, GPU memory management, Integration scheme, Parameters, Raises, Returns (+2 more)

### Community 125 - "TestConservationMultiSpecies"
Cohesion: 0.25
Nodes (6): map, multimap, ThrustCachingAllocator, allocated, free_blocks, ptrdiff_t

### Community 126 - "Empirical radial profiles"
Cohesion: 0.36
Nodes (8): make_axisym(), make_halo(), test_zero_pruning.py ~~~~~~~~~~~~~~~~~~~~ Benchmarks zero-coefficient pruning (P, Full FIRE halo: all (l,m) up to lmax, non-zero for most., Axisymmetric (m=0, even l only): maximum pruning benefit., test_correctness(), test_speedup(), timeit()

### Community 127 - "CoM galactocentric radius vs time chart"
Cohesion: 0.83
Nodes (4): bound particle fraction vs time chart, CoM galactocentric radius vs time chart, final particle density 2x2 comparison, Chandrasekhar dynamical friction

### Community 128 - "_is_uniform"
Cohesion: 0.33
Nodes (6): [1.1.0] - 2026-02-08, Added, Changed, Documentation, Fixed, Performance

### Community 129 - "nbody_streams — Documentation"
Cohesion: 0.67
Nodes (3): Building docs (future), Module reference, nbody_streams — Documentation

### Community 145 - "`nbody_streams.fields`"
Cohesion: 0.10
Nodes (20): compute_nbody_forces_gpu, compute_nbody_potential_cpu, compute_nbody_potential_gpu, Contents, Example, Example, Example, get_gpu_info (+12 more)

### Community 146 - "`nbody_streams.nbody_io` — HDF5 Snapshot I/O"
Cohesion: 0.20
Nodes (10): `extract_orbits`, HDF5 snapshot format, Internal I/O functions, `_load_restart`, `nbody_streams.nbody_io` — HDF5 Snapshot I/O, `ParticleReader`, `read_snapshot`, `_save_restart` (+2 more)

### Community 289 - "compute_nbody_forces_cpu"
Cohesion: 0.40
Nodes (5): compute_nbody_forces_cpu, Example, Notes, Parameters, Returns

### Community 290 - "Reading API"
Cohesion: 0.67
Nodes (3): `read_coef_string(source, group_name="snap_000", dataset_name="coefs") -> str`, `read_coefs(source, group_name="snap_000", dataset_name="coefs", times=None)`, Reading API

### Community 291 - "[2.3.0] - 2026-05-12"
Cohesion: 0.40
Nodes (5): [2.3.0] - 2026-05-12, Added, Changed, Changed, Fixed

### Community 292 - "Upgrade Notes"
Cohesion: 0.40
Nodes (5): From 1.0.0 to 1.1.0, From 1.1.0 to 1.2.0, From 1.2.0 to 1.3.0, From 1.3.0 to 2.0.0, Upgrade Notes

### Community 293 - "[1.3.0] - 2026-02-24"
Cohesion: 0.50
Nodes (4): [1.3.0] - 2026-02-24, Added, Changed, Fixed

### Community 294 - "[2.1.0] - 2026-03-07"
Cohesion: 0.50
Nodes (4): [2.1.0] - 2026-03-07, Added, Changed, Removed

### Community 295 - "[2.2.0] - 2026-03-24"
Cohesion: 0.50
Nodes (4): [2.2.0] - 2026-03-24, Added, Changed, Upgrade Notes

### Community 296 - "vec<3,double>"
Cohesion: 0.50
Nodes (3): double3, vec<3,double>, __device__

### Community 297 - "vec<4,double>"
Cohesion: 0.50
Nodes (3): double4, vec<4,double>, __device__

### Community 298 - "vec<3,float>"
Cohesion: 0.50
Nodes (3): float3, vec<3,float>, __device__

### Community 299 - "[2.0.0] - 2026-02-28"
Cohesion: 0.67
Nodes (3): [2.0.0] - 2026-02-28, Added, Changed

## Knowledge Gaps
- **518 isolated node(s):** `vec`, `__device__`, `__device__`, `__device__`, `__device__` (+513 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **114 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GPUPotBase` connect `GroupData` to `MultipolePotentialGPU`, `Physics and formulae`, `._from_data`, `make_df_force_extra`, `__init__.py`, `_jeans_sigma_r`, `CylSplinePotentialGPU`, `UniformAccelerationGPU`, `ShiftedPotentialGPU`, `buildTree.cu`, `NFWPotentialGPU`, `test_phase2_analytic.py`, `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the`, `Find the density peak by locating the minimum gravitational potential.      Uses`, `TreeGPU`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `PotentialGPU()` connect `UniformAccelerationGPU` to `test_multi_species.py`, `_spheroid.py`, `test_phase3_cylspline.py`, `NFWPotentialGPU`, `tree_gravity_gpu`, `test_phase2_analytic.py`, `run_nbody_gpu_tree`, `GroupData`, `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the`, `_io.py`, `Find the density peak by locating the minimum gravitational potential.      Uses`, `vec<3,double>`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `CylSplineCoefs` connect `TreeGPU` to `CylSplinePotentialGPU`, `Box`, `ShiftedPotentialGPU`, `buildTree.cu`, `NFWPotentialGPU`, `tree_gravity_gpu`, `test_newtons_third_law.py`, `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the`, `Generate uniformly random points on the surface of a sphere.      Parameters`, `_io.py`, `GroupData`, `Find the density peak by locating the minimum gravitational potential.      Uses`, `.radial_power`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `MultipolePotentialGPU` (e.g. with `DehnenSphericalPotentialGPU` and `DiskAnsatzPotentialGPU`) actually correct?**
  _`MultipolePotentialGPU` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `Species` (e.g. with `ParticleReader` and `_StepWatchdog`) actually correct?**
  _`Species` has 21 INFERRED edges - model-reasoned connections that need verification._
- **What connects `nbody_streams - lightweight direct N-body utilities.`, `nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica`, `Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when     possibl` to the rest of the system?**
  _1173 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_make_ic` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._