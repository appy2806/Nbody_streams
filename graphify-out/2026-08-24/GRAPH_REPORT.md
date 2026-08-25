# Graph Report - Nbody_streams  (2026-08-13)

## Corpus Check
- 99 files · ~193,962 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2592 nodes · 4752 edges · 288 communities (174 shown, 114 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 373 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `96d7d020`
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
- _validate_species
- TreeGPU
- transforms.py
- _jeans_sigma_r
- Treecode.h
- main.py
- vec<4,float>
- plot_density
- computeGridAndBlockSize
- minmax_block
- Writers and materialization
- run_nbody_gpu
- Load a pre-defined spherical spiral grid scaled to a given radius.      The grid
- GPU potential evaluation (PotentialGPU)
- FIRE helpers
- Dynamical friction
- cuda_primitives.h
- Quick-start workflow
- validate_masses
- _io.py
- TestBoundCenterPhi
- test_newtons_third_law.py
- AGAMA_GPU
- Quadrupole
- vector3
- Loading Agama potentials
- test_phase1_multipole.py
- Physics and formulae
- plots.py
- Quick start
- convert_to_vel_los
- EvolvingPotentialGPU
- convert_vectors
- render_surface_density
- _cylspl_potential_kernel.cu
- AGAMA_GPU — Development Notes
- test_phase3_cylspline.py
- cuda_mem
- plot_stream_evolution
- vec<3,double>
- get_smoothing_lengths
- test_tree.py
- Public API
- vec<3,float>
- _multipole_potential_kernel.cu
- make_uneven_grid
- __init__.py
- fit_double_spheroid_profile
- empirical_density_profile
- Potential
- Chandrasekhar dynamical friction
- Potential fitting
- HDF5 I/O
- .test_circular_orbit_decays
- Reading API
- empirical_velocity_rms_profile
- CoM galactocentric radius vs time chart
- vec<4,float>
- nbody_streams — Documentation
- CLAUDE.md
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

## God Nodes (most connected - your core abstractions)
1. `Treecode` - 68 edges
2. `Species` - 58 edges
3. `MultipolePotentialGPU` - 55 edges
4. `PotentialGPU()` - 51 edges
5. `read_coefs()` - 47 edges
6. `MultipoleCoefs` - 41 edges
7. `CylSplinePotentialGPU` - 41 edges
8. `CompositePotentialGPU` - 40 edges
9. `CylSplineCoefs` - 39 edges
10. `ShiftedPotentialGPU` - 38 edges

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

## Communities (288 total, 114 thin omitted)

### Community 0 - "_make_ic"
Cohesion: 0.06
Nodes (44): _com(), _ke(), _make_ic(), _pe(), ndarray, Path, tests/test_physics.py ===================== Physics-level validation tests for n, KE = (1/2) sum_i m_i |v_i|^2. (+36 more)

### Community 1 - "MultipolePotentialGPU"
Cohesion: 0.09
Nodes (21): _build_single(), _is_potential_ini(), _load_potential_ini(), MultipolePotentialGPU, Path, Load from an Agama .coef_cylsp file., Build from an agama.Potential by materialising it as a single Multipole, True when *p* looks like an Agama multi-section potential INI file. (+13 more)

### Community 2 - "sph_kernels.py"
Cohesion: 0.13
Nodes (21): _argsort_morton_2d(), _cubic_spline_2d_gpu(), _cubic_spline_2d_scalar(), _get_smoothing_lengths_cpu(), _get_smoothing_lengths_gpu(), ndarray, sph_kernels.py ============== SPH smoothing-length computation and 2-D surface-d, GPU smoothing-length computation via CuPy KDTree (internal). (+13 more)

### Community 3 - "_load.py"
Cohesion: 0.09
Nodes (26): get_gpu_info(), Get information about available GPU(s).          Returns     -------     info :, nbody_streams - lightweight direct N-body utilities., ndarray, nbody_streams.sim Unified high-level simulation entry point for multi-species N-, Run a direct N-body simulation with one or more particle species.      This is t, run_simulation(), _build_particle_arrays() (+18 more)

### Community 4 - "_nfw_potential"
Cohesion: 0.15
Nodes (14): chandrasekhar_friction(), Chandrasekhar dynamical-friction acceleration at the satellite CoM.      Compute, _nfw_potential(), Constant sigma function for analytic tests., DF force must oppose the velocity., No friction when satellite is at rest., DF acceleration should scale linearly with satellite mass., Validate against a direct evaluation of BT2008 eq. 8.13. (+6 more)

### Community 5 - "test_utils.py"
Cohesion: 0.11
Nodes (21): empirical_circular_velocity_profile(), Generate uniformly random points on the surface of a sphere.      Parameters, r"""Compute the circular velocity profile :math:`v_{\rm circ}(r) = \sqrt{G\,M(<r, uniform_spherical_grid(), plummer_enclosed_mass(), plummer_v_circ(), tests/test_utils.py ===================  Tests for ``nbody_streams.utils`` using, Circular velocity should match analytic Plummer. (+13 more)

### Community 6 - "__init__.py"
Cohesion: 0.07
Nodes (48): ArrayLike, float32, ImportError, KERNEL_TYPES, _cupy.py ~~~~~~~~ Optional-CuPy shim for ``agama_helper``.  CuPy is an *optional, nbody_streams.cuda_kernels CUDA kernels for N-body force and potential computati, _compute_forces_cpu(), compute_nbody_forces_cpu() (+40 more)

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
Cohesion: 0.09
Nodes (25): _clamped_left_cubic_deriv_batch(), CylSplinePotentialGPU, _get_cylspl_kernel(), _get_kernel(), _natural_cubic_deriv_batch(), _prep_xyz(), ndarray, First derivatives at all knots for natural cubic splines — batched over columns. (+17 more)

### Community 11 - "ShiftedPotentialGPU"
Cohesion: 0.10
Nodes (10): _apply_modifiers(), Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Shifted modifier: evaluates ``inner`` at ``xyz - center(t)``.      Parameters, Scaled modifier: ``Phi_s(x,t) = a(t)*s(t) * Phi(x*s(t))`` where ``s(t) = 1/scale, Return (s, a) = (1/scale(t), ampl(t)) with linear extrapolation., Wrap *pot* with Shifted/Scaled modifiers if requested.      ``scale`` may be:, Agama-compatible eval : returns any combination of potential, acceleration, (+2 more)

### Community 12 - "run_simulation"
Cohesion: 0.14
Nodes (14): Contents, Definition, Example, make_plummer_sphere, Named constructors, `nbody_streams` — main package, Parameters, Parameters (+6 more)

### Community 13 - "generate_stream_coords"
Cohesion: 0.17
Nodes (12): nbody_streams.coords - coordinate transformations.  Cartesian/spherical/cylindri, generate_stream_coords(), get_observed_stream_coords(), ndarray, Stream coordinate generation and observable conversions.  Generate stream-aligne, Project galactocentric positions or phase-space vectors into a     pre-computed, Convert galactocentric phase space into stream-aligned coordinates     (phi1, ph, Convert galactocentric phase-space coordinates to observed sky     coordinates ( (+4 more)

### Community 14 - "_AnalyticBase"
Cohesion: 0.17
Nodes (15): CompletedProcess, _check(), test_no_cupy.py ~~~~~~~~~~~~~~~ Regression tests for the optional-CuPy contract, Coefficient dataclasses, text and HDF5 round-trips are pure NumPy., Every GPU entry point fails loudly with an actionable message., With the real environment, ``CUPY_AVAILABLE`` tracks the import., Execute *body* in a fresh interpreter where ``import cupy`` fails., ``import nbody_streams.agama_helper`` must not require CuPy. (+7 more)

### Community 15 - "tree_gravity_gpu"
Cohesion: 0.16
Nodes (22): Compute gravitational accelerations and potential using the GPU tree code     (B, tree_gravity_gpu(), generate_plummer(), Net force = sum(m_i * a_i) should be ~0 for an isolated system., Net torque = sum(r x (m*a)) should be small., Compare 0.5*sum(m*phi_tree) against 0.5*sum(m*phi_direct)., For a spherically symmetric distribution, forces should point inward., Error should not degrade significantly as N grows (same distribution). (+14 more)

### Community 17 - "__init__.py"
Cohesion: 0.21
Nodes (12): create_snapshot_dict(), fit_potential(), ndarray, Path, agama_helper._fit ~~~~~~~~~~~~~~~~~ Fit Agama Multipole and CylSpline potential, Fit Agama Multipole and CylSpline potentials from a multi-species snapshot., Sample *n* positions from a spherically declining density profile., Sample *n* positions from a thin exponential disk. (+4 more)

### Community 18 - "test_phase2_analytic.py"
Cohesion: 0.15
Nodes (8): cuda_alive(), Return True if the CUDA context has no pending errors.      Uses cudaGetLastErro, nbody_streams.tree_gpu ====================== GPU Barnes-Hut tree code for N-bod, ndarray, Run a GPU-tree N-body simulation with KDK leapfrog integration.      Force evalu, Background-thread watchdog for GPU leapfrog loops.      Per-step overhead is ~1, run_nbody_gpu_tree(), _StepWatchdog

### Community 19 - "test_advanced.py"
Cohesion: 0.28
Nodes (8): Checks which high-performance features are actually active., Runs a tiny GPU simulation and verifies snapshot output., Quick helper to generate test particles (Plummer-like spread)., Runs a tiny CPU simulation and verifies snapshot output., setup_dummy_data(), test_cpu_integration(), test_environment_summary(), test_gpu_integration()

### Community 21 - "CellData"
Cohesion: 0.25
Nodes (8): PendingWork, box, cellFirstChildIndex, cellParentIndex, mempool_offset, nCellmax, nSubNodes_y, octant_mask

### Community 22 - "test_newtons_third_law.py"
Cohesion: 0.12
Nodes (29): ndarray, Subdivide every interval of *times* by *factor*, keeping the original nodes., Cubic-spline every coefficient series along the trailing time axis., Resample a coefficient time series onto a new time grid with cubic splines., refine_times(), _spline_block(), spline_resample_coefs(), Tests for cubic-spline resampling of a coefficient time series.  ``spline_resamp (+21 more)

### Community 23 - "_build.py"
Cohesion: 0.67
Nodes (3): main(), nbody_streams.tree_gpu._build Console-script entry point: nbody-build-tree  Comp, _tree_gpu_dir()

### Community 24 - ".radial_power"
Cohesion: 0.03
Nodes (49): Stack time-less coefficient snapshots into one object with a time axis.      Gri, Read one or many Agama Multipole snapshots into a :class:`MultipoleCoefs`., Read an Agama expansion into a structured dataclass — one snapshot or many., read_coefs(), read_mult_coefs(), stack_coefs(), cylsp3_snaps(), cylsp_snaps() (+41 more)

### Community 26 - "`nbody_streams.utils` — Analysis Utilities"
Cohesion: 0.08
Nodes (25): Centre finding, `compute_iterative_boundness` (deprecated), Density-profile fitting, `empirical_circular_velocity_profile`, `empirical_density_profile`, Empirical radial profiles, `empirical_velocity_anisotropy_profile`, `empirical_velocity_dispersion_profile` (+17 more)

### Community 28 - "`nbody_streams.coords` — Coordinate Transforms"
Cohesion: 0.25
Nodes (8): `convert_coords`, `convert_to_vel_los`, `convert_vectors`, Coordinate conventions, `generate_stream_coords`, `get_observed_stream_coords`, `nbody_streams.coords` — Coordinate Transforms, `to_stream_coords`

### Community 31 - "Treecode.h"
Cohesion: 0.11
Nodes (20): int3, double4, Treecode<real_t>::computeForces(), Treecode<real_t>::computeMultipoles(), computeKeys(), __global__, __host__, __out (+12 more)

### Community 32 - "Treecode"
Cohesion: 0.04
Nodes (52): Treecode, BUILD_MAX_WORK, BUILD_PENDING_WORK_BYTES, buildTree, cell_max, computeForces, computeMultipoles, d_build_octCounter (+44 more)

### Community 35 - ".dark"
Cohesion: 0.13
Nodes (12): Convenience constructor for dark-matter particles., _plummer_like(), run_simulation emits PerformanceWarning for large CPU direct N., Very small Plummer-like ICs for integration tests., Without species kwarg, old dark/star HDF5 schema is written., Verify that debug_energy=True works without errors on all dispatcher paths     t, debug_energy=False (default) must not print energy lines., run_simulation must raise TypeError on unknown kwargs for CPU path. (+4 more)

### Community 36 - "Species"
Cohesion: 0.17
Nodes (5): Definition of a particle species in an N-body simulation.      Parameters     --, Convenience constructor for stellar particles., Species, TestParticleReaderMultiSpecies, TestSpeciesDataclass

### Community 37 - "_potential.py"
Cohesion: 0.06
Nodes (39): Raise a descriptive :class:`ImportError` if CuPy is unavailable.      Parameters, require_cupy(), _build_dehnen_gpu(), _build_disk_gpu(), _build_king_gpu(), _build_multipole_data(), _build_spheroid_gpu(), _coerce() (+31 more)

### Community 38 - "nbody_io.py"
Cohesion: 0.29
Nodes (17): check_nan_inf(), direct_sum_max_eps(), generate_plummer(), hdr(), Return (result, elapsed_ms) with a surrounding device sync., Small-N O(N^2) direct-sum with tree_gpu max softening convention:       eps2_ij, rel_err_stats(), subhdr() (+9 more)

### Community 40 - "`nbody_streams.agama_helper`"
Cohesion: 0.18
Nodes (11): Center parameter, Contents, Full example: MW potential pipeline, Gotchas, `nbody_streams.agama_helper`, Overview, `read_coef_string(source, group_name="snap_000", dataset_name="coefs") -> str`, `read_coefs(source, group_name="snap_000", dataset_name="coefs", times=None)` (+3 more)

### Community 41 - "Changelog"
Cohesion: 0.04
Nodes (44): [1.0.0] - 2025-XX-XX, [1.1.0] - 2026-02-08, [1.2.0] - 2026-02-14, [1.3.0] - 2026-02-24, [2.0.0] - 2026-02-28, [2.1.0] - 2026-03-07, [2.2.0] - 2026-03-24, [2.3.0] - 2026-05-12 (+36 more)

### Community 42 - "test_multi_species.py"
Cohesion: 0.17
Nodes (12): Backend-specific kwargs, Dynamical friction kwargs, Example: GPU tree backend, Example: multi-species dark matter + stars, Example: single species, Example: with Chandrasekhar dynamical friction, Example: with external Agama potential, Parameters (+4 more)

### Community 43 - "Box"
Cohesion: 0.08
Nodes (17): _as_times(), MultipoleCoefs, ndarray, Attach or relabel the time axis.          Nothing is interpolated, smoothed or r, Coerce *times* to a 1-D float64 array, rejecting anything else., Structured representation of a Multipole (spherical harmonic BFE) potential., Maximum l order present in *lm_labels*., Sorted unique l values. (+9 more)

### Community 44 - "_force.py"
Cohesion: 0.11
Nodes (21): CDLL, _coerce_eps(), _find_cudart(), _gpu_ptr(), ndarray, _force.py  --  ctypes interface to the GPU tree-code gravity shared library.  Su, Return a contiguous float32 CuPy array of length n for eps.      Accepts:, Load libcudart; try versioned names as fallback. (+13 more)

### Community 45 - "buildTree.cu"
Cohesion: 0.08
Nodes (18): _check_unique_names(), _CoefTimeAxisMixin, Time-less view at time index *i*.          The returned object shares memory wit, Assert internal shape consistency, naming the offending field on failure., Serialise one snapshot back to the Agama CylSpline text format.          Paramet, Refuse a name/group template that collapses several times onto one target., Time-axis behaviour shared by :class:`MultipoleCoefs` and :class:`CylSplineCoefs, ``True`` when this object carries a trailing time axis. (+10 more)

### Community 46 - "NFWPotentialGPU"
Cohesion: 0.11
Nodes (15): NFWPotentialGPU, Navarro-Frenk-White potential optimized with CuPy ElementwiseKernels., PotentialGPU(), GPU potential factory : mirrors the ``agama.Potential`` API.      Usage     ----, Write a potential config file with the given suffix (any extension works)., _is_potential_ini detects multi-section content regardless of extension., [Potential halo] + [Potential disk] headers with a .dat extension., [Potential 0] + [Potential 1] + [Potential 2] (numbered headers). (+7 more)

### Community 47 - "_random_xv"
Cohesion: 0.13
Nodes (10): Write a snapshot compatible with :class:`ParticleReader`.      Two modes     ---, _save_snapshot(), ndarray, _random_xv(), Write an HDF5 file in the old two-species format., Old file with only dark particles (no star group)., Random (N, 6) phase space  - tiny, just for IO tests., TestParticleReaderBackwardCompat (+2 more)

### Community 48 - "API"
Cohesion: 0.07
Nodes (29): AGAMA potential helper (`agama_helper`), Analysis utilities (`utils`), API, Caveats and known limitations, Comparison: direct N-body vs tree methods, Coordinates (`coords`), Dynamical friction with external potentials, Fast stream generation (`fast_sims`) (+21 more)

### Community 49 - "r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I"
Cohesion: 0.09
Nodes (25): double2, __forceinline__ Box<T> ChildBox(), __forceinline__ int Particle4<double>::get_idx(), __forceinline__ int Particle4<double>::get_oct(), __forceinline__ int Particle4<double>::set_idx(), __forceinline__ int Particle4<double>::set_oct(), __forceinline__ int Particle4<float>::get_idx(), __forceinline__ int Particle4<float>::get_oct() (+17 more)

### Community 50 - "Particle4"
Cohesion: 0.11
Nodes (21): countAtRootNode(), __out, Box, centre, __device__, hsize, __forceinline__, __host__ (+13 more)

### Community 51 - "CompositePotentialGPU"
Cohesion: 0.36
Nodes (3): host_mem, n, ptr

### Community 52 - "treewalk_warp"
Cohesion: 0.26
Nodes (16): approxAcc(), __device__, float2, float3, float4, __forceinline__, __global__, int2 (+8 more)

### Community 53 - "treeGPU_interface.cu"
Cohesion: 0.15
Nodes (24): float4, __global__, real_t, k_extract_unsorted(), k_load_pos_mass(), k_load_pos_mass_eps(), Res, w (+16 more)

### Community 54 - "Ellipsoidal radius and particle selection (numba-accelerated).      Computes the"
Cohesion: 0.04
Nodes (27): _AnalyticBase, AnalyticPotentialGPU(), DehnenSphericalPotentialGPU, HernquistPotentialGPU, IsochronePotentialGPU, MiyamotoNagaiPotentialGPU, PlummerPotentialGPU, _analytic_potentials.py ~~~~~~~~~~~~~~~~~~~~~~~ Phase 2: GPU-accelerated analyti (+19 more)

### Community 55 - "Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti"
Cohesion: 0.09
Nodes (24): _load_restart(), _make_times_ns(), ParticleReader, ndarray, Path, nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr, Append or update snapshot.times in output_dir.     Ensures unique snap_index ent, Worker for N-species parallel extraction.      Args (tuple):         snap_index (+16 more)

### Community 56 - "Generate uniformly random points on the surface of a sphere.      Parameters"
Cohesion: 0.17
Nodes (5): _error(), _MissingCuPy, _MissingKernel, Stand-in for a CuPy kernel constructed while CuPy is unavailable., Stub standing in for the ``cupy`` module when it is not installed.

### Community 57 - "Generate num_pts points on a sphere using the Fibonacci spiral.      Points are"
Cohesion: 0.32
Nodes (3): _max_n(), Pre-allocated GPU tree handle for time-stepping loops.      Parameters     -----, TreeGPU

### Community 58 - "Implementation internals"
Cohesion: 0.08
Nodes (24): Availability flag, Build system details, Building the shared library, ctypes interface (`_force.py`), `cuda_alive`, Data structures, Float32 note, `G_DEFAULT` (+16 more)

### Community 59 - "Find the density peak by locating the minimum gravitational potential.      Uses"
Cohesion: 0.06
Nodes (22): DiskAnsatzPotentialGPU, LogHaloPotentialGPU, DiskAnsatz separable disk potential: Phi(R,z) = f(r) * H(z)     where r = sqrt(R, LogHalo: Phi = (v0^2/2) * ln(rc^2 + x^2 + y^2/p^2 + z^2/q^2)      Constructor ma, CompositePotentialGPU, Sum of GPU potential components : mirrors agama.Potential composite.      Each c, _pts(), test_load_paths.py ~~~~~~~~~~~~~~~~~~ Tests for load_agama_potential / load_agam (+14 more)

### Community 60 - "computeMultipoles.cu"
Cohesion: 0.12
Nodes (20): buildOctreeHost(), collect_leaves(), compute_level_begIdx(), computeBoundingBox(), float4, __global__, int2, T (+12 more)

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
Cohesion: 0.15
Nodes (24): _AgamaTimeSpline, Time interpolation of a K-vector, bit-compatible with Agama.      Reproduces ``p, Value of the K-vector at scalar time *t*, as a tuple of K floats., _as_agama_scale(), _pts(), test_time_modifiers.py ~~~~~~~~~~~~~~~~~~~~~~ Numerical tests for the time-depen, Guard: the fixture is coarse enough that not-a-knot is measurably wrong.      If, Our (T,2) [t, scale] -> Agama's (T,3) [t, ampl=1, scale]. (+16 more)

### Community 65 - "_find_cudart"
Cohesion: 0.10
Nodes (32): _as_existing_path(), _check_presence(), _check_stackable(), _describe_grid_mismatch(), generate_lmax_pairs(), _is_source_sequence(), Path, agama_helper._coefs ~~~~~~~~~~~~~~~~~~~ Structured representations of Agama expa (+24 more)

### Community 66 - ".alloc"
Cohesion: 0.18
Nodes (11): Coefficient time axis, `column(l, m) -> int`  (Multipole only), `copy() -> MultipoleCoefs \| CylSplineCoefs`, Introspection properties, Methods (both classes unless noted), Module-level: `stack_coefs(items, times) -> MultipoleCoefs | CylSplineCoefs`, `radial_power` / `total_power` with a time axis, `snapshot(i) -> MultipoleCoefs \| CylSplineCoefs`  (alias: `obj[i]`) (+3 more)

### Community 67 - "make_df_force_extra"
Cohesion: 0.14
Nodes (11): _is_gpu_potential(), make_df_force_extra(), Build a ``force_extra`` closure that applies Chandrasekhar dynamical     frictio, True when *pot* is from the PotentialGPU family (nbody_streams.agama_helper)., DF is a rigid-body force — all particles within the core get the same         ac, Net DF acceleration should point opposite to bulk CoM velocity., Between correction steps, CoM should be predicted, not recalculated., Without Agama the factory should raise ImportError. (+3 more)

### Community 68 - ".test_circular_orbit_decays"
Cohesion: 0.24
Nodes (6): GroupData, __device__, packed_data, __host__, int2, uint4

### Community 69 - "computeMultipoles.cu"
Cohesion: 0.32
Nodes (6): addMonopole(), addQuadrupole(), __device__, double4, real_t, Treecode<float>

### Community 70 - "_validate_species"
Cohesion: 0.20
Nodes (10): Example: basic self-gravity, Example: resume from crash, Example: with external potential, GPU memory management, Integration scheme, Parameters, Raises, Returns (+2 more)

### Community 71 - "TreeGPU"
Cohesion: 0.47
Nodes (3): _is_uniform(), Check whether all elements of *arr* are equal to within relative tolerance., TestIsUniform

### Community 72 - "transforms.py"
Cohesion: 0.17
Nodes (22): _as_3vec(), _cart_to_cyl(), _cart_to_sph(), _cyl_to_cart(), _cyl_to_sph(), _propagate_nans(), ndarray, Coordinate system and vector field conversions.  Cartesian <-> spherical <-> cyl (+14 more)

### Community 73 - "_jeans_sigma_r"
Cohesion: 0.09
Nodes (25): compute_sigma_r(), _jeans_sigma_r(), ndarray, nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica, Local circular-speed approximation to the 1-D velocity dispersion.      Uses the, Compute a radial velocity-dispersion profile from an Agama potential.      Param, Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when     possibl, Jeans-equation estimate of the isotropic 1-D velocity dispersion.      Uses the (+17 more)

### Community 74 - "Treecode.h"
Cohesion: 0.40
Nodes (5): Backend dispatch table, run_nbody_cpu, run_nbody_gpu, run_nbody_gpu_tree, Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles

### Community 75 - "main.py"
Cohesion: 0.15
Nodes (20): _calculate_particle_distances_sq(), _calculate_Rsphall_and_extract(), _compute_weighted_structure_tensor(), _density_peak_center(), fibonacci_sphere_grid(), find_center(), find_center_position(), fit_iterative_ellipsoid() (+12 more)

### Community 76 - "vec<4,float>"
Cohesion: 0.22
Nodes (9): Coefficient dataclasses, `CylSplineCoefs`, `MultipoleCoefs`, `radial_power(l, use_quadrature=True) -> ndarray`, `to_coef_string() -> str`, `to_coef_string() -> str`, `total_power(l, use_quadrature=True) -> float`, `zeroed(keep_lm) -> MultipoleCoefs` (+1 more)

### Community 77 - "plot_density"
Cohesion: 0.15
Nodes (8): Axes, AxesImage, Colormap, plot_density(), Any, Generate a projected density image (imshow) from particle data.      Accepts par, imshow extent should be [-gridsize/2, gridsize/2] on both axes., TestPlotDensity

### Community 80 - "Writers and materialization"
Cohesion: 0.29
Nodes (7): `materialize_potential(*, center=None, interp_linear=True, gpu=False)`, `to_coef_files(out_dir, name_fmt="snap_{i:04d}{ext}") -> list[str]`, `to_coef_string(t=None) -> str`, `to_coef_strings() -> list[str]`, `to_evolving_ini(ini_path, out_dir=None, interp_linear=True) -> str`, `to_h5(path, group_fmt="snap_{i:04d}", dataset_name="coefs", overwrite=True, write_times=True) -> str`, Writers and materialization

### Community 81 - "run_nbody_gpu"
Cohesion: 0.18
Nodes (11): Constants, Contents, Example: direct method with specific kernel, Example: tree method, make_plummer_sphere, Method selection guidance, `nbody_streams.run`, Parameters (+3 more)

### Community 83 - "GPU potential evaluation (PotentialGPU)"
Cohesion: 0.25
Nodes (8): Accuracy (vs Agama CPU), API (matches `agama.Potential`), CuPy is optional, Factory: `PotentialGPU(...)`, GPU potential evaluation (PotentialGPU), Requirements, Supported types, `UniformAcceleration` — non-inertial reference frame

### Community 84 - "FIRE helpers"
Cohesion: 0.33
Nodes (6): `create_fire_evolving_ini(sim_dir, model_pattern, output_filename, snap_range=None, verbose=True) -> str`, FIRE helpers, `load_fire_pot(sim_dir, nsnap, sym="n", lmax=4, kind="whole", keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, file_ext="DR", out_acc=False, halo=None, verbose=True, return_coefs=False, save_modified=False, save_dir=None)`, `read_snapshot_times(sim_dir, sep=r'\s+') -> pandas.DataFrame`, `refine_times(times, factor=10) -> ndarray`, `spline_resample_coefs(coefs, times_new) -> MultipoleCoefs | CylSplineCoefs`

### Community 85 - "Dynamical friction"
Cohesion: 0.09
Nodes (23): Caveats, Centre-of-mass detection, Chandrasekhar formula (BT2008 eq. 8.13), Contents, Core-stalling suppression, Coulomb logarithm, df_* kwargs reference, Dynamical friction (+15 more)

### Community 86 - "cuda_primitives.h"
Cohesion: 0.23
Nodes (18): buildOctant(), addBoxSize(), atomicAdd_double(), __forceinline__(), __device__, real_t, inclusive_scan_warp(), inclusive_segscan_warp() (+10 more)

### Community 87 - "Quick-start workflow"
Cohesion: 0.40
Nodes (5): 1 — Fit and save coefficient files, 2 — Pack text files into HDF5, 3 — Inspect and modify coefficients, 4 — Load Agama potential, Quick-start workflow

### Community 88 - "validate_masses"
Cohesion: 0.15
Nodes (18): empirical_velocity_anisotropy_profile(), fit_dehnen_profile(), fit_plummer_profile(), r"""Compute the velocity anisotropy parameter :math:`\beta(r)`.      .. math::, r"""Fit a triaxial Dehnen profile by mapping to ellipsoidal radius.      Compute, r"""Fit a spherical Plummer profile to particle data.      Bins particles into l, ndarray, nbody_streams.utils._validation ================================  Shared input-v (+10 more)

### Community 89 - "_io.py"
Cohesion: 0.07
Nodes (50): _add_negative_m(), _detect_expansion_type(), Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown., Expand (l, m) pairs to include their negative-m counterparts.      Each (l, m) w, create_fire_evolving_ini(), load_fire_pot(), Path, agama_helper._fire ~~~~~~~~~~~~~~~~~~ FIRE-simulation-specific convenience wrapp (+42 more)

### Community 90 - "TestBoundCenterPhi"
Cohesion: 0.16
Nodes (10): _bound_center_phi(), Phi-energy iterative bound-particle centre.      Finds the centre of the gravita, Return a cluster at r0 on the x-axis with a bulk y-velocity., Simple pairwise Plummer phi (exact for small N test)., Returned CoM should be close to the cluster centre., For a self-gravitating cluster with low velocity dispersion, bound         parti, make_df_force_extra closure should accept and use phi kwarg., Unbound particles (phi + 0.5 v_rel^2 >= 0) should get zero DF. (+2 more)

### Community 91 - "test_newtons_third_law.py"
Cohesion: 0.50
Nodes (4): _newtons_third_law(), Test that net force is ~0 for isolated system.          Returns:         bool: T, Test all precision modes., test_all_precisions()

### Community 92 - "AGAMA_GPU"
Cohesion: 0.10
Nodes (19): Accuracy, AGAMA_GPU, Analytic — `_analytic_potentials.py`, API, Coefficient objects — one snapshot or a whole time series, Composite types, CylSpl BFE - `CylSplinePotentialGPU`, File layout (+11 more)

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
Cohesion: 0.20
Nodes (13): Run GPU-accelerated N-body simulation with leapfrog (KDK) integration., run_nbody_gpu(), _plummer_ic(), ndarray, Path, tests/test_snapshot_schedule.py Regression tests: the GPU-tree backend must writ, The schedule formula itself: exactly `snapshots` entries ending at n_steps., Small cold blob — physics is irrelevant here, only the I/O schedule is. (+5 more)

### Community 99 - "plots.py"
Cohesion: 0.13
Nodes (17): Figure, nbody_streams.viz - visualization and plotting.  Projected density maps, Mollwei, _aggregate_data_chunk(), _extract_particles_at_step(), _gauss_filter_surf_dens(), _generate_ticks(), plot_mollweide(), plot_stream_sky() (+9 more)

### Community 100 - "Quick start"
Cohesion: 0.15
Nodes (13): Choosing a backend, Core package, CPU fallback (no GPU required), GPU support (CuPy), GPU tree code (one-time build), Hello world: Plummer sphere self-gravity, Installation, Next steps (+5 more)

### Community 101 - "convert_to_vel_los"
Cohesion: 0.23
Nodes (6): convert_to_vel_los(), Compute line-of-sight (radial) velocity from Galactocentric phase-space     coor, Tests for nbody_streams.coords — coordinate and vector transforms., Circular orbit at solar position: v_los should be 0., Purely radial velocity along +x: v_los = vx., TestConvertToVelLos

### Community 102 - "EvolvingPotentialGPU"
Cohesion: 0.29
Nodes (4): EvolvingPotentialGPU, _lerp(), Time-evolving GPU potential: wraps a sequence of static GPU potentials     at kn, Linear interp a*(1-alpha) + b*alpha.

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
Cohesion: 0.07
Nodes (40): _prep_xyz(), ndarray, Build the acceleration spline from Agama's ``file=`` argument.      Accepts what, Spatially uniform, optionally time-dependent acceleration ``a(t)``, arising, Acceleration components at time *t*., Agama-compatible eval --- returns any combination of potential, acceleration,, _read_accel_table(), _squeeze() (+32 more)

### Community 108 - "cuda_mem"
Cohesion: 0.25
Nodes (4): cuda_mem, n, ptr, T

### Community 109 - "plot_stream_evolution"
Cohesion: 0.31
Nodes (3): plot_stream_evolution(), Three-panel evolution plot: galactocentric distance, bound fraction     (or 3-D, TestPlotStreamEvolution

### Community 110 - "vec<3,double>"
Cohesion: 0.50
Nodes (3): double3, vec<3,double>, __device__

### Community 111 - "get_smoothing_lengths"
Cohesion: 0.15
Nodes (10): get_smoothing_lengths(), Compute SPH smoothing lengths as the distance to the *k_neighbors*-th     neares, _close_figures(), orbit_data(), Tests for nbody_streams.viz -- visualization functions.  All tests use the Agg b, Close all matplotlib figures after each test., Particles in a dense cluster should get smaller h than isolated ones., 3-D pos should be accepted (D is not restricted to 2). (+2 more)

### Community 112 - "test_tree.py"
Cohesion: 0.33
Nodes (8): error_report(), generate_plummer(), Run with a small N and check that forces point inward., Plummer model ICs matching the C++ binary (double precision + R<100 rejection)., Print relative-error statistics between ref and test arrays., test_accuracy(), test_performance(), test_sanity()

### Community 113 - "Public API"
Cohesion: 0.25
Nodes (8): `create_ic_particle_spray_chen2025`, `create_ic_particle_spray_fardal2015`, `create_particle_spray_stream`, Dynamical friction, `nbody_streams.fast_sims` — Fast Stream Generation, Perturber potential, Public API, `run_restricted_nbody`

### Community 114 - "vec<3,float>"
Cohesion: 0.50
Nodes (3): float3, vec<3,float>, __device__

### Community 115 - "_multipole_potential_kernel.cu"
Cohesion: 0.39
Nodes (5): __device__, multipole_density_kernel(), multipole_eval_device(), multipole_hess_kernel(), quintic_eval()

### Community 116 - "make_uneven_grid"
Cohesion: 0.25
Nodes (8): empirical_velocity_dispersion_profile(), make_uneven_grid(), Create a 1-D grid with unequally spaced nodes.      The grid starts at 0, the se, r"""Compute the velocity dispersion :math:`\sigma_v(r)`.      Parameters     ---, Velocity dispersion should be finite and positive at all radii., Grid starts at 0, second node at xmin, last at xmax., test_empirical_velocity_dispersion(), test_make_uneven_grid()

### Community 117 - "__init__.py"
Cohesion: 0.33
Nodes (6): nbody_streams.utils - analysis and diagnostic utilities., compute_iterative_boundness(), iterative_unbinding(), Iterative unbinding to determine bound particles.      Computes the gravitationa, A virialised Plummer sphere should be >90% bound., test_iterative_unbinding()

### Community 118 - "fit_double_spheroid_profile"
Cohesion: 0.33
Nodes (6): double_power_law_density(), fit_double_spheroid_profile(), r"""Construct a Zhao (1996) double-power-law density profile normalised to total, r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I, Fitted profile should recover the Plummer mass to ~30%., test_fit_double_spheroid_profile()

### Community 119 - "empirical_density_profile"
Cohesion: 0.33
Nodes (6): empirical_density_profile(), r"""Compute the mass-density radial profile :math:`\rho(r)`.      Parameters, plummer_density(), rho(r) = (3M / 4*pi*a^3) (1 + r^2/a^2)^{-5/2}, Density profile should match analytic Plummer within ~20% at intermediate radii., test_empirical_density_profile()

### Community 120 - "Potential"
Cohesion: 0.33
Nodes (4): Potential, When quasispherical DF fails, _common should use Jeans, not MW table., The fallback must NOT use the old hardcoded MW sigma array., TestCommonJeansFallback

### Community 121 - "Chandrasekhar dynamical friction"
Cohesion: 0.50
Nodes (5): Chandrasekhar dynamical friction, Orbital decay mass dependence comparison (Sat A vs Sat B, t_DF proportional to 1/M_sat), Satellite A final surface density (M_sat=5e10 Msun, post-inspiral), Satellite A orbital decay with/without DF (M_sat=5e10 Msun), Satellite B orbit stability plot (M_sat=1e7 Msun, DF negligible)

### Community 122 - "Potential fitting"
Cohesion: 0.36
Nodes (8): make_axisym(), make_halo(), test_zero_pruning.py ~~~~~~~~~~~~~~~~~~~~ Benchmarks zero-coefficient pruning (P, Full FIRE halo: all (l,m) up to lmax, non-zero for most., Axisymmetric (m=0, even l only): maximum pruning benefit., test_correctness(), test_speedup(), timeit()

### Community 123 - "HDF5 I/O"
Cohesion: 0.67
Nodes (3): HDF5 I/O, `write_coef_to_h5(h5_path, coef_string, group_name="snap_000", dataset_name="coefs", overwrite=False, metadata=None)`, `write_snapshot_coefs_to_h5(snapshot_ids, coef_file_patterns, h5_output_paths, group_fmt="snap_{snap:03d}", dataset_name="coefs", overwrite=True, encoding="utf-8", times=None)`

### Community 124 - ".test_circular_orbit_decays"
Cohesion: 0.67
Nodes (3): dim3, computeGridAndBlockSize(), __forceinline__

### Community 125 - "Reading API"
Cohesion: 0.67
Nodes (3): `create_snapshot_dict(pos_dark, mass_dark, pos_star=None, mass_star=None, pos_gas=None, mass_gas=None, temperature_gas=None) -> dict`, `fit_potential(snap, nsnap, *, sym="n", lmax=4, rmax_sel=300.0, rmax_exp=None, file_ext="", save_dir="potential/", halo=None, kind="both", center=None, rotation=None, verbose=True, subsample_factor=1, cold_temp_log10_thresh=4.5) -> dict[str, list[str]]`, Potential fitting

### Community 126 - "empirical_velocity_rms_profile"
Cohesion: 0.50
Nodes (4): empirical_velocity_rms_profile(), r"""Compute the root-mean-square velocity profile :math:`v_{\rm rms}(r)`.      P, RMS velocity should be finite and positive., test_empirical_velocity_rms()

### Community 127 - "CoM galactocentric radius vs time chart"
Cohesion: 0.83
Nodes (4): bound particle fraction vs time chart, CoM galactocentric radius vs time chart, final particle density 2x2 comparison, Chandrasekhar dynamical friction

### Community 128 - "vec<4,float>"
Cohesion: 0.67
Nodes (3): __device__, float2, minmax_block()

### Community 129 - "nbody_streams — Documentation"
Cohesion: 0.67
Nodes (3): Building docs (future), Module reference, nbody_streams — Documentation

### Community 145 - "`nbody_streams.fields`"
Cohesion: 0.08
Nodes (25): compute_nbody_forces_cpu, compute_nbody_forces_gpu, compute_nbody_potential_cpu, compute_nbody_potential_gpu, Contents, Example, Example, Example (+17 more)

### Community 146 - "`nbody_streams.nbody_io` — HDF5 Snapshot I/O"
Cohesion: 0.20
Nodes (10): `extract_orbits`, HDF5 snapshot format, Internal I/O functions, `_load_restart`, `nbody_streams.nbody_io` — HDF5 Snapshot I/O, `ParticleReader`, `read_snapshot`, `_save_restart` (+2 more)

## Knowledge Gaps
- **491 isolated node(s):** `vec`, `__device__`, `__device__`, `__device__`, `__device__` (+486 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **114 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GPUPotBase` connect `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the` to `MultipolePotentialGPU`, `make_df_force_extra`, `_potential.py`, `EvolvingPotentialGPU`, `__init__.py`, `_jeans_sigma_r`, `CylSplinePotentialGPU`, `test_phase3_cylspline.py`, `Box`, `ShiftedPotentialGPU`, `NFWPotentialGPU`, `computeGridAndBlockSize`, `Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti`, `Find the density peak by locating the minimum gravitational potential.      Uses`, `TreeGPU`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `Species` connect `Species` to `_make_ic`, `Physics and formulae`, `_load.py`, `.dark`, `__init__.py`, `TreeGPU`, `_random_xv`, `test_phase2_analytic.py`, `Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `CylSplineCoefs` connect `TreeGPU` to `vec<2,float>`, `_find_cudart`, `MultipolePotentialGPU`, `_potential.py`, `EvolvingPotentialGPU`, `CylSplinePotentialGPU`, `Box`, `ShiftedPotentialGPU`, `buildTree.cu`, `test_newtons_third_law.py`, `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the`, `.radial_power`, `_io.py`, `Find the density peak by locating the minimum gravitational potential.      Uses`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `Species` (e.g. with `ParticleReader` and `_StepWatchdog`) actually correct?**
  _`Species` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `MultipolePotentialGPU` (e.g. with `DehnenSphericalPotentialGPU` and `DiskAnsatzPotentialGPU`) actually correct?**
  _`MultipolePotentialGPU` has 25 INFERRED edges - model-reasoned connections that need verification._
- **What connects `nbody_streams - lightweight direct N-body utilities.`, `nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica`, `Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when     possibl` to the rest of the system?**
  _1044 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_make_ic` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._