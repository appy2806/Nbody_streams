# Graph Report - Nbody_streams  (2026-07-20)

## Corpus Check
- 91 files · ~171,580 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2227 nodes · 3993 edges · 264 communities (150 shown, 114 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 326 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d25226f3`
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
- test_chandrasekhar.py
- make_df_force_extra
- transforms.py
- _jeans_sigma_r
- main.py
- plot_density
- run_nbody_gpu
- Load a pre-defined spherical spiral grid scaled to a given radius.      The grid
- Dynamical friction
- cuda_primitives.h
- validate_masses
- _io.py
- TestBoundCenterPhi
- AGAMA_GPU
- Quadrupole
- vector3
- load_agama_potential
- test_phase1_multipole.py
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
- host_mem
- get_smoothing_lengths
- test_tree.py
- Public API
- _multipole_potential_kernel.cu
- make_uneven_grid
- __init__.py
- fit_double_spheroid_profile
- empirical_density_profile
- Potential
- Chandrasekhar dynamical friction
- vec<3,double>
- vec<2,float>
- empirical_velocity_rms_profile
- CoM galactocentric radius vs time chart
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
4. `CylSplinePotentialGPU` - 41 edges
5. `CompositePotentialGPU` - 40 edges
6. `tree_gravity_gpu()` - 38 edges
7. `ShiftedPotentialGPU` - 36 edges
8. `EvolvingPotentialGPU` - 34 edges
9. `NFWPotentialGPU` - 33 edges
10. `load_agama_potential()` - 33 edges

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

## Communities (264 total, 114 thin omitted)

### Community 0 - "_make_ic"
Cohesion: 0.07
Nodes (42): _com(), _ke(), _make_ic(), _pe(), ndarray, Path, tests/test_physics.py ===================== Physics-level validation tests for n, KE = (1/2) sum_i m_i |v_i|^2. (+34 more)

### Community 1 - "MultipolePotentialGPU"
Cohesion: 0.06
Nodes (29): DiskAnsatzPotentialGPU, LogHaloPotentialGPU, DiskAnsatz separable disk potential: Phi(R,z) = f(r) * H(z)     where r = sqrt(R, LogHalo: Phi = (v0^2/2) * ln(rc^2 + x^2 + y^2/p^2 + z^2/q^2)      Constructor ma, MultipolePotentialGPU, GPU evaluator for an Agama Multipole BFE potential.      Replicates Agama's ``Mu, Scalar args common to all kernels.          All int/double scalars are explicitl, _pts() (+21 more)

### Community 2 - "sph_kernels.py"
Cohesion: 0.13
Nodes (20): _argsort_morton_2d(), _cubic_spline_2d_gpu(), _cubic_spline_2d_scalar(), _get_smoothing_lengths_cpu(), _get_smoothing_lengths_gpu(), ndarray, sph_kernels.py ============== SPH smoothing-length computation and 2-D surface-d, GPU smoothing-length computation via CuPy KDTree (internal). (+12 more)

### Community 3 - "_load.py"
Cohesion: 0.09
Nodes (31): _add_negative_m(), CylSplineCoefs, _detect_expansion_type(), MultipoleCoefs, Path, agama_helper._coefs ~~~~~~~~~~~~~~~~~~~ Structured representations of Agama expa, Return lines from any coef source: file path, HDF5 path, or raw string.      Del, Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown. (+23 more)

### Community 4 - "_nfw_potential"
Cohesion: 0.15
Nodes (14): chandrasekhar_friction(), Chandrasekhar dynamical-friction acceleration at the satellite CoM.      Compute, _nfw_potential(), Constant sigma function for analytic tests., DF force must oppose the velocity., No friction when satellite is at rest., DF acceleration should scale linearly with satellite mass., Validate against a direct evaluation of BT2008 eq. 8.13. (+6 more)

### Community 5 - "test_utils.py"
Cohesion: 0.11
Nodes (21): empirical_circular_velocity_profile(), Generate uniformly random points on the surface of a sphere.      Parameters, r"""Compute the circular velocity profile :math:`v_{\rm circ}(r) = \sqrt{G\,M(<r, uniform_spherical_grid(), plummer_enclosed_mass(), plummer_v_circ(), tests/test_utils.py ===================  Tests for ``nbody_streams.utils`` using, Circular velocity should match analytic Plummer. (+13 more)

### Community 6 - "__init__.py"
Cohesion: 0.07
Nodes (53): ArrayLike, float32, KERNEL_TYPES, nbody_streams.cuda_kernels CUDA kernels for N-body force and potential computati, _compute_forces_cpu(), compute_nbody_forces_cpu(), compute_nbody_forces_gpu(), compute_nbody_potential_cpu() (+45 more)

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
Cohesion: 0.08
Nodes (13): CompositePotentialGPU, _GPUPotBase, Sum of GPU potential components : mirrors agama.Potential composite.      Each c, Build from an agama.Potential by materialising it as a single Multipole, Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Agama-compatible eval : see ``MultipolePotentialGPU.eval`` for details., Shifted modifier: evaluates ``inner`` at ``xyz - center(t)``.      Parameters, Scaled modifier: ``Phi_s(x,t) = a(t)*s(t) * Phi(x*s(t))`` where ``s(t) = 1/scale (+5 more)

### Community 12 - "run_simulation"
Cohesion: 0.06
Nodes (31): Backend dispatch table, Backend-specific kwargs, Contents, Definition, Dynamical friction kwargs, Example, Example: GPU tree backend, Example: multi-species dark matter + stars (+23 more)

### Community 13 - "generate_stream_coords"
Cohesion: 0.17
Nodes (12): nbody_streams.coords - coordinate transformations.  Cartesian/spherical/cylindri, generate_stream_coords(), get_observed_stream_coords(), ndarray, Stream coordinate generation and observable conversions.  Generate stream-aligne, Project galactocentric positions or phase-space vectors into a     pre-computed, Convert galactocentric phase space into stream-aligned coordinates     (phi1, ph, Convert galactocentric phase-space coordinates to observed sky     coordinates ( (+4 more)

### Community 14 - "_AnalyticBase"
Cohesion: 0.14
Nodes (11): _AnalyticBase, AnalyticPotentialGPU(), _prep_xyz(), ndarray, _analytic_potentials.py ~~~~~~~~~~~~~~~~~~~~~~~ Phase 2: GPU-accelerated analyti, Uniform (spatially constant) acceleration: F = (ax, ay, az).     Phi(x,y,z) = -a, Factory function matching Agama's constructor syntax::          pot = AnalyticPo, Agama-compatible eval --- returns any combination of potential, acceleration, (+3 more)

### Community 15 - "tree_gravity_gpu"
Cohesion: 0.11
Nodes (39): Compute gravitational accelerations and potential using the GPU tree code     (B, tree_gravity_gpu(), generate_plummer(), Net force = sum(m_i * a_i) should be ~0 for an isolated system., Net torque = sum(r x (m*a)) should be small., Compare 0.5*sum(m*phi_tree) against 0.5*sum(m*phi_direct)., For a spherically symmetric distribution, forces should point inward., Error should not degrade significantly as N grows (same distribution). (+31 more)

### Community 17 - "__init__.py"
Cohesion: 0.12
Nodes (22): generate_lmax_pairs(), Generate (l, m) pairs for a spherical harmonic expansion.      Parameters     --, create_fire_evolving_ini(), Path, Write an Agama ``Evolving`` ``.ini`` file from a FIRE simulation directory., r"""     Read ``snapshot_times.txt`` from a FIRE simulation directory.      Retu, read_snapshot_times(), create_snapshot_dict() (+14 more)

### Community 18 - "test_phase2_analytic.py"
Cohesion: 0.12
Nodes (13): _extract_int_from_group(), Extract first integer from a group name for numeric sorting (e.g. 'snap_042' → 4, load_agama_evolving_potential(), _parse_evolving_ini(), Path, Build a time-evolving Agama potential from an HDF5 archive or a native     Agama, Parse an Agama ``Evolving`` potential ``.ini`` config file.      Returns     ---, _require_agama() (+5 more)

### Community 19 - "test_advanced.py"
Cohesion: 0.28
Nodes (8): Checks which high-performance features are actually active., Runs a tiny GPU simulation and verifies snapshot output., Quick helper to generate test particles (Plummer-like spread)., Runs a tiny CPU simulation and verifies snapshot output., setup_dummy_data(), test_cpu_integration(), test_environment_summary(), test_gpu_integration()

### Community 21 - "CellData"
Cohesion: 0.18
Nodes (10): collect_leaves(), shuffle_cells(), Treecode<real_t>::buildTree(), computeCellMultipoles(), __out, CellData, __device__, packed_data (+2 more)

### Community 22 - "test_newtons_third_law.py"
Cohesion: 0.50
Nodes (4): _newtons_third_law(), Test that net force is ~0 for isolated system.          Returns:         bool: T, Test all precision modes., test_all_precisions()

### Community 23 - "_build.py"
Cohesion: 0.67
Nodes (3): main(), nbody_streams.tree_gpu._build Console-script entry point: nbody-build-tree  Comp, _tree_gpu_dir()

### Community 24 - ".radial_power"
Cohesion: 0.40
Nodes (3): ndarray, Radial power spectrum for harmonic order *l*.          Parameters         ------, Total power for harmonic order *l* summed over all radial bins.          Paramet

### Community 26 - "`nbody_streams.utils` — Analysis Utilities"
Cohesion: 0.08
Nodes (25): Centre finding, `compute_iterative_boundness` (deprecated), Density-profile fitting, `empirical_circular_velocity_profile`, `empirical_density_profile`, Empirical radial profiles, `empirical_velocity_anisotropy_profile`, `empirical_velocity_dispersion_profile` (+17 more)

### Community 28 - "`nbody_streams.coords` — Coordinate Transforms"
Cohesion: 0.25
Nodes (8): `convert_coords`, `convert_to_vel_los`, `convert_vectors`, Coordinate conventions, `generate_stream_coords`, `get_observed_stream_coords`, `nbody_streams.coords` — Coordinate Transforms, `to_stream_coords`

### Community 31 - "Treecode.h"
Cohesion: 0.09
Nodes (26): int3, double4, Treecode<real_t>::computeForces(), addMonopole(), addQuadrupole(), __device__, double4, real_t (+18 more)

### Community 32 - "Treecode"
Cohesion: 0.04
Nodes (53): Treecode, BUILD_MAX_WORK, BUILD_PENDING_WORK_BYTES, buildTree, cell_max, computeForces, computeMultipoles, d_build_octCounter (+45 more)

### Community 35 - ".dark"
Cohesion: 0.11
Nodes (13): Convenience constructor for dark-matter particles., _plummer_like(), run_simulation emits PerformanceWarning for large CPU direct N., Very small Plummer-like ICs for integration tests., Without species kwarg, old dark/star HDF5 schema is written., Verify that debug_energy=True works without errors on all dispatcher paths     t, debug_energy=False (default) must not print energy lines., run_simulation must raise TypeError on unknown kwargs for CPU path. (+5 more)

### Community 36 - "Species"
Cohesion: 0.13
Nodes (7): Definition of a particle species in an N-body simulation.      Parameters     --, Convenience constructor for stellar particles., Species, TestParticleReaderMultiSpecies, TestSpeciesDataclass, Multi-species runs must conserve the *total* combined energy and momentum     ac, TestConservationMultiSpecies

### Community 37 - "_potential.py"
Cohesion: 0.06
Nodes (47): _apply_modifiers(), _build_dehnen_gpu(), _build_disk_gpu(), _build_king_gpu(), _build_multipole_data(), _build_single(), _build_spheroid_gpu(), _coerce() (+39 more)

### Community 38 - "nbody_io.py"
Cohesion: 0.15
Nodes (10): _aggregate_data_chunk(), plot_mollweide(), Mollweide projection of a 3-D particle field using Healpix.      Supports automa, Helper for parallel pixel aggregation (pandas path)., _close_figures(), orbit_data(), Tests for nbody_streams.viz -- visualization functions.  All tests use the Agg b, Close all matplotlib figures after each test. (+2 more)

### Community 40 - "`nbody_streams.agama_helper`"
Cohesion: 0.05
Nodes (43): 1 — Fit and save coefficient files, 2 — Pack text files into HDF5, 3 — Inspect and modify coefficients, 4 — Load Agama potential, Accuracy (vs Agama CPU), API (matches `agama.Potential`), Center parameter, Coefficient dataclasses (+35 more)

### Community 41 - "Changelog"
Cohesion: 0.05
Nodes (40): [1.0.0] - 2025-XX-XX, [1.1.0] - 2026-02-08, [1.2.0] - 2026-02-14, [1.3.0] - 2026-02-24, [2.0.0] - 2026-02-28, [2.1.0] - 2026-03-07, [2.2.0] - 2026-03-24, [2.3.0] - 2026-05-12 (+32 more)

### Community 42 - "test_multi_species.py"
Cohesion: 0.10
Nodes (23): _is_uniform(), Check whether all elements of *arr* are equal to within relative tolerance., ndarray, nbody_streams.sim Unified high-level simulation entry point for multi-species N-, Run a direct N-body simulation with one or more particle species.      This is t, run_simulation(), _build_particle_arrays(), _emit_performance_warnings() (+15 more)

### Community 43 - "Box"
Cohesion: 0.08
Nodes (22): double2, Box, centre, __device__, hsize, double4, float3, float4 (+14 more)

### Community 44 - "_force.py"
Cohesion: 0.09
Nodes (24): CDLL, _coerce_eps(), cuda_alive(), _find_cudart(), _gpu_ptr(), _max_n(), ndarray, _force.py  --  ctypes interface to the GPU tree-code gravity shared library.  Su (+16 more)

### Community 45 - "buildTree.cu"
Cohesion: 0.09
Nodes (28): dim3, buildOctreeHost(), compute_level_begIdx(), computeBoundingBox(), computeGridAndBlockSize(), countAtRootNode(), __device__, float2 (+20 more)

### Community 46 - "NFWPotentialGPU"
Cohesion: 0.11
Nodes (13): NFWPotentialGPU, Navarro-Frenk-White potential optimized with CuPy ElementwiseKernels., Write a potential config file with the given suffix (any extension works)., _is_potential_ini detects multi-section content regardless of extension., [Potential halo] + [Potential disk] headers with a .dat extension., [Potential 0] + [Potential 1] + [Potential 2] (numbered headers)., Plain [Potential] header → single GPU pot, not composite., TYPE = NFW (uppercase) should be recognised. (+5 more)

### Community 47 - "_random_xv"
Cohesion: 0.13
Nodes (12): Write a snapshot compatible with :class:`ParticleReader`.      Two modes     ---, _save_snapshot(), Validate *species* list against *phase_space* shape.      Checks     ------, _validate_species(), ndarray, _random_xv(), Write an HDF5 file in the old two-species format., Old file with only dark particles (no star group). (+4 more)

### Community 48 - "API"
Cohesion: 0.07
Nodes (29): AGAMA potential helper (`agama_helper`), Analysis utilities (`utils`), API, Caveats and known limitations, Comparison: direct N-body vs tree methods, Coordinates (`coords`), Dynamical friction with external potentials, Fast stream generation (`fast_sims`) (+21 more)

### Community 50 - "Particle4"
Cohesion: 0.13
Nodes (21): __forceinline__ Box<T> ChildBox(), __forceinline__ int Particle4<double>::get_idx(), __forceinline__ int Particle4<double>::get_oct(), __forceinline__ int Particle4<double>::set_idx(), __forceinline__ int Particle4<double>::set_oct(), __forceinline__ int Particle4<float>::get_idx(), __forceinline__ int Particle4<float>::get_oct(), __forceinline__ int Particle4<float>::set_idx() (+13 more)

### Community 51 - "CompositePotentialGPU"
Cohesion: 0.07
Nodes (16): HernquistPotentialGPU, IsochronePotentialGPU, MiyamotoNagaiPotentialGPU, Hernquist: Phi(r) = -GM / (r + a)      Constructor: HernquistPotentialGPU(mass=1, Isochrone: Phi(r) = -GM / (b + sqrt(r^2 + b^2))      Constructor: IsochronePoten, Miyamoto-Nagai disk: Phi = -GM / sqrt(R^2 + (sqrt(z^2+b^2)+a)^2)     where R^2 =, _frel(), GPUTimer (+8 more)

### Community 52 - "treewalk_warp"
Cohesion: 0.16
Nodes (21): approxAcc(), __device__, float2, float3, float4, __forceinline__, __global__, int2 (+13 more)

### Community 53 - "treeGPU_interface.cu"
Cohesion: 0.15
Nodes (24): float4, __global__, real_t, k_extract_unsorted(), k_load_pos_mass(), k_load_pos_mass_eps(), Res, w (+16 more)

### Community 55 - "Fit an adaptive ellipsoid to a particle distribution and compute shape diagnosti"
Cohesion: 0.40
Nodes (5): compute_nbody_forces_cpu, Example, Notes, Parameters, Returns

### Community 56 - "Generate uniformly random points on the surface of a sphere.      Parameters"
Cohesion: 0.50
Nodes (4): `'jeans'` (default), `'local_circular'`, `'quasispherical'`, sigma(r) computation

### Community 57 - "Generate num_pts points on a sphere using the Fibonacci spiral.      Points are"
Cohesion: 0.06
Nodes (30): _load_restart(), _make_times_ns(), ParticleReader, ndarray, Path, nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr, Worker for N-species parallel extraction.      Args (tuple):         snap_index, A class to read N-body simulation data from one or more HDF5 files.      This re (+22 more)

### Community 58 - "Implementation internals"
Cohesion: 0.08
Nodes (24): Availability flag, Build system details, Building the shared library, ctypes interface (`_force.py`), `cuda_alive`, Data structures, Float32 note, `G_DEFAULT` (+16 more)

### Community 62 - "test_chandrasekhar.py"
Cohesion: 0.24
Nodes (7): Iterative shrinking-sphere centre-of-mass estimator.      At each iteration the, _shrinking_sphere_com(), CoM should converge close to the true cloud centre., All particles at same position → CoM = that position exactly., With one dominant dense clump + sparse background, CoM should         land in th, r_sphere must be positive and smaller than the full cloud extent., TestShrinkingSphereCom

### Community 67 - "make_df_force_extra"
Cohesion: 0.14
Nodes (11): _is_gpu_potential(), make_df_force_extra(), Build a ``force_extra`` closure that applies Chandrasekhar dynamical     frictio, True when *pot* is from the PotentialGPU family (nbody_streams.agama_helper)., DF is a rigid-body force — all particles within the core get the same         ac, Net DF acceleration should point opposite to bulk CoM velocity., Between correction steps, CoM should be predicted, not recalculated., Without Agama the factory should raise ImportError. (+3 more)

### Community 72 - "transforms.py"
Cohesion: 0.17
Nodes (22): _as_3vec(), _cart_to_cyl(), _cart_to_sph(), _cyl_to_cart(), _cyl_to_sph(), _propagate_nans(), ndarray, Coordinate system and vector field conversions.  Cartesian <-> spherical <-> cyl (+14 more)

### Community 73 - "_jeans_sigma_r"
Cohesion: 0.09
Nodes (25): compute_sigma_r(), _jeans_sigma_r(), ndarray, nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica, Local circular-speed approximation to the 1-D velocity dispersion.      Uses the, Compute a radial velocity-dispersion profile from an Agama potential.      Param, Convert a CuPy or NumPy array to a plain NumPy array (zero-copy when     possibl, Jeans-equation estimate of the isotropic 1-D velocity dispersion.      Uses the (+17 more)

### Community 75 - "main.py"
Cohesion: 0.15
Nodes (20): _calculate_particle_distances_sq(), _calculate_Rsphall_and_extract(), _compute_weighted_structure_tensor(), _density_peak_center(), fibonacci_sphere_grid(), find_center(), find_center_position(), fit_iterative_ellipsoid() (+12 more)

### Community 77 - "plot_density"
Cohesion: 0.15
Nodes (8): Axes, AxesImage, Colormap, plot_density(), Any, Generate a projected density image (imshow) from particle data.      Accepts par, imshow extent should be [-gridsize/2, gridsize/2] on both axes., TestPlotDensity

### Community 81 - "run_nbody_gpu"
Cohesion: 0.10
Nodes (21): Constants, Contents, Example: basic self-gravity, Example: direct method with specific kernel, Example: resume from crash, Example: tree method, Example: with external potential, GPU memory management (+13 more)

### Community 85 - "Dynamical friction"
Cohesion: 0.11
Nodes (19): Caveats, Centre-of-mass detection, Chandrasekhar formula (BT2008 eq. 8.13), Contents, Core-stalling suppression, Coulomb logarithm, df_* kwargs reference, Dynamical friction (+11 more)

### Community 86 - "cuda_primitives.h"
Cohesion: 0.23
Nodes (18): buildOctant(), addBoxSize(), atomicAdd_double(), __forceinline__(), __device__, real_t, inclusive_scan_warp(), inclusive_segscan_warp() (+10 more)

### Community 88 - "validate_masses"
Cohesion: 0.15
Nodes (18): empirical_velocity_anisotropy_profile(), fit_dehnen_profile(), fit_plummer_profile(), r"""Compute the velocity anisotropy parameter :math:`\beta(r)`.      .. math::, r"""Fit a triaxial Dehnen profile by mapping to ellipsoidal radius.      Compute, r"""Fit a spherical Plummer profile to particle data.      Bins particles into l, ndarray, nbody_streams.utils._validation ================================  Shared input-v (+10 more)

### Community 89 - "_io.py"
Cohesion: 0.16
Nodes (17): _cleanup_tmp_file(), _get_fast_tmp_dir(), Any, Path, agama_helper._io ~~~~~~~~~~~~~~~~ HDF5 archive I/O, fast temporary-file helpers,, Store an Agama coefficient text string in an HDF5 archive.      The archive is c, Batch-write Agama coefficient files for multiple snapshots into HDF5 archives., Return fastest writable temp dir: /dev/shm (RAM-backed on Linux) or system tmp. (+9 more)

### Community 90 - "TestBoundCenterPhi"
Cohesion: 0.16
Nodes (10): _bound_center_phi(), Phi-energy iterative bound-particle centre.      Finds the centre of the gravita, Return a cluster at r0 on the x-axis with a bulk y-velocity., Simple pairwise Plummer phi (exact for small N test)., Returned CoM should be close to the cluster centre., For a self-gravitating cluster with low velocity dispersion, bound         parti, make_df_force_extra closure should accept and use phi kwarg., Unbound particles (phi + 0.5 v_rel^2 >= 0) should get zero DF. (+2 more)

### Community 92 - "AGAMA_GPU"
Cohesion: 0.12
Nodes (15): Accuracy, AGAMA_GPU, Analytic — `_analytic_potentials.py`, API, Composite types, CylSpl BFE - `CylSplinePotentialGPU`, File layout, Known gotchas (+7 more)

### Community 93 - "Quadrupole"
Cohesion: 0.20
Nodes (7): real_t, Quadrupole, __device__, q0, q1, real2_t, real4_t

### Community 94 - "vector3"
Cohesion: 0.24
Nodes (8): maxeach(), mineach(), operator *(), vector3, x, y, z, REAL

### Community 95 - "load_agama_potential"
Cohesion: 0.20
Nodes (12): load_agama_potential(), Load a single-snapshot Agama potential from any coefficient source.      Accepts, gpu=True from a MultipoleCoefs dataclass → MultipolePotentialGPU., Default gpu=False returns agama.Potential, not a GPU object., make_axisym(), make_halo(), test_zero_pruning.py ~~~~~~~~~~~~~~~~~~~~ Benchmarks zero-coefficient pruning (P, Full FIRE halo: all (l,m) up to lmax, non-zero for most. (+4 more)

### Community 96 - "test_phase1_multipole.py"
Cohesion: 0.29
Nodes (14): _make_from_agama(), _make_from_coef_file(), _pts(), test_phase1_multipole.py ~~~~~~~~~~~~~~~~~~~~~~~~ Tests MultipolePotentialGPU ag, Load coef file, cap at lmax, return (agama_pot, gpu_pot)., Export agama_pot as Multipole BFE, load into GPU., _rel_force(), _rel_phi() (+6 more)

### Community 99 - "plots.py"
Cohesion: 0.19
Nodes (12): Figure, nbody_streams.viz - visualization and plotting.  Projected density maps, Mollwei, _extract_particles_at_step(), _gauss_filter_surf_dens(), _generate_ticks(), plot_stream_sky(), ndarray, Plotting functions for stream and N-body visualization.  Public API (flat via `` (+4 more)

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
Cohesion: 0.35
Nodes (10): _pts(), test_phase3_cylspline.py ~~~~~~~~~~~~~~~~~~~~~~~~ Tests CylSplinePotentialGPU ag, Random Cartesian points inside a sphere, avoiding z-axis., _rel_deriv(), _rel_force(), _rel_phi(), test_accuracy(), test_mmax_sweep() (+2 more)

### Community 108 - "cuda_mem"
Cohesion: 0.25
Nodes (4): cuda_mem, n, ptr, T

### Community 109 - "plot_stream_evolution"
Cohesion: 0.31
Nodes (3): plot_stream_evolution(), Three-panel evolution plot: galactocentric distance, bound fraction     (or 3-D, TestPlotStreamEvolution

### Community 110 - "host_mem"
Cohesion: 0.27
Nodes (3): host_mem, n, ptr

### Community 111 - "get_smoothing_lengths"
Cohesion: 0.29
Nodes (5): get_smoothing_lengths(), Compute SPH smoothing lengths as the distance to the *k_neighbors*-th     neares, Particles in a dense cluster should get smaller h than isolated ones., 3-D pos should be accepted (D is not restricted to 2)., TestGetSmoothingLengths

### Community 112 - "test_tree.py"
Cohesion: 0.33
Nodes (8): error_report(), generate_plummer(), Run with a small N and check that forces point inward., Plummer model ICs matching the C++ binary (double precision + R<100 rejection)., Print relative-error statistics between ref and test arrays., test_accuracy(), test_performance(), test_sanity()

### Community 113 - "Public API"
Cohesion: 0.25
Nodes (8): `create_ic_particle_spray_chen2025`, `create_ic_particle_spray_fardal2015`, `create_particle_spray_stream`, Dynamical friction, `nbody_streams.fast_sims` — Fast Stream Generation, Perturber potential, Public API, `run_restricted_nbody`

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

### Community 124 - "vec<3,double>"
Cohesion: 0.50
Nodes (3): double3, vec<3,double>, __device__

### Community 125 - "vec<2,float>"
Cohesion: 0.50
Nodes (3): float2, vec<2,float>, __device__

### Community 126 - "empirical_velocity_rms_profile"
Cohesion: 0.50
Nodes (4): empirical_velocity_rms_profile(), r"""Compute the root-mean-square velocity profile :math:`v_{\rm rms}(r)`.      P, RMS velocity should be finite and positive., test_empirical_velocity_rms()

### Community 127 - "CoM galactocentric radius vs time chart"
Cohesion: 0.83
Nodes (4): bound particle fraction vs time chart, CoM galactocentric radius vs time chart, final particle density 2x2 comparison, Chandrasekhar dynamical friction

### Community 129 - "nbody_streams — Documentation"
Cohesion: 0.67
Nodes (3): Building docs (future), Module reference, nbody_streams — Documentation

### Community 145 - "`nbody_streams.fields`"
Cohesion: 0.10
Nodes (20): compute_nbody_forces_gpu, compute_nbody_potential_cpu, compute_nbody_potential_gpu, Contents, Example, Example, Example, get_gpu_info (+12 more)

### Community 146 - "`nbody_streams.nbody_io` — HDF5 Snapshot I/O"
Cohesion: 0.20
Nodes (10): `extract_orbits`, HDF5 snapshot format, Internal I/O functions, `_load_restart`, `nbody_streams.nbody_io` — HDF5 Snapshot I/O, `ParticleReader`, `read_snapshot`, `_save_restart` (+2 more)

## Knowledge Gaps
- **465 isolated node(s):** `vec`, `__device__`, `__device__`, `__device__`, `__device__` (+460 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **114 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_GPUPotBase` connect `ShiftedPotentialGPU` to `MultipolePotentialGPU`, `_load.py`, `make_df_force_extra`, `_potential.py`, `EvolvingPotentialGPU`, `__init__.py`, `_jeans_sigma_r`, `CylSplinePotentialGPU`, `_AnalyticBase`, `NFWPotentialGPU`, `r"""Fit a spheroid (Zhao / generalised double-power-law) density profile.      I`, `CompositePotentialGPU`, `Ellipsoidal radius and particle selection (numba-accelerated).      Computes the`, `Generate num_pts points on a sphere using the Fibonacci spiral.      Points are`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `Species` connect `Species` to `_make_ic`, `.dark`, `__init__.py`, `test_multi_species.py`, `_random_xv`, `Generate num_pts points on a sphere using the Fibonacci spiral.      Points are`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `PotentialGPU()` connect `_potential.py` to `MultipolePotentialGPU`, `_load.py`, `ShiftedPotentialGPU`, `test_phase3_cylspline.py`, `NFWPotentialGPU`, `__init__.py`, `CompositePotentialGPU`, `load_agama_potential`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `Species` (e.g. with `ParticleReader` and `_StepWatchdog`) actually correct?**
  _`Species` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `MultipolePotentialGPU` (e.g. with `DehnenSphericalPotentialGPU` and `DiskAnsatzPotentialGPU`) actually correct?**
  _`MultipolePotentialGPU` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `CylSplinePotentialGPU` (e.g. with `DehnenSphericalPotentialGPU` and `DiskAnsatzPotentialGPU`) actually correct?**
  _`CylSplinePotentialGPU` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `CompositePotentialGPU` (e.g. with `DehnenSphericalPotentialGPU` and `DiskAnsatzPotentialGPU`) actually correct?**
  _`CompositePotentialGPU` has 23 INFERRED edges - model-reasoned connections that need verification._