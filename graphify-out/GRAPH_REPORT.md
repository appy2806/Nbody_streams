# Graph Report - .  (2026-04-29)

## Corpus Check
- 89 files · ~164,116 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1228 nodes · 2068 edges · 53 communities detected
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 527 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_CPU Sim Core & IO|CPU Sim Core & IO]]
- [[_COMMUNITY_Dynamical Friction|Dynamical Friction]]
- [[_COMMUNITY_Main Entry & Validation|Main Entry & Validation]]
- [[_COMMUNITY_Package Top-Level & ParticleReader|Package Top-Level & ParticleReader]]
- [[_COMMUNITY_CylSpline Serialization|CylSpline Serialization]]
- [[_COMMUNITY_Fast Sims Spray & Perturber|Fast Sims Spray & Perturber]]
- [[_COMMUNITY_Direct Sum Force Computation|Direct Sum Force Computation]]
- [[_COMMUNITY_Snapshot IO & Species|Snapshot IO & Species]]
- [[_COMMUNITY_Agama Multipole Coefficients|Agama Multipole Coefficients]]
- [[_COMMUNITY_Coordinate Transforms|Coordinate Transforms]]
- [[_COMMUNITY_Stream Spray Simulation|Stream Spray Simulation]]
- [[_COMMUNITY_Visualization & Plots|Visualization & Plots]]
- [[_COMMUNITY_CUDA Kernels|CUDA Kernels]]
- [[_COMMUNITY_Agama Potential Loading|Agama Potential Loading]]
- [[_COMMUNITY_Stream Coordinate Generation|Stream Coordinate Generation]]
- [[_COMMUNITY_Test Utilities & Plummer IC|Test Utilities & Plummer IC]]
- [[_COMMUNITY_FIRE Simulation Helpers|FIRE Simulation Helpers]]
- [[_COMMUNITY_Shrinking Sphere CoM|Shrinking Sphere CoM]]
- [[_COMMUNITY_CUDA Primitives|CUDA Primitives]]
- [[_COMMUNITY_Potential Fitting|Potential Fitting]]
- [[_COMMUNITY_ParticleReader Orbit Extraction|ParticleReader Orbit Extraction]]
- [[_COMMUNITY_GPUCPU Integration Tests|GPU/CPU Integration Tests]]
- [[_COMMUNITY_Snapshot Write Helpers|Snapshot Write Helpers]]
- [[_COMMUNITY_CUDA Memory Management|CUDA Memory Management]]
- [[_COMMUNITY_Newton's Third Law Tests|Newton's Third Law Tests]]
- [[_COMMUNITY_GPU Force Computation|GPU Force Computation]]
- [[_COMMUNITY_Legacy GPU Force Code|Legacy GPU Force Code]]
- [[_COMMUNITY_Tree GPU Build System|Tree GPU Build System]]
- [[_COMMUNITY_Empirical Profile Utilities|Empirical Profile Utilities]]
- [[_COMMUNITY_Profile Fitting|Profile Fitting]]
- [[_COMMUNITY_Stream Coordinate API Docs|Stream Coordinate API Docs]]
- [[_COMMUNITY_Agama HDF5 Coefficient IO|Agama HDF5 Coefficient IO]]
- [[_COMMUNITY_Agama Evolving Potential|Agama Evolving Potential]]
- [[_COMMUNITY_GPU Tree vs Direct Test|GPU Tree vs Direct Test]]
- [[_COMMUNITY_Unit System Constants|Unit System Constants]]
- [[_COMMUNITY_Potential Fitting Docs|Potential Fitting Docs]]
- [[_COMMUNITY_Dark Matter Constructor|Dark Matter Constructor]]
- [[_COMMUNITY_Stellar Particle Constructor|Stellar Particle Constructor]]
- [[_COMMUNITY_Multipole lmax|Multipole lmax]]
- [[_COMMUNITY_Multipole l Values|Multipole l Values]]
- [[_COMMUNITY_Multipole m Values|Multipole m Values]]
- [[_COMMUNITY_Jacobi Orbit IC|Jacobi Orbit IC]]
- [[_COMMUNITY_Matplotlib Dependency|Matplotlib Dependency]]
- [[_COMMUNITY_Graphify Rules|Graphify Rules]]
- [[_COMMUNITY_NBODY_UNITS Constant|NBODY_UNITS Constant]]
- [[_COMMUNITY_Double Spheroid Profile|Double Spheroid Profile]]
- [[_COMMUNITY_Iterative Ellipsoid Fit|Iterative Ellipsoid Fit]]
- [[_COMMUNITY_CUDA Alive Check|CUDA Alive Check]]
- [[_COMMUNITY_CPU Potential Computation|CPU Potential Computation]]
- [[_COMMUNITY_GPU Info|GPU Info]]
- [[_COMMUNITY_Restart Load Internal|Restart Load Internal]]
- [[_COMMUNITY_Vector Coordinate Convert|Vector Coordinate Convert]]
- [[_COMMUNITY_LOS Velocity Convert|LOS Velocity Convert]]

## God Nodes (most connected - your core abstractions)
1. `Species` - 111 edges
2. `PerformanceWarning` - 33 edges
3. `max()` - 32 edges
4. `_nfw_potential()` - 30 edges
5. `min()` - 24 edges
6. `_random_xv()` - 23 edges
7. `make_df_force_extra()` - 21 edges
8. `_make_ic()` - 21 edges
9. `tree_gravity_gpu()` - 19 edges
10. `plot_density()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Satellite A final surface density (M_sat=5e10 Msun, post-inspiral)` --references--> `Chandrasekhar dynamical friction`  [INFERRED]
  output/df_tutorial/satA_final_density.png → docs/dynamical_friction.md
- `run_simulation must raise TypeError on unknown kwargs for GPU direct path.` --uses--> `Species`  [INFERRED]
  tests/test_multi_species.py → nbody_streams/species.py
- `final particle density 2x2 comparison` --references--> `Chandrasekhar dynamical friction`  [EXTRACTED]
  output/final_density_2x2.png → README.md
- `bound particle fraction vs time chart` --references--> `Chandrasekhar dynamical friction`  [EXTRACTED]
  output/bound_fraction_vs_time.png → README.md
- `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles` --references--> `Backend dispatch table`  [INFERRED]
  plots/acceleration_timings.png → docs/main.md

## Communities

### Community 0 - "CPU Sim Core & IO"
Cohesion: 0.03
Nodes (85): compute_nbody_forces_cpu(), Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba., _finalize_snapshot_times(), _load_restart(), Worker executed in a separate process. Reads one snapshot from disk and writes, Fast append-only version. Sort at the end with _finalize_snapshot_times()., Call this ONCE at the end of run_nbody_gpu., Save restart file for crash recovery.          Parameters     ----------     pha (+77 more)

### Community 1 - "Dynamical Friction"
Cohesion: 0.03
Nodes (60): _bound_center_phi(), chandrasekhar_friction(), compute_sigma_r(), _jeans_sigma_r(), make_df_force_extra(), nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica, Local circular-speed approximation to the 1-D velocity dispersion.      Uses the, Compute a radial velocity-dispersion profile from an Agama potential.      Param (+52 more)

### Community 2 - "Main Entry & Validation"
Cohesion: 0.03
Nodes (87): _emit_performance_warnings(), Emit :class:`PerformanceWarning` when *N_total* exceeds recommended     threshol, Tree method on CPU never warns for moderate N., TestPerformanceWarnings, plummer_density(), plummer_enclosed_mass(), plummer_v_circ(), tests/test_utils.py ===================  Tests for ``nbody_streams.utils`` using (+79 more)

### Community 3 - "Package Top-Level & ParticleReader"
Cohesion: 0.05
Nodes (54): nbody_streams - lightweight direct N-body utilities., _make_times_ns(), ParticleReader, A class to read N-body simulation data from one or more HDF5 files.      This re, Convert a raw np.loadtxt array into SimpleNamespace(snap=int_array, time=float_a, Read simulation properties from the first HDF5 file.          Supports two HDF5, Scan HDF5 files and map snapshot index -> file path for fast lookups.          S, Read a single snapshot by index or physical time.          Parameters         -- (+46 more)

### Community 4 - "CylSpline Serialization"
Cohesion: 0.03
Nodes (56): generate_lmax_pairs(), lmax(), Serialise back to the Agama CylSpline text format.          Returns         ----, Generate (l, m) pairs for a spherical harmonic expansion.      Parameters     --, generate_plummer(), Plummer model ICs matching the C++ binary (double precision + R<100 rejection)., generate_plummer(), Plummer model ICs matching the C++ binary (double precision + R<100 rejection). (+48 more)

### Community 5 - "Fast Sims Spray & Perturber"
Cohesion: 0.03
Nodes (81): _chandrasekhar module, force_extra hook, make_df_force_extra, Unreleased feat-dynamicFric, add_perturber subhalo, create_ic_particle_spray_chen2025, create_particle_spray_stream, dynamical friction progenitor orbit (+73 more)

### Community 6 - "Direct Sum Force Computation"
Cohesion: 0.04
Nodes (72): _compute_forces_cpu(), compute_nbody_forces_cpu(), compute_nbody_forces_gpu(), compute_nbody_potential_cpu(), compute_nbody_potential_gpu(), _compute_potential_cpu(), _get_force_kernel(), get_gpu_info() (+64 more)

### Community 7 - "Snapshot IO & Species"
Cohesion: 0.05
Nodes (35): Write snapshot(s) compatible with ParticleReader.      Modes:       - single_fil, _save_snapshot(), PerformanceWarning, Validate *species* list against *phase_space* shape.      Checks     ------, Split a combined ``(N_total, 6)`` phase-space array into per-species slices., Warning emitted when particle count exceeds recommended thresholds., _split_by_species(), _validate_species() (+27 more)

### Community 8 - "Agama Multipole Coefficients"
Cohesion: 0.05
Nodes (59): _add_negative_m(), CylSplineCoefs, _detect_expansion_type(), MultipoleCoefs, agama_helper._coefs ~~~~~~~~~~~~~~~~~~~ Structured representations of Agama expa, Return lines from any coef source: file path, HDF5 path, or raw string.      Del, Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown., Structured representation of a Multipole (spherical harmonic BFE) potential. (+51 more)

### Community 9 - "Coordinate Transforms"
Cohesion: 0.05
Nodes (42): _as_3vec(), _cart_to_cyl(), _cart_to_sph(), convert_coords(), convert_to_vel_los(), convert_vectors(), _cyl_to_cart(), _cyl_to_sph() (+34 more)

### Community 10 - "Stream Spray Simulation"
Cohesion: 0.05
Nodes (38): _compute_vel_disp_from_Potential(), _create_perturber_potential(), _dynamical_friction_acceleration(), _find_prog_pot_Nparticles(), _get_prog_GalaxyModel(), _integrate_orbit_with_dynamical_friction(), Private helpers shared by fast_sims methods.  Functions here are implementation, Integrate a satellite orbit with optional dynamical friction.      Dynamical fri (+30 more)

### Community 11 - "Visualization & Plots"
Cohesion: 0.08
Nodes (17): imshow extent should be [-gridsize/2, gridsize/2] on both axes., TestPlotDensity, TestPlotStreamEvolution, nbody_streams.viz - visualization and plotting.  Projected density maps, Mollwei, _aggregate_data_chunk(), _extract_particles_at_step(), _gauss_filter_surf_dens(), _generate_ticks() (+9 more)

### Community 12 - "CUDA Kernels"
Cohesion: 0.1
Nodes (27): nbody_streams.cuda_kernels CUDA kernels for N-body force and potential computati, _compute_forces_cpu(), compute_nbody_forces_cpu(), compute_nbody_forces_gpu(), compute_nbody_potential_cpu(), compute_nbody_potential_gpu(), _compute_potential_cpu(), _get_force_kernel() (+19 more)

### Community 13 - "Agama Potential Loading"
Cohesion: 0.11
Nodes (29): CylSplineCoefs, load_agama_potential, load_fire_pot, MultipoleCoefs, read_coefs, convert_coords, _bound_center_phi, Chandrasekhar dynamical friction (+21 more)

### Community 14 - "Stream Coordinate Generation"
Cohesion: 0.14
Nodes (13): nbody_streams.coords - coordinate transformations.  Cartesian/spherical/cylindri, generate_stream_coords(), get_observed_stream_coords(), Stream coordinate generation and observable conversions.  Generate stream-aligne, Project galactocentric positions or phase-space vectors into a     pre-computed, Convert galactocentric phase space into stream-aligned coordinates     (phi1, ph, Convert galactocentric phase-space coordinates to observed sky     coordinates (, to_stream_coords() (+5 more)

### Community 15 - "Test Utilities & Plummer IC"
Cohesion: 0.29
Nodes (17): check_nan_inf(), direct_sum_max_eps(), generate_plummer(), hdr(), Return (result, elapsed_ms) with a surrounding device sync., Small-N O(N^2) direct-sum with tree_gpu max softening convention:       eps2_ij, rel_err_stats(), subhdr() (+9 more)

### Community 16 - "FIRE Simulation Helpers"
Cohesion: 0.12
Nodes (18): CylSplineCoefs dataclass, FIRE simulation helpers, libtreeGPU.so shared library, load_agama_evolving_potential, load_agama_potential, MultipoleCoefs dataclass, nbody_streams Changelog, read_coefs unified API (+10 more)

### Community 17 - "Shrinking Sphere CoM"
Cohesion: 0.24
Nodes (7): Iterative shrinking-sphere centre-of-mass estimator.      At each iteration the, _shrinking_sphere_com(), CoM should converge close to the true cloud centre., All particles at same position → CoM = that position exactly., With one dominant dense clump + sparse background, CoM should         land in th, r_sphere must be positive and smaller than the full cloud extent., TestShrinkingSphereCom

### Community 18 - "CUDA Primitives"
Cohesion: 0.24
Nodes (5): inclusive_segscan_warp(), lanemask_le(), lanemask_lt(), warpBinExclusiveScan(), warpBinExclusiveScan1()

### Community 19 - "Potential Fitting"
Cohesion: 0.2
Nodes (10): create_snapshot_dict(), fit_potential(), agama_helper._fit ~~~~~~~~~~~~~~~~~ Fit Agama Multipole and CylSpline potential, Fit Agama Multipole and CylSpline potentials from a multi-species snapshot., Sample *n* positions from a spherically declining density profile., Sample *n* positions from a thin exponential disk., Pack particle arrays into a FIRE-like snapshot dictionary.      Creates the mini, _require_agama() (+2 more)

### Community 20 - "ParticleReader Orbit Extraction"
Cohesion: 0.22
Nodes (6): ParticleReader, Read simulation properties from the first HDF5 file.         Robust to missing ', Scan HDF5 files and map snapshot index -> file path for fast lookups.          S, Read a single snapshot by index or physical time.          Parameters         --, Extract orbits for selected particle types across all available snapshots., A class to read N-body simulation data from one or more HDF5 files.      This re

### Community 21 - "GPU/CPU Integration Tests"
Cohesion: 0.28
Nodes (8): Checks which high-performance features are actually active., Runs a tiny GPU simulation and verifies snapshot output., Quick helper to generate test particles (Plummer-like spread)., Runs a tiny CPU simulation and verifies snapshot output., setup_dummy_data(), test_cpu_integration(), test_environment_summary(), test_gpu_integration()

### Community 22 - "Snapshot Write Helpers"
Cohesion: 0.32
Nodes (5): _is_uniform(), Check whether all elements of *arr* are equal to within relative tolerance., Write a snapshot compatible with :class:`ParticleReader`.      Two modes     ---, _save_snapshot(), TestIsUniform

### Community 23 - "CUDA Memory Management"
Cohesion: 0.38
Nodes (3): alloc(), free(), realloc()

### Community 25 - "Newton's Third Law Tests"
Cohesion: 0.5
Nodes (4): _newtons_third_law(), Test that net force is ~0 for isolated system.          Returns:         bool: T, Test all precision modes., test_all_precisions()

### Community 26 - "GPU Force Computation"
Cohesion: 0.4
Nodes (4): compute_nbody_forces_gpu(), get_gpu_info(), Compute direct N-body gravitational forces using GPU acceleration.          Calc, Get information about available GPU(s).          Returns     -------     info :

### Community 27 - "Legacy GPU Force Code"
Cohesion: 0.4
Nodes (4): compute_nbody_forces_gpu(), get_gpu_info(), Compute direct N-body gravitational forces using GPU acceleration.          Calc, Get information about available GPU(s).          Returns     -------     info :

### Community 28 - "Tree GPU Build System"
Cohesion: 0.67
Nodes (3): main(), nbody_streams.tree_gpu._build Console-script entry point: nbody-build-tree  Comp, _tree_gpu_dir()

### Community 30 - "Empirical Profile Utilities"
Cohesion: 0.5
Nodes (4): empirical_circular_velocity_profile, empirical_density_profile, empirical_velocity_dispersion_profile, make_uneven_grid

### Community 31 - "Profile Fitting"
Cohesion: 0.67
Nodes (3): scipy dependency, fit_dehnen_profile, fit_plummer_profile

### Community 32 - "Stream Coordinate API Docs"
Cohesion: 0.67
Nodes (3): generate_stream_coords, get_observed_stream_coords, to_stream_coords

### Community 33 - "Agama HDF5 Coefficient IO"
Cohesion: 0.67
Nodes (3): load_agama_evolving_potential, write_coef_to_h5, write_snapshot_coefs_to_h5

### Community 34 - "Agama Evolving Potential"
Cohesion: 0.67
Nodes (3): create_evolving_ini, create_fire_evolving_ini, read_snapshot_times

### Community 36 - "GPU Tree vs Direct Test"
Cohesion: 1.0
Nodes (1): Direct comparison of GPU tree-code vs nbody_streams direct-sum.

### Community 37 - "Unit System Constants"
Cohesion: 1.0
Nodes (2): kpc km/s Msun unit system, G_DEFAULT constant

### Community 38 - "Potential Fitting Docs"
Cohesion: 1.0
Nodes (2): create_snapshot_dict, fit_potential

### Community 39 - "Dark Matter Constructor"
Cohesion: 1.0
Nodes (1): Convenience constructor for dark-matter particles.

### Community 40 - "Stellar Particle Constructor"
Cohesion: 1.0
Nodes (1): Convenience constructor for stellar particles.

### Community 44 - "Multipole lmax"
Cohesion: 1.0
Nodes (1): Maximum l order present in *lm_labels*.

### Community 45 - "Multipole l Values"
Cohesion: 1.0
Nodes (1): Sorted unique l values.

### Community 46 - "Multipole m Values"
Cohesion: 1.0
Nodes (1): Sorted unique m values (includes negatives).

### Community 48 - "Jacobi Orbit IC"
Cohesion: 1.0
Nodes (1): Short orbit + Jacobi quantities for IC tests.

### Community 49 - "Matplotlib Dependency"
Cohesion: 1.0
Nodes (1): matplotlib dependency

### Community 50 - "Graphify Rules"
Cohesion: 1.0
Nodes (1): graphify knowledge graph rules

### Community 51 - "NBODY_UNITS Constant"
Cohesion: 1.0
Nodes (1): NBODY_UNITS constant

### Community 52 - "Double Spheroid Profile"
Cohesion: 1.0
Nodes (1): fit_double_spheroid_profile

### Community 53 - "Iterative Ellipsoid Fit"
Cohesion: 1.0
Nodes (1): fit_iterative_ellipsoid

### Community 54 - "CUDA Alive Check"
Cohesion: 1.0
Nodes (1): cuda_alive

### Community 55 - "CPU Potential Computation"
Cohesion: 1.0
Nodes (1): compute_nbody_potential_cpu

### Community 56 - "GPU Info"
Cohesion: 1.0
Nodes (1): get_gpu_info

### Community 57 - "Restart Load Internal"
Cohesion: 1.0
Nodes (1): _load_restart internal

### Community 58 - "Vector Coordinate Convert"
Cohesion: 1.0
Nodes (1): convert_vectors

### Community 59 - "LOS Velocity Convert"
Cohesion: 1.0
Nodes (1): convert_to_vel_los

## Knowledge Gaps
- **397 isolated node(s):** `Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo`, `CPU parallelized force computation with Numba.                  Parameters`, `CPU potential kernel matching GPU version.         Returns -1/r equivalent for p`, `CPU parallelized potential computation with Numba.                  Parameters`, `Get compiled CUDA kernel for forces or potential.          Parameters     ------` (+392 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `GPU Tree vs Direct Test`** (2 nodes): `test_compare.py`, `Direct comparison of GPU tree-code vs nbody_streams direct-sum.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Unit System Constants`** (2 nodes): `kpc km/s Msun unit system`, `G_DEFAULT constant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Potential Fitting Docs`** (2 nodes): `create_snapshot_dict`, `fit_potential`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dark Matter Constructor`** (1 nodes): `Convenience constructor for dark-matter particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Stellar Particle Constructor`** (1 nodes): `Convenience constructor for stellar particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Multipole lmax`** (1 nodes): `Maximum l order present in *lm_labels*.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Multipole l Values`** (1 nodes): `Sorted unique l values.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Multipole m Values`** (1 nodes): `Sorted unique m values (includes negatives).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Jacobi Orbit IC`** (1 nodes): `Short orbit + Jacobi quantities for IC tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matplotlib Dependency`** (1 nodes): `matplotlib dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Graphify Rules`** (1 nodes): `graphify knowledge graph rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `NBODY_UNITS Constant`** (1 nodes): `NBODY_UNITS constant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Double Spheroid Profile`** (1 nodes): `fit_double_spheroid_profile`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Iterative Ellipsoid Fit`** (1 nodes): `fit_iterative_ellipsoid`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CUDA Alive Check`** (1 nodes): `cuda_alive`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CPU Potential Computation`** (1 nodes): `compute_nbody_potential_cpu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GPU Info`** (1 nodes): `get_gpu_info`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Restart Load Internal`** (1 nodes): `_load_restart internal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vector Coordinate Convert`** (1 nodes): `convert_vectors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LOS Velocity Convert`** (1 nodes): `convert_to_vel_los`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `max()` connect `CylSpline Serialization` to `CPU Sim Core & IO`, `Dynamical Friction`, `Main Entry & Validation`, `Package Top-Level & ParticleReader`, `Direct Sum Force Computation`, `Agama Multipole Coefficients`, `Visualization & Plots`, `Test Utilities & Plummer IC`?**
  _High betweenness centrality (0.333) - this node is a cross-community bridge._
- **Why does `min()` connect `CylSpline Serialization` to `CPU Sim Core & IO`, `Dynamical Friction`, `Package Top-Level & ParticleReader`, `Direct Sum Force Computation`, `Snapshot IO & Species`, `Visualization & Plots`, `Test Utilities & Plummer IC`, `CUDA Primitives`, `ParticleReader Orbit Extraction`, `Snapshot Write Helpers`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `_jeans_sigma_r()` connect `Dynamical Friction` to `Stream Spray Simulation`, `CylSpline Serialization`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 106 inferred relationships involving `Species` (e.g. with `ParticleReader` and `nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr`) actually correct?**
  _`Species` has 106 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `PerformanceWarning` (e.g. with `nbody_streams.sim Unified high-level simulation entry point for multi-species N-` and `Run a direct N-body simulation with one or more particle species.      This is t`) actually correct?**
  _`PerformanceWarning` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `max()` (e.g. with `_jeans_sigma_r()` and `_sigma_local_circular()`) actually correct?**
  _`max()` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `min()` (e.g. with `.extract_orbits()` and `_save_snapshot()`) actually correct?**
  _`min()` has 22 INFERRED edges - model-reasoned connections that need verification._