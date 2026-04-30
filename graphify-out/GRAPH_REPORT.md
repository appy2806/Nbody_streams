# Graph Report - Nbody_streams  (2026-04-29)

## Corpus Check
- 65 files · ~165,275 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1540 nodes · 3618 edges · 91 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 1619 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]

## God Nodes (most connected - your core abstractions)
1. `Species` - 111 edges
2. `NFWPotentialGPU` - 106 edges
3. `MiyamotoNagaiPotentialGPU` - 106 edges
4. `LogHaloPotentialGPU` - 106 edges
5. `DiskAnsatzPotentialGPU` - 106 edges
6. `MultipoleCoefs` - 82 edges
7. `MultipolePotentialGPU` - 77 edges
8. `CylSplineCoefs` - 76 edges
9. `CompositePotentialGPU` - 75 edges
10. `PlummerPotentialGPU` - 73 edges

## Surprising Connections (you probably didn't know these)
- `Chandrasekhar dynamical friction` --references--> `Satellite A final surface density (M_sat=5e10 Msun, post-inspiral)`  [INFERRED]
  docs/dynamical_friction.md → output/df_tutorial/satA_final_density.png
- `Chandrasekhar dynamical friction` --references--> `final particle density 2x2 comparison`  [EXTRACTED]
  README.md → output/final_density_2x2.png
- `Chandrasekhar dynamical friction` --references--> `bound particle fraction vs time chart`  [EXTRACTED]
  README.md → output/bound_fraction_vs_time.png
- `Backend dispatch table` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png
- `run_nbody_gpu` --references--> `Acceleration timing benchmark: CPU direct vs GPU direct vs FMM/Tree across N particles`  [INFERRED]
  docs/main.md → plots/acceleration_timings.png

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (134): nbody_streams - lightweight direct N-body utilities., _is_uniform(), _load_restart(), _make_times_ns(), ParticleReader, nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr, Append or update snapshot.times in output_dir.     Ensures unique snap_index ent, Worker for N-species parallel extraction.      Args (tuple):         snap_index (+126 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (137): AnalyticPotentialGPU(), DehnenSphericalPotentialGPU, DiskAnsatzPotentialGPU, HernquistPotentialGPU, IsochronePotentialGPU, LogHaloPotentialGPU, MiyamotoNagaiPotentialGPU, NFWPotentialGPU (+129 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (96): _add_negative_m(), _detect_expansion_type(), lmax(), agama_helper._coefs ~~~~~~~~~~~~~~~~~~~ Structured representations of Agama expa, Return lines from any coef source: file path, HDF5 path, or raw string.      Del, Return 'Multipole' or 'CylSpline' from a coef string header, or '' if unknown., Return a copy with all (l, m) terms **not** in *keep_lm* zeroed out.          Ne, Read an Agama Multipole expansion into a :class:`MultipoleCoefs` dataclass. (+88 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (67): _bound_center_phi(), chandrasekhar_friction(), compute_sigma_r(), _jeans_sigma_r(), make_df_force_extra(), nbody_streams._chandrasekhar ============================ Chandrasekhar dynamica, Local circular-speed approximation to the 1-D velocity dispersion.      Uses the, Compute a radial velocity-dispersion profile from an Agama potential.      Param (+59 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (81): nbody_streams.cuda_kernels CUDA kernels for N-body force and potential computati, _compute_forces_cpu(), compute_nbody_forces_cpu(), compute_nbody_forces_gpu(), compute_nbody_potential_cpu(), compute_nbody_potential_gpu(), _compute_potential_cpu(), _get_force_kernel() (+73 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (87): plummer_density(), plummer_enclosed_mass(), plummer_v_circ(), tests/test_utils.py ===================  Tests for ``nbody_streams.utils`` using, Circular velocity should match analytic Plummer., Velocity dispersion should be finite and positive at all radii., RMS velocity should be finite and positive., For an isotropic Plummer sphere, beta ~= 0 on average. (+79 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (49): _close_figures(), orbit_data(), Tests for nbody_streams.viz -- visualization functions.  All tests use the Agg b, imshow extent should be [-gridsize/2, gridsize/2] on both axes., Total mass in grid should approximately match input mass., Morton sort should produce a valid density grid.          Grids are NOT expected, Close all matplotlib figures after each test., Particles in a dense cluster should get smaller h than isolated ones. (+41 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (78): force_extra hook, add_perturber subhalo, create_ic_particle_spray_chen2025, create_particle_spray_stream, dynamical friction progenitor orbit, create_ic_particle_spray_fardal2015, run_restricted_nbody, custom stripping times (+70 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (24): _build_cylspline_data(), _build_dehnen_gpu(), _build_disk_gpu(), _build_king_gpu(), _build_spheroid_gpu(), _clamped_left_cubic_deriv_batch(), _desort(), _determine_asympt_cylspline() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (42): _as_3vec(), _cart_to_cyl(), _cart_to_sph(), convert_coords(), convert_to_vel_los(), convert_vectors(), _cyl_to_cart(), _cyl_to_sph() (+34 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (45): generate_lmax_pairs(), Serialise back to the Agama CylSpline text format.          Returns         ----, Generate (l, m) pairs for a spherical harmonic expansion.      Parameters     --, _build_multipole_data(), _compute_invPhi0(), _compute_outer_extrap(), _solve_quintic_d2(), generate_plummer() (+37 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (38): _compute_vel_disp_from_Potential(), _create_perturber_potential(), _dynamical_friction_acceleration(), _find_prog_pot_Nparticles(), _get_prog_GalaxyModel(), _integrate_orbit_with_dynamical_friction(), Private helpers shared by fast_sims methods.  Functions here are implementation, Integrate a satellite orbit with optional dynamical friction.      Dynamical fri (+30 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (29): CylSplineCoefs, load_agama_potential, load_fire_pot, MultipoleCoefs, read_coefs, convert_coords, _bound_center_phi, Chandrasekhar dynamical friction (+21 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (13): nbody_streams.coords - coordinate transformations.  Cartesian/spherical/cylindri, generate_stream_coords(), get_observed_stream_coords(), Stream coordinate generation and observable conversions.  Generate stream-aligne, Project galactocentric positions or phase-space vectors into a     pre-computed, Convert galactocentric phase space into stream-aligned coordinates     (phi1, ph, Convert galactocentric phase-space coordinates to observed sky     coordinates (, to_stream_coords() (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (5): _AnalyticBase, _prep_xyz(), Agama-compatible eval --- returns any combination of potential, acceleration,, Abstract base --- subclasses implement _phi, _grad, _hess, _rho., _squeeze()

### Community 15 - "Community 15"
Cohesion: 0.1
Nodes (21): _chandrasekhar module, CylSplineCoefs dataclass, FIRE simulation helpers, libtreeGPU.so shared library, load_agama_evolving_potential, load_agama_potential, make_df_force_extra, MultipoleCoefs dataclass (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.2
Nodes (10): create_snapshot_dict(), fit_potential(), agama_helper._fit ~~~~~~~~~~~~~~~~~ Fit Agama Multipole and CylSpline potential, Fit Agama Multipole and CylSpline potentials from a multi-species snapshot., Sample *n* positions from a spherically declining density profile., Sample *n* positions from a thin exponential disk., Pack particle arrays into a FIRE-like snapshot dictionary.      Creates the mini, _require_agama() (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.31
Nodes (8): benchmark(), _frel(), _hess_fd_check(), _pts(), _pts_king(), _rel(), run_comprehensive_benchmark(), test_accuracy()

### Community 18 - "Community 18"
Cohesion: 0.28
Nodes (8): Checks which high-performance features are actually active., Runs a tiny GPU simulation and verifies snapshot output., Quick helper to generate test particles (Plummer-like spread)., Runs a tiny CPU simulation and verifies snapshot output., setup_dummy_data(), test_cpu_integration(), test_environment_summary(), test_gpu_integration()

### Community 19 - "Community 19"
Cohesion: 0.38
Nodes (3): alloc(), free(), realloc()

### Community 21 - "Community 21"
Cohesion: 0.5
Nodes (4): _newtons_third_law(), Test that net force is ~0 for isolated system.          Returns:         bool: T, Test all precision modes., test_all_precisions()

### Community 22 - "Community 22"
Cohesion: 0.67
Nodes (3): main(), nbody_streams.tree_gpu._build Console-script entry point: nbody-build-tree  Comp, _tree_gpu_dir()

### Community 23 - "Community 23"
Cohesion: 0.5
Nodes (2): Radial power spectrum for harmonic order *l*.          Parameters         ------, Total power for harmonic order *l* summed over all radial bins.          Paramet

### Community 25 - "Community 25"
Cohesion: 0.5
Nodes (4): empirical_circular_velocity_profile, empirical_density_profile, empirical_velocity_dispersion_profile, make_uneven_grid

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (3): scipy dependency, fit_dehnen_profile, fit_plummer_profile

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (3): generate_stream_coords, get_observed_stream_coords, to_stream_coords

### Community 28 - "Community 28"
Cohesion: 0.67
Nodes (3): load_agama_evolving_potential, write_coef_to_h5, write_snapshot_coefs_to_h5

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (3): create_evolving_ini, create_fire_evolving_ini, read_snapshot_times

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Direct comparison of GPU tree-code vs nbody_streams direct-sum.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (2): kpc km/s Msun unit system, G_DEFAULT constant

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (2): create_snapshot_dict, fit_potential

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Convenience constructor for dark-matter particles.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Convenience constructor for stellar particles.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Maximum l order present in *lm_labels*.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Sorted unique l values.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Sorted unique m values (includes negatives).

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Short orbit + Jacobi quantities for IC tests.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): r"""Create a stellar stream using the particle-spray method.      The progenitor

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): CPU parallelized force computation with Numba.                  Parameters

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): CPU potential kernel matching GPU version.         Returns -1/r equivalent for p

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): CPU parallelized potential computation with Numba.                  Parameters

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Get compiled CUDA kernel for forces or potential.          Parameters     ------

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Validate inputs and prepare CPU arrays.     Works regardless of whether CuPy is

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Transfer validated CPU data to GPU or use existing GPU arrays.     Only called b

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Prepare CPU arrays from validated data.     Only called by CPU functions.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Compute direct N-body gravitational forces using GPU acceleration.          Calc

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Compute gravitational potential at each particle location on GPU.          Calcu

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Compute gravitational potential using CPU parallelization (Numba).          Fall

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Get information about available GPU(s).          Returns     -------     info :

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Compute direct N-body gravitational forces using GPU acceleration.          Calc

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Get information about available GPU(s).          Returns     -------     info :

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Fallback saver: Stores data in a simple .npy format.          Parameters     ---

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Compute total accelerations on GPU.          Combines self-gravity (GPU) with op

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Compute accelerations using tree algorithm (pyfalcon).          Parameters     -

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Compute accelerations using direct O(N^2) pairwise summation.          Parameter

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Run GPU-accelerated N-body simulation with leapfrog (KDK) integration.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Run GPU-accelerated N-body simulation with leapfrog (KDK) integration.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Run CPU-NUMBA accelerated N-body simulation with leapfrog (KDK) integration.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Generate Plummer sphere in virial equilibrium.          Parameters     ---------

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Place system on orbit in external potential.          Parameters     ----------

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Compute direct N-body gravitational forces using GPU acceleration.          Calc

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Get information about available GPU(s).          Returns     -------     info :

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): CPU parallelized force computation with Numba.                  Parameters

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Compute direct N-body gravitational forces using GPU acceleration.          Calc

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Get information about available GPU(s).          Returns     -------     info :

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Worker executed in a separate process. Reads one snapshot from disk and writes

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): A class to read N-body simulation data from one or more HDF5 files.      This re

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Read simulation properties from the first HDF5 file.         Robust to missing '

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Scan HDF5 files and map snapshot index -> file path for fast lookups.          S

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Read a single snapshot by index or physical time.          Parameters         --

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Extract orbits for selected particle types across all available snapshots.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Fast append-only version. Sort at the end with _finalize_snapshot_times().

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Call this ONCE at the end of run_nbody_gpu.

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Write snapshot(s) compatible with ParticleReader.      Modes:       - single_fil

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Save restart file for crash recovery.          Parameters     ----------     pha

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): Load restart file if it exists.          Parameters     ----------     output_di

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): matplotlib dependency

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): graphify knowledge graph rules

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): NBODY_UNITS constant

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): fit_double_spheroid_profile

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): fit_iterative_ellipsoid

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): cuda_alive

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): compute_nbody_potential_cpu

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (1): get_gpu_info

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (1): _load_restart internal

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (1): convert_vectors

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): convert_to_vel_los

## Knowledge Gaps
- **399 isolated node(s):** `Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo`, `CPU parallelized force computation with Numba.                  Parameters`, `CPU potential kernel matching GPU version.         Returns -1/r equivalent for p`, `CPU parallelized potential computation with Numba.                  Parameters`, `Get compiled CUDA kernel for forces or potential.          Parameters     ------` (+394 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 23`** (4 nodes): `.radial_power()`, `.total_power()`, `Radial power spectrum for harmonic order *l*.          Parameters         ------`, `Total power for harmonic order *l* summed over all radial bins.          Paramet`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (2 nodes): `test_compare.py`, `Direct comparison of GPU tree-code vs nbody_streams direct-sum.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `kpc km/s Msun unit system`, `G_DEFAULT constant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (2 nodes): `create_snapshot_dict`, `fit_potential`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Convenience constructor for dark-matter particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Convenience constructor for stellar particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Maximum l order present in *lm_labels*.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Sorted unique l values.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Sorted unique m values (includes negatives).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Short orbit + Jacobi quantities for IC tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `r"""Create a stellar stream using the particle-spray method.      The progenitor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `CPU parallelized force computation with Numba.                  Parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `CPU potential kernel matching GPU version.         Returns -1/r equivalent for p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `CPU parallelized potential computation with Numba.                  Parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Get compiled CUDA kernel for forces or potential.          Parameters     ------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Validate inputs and prepare CPU arrays.     Works regardless of whether CuPy is`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Transfer validated CPU data to GPU or use existing GPU arrays.     Only called b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Prepare CPU arrays from validated data.     Only called by CPU functions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Compute direct N-body gravitational forces using GPU acceleration.          Calc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Compute gravitational potential at each particle location on GPU.          Calcu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Compute gravitational potential using CPU parallelization (Numba).          Fall`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Get information about available GPU(s).          Returns     -------     info :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Compute direct N-body gravitational forces using GPU acceleration.          Calc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Get information about available GPU(s).          Returns     -------     info :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Fallback saver: Stores data in a simple .npy format.          Parameters     ---`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Compute total accelerations on GPU.          Combines self-gravity (GPU) with op`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Compute accelerations using tree algorithm (pyfalcon).          Parameters     -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Compute accelerations using direct O(N^2) pairwise summation.          Parameter`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Run GPU-accelerated N-body simulation with leapfrog (KDK) integration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Run GPU-accelerated N-body simulation with leapfrog (KDK) integration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Run CPU-NUMBA accelerated N-body simulation with leapfrog (KDK) integration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Generate Plummer sphere in virial equilibrium.          Parameters     ---------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Place system on orbit in external potential.          Parameters     ----------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Compute direct N-body gravitational forces using GPU acceleration.          Calc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Get information about available GPU(s).          Returns     -------     info :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Selectable force kernel function.         Returns the 1/r^3 equivalent factor fo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `CPU parallelized force computation with Numba.                  Parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Compute direct N-body gravitational forces using GPU acceleration.          Calc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Get information about available GPU(s).          Returns     -------     info :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Worker executed in a separate process. Reads one snapshot from disk and writes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `A class to read N-body simulation data from one or more HDF5 files.      This re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Read simulation properties from the first HDF5 file.         Robust to missing '`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Scan HDF5 files and map snapshot index -> file path for fast lookups.          S`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Read a single snapshot by index or physical time.          Parameters         --`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Extract orbits for selected particle types across all available snapshots.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Fast append-only version. Sort at the end with _finalize_snapshot_times().`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Call this ONCE at the end of run_nbody_gpu.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Write snapshot(s) compatible with ParticleReader.      Modes:       - single_fil`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Save restart file for crash recovery.          Parameters     ----------     pha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `Load restart file if it exists.          Parameters     ----------     output_di`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `matplotlib dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `graphify knowledge graph rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `NBODY_UNITS constant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `fit_double_spheroid_profile`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `fit_iterative_ellipsoid`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `cuda_alive`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `compute_nbody_potential_cpu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (1 nodes): `get_gpu_info`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (1 nodes): `_load_restart internal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (1 nodes): `convert_vectors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `convert_to_vel_los`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `max()` connect `Community 10` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.318) - this node is a cross-community bridge._
- **Why does `min()` connect `Community 10` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Why does `plot_stream_sky()` connect `Community 13` to `Community 6`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 106 inferred relationships involving `Species` (e.g. with `ParticleReader` and `nbody_streams.nbody_io  I/O utilities for N-body snapshots and restart data.  Pr`) actually correct?**
  _`Species` has 106 INFERRED edges - model-reasoned connections that need verification._
- **Are the 98 inferred relationships involving `NFWPotentialGPU` (e.g. with `CompositePotentialGPU` and `_GPUPotBase`) actually correct?**
  _`NFWPotentialGPU` has 98 INFERRED edges - model-reasoned connections that need verification._
- **Are the 98 inferred relationships involving `MiyamotoNagaiPotentialGPU` (e.g. with `CompositePotentialGPU` and `_GPUPotBase`) actually correct?**
  _`MiyamotoNagaiPotentialGPU` has 98 INFERRED edges - model-reasoned connections that need verification._
- **Are the 98 inferred relationships involving `LogHaloPotentialGPU` (e.g. with `CompositePotentialGPU` and `_GPUPotBase`) actually correct?**
  _`LogHaloPotentialGPU` has 98 INFERRED edges - model-reasoned connections that need verification._