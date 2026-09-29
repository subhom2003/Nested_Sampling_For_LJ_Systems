# Implementation

## 1. Execution path

The command-line workflow is:

```text
JSON config
    |
    v
SimConfig.from_json() -> validation and derived box/cutoff helpers
    |
    v
CLI chooses serial or density-parallel execution
    |
    +--> run_ns(rho)
    |       +--> initialize K live points
    |       +--> remove worst point and constrained-walk a clone
    |       +--> save dead points
    |
    +--> analyse(dead, T_array)
    |       +--> weighted energy moments
    |       +--> Cv, S, F, G, P, Z
    |       +--> locate Cv peak
    |
    +--> save thermo archives and generate plots
    |
    v
summary.txt
```

The density calculations are independent. No live-point population is shared
between densities.

## 2. Module responsibilities

### `config.py`

`SimConfig` is the single source of simulation and analysis parameters. It:

- loads grouped or flat JSON;
- converts numeric values to the expected types;
- validates hard constraints;
- warns about non-FCC system sizes;
- calculates `L`, `r_cut_eff`, and the temperature array;
- creates output directories;
- clamps worker counts to the package's maximum and available CPUs.

`tiny_test_cfg` supplies the intentionally small configuration used by tests.

### `potential.py`

This is the physics kernel. It implements minimum-image displacements, LJ
energy and virial primitives, full and single-atom energy evaluation, analytic
tail corrections, and periodic lattice initialization.

The full energy routines use direct pair loops and NumPy vector operations.
This is simple and transparent but has `O(N^2)` pair-scaling.

### `walker.py`

`LivePoint` stores positions, total configurational energy, box length,
density, and effective cutoff. `make_live_point` creates and burns in initial
points. `walk` deep-copies a donor and performs constrained Metropolis moves.

The walker updates energy incrementally during trial moves, then the initial
constructor performs a full energy recomputation to remove accumulated
floating-point drift.

### `nested_sampling.py`

`run_ns` owns one complete fixed-density nested-sampling calculation. It
stores `DeadPoint` records containing:

```text
iteration, energy, virial, log_weight, rho, n_atoms
```

The function returns both the dead-point list and metadata such as the number
of records, compression constants, final live-point weight, and elapsed time.

`save_dead_points` and `load_dead_points` use compressed NumPy `.npz` files,
which makes the raw sample population reusable for later analysis.

### `thermodynamics.py`

`analyse` stacks dead-point fields into NumPy arrays and evaluates all
temperatures simultaneously using broadcasting. For `n_dead` samples and
`n_T` temperatures, the main temporary likelihood array has shape
`(n_dead, n_T)`.

The module also provides:

- `log_sum_exp` for numerical stability;
- `find_cv_peak` for peak and FWHM extraction;
- `log_dos` for a weighted histogram estimate of `ln g(E)`;
- `.npz` save/load helpers for thermodynamic arrays.

### `parallel.py`

The density sweep uses joblib. One `SeedSequence` child is spawned per
density from the master seed. This gives each density an independent RNG
stream and makes the sampled result independent of worker scheduling order.

The default `loky` backend uses separate worker processes. Each worker runs
the nested-sampling calculation, optionally saves its raw points, analyzes the
temperature grid, saves thermodynamics, and returns the phase-peak summary.

### `cli.py`

The CLI loads configuration, applies command-line overrides, chooses serial or
parallel execution, loads or writes dead-point files, generates overview and
per-density figures, prints a density summary, and writes `summary.txt`.

The serial path is useful for debugging, small tests, and environments where
process-based parallelism is undesirable. The parallel path is selected when
`n_workers > 1`.

### `plotting/`

`plotting/style.py` centralizes publication-style Matplotlib settings and the
color palettes. `plotting/single.py` creates a six-panel figure for one
density. `plotting/overview.py` creates curves and phase-diagram projections
across the density sweep.

Figures are written as 300 DPI PNG files with consistent labels, inward ticks,
minor ticks, and a colorblind-aware palette.

## 3. Initialization details

The box is always cubic, with `L = (N/rho)^(1/3)`. For `N = 4 m^3`, the
initial coordinates are generated from an `m x m x m` conventional FCC
supercell with four basis atoms. Other values use a simple-cubic grid large
enough to hold `N` atoms. Both paths add a small random displacement and fold
coordinates into `[0, L)`.

Each initial configuration receives an unconstrained burn-in of `5N` random
single-atom trial moves. This is a practical relaxation from the lattice
starting state; it is not a formal proof that the initial population is
independent and exactly uniform in configuration space.

## 4. Data conventions

### Dead-point archive

`dead_rhoXXXX.npz` stores one-dimensional arrays:

| Key | Meaning |
| --- | --- |
| `iteration` | Free iteration index or `n_iter` for final live points |
| `energy` | Configurational energy including energy tail correction |
| `virial` | Truncated virial plus volume-weighted virial tail |
| `log_weight` | Natural logarithm of the nested-sampling prior-volume weight |
| `rho` | Number density |
| `n_atoms` | Particle count |

There are exactly `n_iter + n_walkers` records for a successful run.

### Thermodynamic archive

`thermo_rhoXXXX.npz` stores the dictionary returned by `analyse`. Array keys
are stored as compressed numeric arrays; scalar metadata is promoted to
one-element arrays by `save_results`.

Important keys include:

```text
T, rho, n_atoms, volume
log_Z_conf, Z_conf_norm
mean_U, mean_U_per_atom
Cv_conf, Cv_total, Cv_per_atom
F_conf, F_conf_per_atom, F_total, F_total_per_atom
S_conf, S_conf_per_atom, S_total, S_total_per_atom
mean_P, mean_W
G_conf, G_conf_per_atom, G_total, G_total_per_atom
Z_factor
```

## 5. Reproducibility

Reproducibility is controlled by `seed`. In the parallel path,
`SeedSequence(seed).spawn(len(densities))` creates one child sequence per
density. The worker count and scheduling therefore do not determine which RNG
stream belongs to a density.

For scientifically meaningful comparisons, record:

- the exact JSON configuration;
- code version or commit;
- master seed;
- number of walkers and iterations;
- MC steps and displacement size;
- density and temperature grids;
- output archives and summary;
- the number of workers and backend.

The output archives contain sampled data but not a full copy of the input
configuration, so preserve the JSON file alongside results.

## 6. Numerical stability and performance

The thermodynamic analysis uses log-sum-exp rather than directly summing
Boltzmann factors, preventing overflow and underflow over broad energy and
temperature ranges. It also vectorizes all temperatures in one NumPy
broadcasted operation.

The main performance costs are:

- initial population construction: `K` configurations;
- constrained MC: approximately `n_iter * n_mc_steps * N` trial moves;
- pair energy/virial evaluation: `O(N^2)`;
- density-level parallelism and process startup;
- the `(n_dead, n_T)` thermodynamic working array.

The `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` variables set by `scripts/run_ns.sh` prevent nested
thread oversubscription inside worker processes.

## 7. Testing strategy

The tests are deliberately small but cover the main invariants:

- minimum-image wrapping;
- LJ energy and virial values at known distances;
- analytic tail formulas;
- consistency of total and single-atom energies;
- lattice initialization behavior;
- monotonic free dead-point energies;
- exact nested-sampling record count and log weights;
- deep-copy and energy-ceiling behavior in the walker;
- high-temperature ideal-gas limits;
- heat-capacity peak detection and density-of-states integration;
- compressed archive round trips;
- an end-to-end tiny CLI campaign.

Run them with `pytest -q`. Passing unit tests establishes implementation
invariants, not production-level statistical convergence.
