# LJNS: Lennard-Jones Nested Sampling

`ljns` is a Python implementation of nested sampling for a bulk
Lennard-Jones (LJ) fluid. It runs independent fixed-density NVT nested
sampling calculations, re-analyses the resulting density of states over a
temperature grid, and produces thermodynamic curves and phase-diagram
figures.

The code uses reduced LJ units:

| Quantity | Reduced unit |
| --- | --- |
| Length | `sigma` |
| Energy | `epsilon` |
| Temperature | `epsilon / k_B` |
| Pressure | `epsilon / sigma^3` |
| Boltzmann constant | `k_B = 1` |

The project is intended for reproducible computational studies of LJ fluids,
not as a general-purpose molecular-dynamics engine. The implementation is
small, inspectable, and separates the simulation, thermodynamic analysis,
parallel execution, and plotting layers.

## What the code does

For every density `rho` in a configured density grid, the pipeline:

1. Builds `K` live configurations in a periodic cubic box.
2. Removes the highest-energy live point.
3. Replaces it with a constrained Metropolis walk of a surviving point,
   enforcing the nested-sampling energy ceiling.
4. Assigns the standard deterministic nested-sampling volume weight to the
   discarded point.
5. Repeats for `n_iter` iterations and records the remaining live points.
6. Converts all weighted energy/virial points into thermodynamic observables
   over the requested temperature grid.
7. Locates the heat-capacity peak and uses it as the reported
   density-temperature phase-boundary indicator.
8. Saves compressed data products and, optionally, publication-style figures.

The primary scientific outputs are configurational and total heat capacity,
mean potential energy, entropy, Helmholtz and Gibbs free energies, pressure,
compressibility factor, density of states, and the `C_v` peak ridge in the
`T-rho` plane.

## Quick start

### 1. Create an environment

Python 3.7 or newer is supported. A recent Python version is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an editable install, use:

```bash
python -m pip install -e .
```

### 2. Run the smoke-test-sized campaign

```bash
python -m ljns.cli --config config/quick.json --cores 1
```

The equivalent convenience script is:

```bash
./scripts/run_quick.sh
```

This writes results to `runs/quick/`. The quick preset is intended to verify
the complete workflow; it is not a converged production calculation.

### 3. Run a production campaign

```bash
python -m ljns.cli --config config/default.json --cores 64
```

or:

```bash
./scripts/run_ns.sh config/default.json 64
```

The package-level configuration validation clamps the worker count to
`min(64, os.cpu_count())`. The `loky` joblib backend runs one independent
simulation per density.

### 4. Re-analyse saved nested-sampling data

When dead-point archives already exist in the configured output directory,
skip the nested-sampling calculation:

```bash
python -m ljns.cli \
  --config config/default.json \
  --load \
  --override T_min=0.2 \
  --override T_max=3.0 \
  --override n_T=800
```

Changing the temperature grid only requires thermodynamic re-analysis; it
does not require new nested-sampling samples.

## Command-line interface

The main entry point is:

```text
python -m ljns.cli --config CONFIG [OPTIONS]
```

| Option | Meaning |
| --- | --- |
| `--config PATH` | Required JSON configuration file |
| `--cores N` | Worker-process count, clamped by configuration validation |
| `--seed N` | Override the master random seed |
| `--load` | Load `dead_rho*.npz` files instead of running NS |
| `--no-plots` | Disable all plots |
| `--no-per-density-plots` | Keep overview plots but disable six-panel density plots |
| `--densities RHO...` | Replace the configured density list |
| `--output-dir PATH` | Replace the configured output directory |
| `--override key=value` | Repeatable ad-hoc configuration override |
| `-v`, `--verbose` | Enable debug logging |

Examples:

```bash
# Run only three densities without creating figures.
python -m ljns.cli --config config/default.json \
  --densities 0.2 0.5 0.8 --no-plots --cores 1

# Change analysis settings while retaining the simulation settings.
python -m ljns.cli --config config/default.json \
  --load --override T_min=0.1 --override T_max=4.0 --override n_T=1000
```

## Configuration

Configuration files are grouped into `physics`, `ns`, `scan`, `parallel`, and
`io` sections. Values can also be supplied in a flat JSON object, although
the grouped form is clearer.

| Section | Important fields |
| --- | --- |
| `physics` | `n_atoms`, `epsilon`, `sigma`, `r_cut` |
| `ns` | `n_walkers`, `n_iter`, `n_mc_steps`, `atom_step_size` |
| `scan` | `densities`, `T_min`, `T_max`, `n_T`, `cv_smooth_window` |
| `parallel` | `n_workers`, `backend` |
| `io` | output and save/plot switches |
| top level | `seed`, `verbosity` |

The included presets are:

| File | Purpose |
| --- | --- |
| `config/quick.json` | Small end-to-end smoke test |
| `config/default.json` | Broad `N=64` density campaign |
| `config/N512_publication.json` | High-cost `N=512` campaign |

The JSON configuration is authoritative: the CLI loads and validates it
before execution. Validation rejects invalid atom counts, densities,
temperature ranges, and iteration settings; it warns when an atom count is
not an FCC-compatible `4 m^3` size and reports when the cutoff must be
reduced to satisfy the minimum-image convention.

## Output files

Each density produces, when enabled:

```text
OUTPUT/
  dead_rho0.3000.npz
  thermo_rho0.3000.npz
  thermo_rho0.300.png
  ...
  summary.txt
  cv_vs_T_all_densities.png
  phase_diagram_T_rho.png
  phase_diagram_Trho.png
  phase_diagram_Prho.png
  phase_diagram_PT.png
  caloric_curves.png
  entropy_vs_T.png
  free_energy_vs_T.png
  pressure_isochores.png
  compressibility_factor.png
  energy_convergence.png
  cv_3d.png
```

`dead_rho*.npz` contains the arrays `iteration`, `energy`, `virial`,
`log_weight`, `rho`, and `n_atoms`. These are the reusable raw nested-sampling
results.

`thermo_rho*.npz` contains the temperature-dependent arrays returned by
`ljns.thermodynamics.analyse`, including `Cv_per_atom`, `mean_P`,
`F_total_per_atom`, `S_conf_per_atom`, `Z_factor`, and the corresponding
metadata.

The summary table reports the density, `T_peak`, peak heat capacity per atom,
and FWHM estimated by `find_cv_peak`.

## Repository layout

```text
config/                 JSON simulation presets and configuration notes
docs/about/             Theory and implementation guide
scripts/                Shell entry points
src/ljns/
  config.py             Validated configuration and reduced-unit helpers
  potential.py          LJ potential, virial, tails, periodic initialization
  walker.py             Constrained NVT Metropolis live-point walker
  nested_sampling.py    Fixed-density nested-sampling loop and NPZ I/O
  thermodynamics.py     Vectorized thermodynamic re-analysis
  parallel.py           Reproducible density-level joblib parallelism
  cli.py                Command-line orchestration and summary output
  plotting/             Overview and per-density publication plots
tests/                  Unit and end-to-end tests
runs/                   Example/generated run products
results/                Reserved output location
```

## Testing

Run the existing test suite with:

```bash
pytest -q
```

The tests cover periodic wrapping, LJ energy and virial formulas, tail
corrections, FCC/simple-cubic initialization, nested-sampling weights and
monotonicity, walker ceiling behavior, thermodynamic identities, archive
round trips, and a tiny CLI run.

## Scientific interpretation and limitations

The reported `T_peak(rho)` curve is an operational phase-boundary proxy based
on the maximum in `C_v/N`; it is not, by itself, a complete coexistence
calculation. Reliable phase boundaries require convergence checks with
respect to `N`, `K`, `n_iter`, MC mixing, density spacing, temperature
resolution, and independent seeds.

The energy and virial tail corrections assume a uniform pair correlation
function beyond `r_cut`. The effective cutoff is reduced for small boxes when
`L/2 <= r_cut`. Energies use an `O(N^2)` pair loop, so the implementation is
best suited to the system sizes and density scans represented by the presets.

The constrained replacement step uses a finite Metropolis walk. It therefore
approximates sampling from the prior restricted by the nested-sampling
likelihood boundary; increasing `n_mc_steps` and checking independent
replicas is important near sharp transitions.

## Documentation

The detailed documentation is in [`docs/about/`](docs/about/):

- [Theory](docs/about/theory.md): nested sampling, LJ physics, and
  thermodynamic estimators.
- [Implementation](docs/about/implementation.md): module-by-module execution
  path, data structures, numerical methods, and reproducibility.
- [Data and workflow](docs/about/data-and-workflow.md): configuration,
  outputs, plots, convergence, and recommended usage.

## License

The package metadata declares the MIT license. Confirm and complete the
repository's license distribution requirements before publishing if a
standalone `LICENSE` file is required by your hosting or release workflow.
