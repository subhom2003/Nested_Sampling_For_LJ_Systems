# Data and workflow

## 1. Recommended workflow

### Development and smoke testing

Start with the smallest complete run:

```bash
python -m ljns.cli --config config/quick.json --cores 1 --no-per-density-plots
```

Use this to check installation, configuration parsing, archive creation,
summary generation, and basic plot generation. It is intentionally too small
to support strong physical conclusions.

### Production runs

Copy an existing JSON preset and change the output directory before launching
a larger campaign. Keep one output directory per combination of system size,
sampling settings, density grid, and seed.

For example:

```bash
cp config/default.json /tmp/ljns-production.json
python -m ljns.cli --config /tmp/ljns-production.json --cores 64
```

The `N512_publication.json` preset is substantially more expensive because it
uses 512 atoms, 1000 live points, 30,000 iterations, a dense density grid,
and 1000 analysis temperatures.

### Re-analysis

Dead-point archives are the key checkpoint. Once they exist, use `--load` to
change `T_min`, `T_max`, `n_T`, or the heat-capacity smoothing window without
re-running the expensive nested-sampling stage.

## 2. Preset interpretation

### `config/quick.json`

- `N=32`, `K=80`, `n_iter=2000`
- five densities
- four workers by default
- 200 temperatures
- plots enabled

It is appropriate for an end-to-end smoke test and rough visual inspection.

### `config/default.json`

- `N=64`, `K=6000`, `n_iter=20000`
- a dense density grid from `0.01` to `1.0`
- 500 temperatures
- broad production-style output

This is computationally expensive and should be run on a machine with enough
memory and CPU capacity.

### `config/N512_publication.json`

- `N=512`, `K=1000`, `n_iter=30000`
- 126 density values
- temperatures from `0.1` to `10`
- 1000 temperature points

The preset is designed for a high-cost finite-size campaign. Inspect the
requested worker count and memory budget before launching it.

## 3. Plot inventory

The CLI writes overview plots when there is more than one density:

| File | Content |
| --- | --- |
| `cv_vs_T_all_densities.png` | `C_v/N` versus temperature for all densities |
| `phase_diagram_T_rho.png` | `T_peak` versus density |
| `caloric_curves.png` | Mean potential energy per atom |
| `entropy_vs_T.png` | Configurational entropy per atom |
| `free_energy_vs_T.png` | Total Helmholtz free energy per atom |
| `pressure_isochores.png` | Pressure versus temperature |
| `compressibility_factor.png` | `Z = PV/(NT)` versus temperature |
| `energy_convergence.png` | Dead-point energy versus NS iteration |
| `cv_3d.png` | Three-dimensional `C_v(T,rho)` line view |
| `phase_diagram_Trho.png` | Heat-capacity contours and peak ridge |
| `phase_diagram_Prho.png` | Pressure at each `T_peak` |
| `phase_diagram_PT.png` | Pressure-temperature isochores and peak curve |

With `per_density_plots=true`, each density also gets a six-panel figure with
heat capacity, entropy, energy, pressure, free energy, and `ln g(E)`.

## 4. Convergence and validation checklist

Before interpreting a phase diagram, repeat selected densities while varying:

1. `n_walkers` (`K`) to test nested-sampling volume resolution.
2. `n_iter` to test energy-range coverage.
3. `n_mc_steps` and `atom_step_size` to test constrained-walk mixing.
4. `seed` to estimate stochastic variation.
5. `n_atoms` to identify finite-size effects.
6. `n_T` and the density spacing to test interpolation and peak resolution.
7. `r_cut` and the tail-correction assumptions where relevant.

Useful diagnostics are monotonic free dead-point energies, stable
`T_peak(rho)`, sensible high-temperature `Z_factor` approaching one, stable
heat-capacity curves, and agreement between independent seeds.

The FWHM reported in `summary.txt` is a peak-width diagnostic, not a formal
confidence interval. It is obtained after optional Gaussian smoothing and
depends on the analysis grid.

## 5. Important limitations

The implementation has several deliberate approximations:

- A finite constrained MC walk only approximates the restricted prior.
- The initial lattice-plus-burn-in population is not an exact independent
  draw from a uniform configuration-space prior.
- The deterministic compression factor is an approximation to the random
  shrinkage process.
- Tail corrections assume a uniform fluid beyond the cutoff.
- The direct pair loop scales quadratically with `N`.
- The phase-boundary curve is inferred from `C_v` peaks rather than from an
  explicit equality of chemical potentials or Gibbs free energies.
- The ideal-gas kinetic free-energy expression fixes a particular
  normalization convention; absolute free energies should be compared using
  consistent conventions.

These points do not prevent the code from being useful, but they define what
can be claimed from a run and what must be checked independently.

## 6. Publishing results

For a reproducible archive, publish:

- the exact source version;
- the complete JSON configuration;
- the master seed;
- `dead_rho*.npz` archives;
- `thermo_rho*.npz` archives;
- `summary.txt`;
- generated figures;
- the Python and dependency versions;
- convergence comparisons across at least one sampling and one finite-size
  parameter.

Do not treat the PNG figures alone as the primary data. The compressed
dead-point archives are the most important reusable scientific artifact.
