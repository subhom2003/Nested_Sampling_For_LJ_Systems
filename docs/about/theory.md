# Theory

## 1. Physical model

The system contains `N` identical particles in a cubic periodic box of volume
`V = L^3` and number density

```text
rho = N / V,              L = (N / rho)^(1/3).
```

The pair interaction is the Lennard-Jones 12-6 potential

```text
u(r) = 4 epsilon [ (sigma/r)^12 - (sigma/r)^6 ].
```

The potential minimum is at `r = 2^(1/6) sigma`, where `u(r) = -epsilon`.
The implementation uses a truncated-and-shifted *range* in the sense that
only pairs with `r < r_cut` contribute. The code does not add a constant
energy shift at the cutoff; it instead adds analytic uniform-fluid tail
corrections for the omitted long-range contribution.

All calculations use reduced LJ variables:

```text
r* = r / sigma
E* = E / epsilon
T* = k_B T / epsilon
P* = P sigma^3 / epsilon.
```

The package sets `k_B = 1` numerically. Input `epsilon` and `sigma` remain
configurable, although the supplied presets use one for both.

## 2. Periodic boundary conditions

For a displacement `dr`, the minimum-image displacement is

```text
dr_min = dr - L floor(dr/L + 1/2).
```

This folds each Cartesian component into the primary periodic image. Pair
distances are evaluated from this displacement, and each unordered pair is
counted once. The cutoff must satisfy the minimum-image condition
`r_cut < L/2`; `SimConfig.r_cut_eff` lowers the requested cutoff by a small
margin when a density would violate this condition.

## 3. Energy and virial

For a pair, the configurational energy is `u(r)`. The pair virial used by the
code is

```text
w(r) = r du/dr
     = 24 epsilon [ 2 (sigma/r)^12 - (sigma/r)^6 ].
```

The potential module provides:

- `total_energy`: full `O(N^2)` configurational energy;
- `single_atom_energy`: the partial energy needed for an incremental trial
  move;
- `energy_and_virial`: one shared `O(N^2)` pass for both quantities.

The pressure estimator later uses the volume-weighted total virial `W`:

```text
P = [N T + W/3] / V.
```

The kinetic term is the ideal-gas contribution for three translational
degrees of freedom per particle.

## 4. Long-range tail corrections

The cutoff omits interactions beyond `r_c`. The implementation assumes a
uniform fluid there, equivalent to setting the pair correlation function to
one beyond the cutoff. With `s = sigma/r_c`, the analytic corrections are

```text
U_tail = (8 pi / 3) N rho epsilon [ s^9/3 - s^3 ]

(W_tail / V) = (16 pi / 3) rho^2 epsilon [ 2 s^9/3 - s^3 ].
```

The live-point energy includes `U_tail`. The dead-point virial stores

```text
W_dead = W_truncated + (W_tail / V) V.
```

This convention lets thermodynamic analysis use a single energy and virial
value per dead point. Tail corrections are an approximation and are least
trustworthy when structure beyond the cutoff is important.

## 5. Nested sampling

Nested sampling estimates integrals by progressively shrinking a prior volume.
For a configurational energy `E`, the canonical configurational partition
function at temperature `T` is proportional to

```text
Z_conf(T) = integral exp[-E(x)/T] dx.
```

Nested sampling replaces this high-dimensional integral with weighted
configurations ordered by decreasing likelihood. For this code the likelihood
is the Boltzmann factor `exp(-E/T)`, while the nested-sampling run itself is
temperature-independent.

At fixed density:

1. Start with `K = n_walkers` live configurations.
2. Find the live point with the largest energy. It has the smallest
   Boltzmann likelihood and is the next point to remove.
3. Assign it a prior-volume weight.
4. Select a random surviving donor and constrained-walk a clone until it
   remains below the discarded energy.
5. Continue for `n_iter` iterations.
6. Give the final `K` live points equal shares of the remaining volume.

The deterministic compression approximation used by the implementation is

```text
alpha = K / (K + 1)
log(w_k) = k log(alpha) + log[1/(K+1)].
```

The final live points use

```text
log(w_final) = n_iter log(alpha) - log(K).
```

The first `n_iter` entries are the free dead points; the last `K` entries are
the surviving live points. Because the worst point is removed at every
iteration, the free dead-point energies should be non-increasing.

The standard stochastic nested-sampling uncertainty in the explored prior
volume decreases approximately with `K^(-1/2)` in `ln Z`, but finite-MC mixing,
finite iteration count, and correlated replacement walks add practical error
sources.

## 6. Constrained Metropolis walker

The live-point walker samples configuration space under the hard constraint

```text
E_new < E_ceil.
```

For one randomly selected atom, it proposes a displacement with each
Cartesian component sampled uniformly from a symmetric interval of width
`atom_step_size`. The proposed coordinate is wrapped modulo `L`.

The partial old and new energies of that atom give the total-energy change
`dE`. A move is rejected immediately if the new total energy reaches the
ceiling. Otherwise it uses the Metropolis rule

```text
accept = 1                         if dE <= 0
accept = exp(-dE)                  if dE > 0, probabilistically.
```

Each nested-sampling replacement performs
`n_mc_steps * n_atoms` trial moves. The input donor is copied, including a
deep copy of its positions, so a walk cannot mutate the survivor held in the
live population.

Initial live points start from an FCC lattice whenever `N = 4 m^3`; other
sizes use a simple-cubic fallback. Small Gaussian position noise breaks exact
crystal symmetry, followed by an unconstrained `5N`-move burn-in.

## 7. Thermodynamic re-analysis

For dead points indexed by `i`, with energies `E_i` and log weights
`log(w_i)`, the code evaluates at each temperature:

```text
log b_i(T) = log(w_i) - E_i/T
log Z_conf(T) = log sum_i exp[log b_i(T)].
```

The calculation uses a stable log-sum-exp transformation. Normalized weights
are

```text
p_i(T) = exp[log b_i(T) - log Z_conf(T)].
```

Weighted moments are then

```text
<U>   = sum_i p_i E_i
<U^2> = sum_i p_i E_i^2
<W>   = sum_i p_i W_i.
```

The configurational heat capacity is obtained from energy fluctuations:

```text
C_v,conf = [<U^2> - <U>^2] / T^2.
```

The total heat capacity includes the ideal-gas kinetic term:

```text
C_v,total = C_v,conf + 3N/2,
Cv_per_atom = C_v,total / N.
```

The configurational and total free energies are

```text
F_conf = -T log Z_conf

F_kin = -T N [ 3/2 log(2 pi T) + log(V/N) + 1 ]

F_total = F_conf + F_kin.
```

Entropy follows from `S = (U - F)/T`, separately for configurational and
total quantities. The code also returns

```text
G = F + P V
Z_factor = P V / (N T).
```

The temperature grid is generated once by
`np.linspace(T_min, T_max, n_T)`. Because dead points retain their weights,
changing this grid does not require repeating nested sampling.

## 8. Heat-capacity peak and phase proxy

For each density, `find_cv_peak` smooths `Cv_per_atom` only for peak
localization, finds the maximum, and applies a local parabolic interpolation
for sub-grid `T_peak`. The reported peak height is interpolated from the
unsmoothed curve. A full width at half maximum (FWHM) is also estimated from
the smoothed curve.

The density-temperature curve formed by `T_peak(rho)` is an operational
transition or coexistence-boundary proxy. It should not be interpreted as a
complete thermodynamic coexistence construction without additional phase
equilibrium analysis and convergence studies.
