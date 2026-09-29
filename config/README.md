# Configuration presets for the LJ nested-sampling phase diagram pipeline.
#
# Each JSON file contains a complete, self-contained simulation configuration.
# The CLI loads one of these files via `--config` and allows a handful of
# operational overrides (`--cores`, `--seed`, `--load`, `--no-plots`,
# `--override KEY=VAL`).
#
# - quick.json      : Smoke test (5 densities, N=32, K=80, n_iter=2000)
# - default.json    : Recommended production run (N=64, dense density grid)
# - N512_publication.json : High-cost publication/finite-size campaign (N=512)
#
# The configuration schema:
#   physics:    { n_atoms, epsilon, sigma, r_cut }
#   ns:         { n_walkers, n_iter, n_mc_steps, atom_step_size }
#   scan:       { densities[], T_min, T_max, n_T, cv_smooth_window }
#   parallel:   { n_workers, backend }
#   io:         { output_dir, save_dead_points, save_thermo, make_plots, per_density_plots }
#   seed:       int
#   verbosity:  "debug" | "info" | "warning" | "error"
#
# NOTE: n_workers is silently clamped to min(64, os.cpu_count()) by
# SimConfig.validate(). The shell launcher may pass a larger request, but the
# Python configuration remains authoritative.