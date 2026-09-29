"""
Command-line interface for the LJ nested-sampling phase-diagram pipeline.

Usage is driven by a JSON configuration file (see ``config/*.json``).
The CLI accepts only a minimal set of operational overrides:

    python -m ljns.cli --config config/default.json [--cores N] [--seed N]
                          [--load] [--no-plots] [--no-per-density-plots]
                          [--densities ρ1 ρ2 ...] [--override key=val ...]

The ``--override`` flag allows ad-hoc tweaks to any config value
(e.g. ``--override T_min=0.5 --override n_T=1000``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .config import SimConfig, configure_logging, MAX_CORES
from .nested_sampling import run_ns, save_dead_points, load_dead_points
from .thermodynamics import analyse, find_cv_peak, save_results, load_results
from .parallel import run_density_sweep
from .plotting.overview import (
    plot_cv_all, plot_phase_diagram, plot_caloric, plot_entropy,
    plot_free_energy, plot_pressure, plot_compressibility,
    plot_energy_convergence, plot_cv_3d, plot_phase_diagram_Trho,
    plot_phase_diagram_Prho, plot_phase_diagram_PT,
)
from .plotting.single import plot_single_density


def _parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse ``--override key=val`` pairs into a dict with type coercion."""
    out = {}
    for item in overrides:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Override '{item}' must be in 'key=value' form")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try to coerce: int -> float -> bool -> string
        for caster in (int, float):
            try:
                out[k] = caster(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                out[k] = v
    return out


def _apply_overrides(cfg: SimConfig, overrides: Dict[str, Any]) -> SimConfig:
    """Apply override dict to a validated SimConfig (re-validates)."""
    d = cfg.to_dict()
    d.update(overrides)
    # Reconstruct and re-validate
    return SimConfig.from_json.__func__(  # type: ignore
        SimConfig,  # dummy, actually builds from dict below
    )  # will override below

    # Actually just build fresh from merged dict
    return SimConfig(**d)


def build_config_from_args(args: argparse.Namespace) -> SimConfig:
    """Load base JSON, apply CLI overrides, re-validate."""
    # Load base config
    cfg = SimConfig.from_json(args.config)

    # Apply --override
    if args.override:
        for k, v in _parse_overrides(args.override).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                logging.warning("Unknown override key '%s' ignored", k)

    # Operational CLI overrides (take precedence)
    if args.cores is not None:
        cfg.n_workers = min(args.cores, MAX_CORES)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.load is not None:
        cfg.save_dead_points = not args.load  # if --load, don't re-save
    if args.no_plots:
        cfg.make_plots = False
    if args.no_per_density_plots:
        cfg.per_density_plots = False
    if args.densities is not None:
        cfg.densities = [float(d) for d in args.densities]
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    # Final validation
    cfg.validate()
    return cfg


def _run_one_density_serial(
    rho: float, cfg: SimConfig, rng: np.random.Generator,
    skip_ns: bool, make_plots: bool,
) -> tuple:
    """Serial version of one density (used when n_workers=1)."""
    from .nested_sampling import DeadPoint

    print(f"\n{'='*60}")
    print(f"  Density ρ* = {rho:.3f}   L = {cfg.box_length(rho):.3f} σ")
    print(f"  r_cut_eff = {cfg.r_cut_eff(rho):.3f} σ")
    print(f"{'='*60}")

    # NS run or load
    dead_path = os.path.join(cfg.output_dir, f"dead_rho{rho:.4f}.npz")
    if skip_ns and os.path.exists(dead_path):
        print(f"  Loading dead points from {dead_path}")
        dead = load_dead_points(dead_path)
    else:
        dead, meta = run_ns(cfg, rho, rng)
        if cfg.save_dead_points:
            save_dead_points(dead, dead_path)
            print(f"  Saved → {dead_path}")

    # Thermo analysis
    t0 = time.time()
    results = analyse(dead, cfg.T_array, cfg)
    T_pk, Cv_pk, FWHM = find_cv_peak(results, sw=cfg.cv_smooth_window)
    print(f"  Analysis: {time.time()-t0:.2f}s  "
          f"T_peak={T_pk:.4f}  Cv_peak/N={Cv_pk:.3f}  FWHM≈{FWHM:.4f}")

    if cfg.save_thermo:
        res_path = os.path.join(cfg.output_dir, f"thermo_rho{rho:.4f}.npz")
        save_results(results, res_path)

    phase_info = {"T_peak": T_pk, "Cv_peak": Cv_pk, "FWHM": FWHM}

    # Per-density plot
    if make_plots and cfg.per_density_plots:
        path = plot_single_density(results, dead, cfg.output_dir)
        print(f"  Saved: {os.path.basename(path)}")

    return results, phase_info


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LJ Bulk Fluid T-ρ Phase Diagram via NVT Nested Sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON configuration file (see config/*.json)")
    parser.add_argument(
        "--cores", type=int, default=None,
        help=f"Number of worker processes (max {MAX_CORES}, silently clamped)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master RNG seed (overrides config)")
    parser.add_argument(
        "--load", action="store_true",
        help="Skip NS, reload saved dead points from output_dir")
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable all matplotlib output")
    parser.add_argument(
        "--no-per-density-plots", action="store_true",
        help="Disable per-density 6-panel figures (overview plots still made)")
    parser.add_argument(
        "--densities", nargs="+", type=float, default=None,
        help="Override density list (space-separated)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Ad-hoc config override: --override key=val (repeatable)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging")
    args = parser.parse_args(argv)

    # Load & validate config
    cfg = build_config_from_args(args)
    configure_logging("debug" if args.verbose else cfg.verbosity)

    # Print banner
    logging.info("=" * 60)
    logging.info("  LJ Bulk Fluid — T-ρ Phase Diagram via Nested Sampling")
    logging.info("  N = %d   K = %d   n_iter = %d   n_mc = %d",
                 cfg.n_atoms, cfg.n_walkers, cfg.n_iter, cfg.n_mc_steps)
    logging.info("  r_cut = %.1f σ   T ∈ [%.2f, %.2f]  (%d pts)",
                 cfg.r_cut, cfg.T_min, cfg.T_max, cfg.n_T)
    logging.info("  Densities (%d): %s", len(cfg.densities), cfg.densities)
    logging.info("  Workers: %d / %d max (machine has %d)",
                 cfg.n_workers, MAX_CORES, os.cpu_count())
    logging.info("  Output → %s/", cfg.output_dir)
    logging.info("=" * 60)

    cfg.ensure_output_dir()

    t_all = time.time()
    rng = np.random.default_rng(cfg.seed)

    # Run density sweep
    if cfg.n_workers == 1:
        # Serial path (useful for debugging / tiny runs)
        all_results = {}
        phase_data = {}
        all_dead = {}  # for energy convergence plot
        for rho in cfg.densities:
            results, pinfo = _run_one_density_serial(
                rho, cfg, rng, skip_ns=args.load, make_plots=cfg.make_plots)
            all_results[rho] = results
            phase_data[rho] = pinfo
            # Load dead points for energy convergence plot
            if cfg.make_plots:
                dead_path = os.path.join(cfg.output_dir, f"dead_rho{rho:.4f}.npz")
                if os.path.exists(dead_path):
                    all_dead[rho] = load_dead_points(dead_path)
    else:
        # Parallel path
        all_results, phase_data = run_density_sweep(cfg)
        # Load dead points for energy convergence plot
        all_dead = {}
        if cfg.make_plots:
            for rho in cfg.densities:
                dead_path = os.path.join(cfg.output_dir, f"dead_rho{rho:.4f}.npz")
                if os.path.exists(dead_path):
                    all_dead[rho] = load_dead_points(dead_path)

    # Overview plots
    if cfg.make_plots and len(all_results) > 1:
        print("\n  Generating overview figures ...")
        plot_cv_all(all_results, phase_data, cfg.output_dir)
        plot_phase_diagram(phase_data, cfg.output_dir)
        plot_caloric(all_results, phase_data, cfg.output_dir)
        plot_entropy(all_results, phase_data, cfg.output_dir)
        plot_free_energy(all_results, cfg.output_dir)
        plot_pressure(all_results, phase_data, cfg.output_dir)
        plot_compressibility(all_results, cfg.output_dir)
        # New plots
        if all_dead:
            plot_energy_convergence(all_dead, cfg.output_dir)
        plot_cv_3d(all_results, phase_data, cfg.output_dir)
        plot_phase_diagram_Trho(all_results, phase_data, cfg.output_dir)
        plot_phase_diagram_Prho(all_results, phase_data, cfg.output_dir)
        plot_phase_diagram_PT(all_results, phase_data, cfg.output_dir)

    # Summary table
    print(f"\n{'='*62}")
    print(f"  {'ρ*':>6}  {'L/σ':>6}  {'r_cut':>6}  "
          f"{'T_peak':>8}  {'Cv_peak/N':>10}  {'FWHM':>8}")
    print(f"  {'-'*56}")
    for rho in sorted(phase_data.keys()):
        info = phase_data[rho]
        L = cfg.box_length(rho)
        rc = cfg.r_cut_eff(rho)
        print(f"  {rho:>6.3f}  {L:>6.3f}  {rc:>6.3f}  "
              f"{info['T_peak']:>8.4f}  {info['Cv_peak']:>10.4f}  "
              f"{info.get('FWHM', 0):>8.4f}")
    print(f"{'='*62}")

    # Save summary.txt
    summary_path = os.path.join(cfg.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"LJ Bulk Fluid NS Phase Diagram   N={cfg.n_atoms}   "
                f"K={cfg.n_walkers}   n_iter={cfg.n_iter}\n")
        f.write(f"r_cut={cfg.r_cut}   T=[{cfg.T_min},{cfg.T_max}]\n\n")
        f.write(f"{'rho':>7}  {'T_peak':>9}  {'Cv_peak_per_N':>14}  {'FWHM':>8}\n")
        for rho in sorted(phase_data.keys()):
            info = phase_data[rho]
            f.write(f"{rho:>7.4f}  {info['T_peak']:>9.4f}  "
                    f"{info['Cv_peak']:>14.4f}  {info.get('FWHM', 0):>8.4f}\n")
        f.write("\nLiterature (r_cut=2.5):\n")
        f.write("  Triple point:  T*=0.694  rho*=0.845 (liquid)\n")
        f.write("  Critical point: T*=1.316  rho*=0.304  P*=0.128\n")

    print(f"\n  Summary saved → {summary_path}")
    print(f"  Total wall time: {(time.time()-t_all)/60:.1f} min\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())